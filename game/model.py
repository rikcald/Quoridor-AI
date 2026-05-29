import os

import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_model_dir():
    model_folder_path = "./model"
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    return model_folder_path


class ConvBlock(nn.Module):
    """
    Small convolutional block used by the policy-value network.

    The goal here is clarity, not maximum sophistication.
    Later, if needed, this can evolve into a deeper residual architecture.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PolicyValueNet(nn.Module):
    """
    AlphaZero-style network with two outputs:
    - policy logits over the full action space
    - scalar value in [-1, 1]

    Input:
        (batch, 6, 9, 9)

    Output:
        policy_logits: (batch, 144)
        value: (batch, 1)
    """

    def __init__(
        self,
        input_channels=6,
        board_size=9,
        num_actions=144,
        num_filters=64,
    ):
        super().__init__()
        self.board_size = board_size
        self.num_actions = num_actions

        # Shared trunk:
        # learns spatial features such as pawn progress, wall patterns, and bottlenecks.
        self.trunk = nn.Sequential(
            ConvBlock(input_channels, num_filters),
            ConvBlock(num_filters, num_filters),
            ConvBlock(num_filters, num_filters),
        )

        # Policy head:
        # returns one score per action index.
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, num_actions),
        )

        # Value head:
        # returns one scalar in [-1, 1].
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value

    def predict(self, state):
        """
        Convenience helper for inference on one canonical state.

        Returns:
        - policy probabilities over the full action space
        - scalar value
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            state_tensor = torch.tensor(
                state,
                dtype=torch.float,
                device=device,
            ).unsqueeze(0)
            policy_logits, value = self(state_tensor)
            policy = torch.softmax(policy_logits, dim=1)
            return policy.squeeze(0), value.squeeze(0).item()

    def predict_batch(self, states):
        """
        Batched inference for multiple canonical states.

        Returns:
        - policy probabilities over the full action space, shape (batch, num_actions)
        - scalar values, shape (batch,)
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            states_tensor = torch.tensor(
                states,
                dtype=torch.float,
                device=device,
            )

            if len(states_tensor.shape) == 3:
                states_tensor = states_tensor.unsqueeze(0)

            policy_logits, values = self(states_tensor)
            policies = torch.softmax(policy_logits, dim=1)
            return policies, values.squeeze(1)

    def save(self, file_name="PolicyValueNet_model.pth"):
        model_folder_path = _ensure_model_dir()
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class AlphaZeroTrainer:
    """
    Trainer for policy-value learning.

    Expected targets:
    - target_policy: probability distribution from MCTS visit counts
    - target_value: final game result from the current player's perspective
    """

    def __init__(self, model, lr, weight_decay=1e-4, value_loss_weight=1.0):
        self.device = DEVICE
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.value_loss_weight = value_loss_weight
        self.value_criterion = nn.MSELoss()

    def _prepare_batch(self, states, target_policies, target_values):
        states = torch.tensor(states, dtype=torch.float, device=self.device)
        target_policies = torch.tensor(
            target_policies,
            dtype=torch.float,
            device=self.device,
        )
        target_values = torch.tensor(
            target_values,
            dtype=torch.float,
            device=self.device,
        )

        # e.g. one example:
        # state         (6, 9, 9)  -> (1, 6, 9, 9)
        # target_policy (144,)     -> (1, 144)
        # target_value  scalar     -> (1, 1)
        if len(states.shape) == 3:
            states = states.unsqueeze(0)
            target_policies = target_policies.unsqueeze(0)
            target_values = target_values.unsqueeze(0)

        if len(target_values.shape) == 1:
            target_values = target_values.unsqueeze(1)

        return states, target_policies, target_values

    def _policy_loss(self, policy_logits, target_policy):
        """
        Cross-entropy with soft targets from MCTS.

        AlphaZero does not train on one "correct move".
        It trains on the whole root visit distribution.
        """
        log_probs = torch.log_softmax(policy_logits, dim=1)
        return -(target_policy * log_probs).sum(dim=1).mean()

    def train_step(self, states, target_policies, target_values):
        self.model.train()
        states, target_policies, target_values = self._prepare_batch(
            states, target_policies, target_values
        )

        policy_logits, predicted_values = self.model(states)

        policy_loss = self._policy_loss(policy_logits, target_policies)
        value_loss = self.value_criterion(predicted_values, target_values)
        total_loss = policy_loss + self.value_loss_weight * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }
