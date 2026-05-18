import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def _ensure_model_dir():
    model_folder_path = "./model"
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    return model_folder_path


class Basic_Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="Basic_Linear_QNet_model.pth"):
        model_folder_path = _ensure_model_dir()
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Linear_QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.net(x)

    def save(self, file_name="Linear_QNet_model.pth"):
        model_folder_path = _ensure_model_dir()
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """
    Legacy DQN-style trainer kept for the current Q-learning pipeline.
    """

    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # e.g. a single sample shaped (6, 9, 9) becomes (1, 6, 9, 9).
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # The linear Q-network expects flattened board planes.
        state = state.flatten(start_dim=1)
        next_state = next_state.flatten(start_dim=1)

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            q_new = reward[idx].item()
            if not done[idx]:
                q_new = (
                    reward[idx].item()
                    + self.gamma
                    * torch.max(self.model(next_state[idx].unsqueeze(0))).item()
                )

            target[idx][action[idx].item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class ConvBlock(nn.Module):
    """
    Small reusable convolutional block used by the policy-value network.

    It is intentionally simple so the code stays easy to understand before
    moving to deeper residual architectures.
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
    - policy logits over all legal/illegal action slots
    - scalar value in [-1, 1]

    Unlike the old Q-network, this model does not estimate Q(s, a).
    It learns:
    - which actions look promising now (policy head)
    - how good the whole position is (value head)
    """

    def __init__(
        self,
        input_channels=6,
        board_size=9,
        num_actions=144,
        num_filters=64,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.board_size = board_size
        self.num_actions = num_actions

        # Shared spatial trunk.
        # e.g. the network can learn patterns like "pawn near goal" or
        # "wall corridor blocking a path" before branching into the two heads.
        self.trunk = nn.Sequential(
            ConvBlock(input_channels, num_filters),
            ConvBlock(num_filters, num_filters),
            ConvBlock(num_filters, num_filters),
        )

        # Policy head:
        # returns one logit per action index, later masked by the game logic.
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, num_actions),
        )

        # Value head:
        # returns a single scalar in [-1, 1] using tanh.
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
        """
        Input:
            x: (batch, channels, board_size, board_size)

        Output:
            policy_logits: (batch, num_actions)
            value: (batch, 1)
        """
        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value

    def predict(self, state):
        """
        Convenience helper for inference.

        e.g. given one canonical Quoridor state shaped (6, 9, 9), returns:
        - policy probabilities over 144 actions
        - scalar position value
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            policy_logits, value = self(state_tensor)
            policy = torch.softmax(policy_logits, dim=1)
            return policy.squeeze(0), value.squeeze(0).item()

    def save(self, file_name="PolicyValueNet_model.pth"):
        model_folder_path = _ensure_model_dir()
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class AlphaZeroTrainer:
    """
    Trainer for policy-value learning.

    Expected training targets:
    - target_policy: probability distribution from MCTS visit counts
      e.g. shape (batch, 144)
    - target_value: final game result from the current player's perspective
      e.g. +1 for a win, -1 for a loss, 0 for a draw
    """

    def __init__(self, model, lr, weight_decay=1e-4, value_loss_weight=1.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.value_loss_weight = value_loss_weight
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.value_criterion = nn.MSELoss()

    def _prepare_batch(self, state, target_policy, target_value):
        state = torch.tensor(state, dtype=torch.float)
        target_policy = torch.tensor(target_policy, dtype=torch.float)
        target_value = torch.tensor(target_value, dtype=torch.float)

        # e.g. one training example:
        # state         (6, 9, 9)   -> (1, 6, 9, 9)
        # target_policy (144,)      -> (1, 144)
        # target_value  scalar      -> (1, 1)
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
            target_policy = target_policy.unsqueeze(0)
            target_value = target_value.unsqueeze(0)

        if len(target_value.shape) == 1:
            target_value = target_value.unsqueeze(1)

        return state, target_policy, target_value

    def _policy_loss(self, policy_logits, target_policy):
        """
        Cross-entropy with soft targets from MCTS.

        AlphaZero does not train on a single "correct move".
        It trains on the whole visit distribution:
        e.g. if MCTS visits action 10 fifty times and action 25 forty times,
        both actions influence the target policy.
        """
        log_probs = torch.log_softmax(policy_logits, dim=1)
        return -(target_policy * log_probs).sum(dim=1).mean()

    def train_step(self, state, target_policy, target_value):
        state, target_policy, target_value = self._prepare_batch(
            state, target_policy, target_value
        )

        policy_logits, predicted_value = self.model(state)

        policy_loss = self._policy_loss(policy_logits, target_policy)
        value_loss = self.value_criterion(predicted_value, target_value)
        total_loss = policy_loss + self.value_loss_weight * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }

    def train_batch(self, states, target_policies, target_values):
        """
        Alias kept for readability when the caller already knows it is passing
        a batch of self-play examples.
        """
        return self.train_step(states, target_policies, target_values)
