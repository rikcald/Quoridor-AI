import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="Linear_QNet_model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Se è singolo sample, aggiungi batch dimension e.g. (7, size, size) -> (1, 7, size, size) dove 1 è la batch dimension
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)  # Convert done to a tuple if it's a single boolean

        # state, next_state vanno flattenati da (1, 7, size, size) a (1*7*size*size) per essere compatibili con l'input del modello
        # (bs, 7, size, size) a (bs*7*size*size) nel caso in cui bs > 1
        state = state.flatten(start_dim=1)
        next_state = next_state.flatten(start_dim=1)

        # 1. Predict Q values for current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx].item()
            if not done[idx]:
                # Get max Q value for next state
                Q_new = (
                    reward[idx].item()
                    + self.gamma
                    * torch.max(self.model(next_state[idx].unsqueeze(0))).item()
                )

            # Update target Q value for the action taken
            target[idx][action[idx].item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)  # MSELoss between target and pred
        loss.backward()
        self.optimizer.step()
