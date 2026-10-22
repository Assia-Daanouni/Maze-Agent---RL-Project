import torch
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, obs_dim=9, n_actions=4, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x, hidden=None):
        # Si x a 2 dimensions (batch, features), on ajoute une dimension temporelle
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        lstm_out, hidden = self.lstm(x, hidden)
        # lstm_out est (batch, seq_len, hidden)
        return self.fc(lstm_out[:, -1, :]), hidden


class DQNAgent:
    def __init__(self, obs_dim=9, n_actions=4,
                 lr=1e-3, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=20000,
                 target_update_freq=500,
                 seq_len=8, hidden=128):

        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        self.seq_len = seq_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state_seq):
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              np.exp(-self.steps_done / self.eps_decay)
        self.steps_done += 1

        if np.random.random() < eps:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            t = torch.tensor(state_seq, dtype=torch.float32,
                           device=self.device).unsqueeze(0)
            q_vals, _ = self.q_net(t)
            return q_vals.argmax(dim=1).item()

    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            replay_buffer.sample(batch_size)

        s  = torch.tensor(states,      dtype=torch.float32, device=self.device)
        a  = torch.tensor(actions,     dtype=torch.int64,   device=self.device)
        r  = torch.tensor(rewards,     dtype=torch.float32, device=self.device)
        ns = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        d  = torch.tensor(dones,       dtype=torch.float32, device=self.device)

        q_vals, _ = self.q_net(s)
        q_vals = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q, _ = self.target_net(ns)
            targets = r + self.gamma * next_q.max(dim=1).values * (1 - d)

        loss = self.loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
    