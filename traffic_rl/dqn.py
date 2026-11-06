"""Double + Dueling DQN with the usual stabilisers:
replay buffer, soft-updated target network, Huber loss, gradient clipping."""
import numpy as np
import torch
import torch.nn as nn


class DuelingQ(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.v = nn.Linear(hidden, 1)
        self.a = nn.Linear(hidden, n_actions)

    def forward(self, x):
        h = self.body(x)
        a = self.a(h)
        return self.v(h) + a - a.mean(dim=1, keepdim=True)


class Replay:
    def __init__(self, cap, obs_dim):
        self.cap, self.i, self.n = cap, 0, 0
        self.s = np.zeros((cap, obs_dim), np.float32)
        self.s2 = np.zeros((cap, obs_dim), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)

    def add(self, s, a, r, s2, d):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j] = s, a, r, s2, d
        self.i = (j + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b, rng):
        idx = rng.integers(0, self.n, size=b)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.d[idx]))


class DQNAgent:
    def __init__(self, obs_dim, n_actions, seed=0, lr=5e-4, gamma=0.95, tau=0.005,
                 batch=128, buffer=200_000, warmup=2_000):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.n_actions, self.gamma, self.tau, self.batch, self.warmup = n_actions, gamma, tau, batch, warmup
        self.q = DuelingQ(obs_dim, n_actions)
        self.q_tgt = DuelingQ(obs_dim, n_actions)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = Replay(buffer, obs_dim)

    @torch.no_grad()
    def act(self, obs, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return int(self.q(torch.from_numpy(obs)[None]).argmax(1).item())

    def update(self):
        if self.buf.n < self.warmup:
            return None
        s, a, r, s2, d = self.buf.sample(self.batch, self.rng)
        with torch.no_grad():
            a2 = self.q(s2).argmax(1, keepdim=True)                # Double DQN: online selects
            y = r + self.gamma * (1 - d) * self.q_tgt(s2).gather(1, a2).squeeze(1)  # target evaluates
        q = self.q(s).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_tgt.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
        return loss.item()

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path))
        self.q_tgt.load_state_dict(self.q.state_dict())

    def policy(self):
        return lambda obs, env: self.act(obs, 0.0)
