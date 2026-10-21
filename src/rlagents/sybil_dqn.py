"""Sybil-DQN benchmark attacker (Arabi et al., CSR 2021), DISTRIBUTED version.

Rides the repo's attacker multiprocessing pipeline: N actor procs collect (s,a,r,s',done)
transitions into a SHARED replay (manager.list); one learner proc samples the shared replay,
runs DQN updates, and broadcasts weights back via rl_stats. Implemented in PyTorch (the victim's
TF stack runs with eager disabled, which breaks TF2 Keras).

Interface matches PPOAgent (so att_factory / rl_att_factory / learnerproc reuse it):
  get_action, store_experience, sample_replay, clip_exp_replay, train_batch, get_weights, set_weights

Network: 6 -> 16 -> 16 -> n_actions (linear Q-head). gamma=0.95, lr=0.02, eps=0.1.
Action/state/reward are built by src/attackers/sybil_attacker.py.
"""
import random
import numpy as np
import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, state_d, n_actions, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class SybilDQNAgent:
    """Drop-in for PPOAgent in the attacker pipeline. One instance per actor (collect) and one in
    the learner (train); they share `exp_replay` (manager.list) and `rl_stats` (manager.dict)."""

    def __init__(self, neural_network=None, epsilon=0.1, exp_replay=None, n_actions=19,
                 n_steps=1, n_batch=32, n_exp_replay=512, gamma=0.95, rl_stats=None,
                 mode='train', updates=5000, main_args=None, tsc_id=None):
        self.exp_replay = exp_replay          # SHARED manager.list of (s,a,r,s2,done)
        self.rl_stats = rl_stats              # SHARED manager.dict: online/updates/n_exp
        self.mode = mode
        self.updates = updates
        self.tsc_id = tsc_id
        self.main_args = main_args
        self.gamma = getattr(main_args, 'gamma', gamma) if main_args else gamma
        self.gamma = 0.95                     # DQN discount (paper); victim gamma is separate
        self.batch_size = n_batch
        self.state_d = 6
        self.n_actions = int(n_actions[0]) if isinstance(n_actions, (list, tuple)) else int(n_actions)
        self.epsilon = getattr(main_args, 'sybil_eps', 0.1) if main_args else 0.1
        lr = getattr(main_args, 'sybil_lr', 0.02) if main_args else 0.02
        opt = getattr(main_args, 'sybil_opt', 'adam') if main_args else 'adam'
        self.replay_ratio = getattr(main_args, 'drqn_replay_ratio', 4) if main_args else 4
        self.replay_cap = getattr(main_args, 'sybil_replay_cap', 50000) if main_args else 50000
        self.target_freq = 50
        self.device = torch.device('cpu')

        self.online = _MLP(self.state_d, self.n_actions).to(self.device)
        self.target = _MLP(self.state_d, self.n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.opt = (torch.optim.SGD if str(opt).lower() == 'sgd' else torch.optim.Adam)(
            self.online.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self._gstep = 0
        # actor: adopt the learner's broadcast weights if present
        _synced = False
        try:
            if rl_stats is not None and rl_stats.get('online') is not None:
                w = rl_stats['online'].get('dqn')
                if w:
                    self.set_weights(w); _synced = True
        except Exception:
            pass
        # TEST / standalone (no learner to broadcast): load the trained checkpoint from disk,
        # else the attacker runs on RANDOM weights. Path mirrors the learner's save location.
        if (not _synced) and main_args is not None and getattr(main_args, 'mode', 'train') == 'test':
            try:
                from src.helper_funcs import get_fp
                self.load_weights(get_fp(main_args, 'saved_models/sybil/' + str(tsc_id), False))
            except Exception as e:
                print('[SybilDQN] WARNING test-mode disk load failed -> RANDOM weights:', e)

    # ---- acting (actor) ----------------------------------------------------
    def get_action(self, state, epsilon=None, curr_phase=None):
        """epsilon-greedy. Signature matches PPOAgent.get_action(state, epsilon, curr_phase).
        Returns the action index (int)."""
        eps = self.epsilon if epsilon is None else epsilon
        if self.mode == 'train' and random.random() < eps:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            q = self.online(torch.as_tensor(np.asarray(state, np.float32)).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    # ---- experience (actor -> shared replay) -------------------------------
    def store_experience(self, s, a, r, s2, done):
        if self.exp_replay is None or (self.rl_stats is not None and
                                       self.rl_stats.get('updates', 0) >= self.updates):
            return
        self.exp_replay.append((np.asarray(s, np.float32), int(a), float(r),
                                np.asarray(s2, np.float32), float(done)))
        if self.rl_stats is not None:
            self.rl_stats['n_exp'] = self.rl_stats.get('n_exp', 0) + 1

    # ---- replay (learner) --------------------------------------------------
    def sample_replay(self):
        # return a random minibatch (not the full buffer) -> O(batch) IPC, not O(replay)
        if self.exp_replay is None:
            return []
        n = len(self.exp_replay)
        if n == 0:
            return []
        idx = random.sample(range(n), min(self.batch_size, n))
        return [self.exp_replay[i] for i in idx]

    def clip_exp_replay(self):
        # DQN is OFF-POLICY: keep a persistent ROLLING replay (do NOT discard like PPO on-policy).
        # Trim oldest transitions only when over capacity. This decouples collection from updates
        # (otherwise every update would need `nreplay` freshly-collected transitions -> ~550 episodes).
        if self.exp_replay is not None:
            over = len(self.exp_replay) - self.replay_cap
            if over > 0:
                del self.exp_replay[:over]

    # ---- training (learner) ------------------------------------------------
    def train_batch(self, sample_batch):
        """DQN updates over the collected batch: replay_ratio passes of minibatch SGD."""
        data = [b for b in sample_batch if b is not None]
        if len(data) < min(self.batch_size, 4):
            return
        for _ in range(self.replay_ratio):
            batch = data if len(data) <= self.batch_size else random.sample(data, self.batch_size)
            s = torch.as_tensor(np.stack([b[0] for b in batch]))
            a = torch.as_tensor(np.array([b[1] for b in batch]), dtype=torch.long).unsqueeze(1)
            r = torch.as_tensor(np.array([b[2] for b in batch], np.float32))
            s2 = torch.as_tensor(np.stack([b[3] for b in batch]))
            d = torch.as_tensor(np.array([b[4] for b in batch], np.float32))
            q = self.online(s).gather(1, a).squeeze(1)
            with torch.no_grad():
                q_next = self.target(s2).max(dim=1).values
                y = r + (1.0 - d) * self.gamma * q_next
            loss = self.loss_fn(q, y)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            self._gstep += 1
            if self._gstep % self.target_freq == 0:
                self.target.load_state_dict(self.online.state_dict())
        if self.rl_stats is not None:
            self.rl_stats['updates'] = self.rl_stats.get('updates', 0) + 1
            self.rl_stats['last_loss'] = float(loss.item())
            if 'online' in self.rl_stats:
                self.rl_stats['online']['dqn'] = self.get_weights()   # broadcast to actors

    # ---- weight sync (learner <-> actor, via rl_stats numpy arrays) --------
    def get_weights(self):
        return {k: v.cpu().numpy() for k, v in self.online.state_dict().items()}

    def set_weights(self, w):
        if not w:
            return
        sd = {k: torch.as_tensor(v) for k, v in w.items()}
        self.online.load_state_dict(sd); self.target.load_state_dict(sd)

    # ---- io ----------------------------------------------------------------
    def save_weights(self, path):
        import os
        p = path if path.endswith('.pt') else path + '.pt'
        os.makedirs(os.path.dirname(p), exist_ok=True)
        torch.save(self.online.state_dict(), p)

    def load_weights(self, path):
        import os
        p = path if path.endswith('.pt') else path + '.pt'
        if os.path.exists(p):
            sd = torch.load(p, map_location='cpu')
            self.online.load_state_dict(sd); self.target.load_state_dict(sd)
            print(f'[SybilDQN] loaded weights from {p}')
