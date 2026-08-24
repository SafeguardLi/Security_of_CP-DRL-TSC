import numpy as np
from src.rlagent import RLAgent
from src.jsma import init_attack


class PressLightAgent(RLAgent):
    def __init__(self, networks, epsilon, n_actions, n_steps, n_batch, gamma, mode,
                 updates, main_args, tsc_id):
        super().__init__(networks, epsilon, n_actions, n_steps, n_batch, gamma, mode, updates)
        self.main_args = main_args
        self.tsc_id = tsc_id
        # white-box JSMA state (lazy-initialized on first get_advX, like A2CAgent)
        self.attack_flag = False
        self.jsma = None
        self.classifier = None

    def get_action(self, state, epsilon=1e-5, surrogate_act=False):
        """Return (greedy action, raw Q-values).

        DQN acts greedily (argmax Q). We return the RAW Q-vector — NOT softmax(Q) —
        because a DQN has no calibrated action distribution (Q scale is arbitrary). The
        attacker's impact reward for a DQN victim is computed from the Q-margin directly
        (see NextPhasePressLightAttackTSC._step_impact). `surrogate_act` is accepted for
        interface parity with A2CAgent.get_action and ignored (white-box attack always
        queries the real DQN).
        """
        self.epsilon = epsilon
        q_vals = self.networks.forward(state[np.newaxis, ...], 'online')[0]
        if np.random.uniform(0.0, 1.0) < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = int(np.argmax(q_vals))
        return action, np.asarray(q_vals, dtype=np.float64)

    def init_attacker(self, input_dim=None, classifier_path=None):
        if self.attack_flag:
            return  # idempotent — only initialize once
        self.jsma_params = {
            "theta": 1.0, "gamma": 0.1, "clip_min": 0.0, "clip_max": 1.0, "y_target": None,
        }
        if input_dim is None:
            input_dim = 20  # phase-based presslight default (P=4)
        # Correctness: the DQN victim MUST hold trained weights (white-box attack on a
        # random-init model is meaningless).
        if not getattr(self.main_args, 'load', False):
            raise RuntimeError(
                f"White-box JSMA on PressLight TSC {self.tsc_id}: victim DQN is not loaded "
                f"(args.load is False). Run with -load -tsc_updates <N>.")
        # JSMA may select the ATTACKABLE inc AND out blocks of the phase-based presslight
        # state: state = inc(A) + out(P) + phase_one_hot(P+1) + time(1), A=(P-1)*s+1, dim=5P.
        # inc [0:A] = fake incoming CVs; out [A:A+P] = fake DOWNSTREAM CVs (out-injection).
        # phase/time are not spoofable. Recover A,P from input_dim.
        s = int(getattr(self.main_args, 'num_segments', 3))
        P = max(1, int(round(input_dim / 5.0)))
        A = (P - 1) * s + 1
        # -inc_only_attack: restrict JSMA to the INCOMING block [0:A] only (spoof approaching CVs),
        # excluding the OUT block [A:A+P] (downstream/departure-lane spoofing). Conservative threat
        # model; ablation to quantify how much the out-injection contributes to the attack.
        if getattr(self.main_args, 'inc_only_attack', False):
            feature_range = (0, A)
        else:
            feature_range = (0, A + P)
        sur_dir = getattr(self.main_args, 'surrogate_dir', None)
        if sur_dir:
            # BLACKBOX: JSMA selects features from the trained per-intersection BinaryClassifier
            # SURROGATE (not the real DQN). The real TSC still decides via the victim; only the
            # feature choice comes from the surrogate gradient. See train_surrogate_presslight.py.
            import os
            cpath = os.path.join(sur_dir, f"{self.tsc_id}_surrogate.pt")
            print(f"[attacker init] TSC {self.tsc_id}: BLACKBOX surrogate={cpath}  feature_range={feature_range}")
            self.jsma, self.classifier = init_attack(
                None, self.jsma_params, input_dim=input_dim, white_box=False,
                classifier_path=cpath, feature_range=feature_range)
        else:
            # WHITE-BOX: JSMA on the real DQN victim (softmax head via recompile_loss).
            print(f"[attacker init] TSC {self.tsc_id}: WHITE-BOX feature_range={feature_range} "
                  f"({'INC-ONLY' if feature_range[1]==A else 'inc+out'}; A={A} P={P})")
            self.jsma, self.classifier = init_attack(
                self.networks, self.jsma_params, input_dim=input_dim, white_box=True,
                feature_range=feature_range, recompile_loss='categorical_crossentropy')
        self.attack_flag = True

    def get_advX(self, state, curr_phase, att_action):
        state_cp = state.copy()
        target_action = att_action[0]
        if not self.attack_flag:
            # lazy init: use the actual victim-state dim (handles P=4 dim20 vs P=2 dim10)
            self.init_attacker(input_dim=state_cp.shape[-1])
        one_hot_target = np.zeros((1, self.n_actions), dtype=np.float32)
        one_hot_target[0, target_action] = 1
        self.jsma_params["y_target"] = one_hot_target
        adv_x, feature_ids = self.jsma.generate(x=state_cp[np.newaxis, ...], y=one_hot_target)
        feature_ids = np.array(feature_ids).reshape(-1)
        if feature_ids is None:
            feature_ids = []
        return adv_x, feature_ids, target_action
