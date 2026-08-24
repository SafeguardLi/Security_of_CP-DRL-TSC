import numpy as np
from collections import deque

from src.trafficsignalcontroller import TrafficSignalController


class NextPhasePressLightTSC(TrafficSignalController):
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, green_t, g_max,
                 rlagent, tsc_type, epsilon, eps_min, eps_factor,
                 num_segments, detect_r, args):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t, detect_r)
        self.green_t = green_t
        self.g_max = g_max
        self.green_t_cnt = 10
        self.rlagent = rlagent
        self.tsc_type = tsc_type
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_factor = eps_factor
        self.num_segments = num_segments
        self.args = args

        self.phase_deque = deque()
        self.delay_green = False
        self.acting = False
        self.curr_phase_idx = 0

        self.phase_to_one_hot = self.input_to_one_hot(self.green_phases + [self.all_red])
        self.int_to_phase = self.int_to_input(self.green_phases)

        self.state_action_record = []
        self.sa_collect = []   # (state, action, phase, q_vals) for surrogate training (-collect_sa)
        self.data = None

    def update(self, data, cv_data, uv_data, mask):
        self.data = data
        self.cv_data = cv_data
        self.uv_data = uv_data

    def next_phase(self):
        if len(self.phase_deque) == 0:
            next_phase = self.get_next_phase()
            phases = self.get_intermediate_phases(self.phase, next_phase)
            self.phase_deque.extend(phases + [next_phase])
        return self.phase_deque.popleft()

    def next_phase_duration(self, current_phase):
        if self.phase in self.green_phases:
            if self.phase == current_phase:
                return self.green_t_cnt
            else:
                return self.green_t
        elif 'y' in self.phase:
            return self.yellow_t
        else:
            return self.red_t

    def proceed_phase(self):
        self.curr_phase_idx += 1
        if self.curr_phase_idx >= len(self.green_phases):
            self.curr_phase_idx = 0

    def get_next_phase(self):
        if self.phase == self.all_red and not self.delay_green:
            self.delay_green = True
            self.acting = False
            return self.all_red
        self.delay_green = False

        # PHASE-BASED pressure state (matches the retrained victim + get_state('presslight')):
        #   [ inc CV per phase (segmented), out CV per phase, phase one-hot, time ].
        # No attacker in benign runs, so self.fake_traj_dict stays empty -> inc = real CVs.
        state = np.concatenate([
            self.get_state('presslight', num_segments=self.num_segments),
            self.phase_to_one_hot[self.phase],
            np.array([self.phase_duration / self.g_max])
        ])

        action_idx, q_vals = self.rlagent.get_action(state, self.epsilon)

        if (self.phase_duration > self.g_max) and (action_idx == 1):
            self.proceed_phase()
            next_phase = self.int_to_phase[self.curr_phase_idx]
        elif action_idx == 1:
            next_phase = self.int_to_phase[self.curr_phase_idx]
        else:
            self.proceed_phase()
            next_phase = self.int_to_phase[self.curr_phase_idx]

        self.state_action_record.append([state, action_idx, q_vals])
        # BLACKBOX surrogate data collection (-collect_sa, benign): log the CLEAN (state -> action)
        # pair. Same 4-tuple format as the attack controller so train_surrogate_presslight.py reads it.
        if getattr(self.args, 'collect_sa', False):
            self.sa_collect.append((np.asarray(state, dtype=np.float32),
                                    int(action_idx),
                                    int(self.curr_phase_idx),
                                    np.asarray(q_vals, dtype=np.float32)))
        self.acting = True
        return next_phase
