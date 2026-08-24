import random
import numpy as np
from itertools import cycle
from collections import deque
import traci

from src.trafficsignalattacker import TrafficSignalAttacker

from src.jsma import init_attack

class NextPhaseAttacker(TrafficSignalAttacker):
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, green_t,g_max, rlagent, 
                 tsc_type, epsilon, eps_min, eps_factor, num_segments, detect_r, sync, all_veh_r, args):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t, detect_r)
        self.num_segments = num_segments 
        self.green_t = green_t 
        self.g_max = g_max
        self.t = 0
        
        self.phase_deque = deque()
        self.data = None
        self.delay_green = False
        self.phase_to_one_hot = self.input_to_one_hot(self.green_phases+[self.all_red])
        self.int_to_phase = self.int_to_input(self.green_phases)
        self.rlagent = rlagent
        self.tsc_type = tsc_type 
        self.acting = False
        self.action_first=True
        self.s = None
        self.a = None
        self.a_dist = None
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_factor = eps_factor
        self.sync = sync
        assert self.epsilon >= self.eps_min, 'epsilon can not be smaller than eps_min'
        
        if self.tsc_type in ['cavlight']:
            self.state_action_record = []
            self.state_comparison = []
        else:
            self.state_action_record = None

        self.all_veh_r = all_veh_r

        self.args = args
        self.max_attack_scale = self.args.max_attack_scale

    def get_attacked_state(self, tsc_state, att_action, norm_CV, norm_CV_spd, fake_vehicle_num, OPT = False, jsma_features = None):
        # if att_action != None:
        #     target_action, action_dist, old_prob_1, action_rule, is_policy_action = att_action
        
        # if jsma_features is None:
        #     return tsc_state

        # elif len(jsma_features) >0:
        #     attacked_state = self.get_fake_veh_jsma_guide(jsma_features, tsc_state, norm_CV, norm_CV_spd, fake_vehicle_num)     
        #     return attacked_state     
           
        # else:
        #     return tsc_state
        
        # If traj gen enabled, Attack state is generated in get_state, we just return tsc_state here:
        return tsc_state

    def get_advX_ISIG(self,state, curr_phase):
        state_cp = state.copy()

        if not self.attack_flag:
            if (self.mode == 'test'):
                self.jsma_params = {
                        "theta": 1.0,
                        "gamma": 0.1,
                        "clip_min": 0.0,
                        "clip_max": 1.0,
                        "y_target": None,
                    }
                self.jsma, self.classifier = init_attack(self.networks['actor'], self.jsma_params)

                self.attack_flag = True  
        
        # define target action based on rule
        action_pair = {0: 1,  # current SB, next NB, stay
                         1: 1,  # current NB, next L, stay
                         2: 1, # current L, next THR, stay
                         3: 0} # current Main-THR, switch
        
        target_action = action_pair[curr_phase]

        one_hot_target = np.zeros((1, self.n_actions), dtype=np.float32)
        one_hot_target[0, target_action] = 1
        self.jsma_params["y_target"] = one_hot_target
        adv_x, feature_ids = self.jsma.generate(x=state_cp[np.newaxis, ...],y= one_hot_target) 

        feature_ids = np.array(feature_ids).reshape(-1)

        print("adv_x: ",adv_x, "feature_ids:",feature_ids) 
        
        return adv_x, feature_ids, target_action
    
    def get_attacked_veh_info(self,att_action, cv_data):
        pass

    def get_fake_veh_jsma_guide(self, jsma_features, tsc_state, norm_CV, norm_CV_spd, scale_action):
        if (len(np.array(jsma_features).reshape(-1)) > 0): 
            tsc_state_cp = tsc_state.copy()
            for i in range(10,20):
                if (i in jsma_features): 
                    tsc_state_cp[i] = tsc_state_cp[i]*norm_CV + scale_action 
                else: 
                    tsc_state_cp[i] = tsc_state_cp[i]*norm_CV 
            
            CV_NUM = tsc_state_cp[10:20].copy()

            CV_NORM = np.linalg.norm(np.array(CV_NUM))
            for i in range(10,20):
                tsc_state_cp[i] = tsc_state_cp[i]/CV_NORM
            
            return tsc_state_cp
        else:
            return tsc_state

    def get_fake_veh(self, approach_action, scale_action, tsc_state, norm_CV, norm_CV_spd):
        # function: generate the modification on TSC state
        # input:
        #   approach_action: int, from 0 to 4, indicating the approach where fake vehicles are inserted
        #   scale_action: list, length 4, each value represents the number of vechiles inserted to the corresponding segment
        #   tsc_state: normal state of TSC actor
        # output:
        #   attacked tsc state: attacked state, where normal state is added with fake vehicles
        # we define the approach id corresponding phase here:
        # app_id = 0, EW -> impact Phase EW-THR,L -> phase 1, 2
        #          1, WE -> Phase WE-THR,L -> phase 1, 2
        #          2, Northbound -> phase S-N -> phase 3
        #          3, Southbound -> phase N-S -> phase 4
        # we also define the scale_action element meaning:
        # 0 -> THR segment 1,
        # 1 -> THR segment 2,
        # 2 -> THR segment 3,
        # 3 -> L segment
        # the TSC state is phase-based -> link approach to phase(s) and related segment idxs in the TSC state
        tsc_state_cp = tsc_state.copy()

        app_to_seg_list = {0:[17,18,19,16], 
                        1:[17,18,19,16], 
                        2:[13,14,15],
                        3:[10,11,12]
                        }
        state_idx_ls = app_to_seg_list[approach_action]

        for i in range(10,20):
            tsc_state_cp[i] = tsc_state_cp[i]*norm_CV

        if approach_action in [0,1]:
            for i in range(len(scale_action)):
                tsc_state_cp[state_idx_ls[i]] = tsc_state_cp[state_idx_ls[i]] + scale_action[i] 
        else:
            for i in range(len(scale_action)-1):
                tsc_state_cp[state_idx_ls[i]] = tsc_state_cp[state_idx_ls[i]] + scale_action[i] 
            tsc_state_cp[state_idx_ls[0]] = tsc_state_cp[state_idx_ls[0]] + scale_action[-1] 

        CV_NUM = tsc_state_cp[10:20].copy()
        
        CV_NORM = np.linalg.norm(np.array(CV_NUM))

        for i in range(10,20):
            tsc_state_cp[i] = tsc_state_cp[i]/CV_NORM

        print("Attacked approach_action:",approach_action,
              "\n Attacked scale_action",scale_action)

        return tsc_state_cp

    def get_state_opt(self):
        pass
    
    def get_lane_avg_speed(self, data, lane):
        if lane in data.keys():
            cnt = 0
            speed = 0
            for car, info in data[lane].items():
                cnt += 1
                speed += info[traci.constants.VAR_SPEED]

            avg_speed = speed/cnt if cnt!=0 else 17.88
            
        else:
            avg_speed = 17.88 

        return avg_speed


    def get_fake_table(self, attacker_action, cv_data):
        if attacker_action != None:
            fake_veh_ls = {}

            approach_action, action_dist, old_prob_1, action_dist_2, scale_action, old_prob_2 = attacker_action
            # Input: attacker's action, cv_data

            # use cv_data to calculate avg speed. 
            # hard code position and lane_id based on the app_action and segment info.

            # Output: Fake veh info dictionary
            # fake_veh_ls = {veh_id:{
            #                       traci.constants.VAR_LANE_ID: lane_id,
            #                       traci.constants.VAR_POSITION: veh_position,
            #                       traci.constants.VAR_SPEED: veh_speed
            # }} 

            # approach -> lane and veh_position_list
            # scale -> veh id list
            # cv_data -> veh_speed -> find veh on the same lane and calculate their avg speed
            app_to_laneID_ls = {0:['-11.0.00_0','-18.0.00_0','-4.0.00_0','-11.0.00_3'], 
                        1:['3.0.00_0','7.0.00_0','15.0.00_0','3.0.00_3'],
                        2:['16.0.00_1','8.0.00_1','12.0.00_1','16.0.00_3'], 
                        3:['-17.0.00_0','-20.0.00_0','-0.0.00_0','-17.0.00_2']
                        }
            
            lane2pos = {'-11.0.00_0':(487.60,410.40),
                        '-18.0.00_0':(437.01,439.36),
                        '-4.0.00_0':(366.30,368.40),
                        '-11.0.00_3':(494.88,422.87),
                        '3.0.00_0':(552.86,447.99),
                        '7.0.00_0':(598.84,465.24),
                        '15.0.00_0':(651.82,484.98),
                        '3.0.00_3':(548.48,456.89),
                        '16.0.00_1':(510.00,477.00),
                        '8.0.00_1':(520.00,548.00),
                        '12.0.00_1':(532.00,580.00),
                        '16.0.00_3':(502.00,463.00),
                        '-17.0.00_0':(543.00,403.00),
                        '-20.0.00_0':(542.00,432.00),
                        '-0.0.00_0':(544.00,227.00),
                        '-17.0.00_2':(536.00,404.00)}
            
            laneID_ls = app_to_laneID_ls[approach_action]

            for idx, scale in enumerate(scale_action):
                veh_id_ls = ['fake_veh_'+ str(idx) + '_' + str(i) for i in range(int(scale))]
                lane_id = laneID_ls[idx]
                speed = self.get_lane_avg_speed(cv_data,lane_id)
                position = lane2pos[lane_id] 
                for veh_id in veh_id_ls:
                    fake_veh_ls[veh_id] = {traci.constants.VAR_LANE_ID: lane_id,
                                           traci.constants.VAR_POSITION: position,
                                            traci.constants.VAR_SPEED: speed
                                            }

            print(fake_veh_ls)

            return fake_veh_ls

        else:
            print("Attacker action is None!")
            return None


    def get_state(self, tsc_state, CTM_state):
        return np.concatenate([tsc_state, CTM_state])

    def canonicalize_state(self, tsc_state, CTM_state, P_MAX=4, K_MAX=4, include_ctm=True, n_app=None):
        """Pad the per-intersection attacker state into a geometry-agnostic canonical
        vector so ONE shared attacker network can train/deploy across intersections
        with different #phases and #approaches.

        include_ctm=True  (cavlight): actor state[1] block + CTM block + geometry (46-dim).
        include_ctm=False (presslight, phase-based pressure state): the attacker observes
          the per-phase [inc(A), out(P), phase_onehot(P+1), time(1)] block; there is no CTM
          — canonical = [inc(A_MAX), out(P_MAX), phase(P_MAX+1), time(1)] + geometry (22-dim).
          Pass n_app explicitly for the geometry indicator (K can't be derived from a CTM
          state that isn't there).

        Inputs (per-intersection, native dims):
          tsc_state (actor state[1]) = [avg_speed(A), cv_count(A), phase_onehot(P+1), progress(1)]
              with A = (P-1)*num_segments + 1  (left-turn phase = 1 slot; each through phase = num_segments slots).
          CTM_state = [approach_density(K*num_segments), phase_onehot(P+1), progress(1)].

        Output (fixed canonical dim, e.g. 46 for P_MAX=K_MAX=4, num_segments=3):
          concat([ canon_state1, canon_ctm, geometry ]) where
            canon_state1 = avg(A_MAX) + cv(A_MAX) + phase_onehot(P_MAX+1) + progress(1)
            canon_ctm    = density(K_MAX*num_segments) + phase_onehot(P_MAX+1) + progress(1)
            geometry     = [P/P_MAX, K/K_MAX]
        Real features occupy the LEADING canonical slots (phase ordering puts the
        left-turn phase first, then through phases in order, so a smaller P simply
        omits trailing through-phase slots -> contiguous padding). The all-red
        one-hot bit is re-aligned to the last canonical slot (index P_MAX) so it is
        consistent across geometries. num_segments (S) is assumed shared across
        intersections; K (#approaches) is derived from CTM_state length.
        """
        S = self.num_segments
        P = len(self.green_phases)
        A = (P - 1) * S + 1
        A_MAX = (P_MAX - 1) * S + 1

        def pad_onehot(oh):
            c = np.zeros(P_MAX + 1)
            c[:P] = oh[:P]        # green-phase bits -> leading slots
            c[P_MAX] = oh[P]      # all-red bit -> last canonical slot
            return c

        if include_ctm:
            # cavlight actor state[1] block: [avg_speed(A), cv_count(A), phase(P+1), progress(1)]
            avg = tsc_state[0:A]
            cv = tsc_state[A:2 * A]
            oh = tsc_state[2 * A:2 * A + P + 1]
            prog = tsc_state[2 * A + P + 1:2 * A + P + 2]
            c_avg = np.zeros(A_MAX); c_avg[:A] = avg
            c_cv = np.zeros(A_MAX); c_cv[:A] = cv
            canon_s1 = np.concatenate([c_avg, c_cv, pad_onehot(oh), prog])

            K = int(round((len(CTM_state) - (P + 1) - 1) / S))  # K*S + (P+1) + 1 = len
            dens = CTM_state[0:K * S]
            ctm_oh = CTM_state[K * S:K * S + P + 1]
            ctm_prog = CTM_state[K * S + P + 1:K * S + P + 2]
            c_dens = np.zeros(K_MAX * S); c_dens[:K * S] = dens
            canon_ctm = np.concatenate([c_dens, pad_onehot(ctm_oh), ctm_prog])
            geom = np.array([P / float(P_MAX), K / float(K_MAX)])
            return np.concatenate([canon_s1, canon_ctm, geom])

        # presslight phase-based pressure state (no CTM):
        #   tsc_state = [inc(A), out(P), phase_onehot(P+1), time(1)]
        #   inc is the per-phase segmented incoming block (A=(P-1)*S+1, attackable);
        #   out is the per-phase outgoing block (P slots). Pad inc->A_MAX, out->P_MAX.
        inc = tsc_state[0:A]
        out = tsc_state[A:A + P]
        oh = tsc_state[A + P:A + P + (P + 1)]
        prog = tsc_state[A + P + (P + 1):A + P + (P + 1) + 1]
        c_inc = np.zeros(A_MAX); c_inc[:A] = inc
        c_out = np.zeros(P_MAX); c_out[:P] = out
        canon = np.concatenate([c_inc, c_out, pad_onehot(oh), prog])
        K = n_app if n_app is not None else K_MAX
        geom = np.array([P / float(P_MAX), K / float(K_MAX)])
        return np.concatenate([canon, geom])

    def get_reward(self, attacker_action, total_delay, fake_veh_gen_rate, att_success_idx, s_eff, success_prob=0.0, impact_factor=0.0, marginal=False):
        """
        Impact-Based Reward:
        Reward = (Delay Score) * (Impact Factor)

        marginal=False (default, CAVLight): total_delay is ABSOLUTE system delay; the
          delay score is non-negative (rewards high absolute delay).
        marginal=True (PressLight, -marginal_delay): total_delay is the SIGNED delay
          increase vs a benign baseline (see nextphaserltsc). The delay score is a SIGNED
          quadratic so an attack that REDUCES delay earns NEGATIVE reward — removing the
          perverse incentive where a self-correcting victim's attack still scores high on
          absolute delay.
        """
        if attacker_action is None: return 0, 0

        delay_baseline = 0 #1.0

        # Base Delay Score
        if marginal:
            # total_delay is the SIGNED marginal delay vs the FROZEN baseline (~±1-2, small).
            # Use a LINEAR signed score, NOT the /10 + quadratic that was calibrated for the
            # large ABSOLUTE delay (~4.5): squaring a small marginal annihilates it (1.5 ->
            # (0.15)^2/5 ~ 5e-4), leaving it ~25x below the jsma penalty so the policy learns
            # only to flip, not to flip HARMFULLY. Linear preserves small signals and the sign
            # (attack that REDUCES delay -> negative reward). Scale so a typical marginal
            # dominates the jsma penalty (jsma_lambda=0.1) and drives the delay objective.
            MARGINAL_SCALE = 2.0   # marginal delay is small (~0.02-0.1); scale so the delay
                                   # term DOMINATES the jsma penalty (jsma_lambda=0.1) and
                                   # drives the delay objective, not just flip-to-avoid-penalty.
            positive_delay_score = total_delay * MARGINAL_SCALE   # signed
        else:
            positive_delay_score = max(total_delay/ 10.0 - delay_baseline, 0.0) ** 2 / 5

        jsma_penalty = s_eff - 1 


        if att_success_idx == 1:
            # TIER 1: SUCCESS (Jackpot)
            # If we flipped the switch, we get the FULL value of our impact.
            # We add a 20% bonus to make the "jump" from failure to success extremely attractive.
            scale_factor = 1
        else:
            # TIER 2: FAILURE (Breadcrumbs)
            # If we failed, we get a small fraction (e.g., 10-20%) of credit.
            # This solves sparse rewards: "You lowered confidence by 40%, here is a cookie."
            # But it solves farming: "If you had flipped the switch, you would have gotten a cake."
            scale_factor = 0.1
        
        # MULTIPLIER LOGIC:
        # If Impact is 0 (Attack did nothing), Reward is 0.
        # If Impact is 1 (Total flip), Reward is 100% of Delay.
        # If Impact is 0.5 (Reduced confidence), Reward is 50% of Delay.
        # use att_success_idx as a hard gate to force agent to flip the decision rather than just making TSC doubt
        reward_delay = positive_delay_score * impact_factor * scale_factor * s_eff


        
        
        print(f"Delay: {positive_delay_score:.2f} | Avg Impact: {impact_factor:.2f} | Final Reward: {reward_delay:.4f}")

        # LOGGING FIX: record the per-decision attacker reward so simproc's per-episode dump
        # (reward_store[t].append(attacker.ep_rewards)) is non-empty. The victim does this in its
        # own get_reward (trafficsignalcontroller.py:1202); the attacker never did, so every
        # train_rewards_*_att_*.pkl episode was []. ep_rewards is re-init to [] each episode via
        # sim.close(), so no manual per-episode reset is needed.
        self.ep_rewards.append(reward_delay)

        return reward_delay, jsma_penalty


    def store_experience(self, state, action, next_state, r_delay, r_jsma, terminal, s_eff):
        self.rlagent.store_experience(state, action, next_state, r_delay, r_jsma, terminal, s_eff)
        
    def update(self, data, cv_data, uv_data, mask): 
        self.data = data
        self.cv_data = cv_data
        self.uv_data = uv_data