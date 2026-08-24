import os
import numpy as np
from src.rlagent import RLAgent
from src.jsma import init_attack
import torch
import torch.nn as nn
import torch.nn.functional as F

class A2CAgent(RLAgent):
    def __init__(self, networks, epsilon, n_actions, n_steps, n_batch, gamma, mode,
                 updates, main_args, tsc_id):
        super().__init__(networks, epsilon, n_actions, n_steps, n_batch, gamma,
                         mode, updates)
        self.main_args = main_args
        self.tsc_id = tsc_id
        self.noise = True
        self.global_critic = main_args.global_critic 
        self.attack_flag = False
    
    def init_attacker(self, input_dim=None, classifier_path=None):
        if self.attack_flag:
            return  # idempotent — only initialize once
        self.jsma_params = {
                    "theta": 1.0,
                    "gamma": 0.1,
                    "clip_min": 0.0,
                    "clip_max": 1.0,
                    "y_target": None,
                }
        if input_dim is None:
            input_dim = 26  # Plymouth default
        # White-box attack: JSMA targets the real cavlight actor directly, not a
        # surrogate binary classifier. The surrogate was only for the isolated-TSC
        # case where the victim model was not accessible.
        # Correctness requirement: the actor MUST hold trained weights. Without
        # -load it keeps random init and the white-box attack is meaningless.
        if not getattr(self.main_args, 'load', False):
            raise RuntimeError(
                f"White-box JSMA on TSC {self.tsc_id}: actor is not loaded "
                f"(args.load is False). The victim would be a random-weight model, "
                f"making the attack meaningless. Run with -load -tsc_updates <N>."
            )
        # Restrict JSMA to the CV-count block of state[1] (only features that fake
        # vehicles can move). state[1] = avg_speed(avg_len) + cv_count(avg_len) +
        # phase_one_hot(n_phases+1) + progress(1), with avg_len = (n_phases-1)*s+1.
        # Recover avg_len from input_dim so the range is correct per geometry:
        #   4-phase (input_dim=26) -> (10, 20);  2-phase (input_dim=12) -> (4, 8).
        feature_range = (10, 21)
        s = int(getattr(self.main_args, 'num_segments', 3))
        n_phases = int(round((input_dim + 2 * s - 4) / (2 * s + 1)))
        avg_len = (n_phases - 1) * s + 1
        if avg_len >= 1 and 2 * avg_len <= input_dim:
            feature_range = (avg_len, 2 * avg_len)
        sur_dir = getattr(self.main_args, 'surrogate_dir', None)
        if sur_dir:
            # BLACKBOX: JSMA (get_advX) and the surrogate classifier (surrogate_act decisions) come
            # from the trained per-intersection surrogate, NOT the real actor. The real TSC still
            # decides via get_action(..., surrogate_act=False).
            import os
            cpath = os.path.join(sur_dir, f"{self.tsc_id}_surrogate.pt")
            self.jsma, self.classifier = init_attack(None, self.jsma_params,
                                                     input_dim=input_dim, white_box=False,
                                                     classifier_path=cpath, feature_range=feature_range)
        else:
            self.jsma, self.classifier = init_attack(self.networks['actor'], self.jsma_params,
                                                     input_dim=input_dim, white_box=True,
                                                     feature_range=feature_range)
        self.attack_flag = True

    def get_advX(self,state, curr_phase, att_action):
        state_cp = state.copy()
        target_action = att_action[0]
        # attack_scale = att_action[-2]

        if not self.attack_flag:
            # lazy init: use actual state dim (handles corr3 where dim != 26)
            self.init_attacker(input_dim=state_cp.shape[-1])
        
        # # define target action based on rule
        # action_pair = {0: 1,  # current SB, next NB, stay
        #                  1: 1,  # current NB, next L, stay
        #                  2: 1, # current L, next THR, stay
        #                  3: 0} # current Main-THR, switch
        
        # target_action = action_pair[curr_phase]

        one_hot_target = np.zeros((1, self.n_actions), dtype=np.float32)
        one_hot_target[0, target_action] = 1
        self.jsma_params["y_target"] = one_hot_target
        adv_x, feature_ids = self.jsma.generate(x=state_cp[np.newaxis, ...],y= one_hot_target) 

        feature_ids = np.array(feature_ids).reshape(-1)

        if feature_ids is None:
            feature_ids = []

        # print("adv_x: ",adv_x, "feature_ids:",feature_ids) ####
        
        return adv_x, feature_ids, target_action

    def get_action(self, state, epsilon = 1e-5, surrogate_act = False):

        ###choose action according to the probability distribution
        _sample_actions = np.zeros((1,self.networks['actor'].output_d))
        _q_values = np.zeros((1,self.networks['actor'].output_d))
        # _advantage = np.zeros((1,1))
        if surrogate_act:
            if not self.attack_flag:
                # lazy init: state_real_cv has same dim as classifier input
                self.init_attacker(input_dim=state.shape[-1])
            # action_dist = self.classifier.predict(state[np.newaxis, ...])
            x_torch = state[np.newaxis, ...].astype(np.float32) #torch.tensor().to(torch.float32)
            action_dist = self.classifier.predict(x_torch) #self.networks['actor'].forward(adv_x[np.newaxis, ...],_sample_actions, _q_values,'online')
            # action_dist = action_dist.squeeze() 
            action_dist = action_dist.squeeze()
        else:
            action_dist = self.networks['actor'].forward(state[np.newaxis, ...],_sample_actions, _q_values,'online')
            action_dist = action_dist.squeeze()  # wz: to remove the unnecessary dimension
        
        eps = 1e-5  # for testimg, we dont wanna random action
        if np.random.uniform(0.0, 1.0) < eps: 
            ###act randomly
            print("random action selected")
            action = np.random.randint(self.n_actions)
        else:
            # action = np.round(action_dist) 
            action = np.random.choice(np.arange(len(action_dist)), p=action_dist)

        ###return action integer
        return action, action_dist

    def actions_to_one_hot(self, actions, output_d):
        # wz: this function convert selected actions back to their original dimension
        # e.g actions = [1,0,1,1] -> actions = [[0,1],[1,0],[1,0],[1,0]]
        return np.eye(output_d)[actions]