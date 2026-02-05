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
    
    def init_attacker(self):
        self.jsma_params = {
                    "theta": 1.0,
                    "gamma": 0.1,
                    "clip_min": 0.0,
                    "clip_max": 1.0,
                    "y_target": None,
                }
        self.jsma, self.classifier = init_attack(self.networks['actor'], self.jsma_params)

        self.attack_flag = True  

    def get_advX(self,state, curr_phase, att_action):
        state_cp = state.copy()
        target_action = att_action[0]
        # attack_scale = att_action[-2]

        if not self.attack_flag:
            self.init_attacker()
        
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