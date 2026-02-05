import numpy as np
import torch # Added for PyTorch compatibility

class AttRLAgent:
    def __init__(self, networks, epsilon, exp_replay, n_actions, n_steps, n_batch, n_exp_replay, gamma, rl_stats, mode, updates):
        ###this is a dict, keys = 'online', 'target'
        self.networks = networks
        self.epsilon = epsilon
        self.exp_replay = exp_replay
        # self.hist_exp_replay = [] #exp_replay.copy()
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.n_batch = n_batch
        self.n_exp_replay = n_exp_replay
        self.gamma = gamma
        self.experience_trajectory = []
        self.rl_stats = rl_stats
        # self.exp_replay = exp_replay
        self.mode = mode
        self.updates = updates
        self.global_critic = 'none' # Only overwritten in presslight_a2c
        self.last_exp = [] # wz: for marl info sharing




    def get_action(self, state):
       pass 

    def store_experience(self, state, action, next_state, r_delay, r_jsma, terminal, s_eff):
        # The logic to append to trajectory and then to exp_replay is suitable for PPO trajectory storage
        # (It stores a full trajectory when n_steps or terminal is hit)
        
        if self.rl_stats['updates'] < self.updates:
            experience = {'s':state, 'a':action,                                     
                          'next_s':next_state, 'r_delay':r_delay, 'r_jsma':r_jsma, 
                          'terminal':terminal, 's_eff':s_eff}
                                                                                     
            #append experience to trajectory
            self.experience_trajectory.append(experience)
                                                                                    
            ###check if need to add trajectory to exp replay
            if len(self.experience_trajectory) == self.n_steps or terminal == True:
                
                self.exp_replay.append(self.experience_trajectory)
                
                #rl stats bookkeeping
                self.rl_stats['n_exp'] += 1
                self.experience_trajectory = []
            
            print("###### elf.rl_stats['n_exp']",self.rl_stats['n_exp'])
                


    def train_batch(self, update_freq):
        # The PPOAgent subclass overrides this and must only proceed if self.buffer_ready is True
        pass

    def process_batch(self, sample_batch):
        pass 

    def process_trajectory(self):
        pass

    def compute_targets(self, rewards, R):
        ###compute targets using discounted rewards
        target_batch = []

        for i in reversed(range(len(rewards))):
            R = rewards[i] + (self.gamma * R)
            target_batch.append(R)

        target_batch.reverse()
        return target_batch

    def sample_replay(self):        
        return list(self.exp_replay)


    def clip_exp_replay(self):
        # LearnerProc ensures this is called only after training. ---        
        # Only clear if there is data to clear (safety check)
        if len(self.exp_replay) > 0: 
            self.exp_replay[:] = []         
            self.rl_stats['n_exp']= 0   
        else:
            # The LearnerProc should ensure the buffer is full before calling this
            pass 


    def send_weights(self):
        pass

    def retrieve_weights(self):
        pass