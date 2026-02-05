import numpy as np

class RLAgent:
    def __init__(self, networks, epsilon, n_actions, n_steps, n_batch, gamma, mode, updates):
        ###this is a dict, keys = 'online', 'target'
        self.networks = networks
        self.epsilon = epsilon
        # self.hist_exp_replay = [] #exp_replay.copy()
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.n_batch = n_batch
        self.gamma = gamma
        self.experience_trajectory = []
        # self.exp_replay = exp_replay
        self.mode = mode
        self.updates = updates
        self.global_critic = 'none' # Only overwritten in presslight_a2c
        self.last_exp = [] # wz: for marl info sharing

    def get_action(self, state):
       pass 

    def store_experience(self, state, action, next_state, reward, terminal):
        pass

    def store_experience_to_buffer(self, experience):
        pass

    def train_batch(self, update_freq):
        pass

    def process_batch(self, sample_batch):
        pass 

    def process_trajectory(self):
        pass

    def compute_targets(self, rewards, R):
        pass

    def sample_replay(self):
        pass

    def clip_exp_replay(self):
        pass

    def send_weights(self):
        pass

    def retrieve_weights(self):
        pass

