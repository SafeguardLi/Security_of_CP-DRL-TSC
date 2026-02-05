import os
import numpy as np
#import tensorflow as tf
#import tensorflow.keras.backend as K

#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Dense, Reshape, Flatten

class NeuralNet:
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, learner=False, nntype = None, temp =1.0):
        self.models = {}
        self.JSMA_act = True
        if nntype in ['ppo_act_app','ppo_act_scale','critic']:
            self.models['online'] = self.create_model(input_d, hidden_d, hidden_act, output_d, output_act, nntype, temp)
        else:
            self.models['online'] = self.create_model(input_d, hidden_d, hidden_act, output_d, output_act, nntype, temp)
            if learner:
                # WZ: for policy-based method like PPO and A2C, we dont need such separate networks 
                self.models['target'] = self.create_model(input_d, hidden_d, hidden_act, output_d, output_act, nntype, temp)

    def create_model(self, input_d, hidden_d, hidden_act, output_d, output_act, nntype):
        pass

    def forward(self, _input, nettype):
        return self.models[nettype].predict(_input)

    def backward(self):
        pass

    def transfer_weights(self):
        pass

    def get_weights(self, nettype):
        pass

    def set_weights(self, weights, nettype):
        pass

    def save_weights(self, nettype, path, fname):
        pass

    def load_weights(self, path):
        pass
