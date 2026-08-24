import os
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers.legacy import Adam

from src.neuralnet import NeuralNet


class DQN(NeuralNet):
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre,
                 learner=False, nntype=None, temp=1.0):
        super().__init__(input_d, hidden_d, hidden_act, output_d, output_act,
                         learner=learner, nntype=nntype, temp=temp)
        for model in self.models:
            self.models[model].compile(Adam(lr=lr, epsilon=lre), loss='mse')

    def create_model(self, input_d, hidden_d, hidden_act, output_d, output_act, nntype, temp):
        model_in = Input((input_d,))
        x = model_in
        for units in hidden_d:
            x = Dense(units, activation=hidden_act, kernel_initializer='he_uniform')(x)
        model_out = Dense(output_d, activation=output_act, kernel_initializer='he_uniform')(x)
        return Model(model_in, model_out)

    def forward(self, _input, nettype):
        return self.models[nettype].predict(_input, verbose=0)

    def get_weights(self, nettype):
        return self.models[nettype].get_weights()

    def set_weights(self, weights, nettype):
        self.models[nettype].set_weights(weights)

    def load_weights(self, path):
        path += '.h5'
        if os.path.exists(path):
            self.models['online'].load_weights(path)
            if 'target' in self.models:
                self.models['target'].load_weights(path)
        else:
            assert 0, f'Failed to load DQN weights: {path} does not exist.'
