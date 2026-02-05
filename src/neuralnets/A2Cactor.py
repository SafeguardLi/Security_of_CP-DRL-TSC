'''import os

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# from tensorflow.initializers import he_uniform

from src.neuralnet import NeuralNet
from src.neuralnets.varstate import VariableState
from src.picklefuncs import save_data, load_data
from src.helper_funcs import check_and_make_dir, write_line_to_csv'''

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# tf.compat.v1.enable_eager_execution()

from src.neuralnet import NeuralNet
from src.helper_funcs import check_and_make_dir, write_line_to_csv, get_fp


class A2CActor(NeuralNet):
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre, learner=False, nntype=None, temp = 1.0):
        super().__init__(input_d, hidden_d, hidden_act, output_d, output_act, learner=learner, nntype=nntype, temp = 1.0)
        self.nntype = nntype
        self.output_d = output_d
        self.temperature = temp
        # print('\n temperature for Actor is: ',temp) #####
        self.JSMA_act = True

        for model in self.models:
            if self.JSMA_act:
                self.models[model].compile(optimizer='adam', loss='categorical_crossentropy')
            else:
                self.models[model].compile(optimizer='adam')
                #self.models[model].run_eagerly = True # add run_eargerly=True for debugging

    #@tf.function
    def create_model(self, input_d, hidden_d, hidden_act, output_d, output_act, nntype, temp):
        # ref: https://ai.stackexchange.com/questions/18753/how-to-set-the-target-for-the-actor-in-a2c
        self.temperature = temp
        # print('\n inside create_model \n temperature for Actor is: ', temp)  #####
        if nntype in ['cavlight']:
            model_in = Input(shape=(input_d,), name='state_in')
            #advantage = Input(shape=(1,), name='advantage')
            q_values = Input(shape=(output_d,), name='q_values')
            sampled_actions = Input(shape=(output_d,), name='sampled_actions')

            layers = {}
            # we receive WARNING:tensorflow:Output dense_2 missing from loss dictionary. We assume this was done on purpose. The fit and evaluate APIs will not be expecting any data to be passed to dense_2.
            # but it seems not influence the result
            for i in range(len(hidden_d)):
                if i == 0:
                    layers[i] = Dense(hidden_d[i], activation=hidden_act, kernel_initializer='he_uniform')(model_in)
                else:
                    layers[i] = Dense(hidden_d[i], activation=hidden_act, kernel_initializer='he_uniform')(layers[i-1])

            dropout_layer = Dropout(0.5,input_shape=(hidden_d[i],))(layers[len(hidden_d)-1])
            model_out = Dense(output_d, activation=self.softmax_temp, kernel_initializer='he_uniform')(dropout_layer)
            #model_out = Dense(output_d, activation=tf.keras.activations.softmax, kernel_initializer='he_uniform')(dropout_layer)
            #model_out = Dense(output_d, activation=output_act, kernel_initializer='he_uniform')(dropout_layer)

        else:
            raise NotImplementedError

        # final_model = Model([model_in, sampled_actions, advantage], model_out)
        # final_model.add_loss(self.a2c_loss(model_out, sampled_actions, advantage))
        

        if self.JSMA_act:
            final_model = Model(model_in, model_out)
        else:
            final_model = Model([model_in, sampled_actions, q_values], model_out)
            final_model.add_loss(self.a2c_loss(model_out, sampled_actions, q_values))

        return final_model

    def softmax_temp(self, x):
        # ref: https://stackoverflow.com/questions/63471781/making-custom-activation-function-in-tensorflow-2-0
        # ref: http://www.kasimte.com/2020/02/14/how-does-temperature-affect-softmax-in-machine-learning.html
        # ref: https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax/
        e_x = tf.exp(tf.divide(x - tf.reduce_max(x, axis=-1, keepdims=True), self.temperature))
        output = tf.divide(e_x, tf.reduce_sum(e_x, axis=-1, keepdims=True))
        return output

    #@tf.function
    # ref: https://stackoverflow.com/questions/59283049/print-all-terms-of-loss-function-tensorflow-2-0
    def a2c_loss(self, mu, sampled_actions, q):
        '''
        This function return loss based on the formula: -log(p(a|s)) * (advantage).
        Input:
            mu: shape = (batch_size, output_d), policy network output, action probability distribution under state s
            q: shape = (batch_size, output_d), all Q(s,a) values for each action at state s
            sampled_actions: shape = (batch_size, output_d), actions selected under state s, one-hot vector
            advantage = (batch_size, 1), the calculated advantage, r + gamma*V(s') - V(s)
        Output:
            a loss function in tensorflow
        '''
        # TODO: Confirm if mu should be stochastic or one-hot
        V = tf.reduce_sum(tf.multiply(q, mu), axis=1)  # weighted sum of q with weight as the pi (i.e. mu)
        # use one-hot sampled action to select corresponding q value and mu
        # use reduced_sum to get rid of zero values
        q_a = tf.reduce_sum(tf.multiply(q, sampled_actions), axis=1)
        mu_a = tf.reduce_sum(tf.multiply(mu, sampled_actions), axis=1)
        loss = tf.reduce_sum(tf.multiply(-tf.math.log(mu_a + 1e-5), tf.subtract(q_a, V))) #advantage))
        return tf.multiply(loss, sampled_actions)*1e-2  #, axis=1)
        # TODO: add 0 for output nodes with action we didnt choose? -> tf.multiply(loss,sampled_actions)
        # TODO: should we give axis = 1 in the return line? but if so, when batch_size = 1 it will raise an error.


    def forward(self, _input, actions, q_value, nettype):
        if self.nntype in ['cavlight']:
            if self.JSMA_act:
                return self.models[nettype].predict(_input)
            else:
                return self.models[nettype].predict((_input, actions, q_value))
        else:
            raise NotImplementedError


    def backward(self, states, actions, q_value, updates, gamma, main_args, tsc_id):
    # ref: https://ai.stackexchange.com/questions/18753/how-to-set-the-target-for-the-actor-in-a2c
        if self.nntype in ['cavlight']:
            history = self.models['online'].fit((states, actions, q_value), batch_size=main_args.batch, epochs=1, verbose=1)

            if (updates != None) & (gamma != None) & (main_args != None):
                # wz: prepared for grid search
                self.fp_loss_history = get_fp(main_args,'log')
                #self.fp_loss_history = os.path.join('experiments', f'{main_args.tsc}',
                #                                    f'Global_{main_args.global_critic}',
                #                                    f'CV_pen_rate_{main_args.pen_rate}',
                #                                    f'{main_args.sim}_{main_args.flow_type}_{main_args.turn_type}',
                #                                    f'gamma_{main_args.gamma}', f'eps_{main_args.eps}',f'temp_{main_args.temperature}', 'log')
                check_and_make_dir(self.fp_loss_history)
                self.fp_loss_history = os.path.join(self.fp_loss_history, str(tsc_id) + 'actor_loss_history.csv')
                loss = np.sum(history.history['loss'])
                write_line_to_csv(self.fp_loss_history, [loss])
            else:
                raise ValueError('Gamma is None')
        else:
            raise NotImplementedError

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        self.set_weights(self.get_weights('online'), 'target')

    def get_weights(self, nettype):
        return self.models[nettype].get_weights()

    def set_weights(self, weights, nettype):
        return self.models[nettype].set_weights(weights)

    def save_weights(self, nettype, path, fname):
        check_and_make_dir(path)
        self.models[nettype].save_weights(path + fname + '.h5', save_format='h5', overwrite='True')

    def load_weights(self, path):
        path += '.h5'
        # print('load weight path', path)
        if os.path.exists(path):
            self.models['online'].load_weights(path)
        else:
            # raise not found exceptions
            assert 0, 'Failed to load weights, supplied weight file path ' + str(path) + ' does not exist.'


'''
class DDPGActorNet:
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre, name, batch_size, sess, nntype):
        # create model and all necessary parts
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            self.input = tf.compat.v1.placeholder(tf.float32,
                                                  shape=[None, input_d],
                                                  name='inputs')

            self.action_gradient = tf.compat.v1.placeholder(tf.float32,
                                                            shape=[None, output_d],
                                                            name='gradients')

            dense1 = tf.compat.v1.layers.dense(self.input, units=hidden_d[0],
                                               kernel_initializer=tf.compat.v1.keras.initializers.he_uniform())  # HeUniform()) #he_uniform())

            batch1 = tf.compat.v1.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.elu(batch1)
            dense2 = tf.compat.v1.layers.dense(layer1_activation, units=hidden_d[1],
                                               kernel_initializer=tf.compat.v1.keras.initializers.he_uniform())  # HeUniform()) #he_uniform())

            batch2 = tf.compat.v1.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.elu(batch2)
            if nntype == 'ddpg':
                mu = tf.compat.v1.layers.dense(layer2_activation, units=output_d,
                                               activation='tanh',
                                               kernel_initializer=tf.compat.v1.keras.initializers.he_uniform())  # HeUniform()) #he_uniform())
            elif nntype == 'presslight_ddpg':
                mu = tf.compat.v1.layers.dense(layer2_activation, units=output_d,
                                               activation='softmax',
                                               kernel_initializer=tf.compat.v1.keras.initializers.he_uniform())  # HeUniform()) #he_uniform())
            self.mu = mu

            self.params = tf.compat.v1.trainable_variables(scope=name)
            # print(name)

            self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient,
                                                             unconnected_gradients='zero')  # wz: the negative action_gradient is because the nn is updated by gradient ascent

            self.actor_gradients = list(map(lambda x: tf.math.divide(x, batch_size), self.unnormalized_actor_gradients))

            # wz: this is the original version of backward training of actor, which use gradient calculated from critic nn
            if nntype == 'ddpg':
                self.optimize = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, epsilon=lre).apply_gradients(
                    zip(self.actor_gradients, self.params))

            # wz: here we use loss to train actor nn
            elif nntype == 'presslight_ddpg':
                # wz: add sampled actions to select q value and mu in loss calculation
                self.sampled_actions = tf.compat.v1.placeholder(tf.float32,
                                                                shape=[None, output_d],
                                                                name='sampled_actions')
                self.q = tf.compat.v1.placeholder(tf.float32,
                                                  shape=[None, output_d],
                                                  name='q-value')

                self.V = tf.reduce_sum(
                    tf.multiply(self.q, self.mu))  # weighted sum of q with weight as the pi (i.e. mu)
                # use one-hot sampled action to select corresponding q value and mu
                # use reduced_sum to get rid of zero values
                self.q_a = tf.reduce_sum(tf.multiply(self.q, self.sampled_actions), axis=1)
                self.mu_a = tf.reduce_sum(tf.multiply(self.mu, self.sampled_actions), axis=1)
                self.loss = self.a2c_loss(self.mu_a, self.q_a, self.V)
                self.optimize = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, epsilon=lre).minimize(
                    self.loss)
            else:
                raise NotImplementedError

            # for training/saving
            self.varstate = VariableState(sess, self.params)'''