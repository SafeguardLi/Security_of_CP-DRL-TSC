# https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.optimizers import Adam

from src.neuralnet import NeuralNet
from src.helper_funcs import check_and_make_dir, write_line_to_csv, get_fp

class A2CCritic(NeuralNet):
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre, learner=False, nntype=None, temp = 1.0):
        super().__init__(input_d, hidden_d, hidden_act, output_d, output_act, learner=learner, nntype=nntype, temp = 1.0)
        self.nntype = nntype
        self.output_d = output_d

        for model in self.models:
            self.models[model].compile(optimizer='adam', loss='mse')

    def create_model(self, input_d, hidden_d, hidden_act, output_d, output_act,
                     nntype, temp):
        if nntype in ['presslight_a2c', 'a2c','a2c_r','a2c_ps','a2c_psr','a2c_sr','cavlight']:
            model_in = Input((input_d,))
            for i in range(len(hidden_d)):
                if i == 0:
                    model = Dense(hidden_d[i], activation=hidden_act, kernel_initializer='he_uniform')(model_in)
                else:
                    model = Dense(hidden_d[i], activation=hidden_act, kernel_initializer='he_uniform')(model)

            model_out = Dense(output_d, activation=output_act, kernel_initializer='he_uniform')(model)

        else:
            raise NotImplementedError
        return Model(model_in, model_out)

    def forward(self, _input, nettype):
        if self.nntype in ['presslight_a2c', 'a2c','a2c_r','a2c_ps','a2c_psr','a2c_sr','cavlight']:
            return self.models[nettype].predict(_input)
        else:
            raise NotImplementedError

    def backward(self, _input, _target, updates = None, gamma = None, main_args = None, tsc_id = None):
        if self.nntype in ['presslight_a2c', 'a2c','a2c_r','a2c_ps','a2c_psr','a2c_sr','cavlight']:
            history = self.models['online'].fit(_input, _target, batch_size=main_args.batch, epochs=1, verbose=0) #main_args.batch

            if (updates != None) & (gamma != None) & (main_args != None):
                # wz: prepared for grid search
                self.fp_loss_history = get_fp(main_args,'log')
                #self.fp_loss_history = os.path.join('experiments', f'{main_args.tsc}',
                #                                    f'Global_{main_args.global_critic}',
                #                                    f'CV_pen_rate_{main_args.pen_rate}',
                #                                    f'{main_args.sim}_{main_args.flow_type}_{main_args.turn_type}',
                #                                    f'gamma_{main_args.gamma}', f'eps_{main_args.eps}',f'temp_{main_args.temperature}', 'log')
                check_and_make_dir(self.fp_loss_history)
                self.fp_loss_history = os.path.join(self.fp_loss_history, str(tsc_id) + 'critic_loss_history.csv')
                write_line_to_csv(self.fp_loss_history, history.history['loss'])
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
        print('load weight path', path)
        if os.path.exists(path):
            self.models['online'].load_weights(path)
        else:
            # raise not found exceptions
            assert 0, 'Failed to load weights, supplied weight file path ' + str(path) + ' does not exist.'


'''class A2CCriticNet:
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre, name, sess, nntype):
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            # build critic network
            self.nntype = nntype
            self.input = tf.compat.v1.placeholder(tf.float32,
                                                  shape=[None, input_d],
                                                  name='inputs')

            self.actions = tf.compat.v1.placeholder(tf.float32,
                                                    shape=[None, output_d],
                                                    name='actions')

            dense1 = tf.compat.v1.layers.dense(self.input, units=hidden_d[0],
                                               kernel_initializer=tf.compat.v1.keras.initializers.he_uniform())  # he_uniform())
            batch1 = tf.compat.v1.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.elu(batch1)

            dense2 = tf.compat.v1.layers.dense(layer1_activation, units=hidden_d[1],
                                               kernel_initializer=tf.compat.v1.keras.initializers.he_uniform())  # he_uniform())
            batch2 = tf.compat.v1.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.elu(batch2)


            if nntype == 'presslight_ddpg':

                self.q = tf.compat.v1.layers.dense(layer2_activation, units=output_d,
                                                   kernel_initializer=tf.compat.v1.keras.initializers.he_uniform(),
                                                   # he_uniform(),
                                                   kernel_regularizer=tf.keras.regularizers.l2(0.01))

                self.q_target = tf.compat.v1.placeholder(tf.float32,
                                                         shape=[None, output_d],
                                                         name='targets')




            elif nntype == 'ddpg':

                state_actions = tf.concat([layer2_activation, self.actions], axis=-1)

                dense3 = tf.compat.v1.layers.dense(state_actions, units=hidden_d[1],
                                                   kernel_initializer=tf.compat.v1.keras.initializers.he_uniform())  # he_uniform())
                batch3 = tf.compat.v1.layers.batch_normalization(dense3)
                layer3_activation = tf.nn.elu(batch3)

                # wz: why does the unites = 1? can it be the output?
                self.q = tf.compat.v1.layers.dense(layer3_activation, units=1,
                                                   kernel_initializer=tf.compat.v1.keras.initializers.he_uniform(),
                                                   # he_uniform(),
                                                   kernel_regularizer=tf.keras.regularizers.l2(0.01))

                self.q_target = tf.compat.v1.placeholder(tf.float32,
                                                         shape=[None, 1],
                                                         name='targets')
                self.action_gradients = tf.gradients(self.q, self.actions)
            else:
                print(nntype)
                raise NotImplementedError

            self.loss = tf.compat.v1.losses.mean_squared_error(self.q_target, self.q)

            self.params = tf.compat.v1.trainable_variables(scope=name)

            # optimizer
            self.optimize = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, epsilon=lre).minimize(self.loss)

            # saving stuff
            self.varstate = VariableState(sess, self.params)'''