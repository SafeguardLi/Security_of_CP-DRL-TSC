# from art.attacks.evasion import SaliencyMapMethod
from src.saliency_map import SaliencyMapMethod
from art.estimators.classification import KerasClassifier, PyTorchClassifier

import os
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
# tf.compat.v1.disable_eager_execution() ###

# from tensorflow.keras.layers import Input, Dense, Dropout
# from tensorflow.keras.losses import categorical_crossentropy
# from tensorflow.keras.models import Sequential, Model


# def BuildModel(weights):
#     # tf.compat.v1.disable_eager_execution() ###

#     n_phases = 3
#     num_segments = 3
#     input_d = 2 * n_phases * num_segments + 2 * n_phases + 2 
#     output_d = 3
#     n_hidden = 2
#     hidden_d = [input_d*3] * n_hidden
#     hidden_act = 'elu'
#     # lr=1e-4
#     # lre=1e-7
    

#     model_in = Input(shape=(input_d,), name='state_in')
#     # q_values = Input(shape=(output_d,), name='q_values')
#     # sampled_actions = Input(shape=(output_d,), name='sampled_actions')
#     layers = {}
#     for i in range(len(hidden_d)):
#         if i == 0:
#             layers[i] = Dense(hidden_d[i], activation=hidden_act, kernel_initializer='he_uniform')(model_in)
#         else:
#             layers[i] = Dense(hidden_d[i], activation=hidden_act, kernel_initializer='he_uniform')(layers[i-1])

#     dropout_layer = Dropout(0.5,input_shape=(hidden_d[i],))(layers[len(hidden_d)-1])
#     model_out = Dense(output_d, activation=softmax_temp, kernel_initializer='he_uniform')(dropout_layer)
#     final_model = Model(model_in, model_out)
#     # final_model.add_loss(self.a2c_loss(model_out, sampled_actions, q_values))

#     final_model.set_weights(weights)

#     final_model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=["accuracy"]) #, metrics=["accuracy"])  categorical_crossentropy


#     classifier = KerasClassifier(model= final_model, clip_values=(0, 2), use_logits=False)
    
#     return classifier #final_model

# def softmax_temp( x):
#     # ref: https://stackoverflow.com/questions/63471781/making-custom-activation-function-in-tensorflow-2-0
#     # ref: http://www.kasimte.com/2020/02/14/how-does-temperature-affect-softmax-in-machine-learning.html
#     # ref: https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax/
#     temperature = 100
#     e_x = tf.exp(tf.divide(x - tf.reduce_max(x, axis=-1, keepdims=True), temperature))
#     output = tf.divide(e_x, tf.reduce_sum(e_x, axis=-1, keepdims=True))
#     return output

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=26):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

def init_attack(load_model, jsma_params, input_dim=26, classifier_path=None, white_box=False,
                feature_range=(10, 21), recompile_loss=None):
    if white_box and load_model is not None:
        # White-box: JSMA attacks the real TSC victim directly.
        # load_model is the victim network; its 'online' model is a clean Keras
        # Model(state_in -> output), so ART can read its Jacobian via class_gradient.
        keras_model = load_model.models['online']
        if recompile_loss is not None:
            # DQN victim (PressLight): 'online' outputs LINEAR Q-values. Two problems for JSMA:
            #   (1) it was compiled with loss='mse', which ART's KerasClassifier can't resolve;
            #   (2) the raw-Q gradient is a POOR saliency signal. Restricted to the spoofable
            #       inc+out features (the real feature_range), raw-Q JSMA flips the argmax only
            #       ~13% of the time and needs a large (~6x) perturbation the discrete fake-
            #       vehicle injection can't realize -> attack lands ~0% in the corridor.
            # FIX: attack a SOFTMAX head on top of Q. softmax gives a boundary-aligned gradient,
            # so JSMA flips the argmax ~100% with a SMALL, injection-realizable perturbation.
            # softmax is MONOTONIC: argmax(softmax(Q)) == argmax(Q), so the victim's actual
            # decision is UNCHANGED. This builds a SEPARATE surrogate Model sharing the frozen
            # weights; load_model ('online') is never modified. cavlight is unaffected — its
            # actor already outputs softmax and passes recompile_loss=None, so it skips this.
            # (Verified offline on 62500824 deploy states: raw-Q flip 13% -> softmax flip 100%,
            #  mean|dx| 21 -> 3.75.)
            jsma_model = tf.keras.models.Model(
                keras_model.input,
                tf.keras.layers.Activation('softmax')(keras_model.output))
            jsma_model.compile(optimizer='adam', loss=recompile_loss)
            keras_model = jsma_model
        classifier = KerasClassifier(model=keras_model, clip_values=(0, 2), use_logits=False)
        jsma = SaliencyMapMethod(classifier=classifier, feature_range=feature_range)
        return jsma, classifier

    # Surrogate (BLACKBOX) path: JSMA analyzes a trained BinaryClassifier surrogate standing in for
    # the victim actor. The victim's real model is never touched here — features are chosen purely
    # from the surrogate's gradient, then injected and fed to the real TSC by the caller.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier(input_dim=input_dim).to(device)

    if classifier_path is not None and os.path.exists(classifier_path):
        model.load_state_dict(torch.load(classifier_path, map_location=torch.device('cpu')))
        print(f"[blackbox JSMA] loaded surrogate weights from {classifier_path}")
    else:
        print(f"[blackbox JSMA] WARNING: surrogate weights not found at {classifier_path} — using "
              f"RANDOM init (JSMA direction still defined but NOT the victim's).")

    model.eval()

    classifier = PyTorchClassifier(model=model, clip_values=(0, 2), loss=nn.BCELoss(), input_shape=(input_dim,), nb_classes=2)

    # Restrict JSMA to the spoofable CV block (same as white-box) so only injectable features move.
    jsma = SaliencyMapMethod(classifier=classifier, feature_range=feature_range)

    return jsma, classifier