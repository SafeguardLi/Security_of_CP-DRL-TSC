# from art.attacks.evasion import SaliencyMapMethod
from src.saliency_map import SaliencyMapMethod
from art.estimators.classification import KerasClassifier, PyTorchClassifier

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
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(26, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

def init_attack(load_model, jsma_params):
    # weights = load_model.get_weights('online')
    # classifier = BuildModel(weights)

    # set classifier as torch one rather than keras

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier().to(device)

    # Load the saved weights
    model.load_state_dict(torch.load("/home/Documents/DRL-attacker/experiments/cavlight/CAV_pen_rate_5.0/plymouth_bin_real_real/saved_models/binary_classifier.pth", map_location=torch.device('cpu')))
    # Set to evaluation mode
    model.eval()

    # Trigger White-box attack, we change the classifier here.

    classifier = PyTorchClassifier(model=model, clip_values=(0, 2),loss=nn.BCELoss(), input_shape = (26,), nb_classes = 2)

    # classifier = KerasClassifier(model= load_model.models['online'], clip_values=(0, 2), use_logits=False) #-> ART repo speicifies loss and our customized loss model cant be used
    
    jsma = SaliencyMapMethod(classifier=classifier) # we can pass JSMA parameters here
    # theta: float = 0.1, gamma: float = 1.0, batch_size: int = 1, verbose: bool = True

    # grads = self.estimator.class_gradient(x, label=target)
    # https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/9ed6105b0152bc9b0c902ab3734c1d8acf476315/art/attacks/evasion/saliency_map.py#L205C9-L205C63
    # TODO: use the jsma.estimator.class_gradient to generate saliency map and save it for later viz
    # targets = np.argmax(y, axis=1)
    # x: a batch of input
    # will give back the saliency map for all features towards the target action

    return jsma, classifier