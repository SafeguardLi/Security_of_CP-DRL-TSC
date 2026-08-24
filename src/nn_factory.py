import os

# import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from src.neuralnets.A2Ccritic import A2CCritic
from src.neuralnets.A2Cactor import A2CActor
from src.neuralnets.PPO_actor import PPOActor
from src.neuralnets.PPO_critic import PPOCritic
from src.neuralnets.dqn import DQN
from src.helper_funcs import get_fp

import torch.nn as nn

def nn_factory( nntype, input_d, output_d, args, learner, load, tsc, n_hidden, sess=None):
    # wz: call corresponding function to create neural networks
    nn = None
    if nntype in ['cavlight']:
        cri_input_d = input_d[0]
        act_input_d = input_d[1]
        # hidden_layers = [[cri_input_d*n_hidden, cri_input_d*n_hidden],[act_input_d*n_hidden, act_input_d*n_hidden]]

        # now, we set n_hidden as the number of layers, rather a scaler for num of nerons as before; a fixed scaler is applied
        hidden_layers = [[cri_input_d*3] * n_hidden, [act_input_d*3] * n_hidden]
    else:
        hidden_layers = [input_d*n_hidden, input_d*n_hidden] # number of neurons for each layer

    if nntype in ['cavlight']:
        nn = {}
        nn['actor'] = A2CActor(act_input_d, hidden_layers[1], args.hidden_act, output_d,
                                'softmax', args.lr, args.lre, learner=learner, nntype = nntype, temp = args.temperature)
        if learner:
            nn['critic'] = A2CCritic(cri_input_d, hidden_layers[0], args.hidden_act, output_d,
                                     'linear', args.lrc, args.lre, learner=learner, nntype = nntype)

    elif nntype in ['presslight']:
        nn = DQN(input_d, hidden_layers, args.hidden_act, output_d, 'linear',
                 args.lr, args.lre, learner=learner, nntype=nntype)

    else:
        #raise not found exceptions
        assert 0, 'Supplied traffic signal control argument type '+str(nntype)+' does not exist.'

    return nn

def get_in_out_d(tsctype, n_phases, num_segments, tsc=None, netdata=None):
    # wz: decide the input and output dimensions of neural networks
    #+1 for the all red phase (i.e., terminal state, no vehicles at intersection)

    # how to take the info-sharing into consideration here?
    # the concated state dim should be: #local state + #neighboring state + #neighboring policy(# action i.e. #n_phase)
    #   1. get the number of neighboring agents (it would be easier if we pass it into here)
    #   2. scale the original dim with the number

    # wz: original state setting is, most recent or current phase (one of n_phase or the all_red phase),
    # the density and queue of incoming lanes at the intersection

    
    if tsctype == 'presslight':
        # PHASE-BASED pressure state (redesigned; matches CVLight victim + get_state):
        #   inc per phase (segmented): (n_phases-1)*num_segments + 1
        #   out per phase:             n_phases
        #   phase one-hot:             n_phases + 1
        #   phase_duration/g_max:      1
        input_d = ((n_phases - 1) * num_segments + 1) + n_phases + (n_phases + 1) + 1
        action_num = 2
        return input_d, action_num

    elif tsctype == 'cavlight':
        ''' original CAVLight state design
            # critic: 
            #   num_veh_in: 2 * ((n_phases - 1) * num_segments + 1) -> CV and UV -> one phase is left turning and has no segments
            #   avg_speed: 2 * ((n_phases - 1) * num_segments + 1)  -> CV and UV
            #   phase_idx: 1 * n_phases + 1 # add 1 for transition phase
            #   phase_duration: 1

            # actor:
            #   num_veh_in:  ((n_phases - 1) * num_segments + 1) -> CV 
            #   avg_speed: ((n_phases - 1) * num_segments + 1)  -> CV 
            #   phase_idx: 1 * n_phases + 1 # add 1 for transition phase
            #   phase_duration: 1
            
        '''
        # assume n_phases = 4
        input_d_critic = 4*((n_phases-1) * num_segments + 1) + n_phases + 2
        input_d_actor = 2*((n_phases-1) * num_segments + 1) + n_phases + 2
        action_num = 2 #n_phases  # 2
        return (input_d_critic, input_d_actor), action_num

    else:
        #raise not found exceptions
        assert 0, 'Supplied traffic signal control argument type '+str(tsc)+' does not exist.'

# att_state = state[1] + CTM_est_state for corr3 intersections.
# state[1] includes phase_to_one_hot and duration appended in nextphaserltsc.py:
#   actor_raw = get_avg_speed + get_num_vehicle_cav via _build_phase_array:
#               (n_phases-1)*num_segments+1 per vector (1 left-turn phase × 1 seg)
#   state[1] = 2*((n_phases-1)*num_segments+1) + (n_phases+1) + 1
# CTM_est_state = n_app*num_segments + (n_phases+1) + 1
_CORR3_N_APP = {'62532012': 3, '62477148': 4, '62500824': 4}

def get_in_out_d_att(n_phases, num_segments, tsc_id=None):
    if tsc_id in _CORR3_N_APP:
        n_app = _CORR3_N_APP[tsc_id]
        phase_d = (n_phases + 1) + 1  # phase_one_hot (n_phases green + 1 AR) + duration
        actor_raw_d = 2 * ((n_phases - 1) * num_segments + 1)
        state1_d = actor_raw_d + phase_d
        ctm_d = n_app * num_segments + phase_d
        input_d = state1_d + ctm_d
        return (input_d, input_d), 2

    # Plymouth formula: 3 THR phases × num_segments + 1 L-turn phase × 1 seg; CTM has speed+count
    input_d_critic = 2*(2*((n_phases-1) * num_segments + 1) + n_phases + 2)
    input_d_actor = 2*(2*((n_phases-1) * num_segments + 1) + n_phases + 2)
    return (input_d_critic, input_d_actor), 2

def get_canonical_att_dim(num_segments, P_MAX=4, K_MAX=4, include_ctm=True):
    """Canonical (padded) attacker-state dim for the shared/generalized attacker.
    Matches nextphaseattacker.canonicalize_state:
      include_ctm=True  (cavlight): [avg(A_MAX), cv(A_MAX), phase(P_MAX+1), prog(1)]
        + CTM block + geometry(2) -> 46 (S=3, P_MAX=K_MAX=4).
      include_ctm=False (presslight, phase-based pressure state):
        [inc(A_MAX), out(P_MAX), phase(P_MAX+1), time(1)] + geometry(2) -> 22 (S=3)."""
    A_MAX = (P_MAX - 1) * num_segments + 1
    if not include_ctm:
        # presslight: inc(A_MAX) + out(P_MAX) + phase(P_MAX+1) + time(1) + geom(2)
        return A_MAX + P_MAX + (P_MAX + 1) + 1 + 2
    state1 = 2 * A_MAX + (P_MAX + 1) + 1
    ctm = K_MAX * num_segments + (P_MAX + 1) + 1
    return state1 + ctm + 2

# Updated factory for PyTorch PPO models
def nn_factory_att(input_d,output_d, args,  learner, load, tsc_att, n_hidden):
    cri_input_d = input_d[0]
    act_input_d = input_d[1]
    hidden_layers = [[cri_input_d*3] * n_hidden, [act_input_d*3] * n_hidden]
    
    nn_dict = {}
    nn_dict['actor_app'] = PPOActor(act_input_d, hidden_layers[1], output_d, 
                               hidden_act=nn.ReLU, lr=args.lr)
    if learner:
        nn_dict['critic'] = PPOCritic(cri_input_d, hidden_layers[0],
                                 hidden_act=nn.ReLU, lr=args.lrc)
    return nn_dict 

def gen_att_neural_networks(args, netdata, tsc_ids, learner, load, n_hidden):
    neural_nets = {}

    if getattr(args, 'shared_att', False):
        # Centralized/generalized attacker: ONE network over the canonical padded
        # state (nextphaseattacker.canonicalize_state), shared by all intersections.
        # Keyed 'shared_att' regardless of geometry. Separate attacker per victim type:
        # presslight has no CTM block so its canonical dim is smaller (28 vs 46).
        include_ctm = (getattr(args, 'tsc', 'cavlight') != 'presslight')
        d = get_canonical_att_dim(args.num_segments, include_ctm=include_ctm)
        neural_nets['shared_att'] = nn_factory_att((d, d), 2, args, learner, load, 'shared_att', n_hidden)
    else:
        for tsc in tsc_ids:
            input_d, output_d = get_in_out_d_att(len(netdata['inter'][tsc]['green_phases']),
                                                 args.num_segments, tsc_id=tsc)
            neural_nets[tsc+'_att'] = nn_factory_att(input_d, output_d, args, learner, load, tsc+'_att', n_hidden)

    if load:
        path_dirs = [args.save_path]

        updates = args.updates if args.mode == 'test' else 0
        if args.mode == 'train':
            try:
                # Find the latest update number from saved models
                path = '/'.join(path_dirs + ['critic'])
                models_path = get_fp(args, path)
                models = [f for f in os.listdir(models_path) if f.endswith('.pt')]
                if getattr(args, 'shared_att', False):
                    models = [m for m in models if m.startswith('shared_att')]  # ignore old per-TSC ckpts
                if models:
                    updates = max([int(model.split('.')[0].split('_')[-1]) for model in models])
            except (FileNotFoundError, ValueError):
                 print("Could not find previous training files to determine update number. Starting from 0.")
                 updates = 0

        if updates > 0 or args.mode == 'test':
            print('Trying to load attacker parameters for update '+str(updates)+'...')
            for nn_key in neural_nets:  # 'shared_att' or '{tsc}_att'
                for n in neural_nets[nn_key]:
                    fname = '_'.join([nn_key, str(updates)])  # e.g. shared_att_15000 or 62477148_att_15000
                    path = get_fp(args, '/'.join(path_dirs+[n]))
                    filepath = os.path.join(path, fname)
                    print("LOAD PATH:", filepath)
                    neural_nets[nn_key][n].load_weights(filepath)
            print('... finished loading attacker parameters')

    return neural_nets

def gen_neural_networks(args, netdata, tsctype, tsc_ids, learner, load, n_hidden):
        neural_nets = {}
        tsc_names = ['cavlight', 'presslight']
        if tsctype in tsc_names:
            sess = None

            for tsc in tsc_ids:
                input_d, output_d = get_in_out_d(tsctype,
                                                 len(netdata['inter'][tsc]['green_phases']),
                                                 args.num_segments,
                                                 tsc=tsc, netdata=netdata)

                neural_nets[tsc] = nn_factory(tsctype,
                                              input_d,
                                              output_d,
                                              args,
                                              learner,
                                              load,
                                              tsc,
                                              n_hidden,
                                              sess=sess)

            if load:
                path_dirs = [args.save_path]
                updates = args.tsc_updates

                print('Trying to load '+str(tsctype)+' parameters for update '+str(updates)+'...')

                for tsc in tsc_ids:
                    if tsctype in ['cavlight']:
                        for n in neural_nets[tsc]:
                            fname = '_'.join([tsc, str(updates)])
                            path = '/'.join(path_dirs + [n, fname])
                            path = get_fp(args, path, True)
                            neural_nets[tsc][n].load_weights(path)
                    elif tsctype in ['presslight']:
                        fname = '_'.join([tsc, str(updates)])
                        path = '/'.join(path_dirs + [fname])
                        path = get_fp(args, path, True)
                        neural_nets[tsc].load_weights(path)

                print('... successfully loaded '+str(tsctype)+' parameters')
        return neural_nets

