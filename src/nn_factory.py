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
from src.helper_funcs import get_fp

import sumolib

import os
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

    #original: if nntype == 'dqn':
    if nntype in ['cavlight']:
        nn = {}
        nn['actor'] = A2CActor(act_input_d, hidden_layers[1], args.hidden_act, output_d,
                                'softmax', args.lr, args.lre, learner=learner, nntype = nntype, temp = args.temperature)
        if learner:
            #only need ddpg critic on learner procs
            # wz: note the the output_d is not for the output dim of the critic nn but for the actor's output dim
            nn['critic'] = A2CCritic(cri_input_d, hidden_layers[0], args.hidden_act, output_d,
                                     'linear', args.lrc, args.lre, learner=learner, nntype = nntype) 
            # 20230709 modification:
            # 1. critic should use lrc as learning rate.
            # 2. the critic network is changed to a state value network -> the output_d is 1 now.

    else:
        #raise not found exceptions
        assert 0, 'Supplied traffic signal control argument type '+str(tsc)+' does not exist.'

    return nn

def get_in_out_d(tsctype, n_phases, num_segments):
    # wz: decide the input and output dimensions of neural networks
    #+1 for the all red phase (i.e., terminal state, no vehicles at intersection)

    # how to take the info-sharing into consideration here?
    # the concated state dim should be: #local state + #neighboring state + #neighboring policy(# action i.e. #n_phase)
    #   1. get the number of neighboring agents (it would be easier if we pass it into here)
    #   2. scale the original dim with the number

    # wz: original state setting is, most recent or current phase (one of n_phase or the all_red phase),
    # the density and queue of incoming lanes at the intersection

    
    if tsctype == 'cavlight': 
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

def get_in_out_d_att(n_phases, num_segments):
    # attcker's state is the SAME as CAVLight actor's state.
    # 4 phases, THR/THRL phases with 3 segments, L phase only 1 segment
    # avg_speed: 2*segments + 2*1, num_veh_inc: 2* segments + 2*1, num_veh_out: n_phases; phase indicator: n_phases + 1, phase duration: 1
    # attacker's crtic: the same state as attacker's actor
    
    # Update the input state: now action selector's state is the concatenation of real TSC state and CTM predicted state
    input_d_critic = 2*(2*((n_phases-1) * num_segments + 1) + n_phases + 2)
    input_d_actor = 2*(2*((n_phases-1) * num_segments + 1) + n_phases + 2)
    
    # action output for different actors; phase_idx actor is switch-or-stay and attack_scale actor is one value between 0 and 1
    return (input_d_critic, input_d_actor), 2 #(2, 1)

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
       
    for tsc in tsc_ids:
        input_d, output_d = get_in_out_d_att( len(netdata['inter'][tsc]['green_phases']),
                                            args.num_segments)
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
                if models:
                    updates = max([int(model.split('.')[0].split('_')[-1]) for model in models])
            except (FileNotFoundError, ValueError):
                 print("Could not find previous training files to determine update number. Starting from 0.")
                 updates = 0

        if updates > 0 or args.mode == 'test':
            print('Trying to load attacker parameters for update '+str(updates)+'...')
            for tsc in tsc_ids:                                    
                for n in neural_nets[tsc+'_att']:
                    fname = '_'.join([tsc]+['att']+ [str(updates)])
                    path = get_fp(args, '/'.join(path_dirs+[n]))
                    filepath = os.path.join(path, fname)
                    print("LOAD PATH:", filepath)
                    neural_nets[tsc+'_att'][n].load_weights(filepath)
            print('... finished loading attacker parameters')
    
    return neural_nets

def gen_neural_networks(args, netdata, tsctype, tsc_ids, learner, load, n_hidden):
        neural_nets = {}
        tsc_names = ['cavlight']
        if tsctype in tsc_names:
            sess = None

            #get desired neural net for each traffic signal controller

            net_fp = 'networks/' + args.sim + '/' + args.sim + '.net.xml'
            net = sumolib.net.readNet(net_fp)

            for tsc in tsc_ids:


                input_d, output_d = get_in_out_d(tsctype,
                                                 len(netdata['inter'][tsc]['green_phases']),
                                                 args.num_segments)

                neural_nets[tsc] = nn_factory(tsctype, 
                                              input_d, 
                                              output_d, 
                                              args, 
                                              learner, 
                                              load, 
                                              tsc,
                                              n_hidden,
                                              sess=sess)

            #load the saved weights
            # TODO: modify this for attacker
            if load:
                
                path_dirs = [args.save_path]
                updates = args.tsc_updates
                

                print('Trying to load '+str(tsctype)+' parameters for update '+str(updates)+'...')

                for tsc in tsc_ids:                                    
                    if tsctype in ['cavlight']:
                        for n in neural_nets[tsc]:
                            fname = '_'.join([tsc]+ [str(updates)])
                            path = '/'.join(path_dirs+[n,fname])
                            path = get_fp(args, path, True) # load TSC
                            neural_nets[tsc][n].load_weights(path)

                print('... successfully loaded '+str(tsctype)+' parameters')
        return neural_nets

