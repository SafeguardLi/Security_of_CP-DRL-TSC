import sys, os, time
from multiprocessing import *
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from src.sumosim import SumoSim
from src.nn_factory import gen_neural_networks, gen_att_neural_networks
from src.picklefuncs import save_data
from src.helper_funcs import check_and_make_dir, get_time_now, write_to_log, get_fp
from collections import defaultdict

class SimProc(Process):
    def __init__(self, idx, args, barrier, netdata, rl_stats, exp_replays, eps, offset):
        Process.__init__(self)
        self.idx = idx
        self.args = args
        self.barrier = barrier
        self.netdata = netdata
        self.sim = SumoSim(args.cfg_fp, args.sim_len, args.tsc, args.nogui, netdata, args, idx)
        self.rl_stats = rl_stats
        self.exp_replays = exp_replays
        self.eps = eps
        self.offset = offset
        self.initial = True 

    def run(self):
        learner = False
        # if self.args.load == True and self.args.mode == 'test':
        if self.args.load == True: # wz: get rid of the condition for test, so we can use pre_train method
            # origianlly, whenever we run training, our tsc's NNs, which are used to generate experience, will be initialized everytime
            # now, if we choose to load the model, our tsc's NNs will use our pre-trained model as well.
            load = True
        else:
            load = False

        # for each tsc agent, generate a neural network (dense layers)
        neural_networks = gen_neural_networks(self.args, 
                                              self.netdata, 
                                              self.args.tsc, 
                                              self.netdata['inter'].keys(),
                                              learner,
                                              True,
                                              self.args.n_hidden) # always load TSC NN
        
        # generate attacker NN
        att_neural_networks = gen_att_neural_networks(self.args, 
                                              self.netdata, 
                                              self.netdata['inter'].keys(),
                                              learner,
                                              load,
                                              self.args.n_hidden)

        print('sim proc '+str(self.idx)+' waiting at barrier ---------')
        write_to_log(' ACTOR #'+str(self.idx)+' WAITING AT SYNC WEIGHTS BARRIER...')
        self.barrier.wait()
        write_to_log(' ACTOR #'+str(self.idx)+'  BROKEN SYNC BARRIER...')
        # TODO: Oct 21, 2024
        # 1. update mode: this repo treat attacker as the main function and therefore, train and test are all about attackers
        # 2. We will use multiple actors for the hierarchical structure of actions: one phase_select actor and one scale_select actor
        
        if self.args.l > 0 and self.args.mode == 'train':
            att_neural_networks = self.sync_nn_weights(att_neural_networks, self.args.mode)
     
        #grab weights from learner or load from file

        if self.args.mode == 'train':
            reward_store = defaultdict(list)
            state_store = defaultdict(list)
            while not self.finished_updates():
                self.run_sim(neural_networks, att_neural_networks)
                if (self.eps == 1.0 or self.eps < 0.02):
                    self.write_to_csv(self.sim.sim_stats())
                #self.write_travel_times()
                for t in self.sim.attacker:
                    reward_store[t].append(self.sim.attacker[t].ep_rewards) # wz: extend(self.sim.tsc[t].ep_rewards)
                    state_store[t].append(self.sim.attacker[t].state_record)
                self.sim.close()
                time.sleep(2)
            for t in reward_store.keys():
                #wz: change for grid search. originally wrote by ujwal
                path = 'log/train_reward/processid_'+str(self.idx)+'/'
                path = get_fp(self.args, path)
                #path = os.path.join('experiments',f'{self.args.tsc}',f'Global_{self.args.global_critic}',f'CV_pen_rate_{self.args.pen_rate}',f'{self.args.sim}_{self.args.flow_type}_{self.args.turn_type}',f'gamma_{self.args.gamma}',f'eps_{self.args.eps}',f'temp_{self.args.temperature}',path)
                check_and_make_dir(path)
                save_data(os.path.join(path,'train_rewards_'+ str(t) + '_updates_' +str(self.args.updates) + '.pkl'), reward_store[t])
            for t in state_store.keys():
                #wz: change for grid search. originally wrote by ujwal
                path = 'log/train_state/processid_'+str(self.idx)+'/'
                path = get_fp(self.args, path)
                #path = os.path.join('experiments',f'{self.args.tsc}',f'Global_{self.args.global_critic}',f'CV_pen_rate_{self.args.pen_rate}',f'{self.args.sim}_{self.args.flow_type}_{self.args.turn_type}',f'gamma_{self.args.gamma}',f'eps_{self.args.eps}',f'temp_{self.args.temperature}',path)
                check_and_make_dir(path)
                save_data(os.path.join(path,'train_traffic_state_'+ str(t) + '_updates_' +str(self.args.updates)+ '.pkl'), state_store[t])


        elif self.args.mode == 'test':
            print(str(self.idx)+' test  waiting at offset ------------- '+str(self.offset))
            print(str(self.idx)+' test broken offset =================== '+str(self.offset))
            self.initial = False
            #just run one sim for stats
            self.run_sim(neural_networks, att_neural_networks)
            if (self.eps == 1.0 or self.eps < 0.02) and self.args.mode == 'test':
                self.write_to_csv(self.sim.sim_stats())
                with open( str(self.eps)+'.csv','a+') as f:
                    f.write('-----------------\n')
            self.write_sim_tsc_metrics()
            #for t in self.sim.tsc:
            #    check_and_make_dir('log/test_reward/')
            #    save_data('log/test_reward/test_rewards_' + str(t) + '_updates_' + str(self.args.updates) + '_eps_' + str(self.args.eps) + '.pkl', reward_store[t])
            # self.write_travel_times()
            self.sim.close()
            time.sleep(2)
        

        print('------------------\nFinished on sim process '+str(self.idx)+' Closing\n---------------')

    def run_sim(self, neural_networks, att_neural_networks = None):
        '''
        wz: generate sim (generate networks and vehicles) -> generate tsc -> run sim
        '''
        start_t = time.time()
        self.sim.gen_sim() # wz: vehicles are generated from here

        if self.initial is True:
            #if the initial sim, run until the offset time reached
            self.initial = False
            self.sim.run_offset(self.offset)
            print(str(self.idx)+' train  waiting at offset ------------- '+str(self.offset)+' at '+str(get_time_now()))
            write_to_log(' ACTOR #'+str(self.idx)+' FINISHED RUNNING OFFSET '+str(self.offset)+' to time '+str(self.sim.t)+' , WAITING FOR OTHER OFFSETS...')
            self.barrier.wait()
            print(str(self.idx)+' train  broken offset =================== '+str(self.offset)+' at '+str(get_time_now()))
            write_to_log(' ACTOR #'+str(self.idx)+'  BROKEN OFFSET BARRIER...')

        # wz: create tsc (initialize parameters of tsc)
        self.sim.create_tsc(self.eps, neural_networks)
        write_to_log('ACTOR #'+str(self.idx)+'  START RUN SIM...')

        if att_neural_networks != None:
            self.sim.create_attacker(self.rl_stats, self.exp_replays, self.eps, att_neural_networks)

        # wz: run simulation on vehicle and tsc, time step by time step until reaching simu_len
        self.sim.run()
        # wz: record
        print('sim finished in '+str(time.time()-start_t)+' on proc '+str(self.idx))
        write_to_log('ACTOR #'+str(self.idx)+'  FINISHED SIM...')

    def write_sim_tsc_metrics(self):
        #get data dict of all tsc in sim
        #where each tsc has dict of all metrics
        tsc_metrics = self.sim.get_tsc_metrics()
        #create file name and path for writing metrics data
        #now = datetime.datetime.now()
        #fname = str(self.idx)+'_'+str(now).replace(" ","-")
        fname = get_time_now()
        #write all metrics to correct path
        #path = 'metrics/'+str(self.args.tsc)
        # if self.args.mask == False:
            # wz: path is changed for grid search
            # wz: add exporation rate
        path = os.path.join('metrics', 'updates_' + str(self.args.updates))
        #path = os.path.join('experiments', f'{self.args.tsc}',
        #                    f'Global_{self.args.global_critic}',
        #                    f'CV_pen_rate_{self.args.pen_rate}',
        #                    f'{self.args.sim}_{self.args.flow_type}_{self.args.turn_type}',
        #                    f'gamma_{self.args.gamma}',f'eps_{self.args.eps}',f'temp_{self.args.temperature}', path)
        path = get_fp(self.args, path)
        check_and_make_dir(path)
        for tsc in tsc_metrics:
            for m in tsc_metrics[tsc]:
                mpath = path + '/'+str(m)+'/'+str(tsc)+'/'
                check_and_make_dir(mpath)
                save_data(mpath + fname + '_' + str(self.eps) + '_.p', tsc_metrics[tsc][m])

            # wz: add phase record
            check_and_make_dir(path + '/phase_record/'+str(tsc) +'/')
            save_data(path + '/phase_record/'+str(tsc)+'/' + fname + '.p', self.sim.phase_hist[tsc])
            # # yx: add action dist record
            # if tsc in self.sim.action_hist:
            #     check_and_make_dir(path + '/action_record/'+str(tsc) +'/')
            #     save_data(path + '/action_record/'+str(tsc)+'/' + fname + '.p', self.sim.action_hist[tsc])

            # wz: export queue estimation related records
            if self.args.estimate_queue:
                check_and_make_dir(path + '/queue_estimation/velocity_hist/'+str(tsc)+'/')
                check_and_make_dir(path + '/queue_estimation/rho_hist/' + str(tsc) + '/')
                check_and_make_dir(path + '/queue_estimation/EQ_hist/' + str(tsc) + '/')
                check_and_make_dir(path + '/queue_estimation/TQ_hist/' + str(tsc) + '/')
                save_data(path + '/queue_estimation/velocity_hist/'+str(tsc)+'/' + fname + '.p', self.sim.velo_hist_all)
                save_data(path + '/queue_estimation/rho_hist/'+str(tsc)+'/' + fname + '.p', self.sim.rho_hist_all)
                save_data(path + '/queue_estimation/EQ_hist/'+str(tsc)+'/' + fname + '.p', self.sim.EQ_hist_all)
                save_data(path + '/queue_estimation/TQ_hist/'+str(tsc)+'/' + fname + '.p', self.sim.TQ_hist_all)

        global_metrics = self.sim.get_global_metrics()
        for m in global_metrics:
            mpath = path + '/'+str(m)+'/'
            check_and_make_dir(mpath)
            save_data(mpath + fname + '_' + str(self.eps) + '_.p', global_metrics[m])



    def write_to_csv(self, data):
        with open( str(self.eps)+'.csv','a+') as f:
            f.write(','.join(data)+'\n')

  
    def finished_updates(self):
        # TODO: modify the condition to be attacker based
        for tsc in self.netdata['inter'].keys():
            print(tsc+'  exp replay size '+str(len(self.exp_replays[tsc+'_att'])))
            print(tsc+'  updates '+str(self.rl_stats[tsc+'_att']['updates']))
            if self.rl_stats[tsc+'_att']['updates'] < self.args.updates:
                return False
        return True

    # def sync_nn_weights(self, neural_networks, mode):
    #     # TODO: modify for attacker
    #     if mode == 'train':
    #         for nn in neural_networks:
                
    #             if self.args.tsc in ['cavlight','mmitiss']:
    #                 #sync actor weights
    #                 actor1_weights = self.rl_stats[nn]['online']['actor_app']
    #                 # actor2_weights = self.rl_stats[nn]['online']['actor_scale']
    #                 if (actor1_weights != None): # and (actor2_weights != None):
    #                     neural_networks[nn]['actor_app'].set_weights(actor1_weights, 'online')
    #                     # neural_networks[nn]['actor_scale'].set_weights(actor2_weights, 'online')
    #                 else:
    #                     print("Get NONE as NN weight")
    #             else:
    #                 #raise not found exceptions
    #                 assert 0, 'Supplied RL traffic signal controller attacker '+str(self.args.tsc)+'_att does not exist.'
        
    #     return neural_networks

    def sync_nn_weights(self, neural_networks, mode):
        if mode == 'train':
            for nn_name in neural_networks: # e.g., 'INT_1_att'
                # **FIX**: Added logic to sync both actor and critic weights
                actor_weights = self.rl_stats[nn_name]['online'].get('actor_app')
                critic_weights = self.rl_stats[nn_name]['online'].get('critic')

                if actor_weights is not None:
                    neural_networks[nn_name]['actor_app'].set_weights(actor_weights, 'online')
                else:
                    print(f"Warning: Did not find actor_app weights for {nn_name}")

                # This part was missing
                if critic_weights is not None:
                    # The critic model exists only in the learner, so check if it exists here
                    if 'critic' in neural_networks[nn_name]:
                         neural_networks[nn_name]['critic'].set_weights(critic_weights, 'online')
                else:
                    print(f"Warning: Did not find critic weights for {nn_name}")
        
        return neural_networks