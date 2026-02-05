import time, os
from multiprocessing import *
# TensorFlow imports are removed
import numpy as np

from src.nn_factory import gen_neural_networks, gen_att_neural_networks
from src.rl_factory import rl_factory
from src.helper_funcs import write_line_to_file, check_and_make_dir, get_time_now, write_to_log, get_fp
from src.picklefuncs import save_data, load_data
from src.rl_att_factory import rl_att_factory

class LearnerProc(Process):
    def __init__(self, idx, args, barrier, netdata, agent_ids, rl_stats, exp_replay):
        Process.__init__(self)
        self.idx = idx
        self.args = args
        self.barrier = barrier
        self.netdata = netdata
        self.agent_ids = agent_ids 
        self.rl_stats = rl_stats
        self.exp_replay = exp_replay
        self.save_t = 0
        path_dirs = [self.args.save_replay]
        path = '/'.join(path_dirs)+'/'
        self.replay_fp = get_fp(self.args, path)

        if self.idx == 0:
            path = 'tmp/'                                                                    
            check_and_make_dir(path)
            now = get_time_now()
            self.updates_path = path + str(self.args.tsc)+'_'+str(now)+'_agent_updates.csv' 
            self.replay_path = path + str(self.args.tsc)+'_'+str(now)+'_agent_replay.csv' 
            self.n_exp_path = path + str(self.args.tsc)+'_'+str(now)+'_agent_nexp.csv' 
            self.tsc_ids = list(sorted(list(self.netdata['inter'].keys())))
            write_line_to_file( self.updates_path, 'a+', ','.join([now]+self.tsc_ids) )
            write_line_to_file( self.replay_path, 'a+', ','.join([now]+self.tsc_ids) )
            write_line_to_file( self.n_exp_path, 'a+', ','.join([now]+self.tsc_ids) )

    def run(self):
        learner = True

        att_neural_networks = gen_att_neural_networks(self.args, 
                                              self.netdata, 
                                              self.agent_ids,
                                              learner,
                                              self.args.load,
                                              self.args.n_hidden)

        print('learner proc trying to send weights------------')
        write_to_log(' LEARNER #'+str(self.idx)+' SENDING WEIGHTS...')

        att_neural_networks = self.distribute_weights(att_neural_networks) 

        print('learner waiting at barrier ------------')
        write_to_log(' LEARNER #'+str(self.idx)+' FINISHED SENDING WEIGHTS, WAITING AT BARRIER...')
        self.barrier.wait()
        write_to_log(' LEARNER #'+str(self.idx)+' GENERATING AGENTS...')

        if self.args.load_replay:
            self.load_replays()

        agents = self.gen_agents(att_neural_networks)

        print(att_neural_networks)
        print(self.agent_ids)

        print('learner proc '+str(self.idx)+' waiting at offset barrier------------')
        write_to_log(' LEARNER #'+str(self.idx)+' FINISHED GEN AGENTS, WAITING AT OFFSET BARRIER...')
        self.barrier.wait()
        write_to_log(' LEARNER #'+str(self.idx)+' BROKEN OFFSET BARRIER...')
        print('learner proc '+str(self.idx)+' broken offset barrier ------------')

        self.save_t = time.time()
        othert = time.time()
        
        while not self.finished_learning(self.agent_ids):
            for tsc in self.agent_ids:
                agent_id = tsc + '_att'
                agent = agents[agent_id] # Reference the agent object
                
                # Check the actual length of the experience replay list
                exp_buffer_list = self.exp_replay[agent_id]
                exp_buffer_len = len(exp_buffer_list)
                
                # The condition for training is that the buffer is full (nreplay)
                if exp_buffer_len >= self.args.nreplay:

                    # 1. SAMPLE THE BATCH (Safely grab the data) 
                    # --- FIX G: Use agent.sample_replay() to get the copy of the full list ---
                    sample_batch = agent.sample_replay()
                    
                    # Check if the sampled batch is valid before proceeding to training
                    if len(sample_batch) == 0:
                        # This should no longer happen with the fixed AttRLAgent.sample_replay
                        print(f"!!!!FATAL WARNING: Sampled batch for {agent_id} was empty. Skipping update.")
                        continue 

                    if self.rl_stats[agent_id]['updates'] == 0:
                        if self.args.save:
                            # Pass the sampled batch to save_replays_local
                            self.save_replays_local(tsc, sample_batch) 
                        print('\n \n ',agent_id, ' exp replay full, beginning batch updates******** \n \n')
                        self.rl_stats[agent_id]['n_exp'] = exp_buffer_len
                        
                    if self.rl_stats[agent_id]['updates'] < self.args.updates:
                        
                        # 2. Train the agent, passing the copied batch directly
                        agent.train_batch(sample_batch)
                        
                        # 3. Clear the buffer entirely after training
                        # This method now clears the shared list and resets the shared counter.
                        agent.clip_exp_replay() 
                        
                else:
                    # Report current collection status if the buffer is not yet full
                    if (exp_buffer_len%100==0) and (exp_buffer_len!=0):
                        print("Current Exp len:",exp_buffer_len)

            t = time.time()
            if t - othert > 90:
                othert = t
                n_replay = [str(len(self.exp_replay[i+'_att'])) for i in self.agent_ids]
                updates = [str(self.rl_stats[i+'_att']['updates']) for i in self.agent_ids]
                nexp = [str(self.rl_stats[i+'_att']['n_exp']) for i in self.agent_ids]
                write_to_log(' LEARNER #'+str(self.idx)+'\n'+str(self.agent_ids)+'\n'+str(nexp)+'\n'+str(n_replay)+'\n'+str(updates))                           

            if self.args.save:
                for tsc in self.agent_ids:
                    if self.update_to_save(tsc):
                        self.save_weights({tsc+'_att': att_neural_networks[tsc+'_att']})
        write_to_log(' LEARNER #'+str(self.idx)+' FINISHED TRAINING LOOP ===========')


    def update_to_save(self, tsc):
        return self.rl_stats[tsc+'_att']['updates'] > 0 and self.rl_stats[tsc+'_att']['updates'] % self.args.save_u == 0

    def time_to_save(self):
        t = time.time()                        
        if t - self.save_t > self.args.save_t:
            self.save_t = t
            return True
        return False

    def write_progress(self):
        self.write_training_progress()
        self.write_replay_progress()
        self.write_n_exp_progress()

    def finished_learning(self, agent_ids):
        for agent in agent_ids:
            if self.rl_stats[agent+'_att']['updates'] < self.args.updates:
                return False
        return True

    def gen_agents(self, neural_networks):
        agents = {}
        for agent in self.agent_ids:
            n_action = [2,1]
            agents[agent+'_att'] = rl_att_factory('ppo_att', self.args,
                              neural_networks[agent+'_att'], self.exp_replay[agent+'_att'], 
                              self.rl_stats[agent+'_att'], n_action, self.args.eps, agent)
        return agents
        
    def distribute_weights(self, neural_networks):
        for nn_name in neural_networks: # nn_name is 'INT_1_att', etc.
            actor1_weights = neural_networks[nn_name]['actor_app'].get_weights('online')
            critic_weights = neural_networks[nn_name]['critic'].get_weights('online')
            
            # Synchronize target/online weights (if you had target networks)
            neural_networks[nn_name]['actor_app'].set_weights(actor1_weights, 'online')
            neural_networks[nn_name]['critic'].set_weights(critic_weights, 'online')
            
            # Send weights to sim processes via the shared dictionary
            self.rl_stats[nn_name]['online']['actor_app'] = actor1_weights
            self.rl_stats[nn_name]['online']['critic'] = critic_weights
            
        return neural_networks

    def save_weights(self, neural_networks):
        path_dirs = [self.args.save_path]
        for nn in neural_networks:
            up_n = '_' + str(self.rl_stats[nn]['updates']) # + self.rl_stats[nn]['pre_updates'])
            
            path = '/'.join(path_dirs+['critic'])+'/'
            path = get_fp(self.args, path)
            check_and_make_dir(path)
            neural_networks[nn]['critic'].save_weights('online', path, nn + up_n)

            path = '/'.join(path_dirs+['actor_app'])+'/'
            path = get_fp(self.args, path)
            check_and_make_dir(path)
            neural_networks[nn]['actor_app'].save_weights('online', path, nn + up_n)

    def write_training_progress(self):
        updates = [str(self.rl_stats[i+'_att']['updates']) for i in self.tsc_ids]
        write_line_to_file( self.updates_path, 'a+', ','.join([get_time_now()]+updates) )

    def write_replay_progress(self):
        n_replay = [str(len(self.exp_replay[i+'_att'])) for i in self.tsc_ids]
        write_line_to_file( self.replay_path, 'a+', ','.join([get_time_now()]+n_replay) )

    def write_n_exp_progress(self):
        n_replay = [str(self.rl_stats[i+'_att']['n_exp']) for i in self.tsc_ids]
        write_line_to_file( self.n_exp_path, 'a+', ','.join([get_time_now()]+n_replay) )

    def save_replays_local(self, tsc_id, sample_replay):
        check_and_make_dir(self.replay_fp)
        save_data(self.replay_fp+tsc_id+'.p', sample_replay)
        print('FINISHED SAVING REPLAY FOR '+str(tsc_id))

    def load_replays(self):
        for _id in self.agent_ids:
            replay_fp = self.replay_fp+_id+'.p' 
            if os.path.isfile(replay_fp):
                data = load_data(replay_fp)
                rewards = []
                for traj in data:
                    for exp in traj:
                        rewards.append(abs(exp['r']))
                    self.exp_replay[_id+'_att'].append(traj) 
                # print('mean '+str(np.mean(rewards))+' std '+str(np.std(rewards))+' median '+str(np.median(rewards)))
                self.rl_stats[_id+'_att']['r_max'] = max(rewards)
                self.rl_stats[_id+'_att']['n_exp'] = len(self.exp_replay[_id+'_att'])
                # print(str(self.idx)+' LARGEST REWARD '+str(self.rl_stats[_id+'_att']['r_max']))
                print('SUCCESSFULLY LOADED REPLAY FOR '+str(_id))
            else:
                print('WARNING, tried to load experience replay at '+str(replay_fp)+' but it does not exist, continuing without loading...')