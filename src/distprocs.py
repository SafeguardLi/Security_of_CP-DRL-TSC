import sys, os, subprocess, time
from multiprocessing import *
from src.simproc import SimProc
from src.learnerproc import LearnerProc
from src.networkdata import NetworkData
from src.sumosim import SumoSim
from src.helper_funcs import get_fp

import numpy as np

def get_sim(sim_str):
                                   
    cfg_fp = 'networks/' + sim_str + '/' + sim_str + '.sumocfg'
    net_fp = 'networks/' + sim_str + '/' + sim_str + '.net.xml'
    
    return cfg_fp, net_fp

class DistProcs:
    def __init__(self, args, tsc, mode):
        self.args = args

        if mode == 'train':
            if args.l < 1:
                args.l = 1
        elif mode == 'test':
            if args.l > 0:
                args.l = 0
        
        else:
            print('Input argument tsc '+str(tsc)+' not found, please provide valid tsc.')
            return

        if args.n < 0:
            args.n = 1

        if args.sim:
            args.cfg_fp, args.net_fp = get_sim(args.sim)

        # args.nreplay = int(args.nreplay/args.nsteps)

        barrier = Barrier(args.n+args.l)

        nd = NetworkData(args.net_fp)
        netdata = nd.get_net_data()

        sim = SumoSim(args.cfg_fp, args.sim_len, args.tsc, True, netdata, args, -1)
        sim.gen_sim()
        netdata = sim.update_netdata()
        sim.close()

        tsc_ids = netdata['inter'].keys()

        if mode in ['train','test']:
            rl_stats = self.create_mp_stats_dict(tsc_ids)
            exp_replays = self.create_mp_exp_replay(tsc_ids)           
        
        eps_rates = self.get_exploration_rates(args.eps, args.n, args.mode, args.sim)
        offsets = self.get_start_offsets(args.mode, args.sim_len, args.offset, args.n)

        sim_procs = [ SimProc(i, args, barrier, netdata, rl_stats, exp_replays, eps_rates[i], offsets[i]) for i in range(args.n)]

        if args.l > 0:
            learner_agents = self.assign_learner_agents( tsc_ids, args.l) 
            print('===========LEARNER AGENTS')
            for l in learner_agents:
                print('============== '+str(l))
            learner_procs = [ LearnerProc(i, args, barrier, netdata, learner_agents[i], rl_stats, exp_replays) for i in range(args.l)]
        else:
            learner_procs = []

        self.procs = sim_procs + learner_procs

    def run(self):
        print('Starting up all processes...')
        for p in self.procs:
            p.start()
                              
        for p in self.procs:
            p.join()

        print('...finishing all processes')

    def create_mp_stats_dict(self, tsc_ids):
        manager = Manager()
        rl_stats = manager.dict({})

        updates = 0
        if self.args.load and self.args.mode == 'train':
            try:
                path_dirs = [self.args.save_path, 'critic']
                models_path = get_fp(self.args, '/'.join(path_dirs))
                models = [f for f in os.listdir(models_path) if f.endswith('.pt')]
                if models:
                    updates = max([int(model.split('.')[0].split('_')[-1]) for model in models])
            except (FileNotFoundError, ValueError):
                print("Could not find previous models to determine update number. Starting from 0.")
                updates = 0

        for i in tsc_ids:
            rl_stats[i+'_att'] = manager.dict({})
            rl_stats[i+'_att']['n_exp'] = 0
            rl_stats[i+'_att']['updates'] = updates
            rl_stats[i+'_att']['max_r'] = 1.0
            
            # **FIX**: Initialize all expected keys for the 'online' dict.
            # This prevents the KeyError by ensuring 'critic' always exists.
            # Using manager.dict for the nested dictionary is crucial for multiprocessing.
            rl_stats[i+'_att']['online'] = manager.dict({
                'actor_app': None, 
                'critic': None
            })
            
            rl_stats[i+'_att']['target'] = None
            rl_stats[i+'_att']['pre_updates'] = updates

        # These stats appear to be global, not per-agent
        rl_stats['n_sims'] = 0
        rl_stats['total_sims'] = 104
        rl_stats['delay'] = manager.list()
        rl_stats['queue'] = manager.list()
        rl_stats['throughput'] = manager.list()

        return rl_stats

    def create_mp_exp_replay(self, tsc_ids):
        manager = Manager()
        return manager.dict({ tsc+'_att': manager.list() for tsc in tsc_ids })

    def assign_learner_agents(self, agents, n_learners):
        learner_agents = [ [] for _ in range(n_learners)]
        for agent, i in zip(agents, range(len(agents))):
            learner_agents[i%n_learners].append(agent)
        return learner_agents

    def get_exploration_rates(self, eps, n_actors, mode, net):
        if mode in ['test']:
            return [0.001 for _ in range(n_actors)]
        elif mode in ['train']:
            return np.linspace(1.0, eps, num = n_actors) 

    def get_start_offsets(self, mode, simlen, offset, n_actors):
        if mode in ['test']:
            return [0]*n_actors
        elif mode in ['train']:
            return np.linspace(0, simlen*offset, num = n_actors)