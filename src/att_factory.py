
from src.attackers.nextphaseattacker import NextPhaseAttacker

from src.rl_att_factory import rl_att_factory

def att_factory(att_type, tl, args, netdata, rl_stats, exp_replay, neural_network, eps, conn):
    if att_type == 'ppo_att':
        # TODO: action for app_selector actor is 4+1, for scale_selector actor is 3
        n_action = [2] #[2,1] # hard code for 3 segments and 4 approaches #TODO
        att_agent = rl_att_factory(att_type, args,
                              neural_network, exp_replay, rl_stats, n_action, eps, tl)
        return NextPhaseAttacker(conn, tl, args.mode, netdata, args.r, args.y,
                              args.g_min, args.g_max, att_agent, att_type, eps, args.eps_min, 
                              args.eps_factor, args.num_segments, 
                              args.detect_r,args.sync, args.all_veh_r, args)
    else:
        #raise not found exceptions
        assert 0, 'Supplied attacker argument type '+str(att_type)+' does not exist.'
