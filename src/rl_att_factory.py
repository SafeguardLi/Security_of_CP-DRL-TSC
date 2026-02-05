from src.rlagents.PPO import PPOAgent

def rl_att_factory(rl_type, args, neural_network, exp_replay, rl_stats, n_actions, eps, tsc_id):
    if rl_type in ["ppo_att"]:
        return PPOAgent(neural_network,
                        eps,
                        exp_replay,
                        n_actions,
                        args.nsteps,
                        args.batch,
                        args.nreplay,
                        args.gamma,
                        rl_stats,
                        args.mode,
                        args.updates,
                        args,
                        tsc_id)
    else:
        #raise not found exceptions
        assert 0, 'Supplied rl argument type '+str(rl_type)+' does not exist.'
