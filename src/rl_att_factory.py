from src.rlagents.PPO import PPOAgent
from src.rlagents.sybil_dqn import SybilDQNAgent
from src.attackers import sybil_attacker as _sa

def rl_att_factory(rl_type, args, neural_network, exp_replay, rl_stats, n_actions, eps, tsc_id):
    if rl_type == "sybil_dqn":
        _L = int(getattr(args, 'sybil_scale_levels', 3))
        return SybilDQNAgent(neural_network, eps, exp_replay, _sa.n_actions(_L),
                             args.nsteps, args.batch, args.nreplay, args.gamma,
                             rl_stats, args.mode, args.updates, args, tsc_id)
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
