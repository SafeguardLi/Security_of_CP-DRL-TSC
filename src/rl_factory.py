from src.rlagents.A2C import A2CAgent

def rl_factory(rl_type, args, neural_network, n_actions, eps, tsc_id):
    if rl_type in ['cavlight']:
        return A2CAgent(neural_network,
                        eps,
                        n_actions,
                        args.nsteps,
                        args.batch,
                        args.gamma,
                        args.mode,
                        args.updates,
                        args,
                        tsc_id)
    else:
        #raise not found exceptions
        assert 0, 'Supplied rl argument type '+str(rl_type)+' does not exist.'
