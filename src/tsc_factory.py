from src.trafficsignalcontrollers.uniformcycletsc import UniformCycleTSC
from src.trafficsignalcontrollers.websterstsc import WebstersTSC
from src.trafficsignalcontrollers.maxpressuretsc import MaxPressureTSC
from src.trafficsignalcontrollers.optimizationtsc import OptimizationTSC
from src.trafficsignalcontrollers.sotltsc import SOTLTSC
from src.trafficsignalcontrollers.nextphaserltsc import NextPhaseRLTSC
from src.trafficsignalcontrollers.actuatedtsc import ActuatedTSC
from src.trafficsignalcontrollers.MMITISStsc import mmitissTSC
from src.rl_factory import rl_factory

def tsc_factory(tsc_type, tl, args, netdata, neural_network, eps, conn):
    if tsc_type == 'websters':
        return WebstersTSC(conn, tl, args.mode, netdata, args.r, args.y,
                           args.g_min, args.c_min,
                           args.c_max, args.sat_flow,
                           args.update_freq, args.detect_r)
    elif tsc_type == 'sotl':
        return SOTLTSC(conn, tl, args.mode, netdata, args.r, args.y,
                       args.g_min, args.theta, args.omega,
                       args.mu, args.detect_r)
    elif tsc_type == 'uniform':
        return UniformCycleTSC(conn, tl, args.mode, netdata, args.r, args.y, args.g_min, args.detect_r)
    elif tsc_type == 'maxpressure':
        return MaxPressureTSC(conn, tl, args.mode, netdata, args.r, args.y,
                              args.g_min, args.detect_r,args.estimate_queue, args.pen_rate, args.pen_rate_est, args)
    elif tsc_type == 'actuated':
        return ActuatedTSC(conn, tl, args.mode, netdata, args.r, args.y,
                           args.g_min, args.detect_r)
    elif tsc_type in ['cavlight']:
        tsc_agent = rl_factory(tsc_type, args,
                              neural_network, 2, eps, tl) #n_action is 2 for RL-TSC, switch or stay
        return NextPhaseRLTSC(conn, tl, args.mode, netdata, args.r, args.y,
                              args.g_min, args.g_max, tsc_agent, tsc_type, eps, args.eps_min, args.eps_factor, args.estimate_queue, args.num_segments, args.cong_thresh, args.detect_r,args.sync, args.all_veh_r,args.act_ctm, args)
    elif tsc_type == 'opt':
        return OptimizationTSC(conn, tl, args.mode, netdata, args.r, args.y,
                              args.g_min, args.detect_r, args.g_max, args.act_ctm)
    elif tsc_type == 'mmitiss':
        return mmitissTSC(conn, tl, args.mode, netdata, args.r, args.y,
                              args.g_min, args.detect_r, args.g_max, args.act_ctm, args)
    else:
        #raise not found exceptions
        assert 0, 'Supplied traffic signal control argument type '+str(tsc_type)+' does not exist.'
