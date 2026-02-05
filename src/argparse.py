import argparse, os
import argparse, os

def parse_cl_args():
    parser = argparse.ArgumentParser()

    #multi proc params
    parser.add_argument("-n", type=int, default=os.cpu_count()-1, dest='n', help='number of sim procs (parallel simulations) generating experiences, default: os.cpu_count()-1')
    parser.add_argument("-l", type=int, default=1, dest='l', help='number of parallel learner procs producing updates, default: 1')

    ##sumo params
    parser.add_argument("-sim", type=str, default=None, dest='sim', help='simulation scenario, default: lust, options:lust, single, double')
    parser.add_argument("-port", type=int, default=2000, dest='port', help='port to connect self.conn.server, default: 1000')
    parser.add_argument("-netfp", type=str, default='networks/double.net.xml', dest='net_fp', help='path to desired simulation network file, default: networks/double.net.xml')
    parser.add_argument("-sumocfg", type=str, default='networks/double.sumocfg', dest='cfg_fp', help='path to desired simulation configuration file, default: networks/double.sumocfg' )
    parser.add_argument("-mode", type=str, default='train', dest='mode', help='reinforcement mode, train (agents receive updates) or test (no updates), default:train, options: train, test'  )
    parser.add_argument("-tsc", type=str, default='websters', dest='tsc', help='traffic signal control algorithm, default:websters; options:sotl, maxpressure, dqn, ddpg'  )
    parser.add_argument("-simlen", type=int, default=36000, dest='sim_len', help='length of simulation in seconds/steps')
    parser.add_argument("-nogui", default=False, action='store_true', dest='nogui', help='disable gui, default: False')
    parser.add_argument("-scale", type=float, default=1.4, dest='scale', help='vehicle generation scale parameter, higher values generates more vehicles, default: 1.0')
    parser.add_argument("-demand", type=str, default='dynamic', dest='demand', help='vehicle demand generation patter, single limits vehicle network population to one, dynamic creates changing vehicle population, default:dynamic, options:single, dynamic')
    parser.add_argument("-flow_type", type=str, default='light', dest='flow_type', help='flow file specifier ')
    parser.add_argument("-turn_type", type=str, default='s', dest='turn_type', help='turn file specifier ')

    parser.add_argument("-offset", type=float, default=0.25, dest='offset', help='max sim offset fraction of total sim length, default: 0.3')

    # yx: sumo tsc program

    parser.add_argument("-tsc_program", type=str, default='0', dest='tsc_program', help='program ID for each traffic signal controller, default: 0')
    parser.add_argument("-no_random_flow", default=False, action='store_true', dest='no_random_flow', help='disable --randomize-flows, default: False')
    parser.add_argument("-record_position", default=False, action='store_true', dest='record_position', help='enable position record (the metric file would be very large once enabled) during testing, default: False')

    # wz: sumo CV scenario


    parser.add_argument("-num_segments",type=int,  default=1, dest='num_segments',
                        help='Number of segments to divide the road into for state space representation. (Note: Only compatible with presslight)')
    parser.add_argument("-cong_thresh",type=float,  default=1.0, dest='cong_thresh',
                        help='Congestion threshold for experience rejection')
    parser.add_argument("-global_critic",type=str,  default= 'none', dest='global_critic',
                        help='Decide how to pass global information to state space of critic, default: none, Option: none, total, sep, net. (Note: Only compatible with presslight_a2c)')
    parser.add_argument("-estimate_queue", default=False, action='store_true', dest='estimate_queue',
                        help='Whether to use Saif\'s Queue Estimation technique')
    parser.add_argument("-pen_rate_est", default=False, action='store_true', dest='pen_rate_est',
                        help='Whether to use penetration rate for queue estimation')
    parser.add_argument("-pen_rate", type=float, default=100.0, dest='pen_rate',
                        help='penetration rate (percentage of CVs in all vehicles), default: 100.0')
    parser.add_argument("-succ_detect_rate", type=float, default=100.0, dest='succ_detect_rate',
                        help='successful detection rate (for each UV, the probability it is detected by the CAV within the range), default: 100.0')
    parser.add_argument("-detect_range", type=float, default=80.0, dest='detec_range',
                        help='Detection range for each CAV, default: 80.0 (meters)')
    parser.add_argument("-mask", type=bool, default=True, dest='mask',
                        help='enable to mask normal vehicles, default: True')
    # wz: seed to generate vehicles randomly
    parser.add_argument("-seed", type=int, default= 123, dest='seed',
                        help='an int seed to generate vehicles with flow file randomly, default: 123')
    parser.add_argument("-dynamic_pen_rate", default=False, action='store_true', dest='dynamic_pen_rate',
                        help='to use a dynamic penetration rate along the simulation, following a sine function')

    #shared tsc params
    parser.add_argument("-gmin", type=int, default=100, dest='g_min', help='minimum green phase time (0.1s), default: 100')
    parser.add_argument("-y", type=int, default=40, dest='y', help='yellow change phase time (0.1s), default: 40')
    parser.add_argument("-r", type=int, default=10, dest='r', help='all red stop phase time (0.1s), default: 10')
    parser.add_argument("-detect_r", type=float, default=200, dest='detect_r', help='communication range of CAVs or intersections (m), default: 200.0')

    #websters params
    parser.add_argument("-cmin", type=int, default=600, dest='c_min', help='minimum cycle time (s), default: 60')
    parser.add_argument("-cmax", type=int, default=1800, dest='c_max', help='maximum cycle time (s), default: 180')
    parser.add_argument("-satflow", type=float, default=0.38, dest='sat_flow', help='lane vehicle saturation rate (veh/s), default: 0.38')
    parser.add_argument("-f", type=int, default=900, dest='update_freq', help='interval over which websters timing are computed (s), default: 900')

    #maxpressure params

    #self organizing traffic lights
    parser.add_argument("-theta", type=int, default=45, dest='theta', help='threshold to change signal (veh*s), default: 45')
    parser.add_argument("-omega", type=int, default=1, dest='omega', help='sotl param (veh*s), default: 1')
    parser.add_argument("-mu", type=int, default=3, dest='mu', help='sotl param(veh*s), default: 3')

    #rl params
    parser.add_argument("-eps", type=float, default=0.01, dest='eps', help='reinforcement learning explortation rate, default: 0.01')
    parser.add_argument("-nsteps", type=int, default=1, dest='nsteps', help='n step returns/max experience trajectory, default: 1')
    parser.add_argument("-nreplay", type=int, default=10000, dest='nreplay', help='maximum size of experience replay, default: 10000')
    parser.add_argument("-batch", type=int, default=32, dest='batch', help='batch size to sample from replay to train neural net, default: 32')
    parser.add_argument("-gamma", type=float, default=0.99, dest='gamma', help='reward discount factor, default: 0.99')
    parser.add_argument("-updates", type=int, default=10000, dest='updates', help='total number of batch updates for training, default: 10000')
    parser.add_argument("-tsc_updates", type=int, default=10000, dest='tsc_updates', help='total number of batch updates for training, default: 10000')

    parser.add_argument("-target_freq", type=int, default=50, dest='target_freq', help='target network batch update frequency, default: 50')
    parser.add_argument("-max_r", default=False, action='store_true', dest='max_r', help='reshape reward with max_r, default: False')

    #wz: rl decaying epsilon
    parser.add_argument("-eps_min", type=float, default=0.001, dest='eps_min',
                        help='reinforcement learning minimum explortation rate, noted that eps can not be smaller than this, default: 0.01')
    parser.add_argument("-eps_factor", type=float, default=0.9999, dest='eps_factor',
                        help='reinforcement learning decaying rate of explortation rate, default: 0.999')
    parser.add_argument("-decaying_eps", default=False, action='store_true', dest='decaying_eps', help='enable decaying epsilon, default: False')

    #neural net params
    parser.add_argument("-lr", type=float, default=0.0001, dest='lr', help='ddpg actor/dqn neural network learning rate, default: 0.0001')
    parser.add_argument("-lrc", type=float, default=0.001, dest='lrc', help='ddpg critic neural network learning rate, default: 0.001')
    parser.add_argument("-lre", type=float, default=0.00000001, dest='lre', help='neural network optimizer epsilon, default: 0.00000001')
    parser.add_argument("-hidden_act", type=str, default='elu', dest='hidden_act', help='neural network hidden layer activation, default: elu')
    parser.add_argument("-n_hidden", type=int, default=2, dest='n_hidden', help='neural network hidden layer scaling factor, default: 2; for presslight-a2c, it is the number of layers')
    
    parser.add_argument("-save_path", type=str, default='saved_models', dest='save_path', help='dir to save neural network weights, default: saved_models')
    parser.add_argument("-save_replay", type=str, default='saved_replays', dest='save_replay', help='dir to save experience replays, default: saved_replays')
    parser.add_argument("-load_replay", default=False, action='store_true', dest='load_replay', help='load experience replays if they exist')

    parser.add_argument("-save_t", type=int, default=1200, dest='save_t', help='interval in seconds between saving neural networks on learners, default: 120 (s)')
    parser.add_argument("-save_u", type=int, default=1000, dest='save_u', help='interval in updates between saving neural networks on learners, default: 1000 updates')
    parser.add_argument("-save", default=False, action='store_true', dest='save', help='use argument to save neural network weights')
    parser.add_argument("-load", default=False, action='store_true', dest='load', help='use argument to load neural network weights assuming they exist')

    #ddpg rl params
    parser.add_argument("-tau", type=float, default=0.005, dest='tau', help='ddpg online/target weight shifting tau, default: 0.005')
    parser.add_argument("-gmax", type=int, default=400, dest='g_max', help='maximum green phase time (s), default: 40')


    parser.add_argument("-temperature", type=float, default=1.0, dest='temperature',
                        help='temperature in softmax of Actor NNs output layer, default: 1.0')

    #marl
    parser.add_argument("-marl", type=str, default='sarl', dest='marl', help='multiagent RL setting, default: sarl, single agent RL.'
                                                                             'other types include: r_share, s_share, sr_share, srp_share,'
                                                                             'where r is reward, s is state, p is policy')
    parser.add_argument("-sync", default=False, action='store_true', dest='sync', help='to synchronize the action of agents or not')
    parser.add_argument("-all_veh_r", default=False, action='store_true', dest='all_veh_r',
                        help='to give both CV and Non-CV info in reward calculation')

    # CAV
    parser.add_argument("-data_source", type=str, default='sumo', dest='data_source',
                           help='data source for training or testing, either from SUMO traci or from perception in CARLA, default:sumo, options: sumo, carla')

    parser.add_argument("-sumo_detect", action='store_true', help='enable sumo to do perception, default: False')
    parser.add_argument("-drl_att", action='store_true', help='enable sumo to do perception, default: False')
    parser.add_argument('-detect_mode',
                           type=str,
                           choices=['CAV','CAV_real', 'CAV_w_intersection', 'intersection'],
                           help="select detectors in the system (default: intersection)",
                           default='intersection')
    parser.add_argument("-act_ctm", action='store_true', help='enable CTM in state estimation, default: False')
    parser.add_argument("-act_lp", action='store_true', help='enable loop detector in CTM source node update, default: False')


    parser.add_argument("-max_attack_scale", type=int, default= 15, dest='max_attack_scale',
                        help='scale of attack, i.e. maximum num of fake veh to be insert, default: 15')

    args = parser.parse_args()
    # if args.tsc == 'actuated':
    #     args.tsc_program = 'actuated_' + args.tsc_program
    return args
