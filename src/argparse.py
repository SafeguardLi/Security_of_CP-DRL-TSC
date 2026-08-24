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
    parser.add_argument("-shared_att", default=False, action='store_true', dest='shared_att', help='centralized/generalized attacker: one shared attacker network over the canonical padded state, trained on merged experience from all intersections')
    parser.add_argument("-att_target", type=str, default='', dest='att_target', help='test-only: deploy the (shared) attacker at ONLY this intersection id; others run benign. Empty = attacker on all intersections')
    parser.add_argument("-inc_only_attack", default=False, action='store_true', dest='inc_only_attack', help='PressLight attack: restrict JSMA to the INCOMING block [0:A] only (spoof approaching CVs); exclude the OUT block (downstream-lane spoofing). Conservative threat model / out-injection ablation.')
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
    parser.add_argument("-att_model", type=str, default=None,
                        choices=['drl', 'minPressure', 'random'],
                        dest='att_model',
                        help='attack model / target-phase selector (all share the JSMA+injection pipeline): '
                             '"drl" = trained PPO attacker; "minPressure" = rule-based pressure heuristic '
                             '(switch away from the highest-pressure phase); "random" = uniform-random target '
                             '(baseline). Omit to run without any attack. default: None')
    parser.add_argument("-sdsm_defense", action='store_true', dest='sdsm_defense',
                        help='enable SDSM consistency defense (occupancy-map cross-check), default: False')
    parser.add_argument("-jsma_no_fallback", action='store_true', dest='jsma_no_fallback',
                        help='when JSMA finds no features, SKIP injection (s_eff=0 -> penalty) '
                             'instead of using the heuristic phase-fallback. default: False')
    parser.add_argument("-collect_sa", action='store_true', dest='collect_sa',
                        help='during benign test, log (state[1], action_dist, action_idx) per '
                             'intersection to _surrogate_data.p for surrogate (blackbox) training. '
                             'default: False')
    parser.add_argument("-surrogate_dir", type=str, default=None, dest='surrogate_dir',
                        help='dir holding trained <tsc>_surrogate.pt. When set, JSMA analyzes the '
                             'SURROGATE (blackbox) instead of the real actor; the real TSC still '
                             'decides. default: None (white-box)')
    parser.add_argument("-tsc_surrogate", action='store_true', dest='tsc_surrogate',
                        help='route the TSC benign DECISION through the surrogate (surrogate_act) '
                             'instead of the real actor, to check the surrogate as a controller '
                             '(tripinfo). Requires -surrogate_dir. default: False')
    parser.add_argument("-out_leave_speed", type=float, default=0.0, dest='out_leave_speed',
                        help='-traj_gen OUTGOING-lane fake-vehicle leaving speed (m/s). 0 = lane '
                             'free-flow speed. In a corridor the downstream is another signal, so a '
                             'LOW speed (queued departure) is realistic and keeps spoofed vehicles on '
                             'the lane longer -> higher sustained out-count. Sweep to measure the '
                             'attack impact vs leaving speed. default: 0 (free-flow)')
    parser.add_argument("-traj_gen", action='store_true', dest='traj_gen',
                        help='ENABLE the vehicle-trajectory-generation module (Gurobi optimizer) for '
                             'corr3: fake vehicles follow physically-realistic kinematic trajectories '
                             'instead of static target-position placement. default: False (static).')
    parser.add_argument("-ctm_vf", type=float, default=0.0, dest='ctm_vf',
                        help='CTM free-flow-speed recalibration (m/s). The built-in corr3 CTM uses '
                             'v_f=17.88, ~14%% above the real network median (~15.65), which makes '
                             'the CTM propagate/discharge vehicles too fast -> systematic UNDER-'
                             'prediction (near-segment worst). Setting -ctm_vf <real_vf> scales the '
                             'free-flow SENDING term by (ctm_vf/v_f) so effective propagation matches '
                             'reality, WITHOUT changing delta_x/cell geometry/detection. 0 = off '
                             '(no change). Try 15.65. default: 0')
    parser.add_argument("-inject_drop_rate", type=float, default=0.0, dest='inject_drop_rate',
                        help='DIAGNOSTIC: randomly DROP this fraction of feature injections (set to {}) '
                             'per decision, to match trajgen\'s Gurobi failure frequency. Run STATIC '
                             'with -inject_drop_rate 0.54: still high impact => PERSISTENCE was the '
                             'driver (few frozen injections suffice); collapses to ~benign => FREQUENCY. '
                             'default: 0 (no drop).')
    parser.add_argument("-opt_fallback", action='store_true', dest='opt_fallback',
                        help='DIAGNOSTIC: under -traj_gen, when the incoming Gurobi optimizer fails '
                             '(infeasible -> {}), fall back to placing the fakes at their TARGET '
                             'positions (frozen, static-style) so trajgen injects on EVERY '
                             'JSMA-success decision. Tests whether injection FREQUENCY (the ~54%%% '
                             'Gurobi failures) is what makes trajgen weak. NOT physically realizable.')
    parser.add_argument("-out_trail", action='store_true', dest='out_trail',
                        help='trajgen out-injection: use PER-TIMESTEP ids so the green cache accumulates '
                             'a DENSE TRAIL of every position the out-queue visited (faithful to the '
                             'isolated optimization_process behavior) instead of a bounded standing '
                             'queue (consistent ids). Stronger sustained out-signal, but can exceed jam '
                             'density on short lanes -> over-saturation. Only under -traj_gen. default: False.')
    parser.add_argument("-future_jsma", action='store_true', dest='future_jsma',
                        help='corr3 attack timeline fix: run JSMA on the CTM-projected FUTURE '
                             'victim state at the next decision time (t_next_Gmin_end) instead of '
                             'the CURRENT observed state[1]. Restores the e9bd9b9/NDSS timeline '
                             '(CTM estimates future state -> JSMA targets it -> trajectory realizes '
                             'the fakes to arrive by that decision). default: False (current state).')
    parser.add_argument("-opt_dynamic_n", action='store_true', dest='opt_dynamic_n',
                        help='trajgen feasibility: when optimization_process returns {} (infeasible), '
                             'retry with FEWER fake vehicles (drop one from the densest same-lane group '
                             'each round) until it solves or hits a floor. Converts a total injection '
                             'failure (esp. GREEN-phase, ~0.09 success: fast heterogeneous-speed fakes '
                             'packed into 5 m headway slots) into a PARTIAL injection. default: False.')
    parser.add_argument("-fake_spacing", type=float, default=5.0, dest='fake_spacing',
                        help='trajgen: target/initial spacing (meters) between adjacent same-lane fake '
                             'vehicles (assign_spot offset + adjust_ini_pos min-gap). Wider spacing gives '
                             'the fixed-order headway constraint slack to absorb per-vehicle speed '
                             'differences (the green-phase gap-collapse). default: 5.0 (original).')
    parser.add_argument("-fake_spd_cap", type=float, default=0.0, dest='fake_spd_cap',
                        help='trajgen: cap the fakes\' TARGET speed (m/s) at this value (0 = no cap). '
                             'On a served (green) approach get_fake_veh_final_state returns ~free-flow '
                             '17.88; capping injects gentler "slowing/forming-queue" fakes whose more '
                             'parallel trajectories satisfy headway -> higher green feasibility. default: 0.')
    parser.add_argument("-opt_init_spd_match", action='store_true', dest='opt_init_spd_match',
                        help='trajgen feasibility #1: set each fake\'s INITIAL speed = its TARGET speed '
                             '(get_initial_fake_state) instead of a hard-coded 17.88 m/s. A fake heading '
                             'into a standing queue is realistically already slowed, so it no longer has '
                             'to burn a full 17.88->0 deceleration -> removes the rear-catches-stopped-front '
                             'headway collapse at its source (the low-fspd failure). Physically MORE '
                             'realistic; does not break trajectory validity. default: False (17.88).')
    parser.add_argument("-opt_acc_low", type=float, default=-3.5, dest='opt_acc_low',
                        help='trajgen feasibility #2: deceleration lower bound (m/s^2) passed to the '
                             'optimizer. Default -3.5 is comfort braking; real emergency braking is '
                             '-6..-8. Setting e.g. -opt_acc_low -5 lets a fake reach a stopped target '
                             'sooner, shrinking the window where a rear fake overruns a stopped front '
                             'one -> higher feasibility, still physical. default: -3.5.')
    parser.add_argument("-marginal_delay", action='store_true', dest='marginal_delay',
                        help='reward the delay INCREASE vs a running benign baseline (signed) '
                             'instead of absolute delay, so delay-reducing attacks are '
                             'penalized. default: False')
    parser.add_argument("-force_flip", action='store_true', dest='force_flip',
                        help='DIAGNOSTIC: bypass JSMA/injection and force the victim to take '
                             'the attacker''s requested target action directly, to isolate the '
                             'reward/policy from the JSMA pipeline. default: False')
    parser.add_argument('-detect_mode',
                           type=str,
                           choices=['CAV','CAV_real', 'CAV_w_intersection', 'intersection'],
                           help="select detectors in the system (default: intersection)",
                           default='intersection')
    parser.add_argument("-act_ctm", action='store_true', help='enable CTM in state estimation, default: False')
    parser.add_argument("-act_lp", action='store_true', help='enable loop detector in CTM source node update, default: False')


    parser.add_argument("-max_attack_scale", type=int, default= 15, dest='max_attack_scale',
                        help='scale of attack, i.e. maximum num of fake veh to be insert, default: 15')
    parser.add_argument("-fake_veh_scale", type=float, default=1.0, dest='fake_veh_scale',
                        help='multiplier on fake-vehicle injection strength (fake_veh_weight). '
                             '>1 = stronger injection to cross the DQN Q-margin. Default 1.0 (no change).')

    args = parser.parse_args()
    # if args.tsc == 'actuated':
    #     args.tsc_program = 'actuated_' + args.tsc_program
    return args
