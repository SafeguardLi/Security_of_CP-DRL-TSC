import os, datetime
import csv
import math
import numpy as np
from memoization import cached


# @cached
def phi(x_,t_, tau = 0.5, sigma = 0.5):
    # print('enter the phi')##
    x = abs(x_)/sigma
    t = abs(t_)/tau
    # print('xt',x.shape,t.shape)##
    return np.exp(-(x+t))

# @cached
def estimate_velocity(x_i, historic_data, max_velocity, time):
    '''
    wz
    input:
        X, the center position of a certian cell on a certain lane
        historic_data: a dictionary of velocity and position pairs of CAVs at each time step {time_step: [(velocity, position, veh_id),...]}
    output:
        the estimated average velocity of the given cell in a certain lane
    '''
    # Z = 0
    # phi_list = []
    # speed_list = []
    # t_i = np.arange(time,time - 41, -1)
    # print(np.array(list(historic_data.values()))[-1])
    # print('dimen',np.array(list(historic_data.values())).shape)##
    hist_Data_np = np.array(list(historic_data.values()))[-40:]
    hist_Data_t = np.array(list(historic_data.keys()))[-40:]
    # print('hist_Data_np:',hist_Data_np.shape)###
    # print('hist_Data_np',hist_Data_np.shape)
    hist_Data_np_s = hist_Data_np[:,:,0]
    hist_Data_np_pos = hist_Data_np[:,:,1]
    # print(hist_Data_np_s.shape,hist_Data_np_pos.shape)##
    delta_x_np = x_i - hist_Data_np_pos
    delta_t_np = time - hist_Data_t
    # print("tttt",delta_t_np)##
    phi_np = phi(delta_x_np, np.expand_dims(delta_t_np,axis=1))
    # if delta_x_np.shape[1] != phi_np.shape[1]:
        # print("xt",delta_x_np,delta_t_np)
    # print('phi_np:',phi_np.shape)####
    Z = np.sum(phi_np) ###TODO:check paper
    
    if np.sum(hist_Data_np_s)!=0:
        # print('z',Z)##
        # print('phi_np',phi_np,Z)##
        if Z==0:
            w_np=phi_np
        else:
            w_np = phi_np/Z
        # print(np.sum(np.multiply(w_np,hist_Data_np_s)))##
        if np.isnan(np.sum(np.multiply(w_np,hist_Data_np_s))):
            print('phi np,z',phi_np,Z)
        return np.sum(np.multiply(w_np,hist_Data_np_s))
    else:
        return max_velocity



    # for t in t_i:
    #     for veh in historic_data[t]:
            # speed, position, _ = veh
            # speed_list.append(speed)
            # delta_x = x_i - position
            # delta_t = time - t
            # phi_0 = phi( delta_x, delta_t)
            # phi_list.append(phi_0)
            # Z = Z + phi_0

    # print(phi_list)
    # if phi_list:
    #     w_list = np.array(phi_list)/Z
    # else:
    #     # print('here'*80)
    #     return max_velocity
    # speed_list = np.array(speed_list)

    # return np.sum(np.multiply(w_list,speed_list))

# @cached
def newell_franklin(v,max_velocity):
    '''
    wz
    input: velocity, a list of estimated average velocity of each cell in a certain lane
    output: rho (density), estimated density of each cell in the lane
    '''
    rho_jam = 0.143
    w = 25/3.6
    v_f = max_velocity
    # rho_list = []
    # for v in velocity:
    if v >= v_f:
        rho = 0
    else:
        rho = rho_jam/(1-v_f/w*np.log(1-v/v_f))
    # rho_list.append(rho)
    return rho

def check_and_make_dir(path):
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            print ("Creation of the directory "+str(path)+" failed")

def write_lines_to_file(fp, write_type, lines):
    with open(fp, write_type) as f:
        f.writelines([l+'\n' for l in lines])

def write_line_to_file(fp, write_type, line):
    with open(fp, write_type) as f:
        f.write(line+'\n')

def get_time_now():
    now = datetime.datetime.now()
    now = str(now).replace(" ","-")
    now = now.replace(":","-")
    return now

def write_to_log(s):
    fp = 'tmp/'
    check_and_make_dir(fp)
    fp += 'log.txt'
    t = get_time_now()
    write_line_to_file(fp, 'a+', t+':: '+s)

def write_line_to_csv(fp, line):
    with open( fp, "a+") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(line)

def get_fp(args, path, is_tsc = False):
    # previous path:
    # os.path.join('experiments', f'{args.tsc}', f'Global_{args.global_critic}',
    # f'CV_pen_rate_{args.pen_rate}', f'{args.sim}_{args.flow_type}_{args.turn_type}', f'gamma_{args.gamma}',
    # f'eps_{args.eps}',f'temp_{args.temperature}', path)
    if is_tsc:
        if not args.sumo_detect:
            fp = os.path.join('experiments', f'{args.tsc}',
                                f'CV_pen_rate_{args.pen_rate}',
                                f'{args.sim}_{args.flow_type}_{args.turn_type}', path)
        else:
            fp = os.path.join('experiments', f'{args.tsc}',
                                f'CAV_pen_rate_{args.pen_rate}',
                                f'{args.sim}_{args.flow_type}_{args.turn_type}', path)
    else:
        fp = os.path.join('experiments', f'attacker_{args.tsc}',
                                f'CAV_pen_rate_{args.pen_rate}',
                                f'{args.sim}_{args.flow_type}_{args.turn_type}', path)
    return fp

