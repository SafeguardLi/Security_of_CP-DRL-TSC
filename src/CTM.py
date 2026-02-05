import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traci
from src.helper_funcs import check_and_make_dir,get_time_now,get_fp
from src.picklefuncs import save_data
import json
import random

class CTM_model:
    def __init__(self, fp="./networks/plymouth/CTM_plymouth_1788_4ph.csv", sim_len = 2400):
        self.OPT_TSC_input = {}
        self.lp_inc_prc = None

        self.stop_cell_est = False # True # 

        self.latest_green_phase = "rrrrrrrrrrrrrGGGGG"

        self.t_next_Gmin_end = 0


        if fp == "./networks/plymouth/CTM_plymouth_1788_4ph.csv":
            self.ctm_version = '1788'
            self.sim_len = sim_len # total simulation length (s)
            self.step_tt  = np.zeros([sim_len]) # travel time
            self.step_delay = np.zeros([sim_len]) # delay

            self.state_comparison_CTM = []
            
            # Initialize Signal Controller
            P = 5 # number of signal phases + AR phase
            self.signal = np.zeros([self.sim_len, P]) # signal[row time][column phase], value 1 represent the phase is chosen
            
            network = pd.read_csv(fp, header = 0,
                        names=["cell_idx", "type", "n_pre_cell", "pr1", "pr2", "pr3",
                                "n_fol_cell","fo1","fo2","fo3", "jam_den", "capacity", 
                                "turn_l","turn_th", "turn_r","phase","demand","intersection",
                                "approach","num_lane","seg_idx","ncell2int","detect_cell","cell2lane","direction",
                                "detect_cell_0","detect_cell_1","detect_cell_2","detect_cell_3","detect_cell_4"])
            network = network.astype(str)
            network.type = network.type.astype(int)
            network.jam_den = network.jam_den.astype(float)
            network.capacity = network.capacity.astype(float)
            network.n_pre_cell = network.n_pre_cell.astype(int)
            network.n_fol_cell = network.n_fol_cell.astype(int)
            network.phase = network.phase.astype(int)
            network.turn_l = network.turn_l.astype(float)
            network.turn_th = network.turn_th.astype(float)
            network.turn_r = network.turn_r.astype(float)
            network.demand = network.demand.astype(float)
            network.approach = network.approach.astype(int)
            network.num_lane = network.num_lane.astype(int)
            network.seg_idx = network.seg_idx.astype(int)
            network.ncell2int = network.ncell2int.astype(int)
            network.detect_cell = network.detect_cell.astype(str)
            network.cell2lane = network.cell2lane.astype(str)
            network.direction = network.direction.astype(str)
            network.detect_cell_0 = network.detect_cell_0.astype(str)
            network.detect_cell_1 = network.detect_cell_1.astype(str)
            network.detect_cell_2 = network.detect_cell_2.astype(str)
            network.detect_cell_3 = network.detect_cell_3.astype(str)
            network.detect_cell_4 = network.detect_cell_4.astype(str)
            self.cell_pd = network
            self.total_demand = sum(network.demand)
            self.cell_dict = network.set_index('cell_idx').T.to_dict()

            # Intialize n, y, z as np.array to store cell status at the each time step
            self.n_cell = len(self.cell_dict) # total number of cells, including all types of cells
            self.n_i_t = np.zeros([self.sim_len,self.n_cell]) # number of vehicle at cell i at time t
            self.y_i_t = np.zeros([self.sim_len,self.n_cell]) # number of vehicle entering cell i at time [t,t+1)
            self.z_i_t = np.zeros([self.sim_len,self.n_cell]) # number of vehicle leaving cell i at time [t,t+1)
            self.avg_diff = [] # average difference between observation and CTM estimation over cells
            self.turn_ratio_dict = {}
            self.diverging_outflow = {}

            # for FD validation
            self.v_i_t = np.zeros([self.sim_len,self.n_cell]) # observed average vehicle speed at cell i at time t
            
    #         ffs=15; %free flow speed m/s
    #         t_step=2; %2s time step in ctm
    #         c_length=ffs*t_step; %cell length
    #         N=4; %Jam Density (one lane)
    #         Q=1; %Capacity (one lane)
    #         %Assuming triangle FD, calculate backward shockwave speed:
    #         kj=1000/(c_length/N);
    #         km=(Q/t_step*3600)/(ffs*3.6);
    #         w=(Q/t_step*3600)/(kj-km)/3.6;  %backwards shockwave speed m/s
    #         alpha=w/ffs;
    #         P=4;  %totally 4 phases
            
            # initialize parameters, assuming triangle FD
            self.v_f = 17.88 #10.90      # m/s
            self.delta_t = 1
            self.delta_x = self.v_f*self.delta_t
            for idx in self.cell_dict.keys():
                # here we calculate the parameter for each cell
                # N = self.cell_dict[idx]['jam_den'] 
                #Q = cell_dict[idx]['capacity']
                n_lane = self.cell_dict[idx]['num_lane']
                N = round(n_lane*self.v_f/7.5)
                q_max = 2000/3600*self.delta_t  #1800/3600*self.delta_t # 1412/3600*self.delta_t # qmax: 1500/3600*self.delta_t          # maximum flow rate per lane, veh/hr/lane -> veh/s/lane
                Q = q_max*n_lane
                #k_jam = 1000/(self.delta_x/(N/n_lane)) #  153 # 133 #              # jam density, veh/km
                k_jam = round(1000/7.5)
                # N = self.delta_x*n_lane*k_jam/1000 # jam number of vehicle per cell
                k_cri = (q_max*3600)/(self.v_f*3.6) # 36 #  #k_cri = (Q/self.delta_t*3600)/(self.v_f*3.6)  # veh/km
                w = (q_max*3600)/(k_jam-k_cri)/3.6   #shockwave speed, m/s
                alpha = w/self.v_f
                self.cell_dict[idx].update({
                    "N":N,
                    "Q":Q,
                    "q_max":q_max,
                    "k_jam":k_jam,
                    "k_cri":k_cri,
                    "w":w,
                    "alpha":alpha
                })
                # print("cell_dict: cell",idx,cell_dict[idx]["w"]) ###
                if self.cell_dict[idx]['type'] == 1: # diverging cell
                    self.turn_ratio_dict[idx] = {self.cell_dict[idx]['fo1']:self.cell_dict[idx]['turn_l'],
                                                self.cell_dict[idx]['fo2']:self.cell_dict[idx]['turn_th'],
                                                self.cell_dict[idx]['fo3']:self.cell_dict[idx]['turn_r']}
                    self.diverging_outflow[idx] = {self.cell_dict[idx]['fo1']:0,
                                                self.cell_dict[idx]['fo2']:0,
                                                self.cell_dict[idx]['fo3']:0}
            # specifically for plymouth, we hard code the lane-approach relationship table
            # given the edge id of the vehicle, we can get the approach idx in CTM; 
            # for veh in turning lane, we only list the intersection cells for that turning direction
            # for veh in normal/diverging/merging cell, we list cells and part of the intersection cells 
            # (based on the distance, veh will not fall into those intersection cells) 

            ################### FOR CTM_0328 ###################
            # lane2cell = {lane id: {approach:,cell_id_list:,num_of_intersection_cells_ahead_the_cell_list:},}
            self.lane2cell = {"-4.0.00_0":{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            "-4.0.00_1":{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            "-4.0.00_2":{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            ':46_2_0':{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            ':46_2_1':{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            ':46_2_2':{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            "-18.0.00_0":{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            "-18.0.00_1":{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            "-18.0.00_2":{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            ':27_2_0':{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            ':27_2_1':{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            ':27_2_2':{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            ':27_2_3':{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1}, 
                            "-11.0.00_0":{"app":"1","cell":["15"]+[str(i) for i in range(12,1,-1)],"nIntCell":0},
                            "-11.0.00_1":{"app":"1","cell":["14"]+[str(i) for i in range(12,1,-1)],"nIntCell":0},
                            "-11.0.00_2":{"app":"1","cell":["14"]+[str(i) for i in range(12,1,-1)],"nIntCell":0},
                            "-11.0.00_3":{"app":"1","cell":["13"]+[str(i) for i in range(12,1,-1)],"nIntCell":0},
                            "-3.0.00_0":{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            "-3.0.00_1":{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            "-3.0.00_2":{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            ':66_4_0':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            ':66_4_1':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            ':66_4_2':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            '-7.0.00_0':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},  
                            '-7.0.00_1':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            '-7.0.00_2':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            ':52_3_0':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            ':52_3_1':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            ':52_3_2':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            '-15.0.00_0':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            '-15.0.00_1':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            '-15.0.00_2':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            ':33_3_0':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            ':33_3_1':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            ':33_3_2':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            '-2.0.00_0':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            '-2.0.00_1':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            '-2.0.00_2':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            '2.0.00_0':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '2.0.00_1':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '2.0.00_2':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':33_0_0':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':33_0_1':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':33_0_2':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '15.0.00_0':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '15.0.00_1':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '15.0.00_2':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':52_0_0':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':52_0_1':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':52_0_2':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '7.0.00_0':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '7.0.00_1':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '7.0.00_2':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':66_0_0':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':66_0_1':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':66_0_2':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':66_0_3':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '3.0.00_0':{"app":"3","cell":["45","44"]+[str(i) for i in range(39,29,-1)],"nIntCell":0},
                            '3.0.00_1':{"app":"3","cell":["43","42"]+[str(i) for i in range(39,29,-1)],"nIntCell":0},
                            '3.0.00_2':{"app":"3","cell":["43","42"]+[str(i) for i in range(39,29,-1)],"nIntCell":0},
                            '3.0.00_3':{"app":"3","cell":["41","40"]+[str(i) for i in range(39,29,-1)],"nIntCell":0},
                            '11.0.00_0':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            '11.0.00_1':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            ':27_0_0':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            ':27_0_1':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            '18.0.00_0':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            '18.0.00_1':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            ':46_0_0':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            ':46_0_1':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            '4.0.00_0':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            '4.0.00_1':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            '-0.0.00_0':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            ':21_0_0':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            ':21_0_1':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            '-5.0.00_0':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            '-5.0.00_1':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            ':156_0_0':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            ':156_0_1':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            '-20.0.00_0':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            ':24_0_0':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            ':24_0_1':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            ':24_0_2':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            '-17.0.00_0':{"app":"5","cell":["75","74"]+[str(i) for i in range(69,59,-1)],"nIntCell":0},
                            '-17.0.00_1':{"app":"5","cell":["73","72"]+[str(i) for i in range(69,59,-1)],"nIntCell":0},
                            '-17.0.00_2':{"app":"5","cell":["71","70"]+[str(i) for i in range(69,59,-1)],"nIntCell":0},
                            '-16.0.00_0':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            '-16.0.00_1':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            ':73_4_0':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            ':73_4_1':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            '-8.0.00_0':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            '-8.0.00_1':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            ':140_3_0':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            ':140_3_1':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            ':140_3_2':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            '-12.0.00_0':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            '-12.0.00_1':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            '-12.0.00_2':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            '12.0.00_0':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            '12.0.00_1':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            ':140_0_0':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            ':140_0_1':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            ':140_0_2':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            '8.0.00_0':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            '8.0.00_1':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            '8.0.00_2':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            ':73_0_0':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            ':73_0_1':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            ':73_0_2':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            ':73_0_3':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            '16.0.00_0':{"app":"7","cell":["109","108","107","106"]+[str(i) for i in range(97,89,-1)],"nIntCell":0},
                            '16.0.00_1':{"app":"7","cell":["105","104","103","102"]+[str(i) for i in range(97,89,-1)],"nIntCell":0},
                            '16.0.00_2':{"app":"7","cell":["101","100","99","98"]+[str(i) for i in range(97,89,-1)],"nIntCell":0},
                            '16.0.00_3':{"app":"7","cell":["101","100","99","98"]+[str(i) for i in range(97,89,-1)],"nIntCell":0},
                            '17.0.00_0':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0},
                            '17.0.00_1':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0},
                            ':24_3_0':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0},
                            '20.0.00_0':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0},
                            ':156_2_0':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0},
                            '5.0.00_0':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0},
                            ':21_2_0':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0},
                            '0.0.00_0':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0}
                            }

        elif fp == "./networks/plymouth/CTM_plymouth_2407_1090.csv":
            self.ctm_version = 'v2'
            self.sim_len = sim_len # total simulation length (s)
            self.step_tt  = np.zeros([sim_len]) # travel time
            self.step_delay = np.zeros([sim_len]) # delay

            self.state_comparison_CTM = []
            
            # Initialize Signal Controller
            P = 4 # number of signal phases + AR phase
            self.signal = np.zeros([self.sim_len, P]) # signal[row time][column phase], value 1 represent the phase is chosen
            
            network = pd.read_csv(fp, header = 0,
                        names=["cell_idx", "type", "n_pre_cell", "pr1", "pr2", "pr3",
                                "n_fol_cell","fo1","fo2","fo3", "jam_den", "capacity", 
                                "turn_l","turn_th", "turn_r","phase","demand","intersection",
                                "approach","num_lane","seg_idx","ncell2int","detect_cell","direction",
                                "detect_cell_0","detect_cell_1","detect_cell_2","detect_cell_3","detect_cell_4",
                                "detect_cell_5","detect_cell_6","detect_cell_7"])
            network = network.astype(str)
            network.type = network.type.astype(int)
            network.jam_den = network.jam_den.astype(float)
            network.capacity = network.capacity.astype(float)
            network.n_pre_cell = network.n_pre_cell.astype(int)
            network.n_fol_cell = network.n_fol_cell.astype(int)
            network.phase = network.phase.astype(int)
            network.turn_l = network.turn_l.astype(float)
            network.turn_th = network.turn_th.astype(float)
            network.turn_r = network.turn_r.astype(float)
            network.demand = network.demand.astype(float)
            network.approach = network.approach.astype(int)
            network.num_lane = network.num_lane.astype(int)
            network.seg_idx = network.seg_idx.astype(int)
            network.ncell2int = network.ncell2int.astype(int)
            network.detect_cell = network.detect_cell.astype(str)
            network.direction = network.direction.astype(str)
            network.detect_cell_0 = network.detect_cell_0.astype(str)
            network.detect_cell_1 = network.detect_cell_1.astype(str)
            network.detect_cell_2 = network.detect_cell_2.astype(str)
            network.detect_cell_3 = network.detect_cell_3.astype(str)
            network.detect_cell_4 = network.detect_cell_4.astype(str)
            network.detect_cell_5 = network.detect_cell_5.astype(str)
            network.detect_cell_6 = network.detect_cell_6.astype(str)
            network.detect_cell_7 = network.detect_cell_7.astype(str)
            self.cell_pd = network
            self.total_demand = sum(network.demand)
            self.cell_dict = network.set_index('cell_idx').T.to_dict()

            # Intialize n, y, z as np.array to store cell status at the each time step
            self.n_cell = len(self.cell_dict) # total number of cells, including all types of cells
            self.n_i_t = np.zeros([self.sim_len,self.n_cell]) # number of vehicle at cell i at time t
            self.y_i_t = np.zeros([self.sim_len,self.n_cell]) # number of vehicle entering cell i at time [t,t+1)
            self.z_i_t = np.zeros([self.sim_len,self.n_cell]) # number of vehicle leaving cell i at time [t,t+1)
            self.avg_diff = [] # average difference between observation and CTM estimation over cells
            self.turn_ratio_dict = {}
            self.diverging_outflow = {}

            # for FD validation
            self.v_i_t = np.zeros([self.sim_len,self.n_cell]) # observed average vehicle speed at cell i at time t
            
            # initialize parameters, assuming triangle FD
            self.v_f = 10.90      # m/s
            self.delta_t = 1
            self.delta_x = self.v_f*self.delta_t
            for idx in self.cell_dict.keys():
                # here we calculate the parameter for each cell
                # N = self.cell_dict[idx]['jam_den'] 
                # Q = cell_dict[idx]['capacity']
                n_lane = self.cell_dict[idx]['num_lane']
                q_max = 1800/3600*self.delta_t #1824/3600*self.delta_t # v2          # maximum flow rate per lane, veh/hr/lane -> veh/s/lane
                Q = q_max*n_lane
                k_jam = 133 #              # jam density, veh/km
                N = self.delta_x*n_lane*k_jam/1000 # jam number of vehicle per cell
                k_cri = 46 #(Q/self.delta_t*3600)/(self.v_f*3.6)  # veh/km  # 46 # v2
                w = (q_max*3600)/(k_jam-k_cri)/3.6   #shockwave speed, m/s
                alpha = w/self.v_f
                self.cell_dict[idx].update({
                    "N":N,
                    "Q":Q,
                    "q_max":q_max,
                    "k_jam":k_jam,
                    "k_cri":k_cri,
                    "w":w,
                    "alpha":alpha
                })
                # print("cell_dict: cell",idx,cell_dict[idx]["w"]) ###
                if self.cell_dict[idx]['type'] == 1: # diverging cell
                    self.turn_ratio_dict[idx] = {self.cell_dict[idx]['fo1']:self.cell_dict[idx]['turn_l'],
                                                self.cell_dict[idx]['fo2']:self.cell_dict[idx]['turn_th'],
                                                self.cell_dict[idx]['fo3']:self.cell_dict[idx]['turn_r']}
                    self.diverging_outflow[idx] = {self.cell_dict[idx]['fo1']:0,
                                                self.cell_dict[idx]['fo2']:0,
                                                self.cell_dict[idx]['fo3']:0}



            ############## FOR CTM V2 #######################
            # app: approach
            # cell: cell id list
            # nIntCell: number of intersection cells
            self.lane2cell = {"-4.0.00_0":{"app":"1","cell":[str(i) for i in range(17,1,-1)],"nIntCell":2},
                            "-4.0.00_1":{"app":"1","cell":[str(i) for i in range(17,1,-1)],"nIntCell":2},
                            "-4.0.00_2":{"app":"1","cell":[str(i) for i in range(17,1,-1)],"nIntCell":2},
                            ':46_2_0':{"app":"1","cell":[str(i) for i in range(17,1,-1)],"nIntCell":2},
                            ':46_2_1':{"app":"1","cell":[str(i) for i in range(17,1,-1)],"nIntCell":2},
                            ':46_2_2':{"app":"1","cell":[str(i) for i in range(17,1,-1)],"nIntCell":2},
                            "-18.0.00_0":{"app":"1","cell":[str(i) for i in range(17,1,-1)],"nIntCell":2},
                            "-18.0.00_1":{"app":"1","cell":[str(i) for i in range(17,1,-1)],"nIntCell":2},
                            "-18.0.00_2":{"app":"1","cell":[str(i) for i in range(17,1,-1)],"nIntCell":2},
                            ':27_2_0':{"app":"1","cell":[str(i) for i in range(17,1,-1)],"nIntCell":2},
                            ':27_2_1':{"app":"1","cell":[str(i) for i in range(17,1,-1)],"nIntCell":2},
                            ':27_2_2':{"app":"1","cell":[str(i) for i in range(17,1,-1)],"nIntCell":2},
                            ':27_2_3':{"app":"1","cell":[str(i) for i in range(17,1,-1)],"nIntCell":2}, 
                            "-11.0.00_0":{"app":"1","cell":["23","22"]+[str(i) for i in range(17,1,-1)],"nIntCell":0},
                            "-11.0.00_1":{"app":"1","cell":["21","20"]+[str(i) for i in range(17,1,-1)],"nIntCell":0},
                            "-11.0.00_2":{"app":"1","cell":["21","20"]+[str(i) for i in range(17,1,-1)],"nIntCell":0},
                            "-11.0.00_3":{"app":"1","cell":["19","18"]+[str(i) for i in range(17,1,-1)],"nIntCell":0},
                            "-3.0.00_0":{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0}, 
                            "-3.0.00_1":{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0}, 
                            "-3.0.00_2":{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0}, 
                            ':66_4_0':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0}, 
                            ':66_4_1':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0}, 
                            ':66_4_2':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0}, 
                            '-7.0.00_0':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0},  
                            '-7.0.00_1':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0}, 
                            '-7.0.00_2':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0},
                            ':52_3_0':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0},
                            ':52_3_1':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0}, 
                            ':52_3_2':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0},
                            '-15.0.00_0':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0},
                            '-15.0.00_1':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0}, 
                            '-15.0.00_2':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0},
                            ':33_3_0':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0},
                            ':33_3_1':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0},
                            ':33_3_2':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0},
                            '-2.0.00_0':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0},
                            '-2.0.00_1':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0},
                            '-2.0.00_2':{"app":"2","cell":[str(i) for i in range(24,42)],"nIntCell":0},
                            '2.0.00_0':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            '2.0.00_1':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            '2.0.00_2':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            ':33_0_0':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            ':33_0_1':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            ':33_0_2':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            '15.0.00_0':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            '15.0.00_1':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            '15.0.00_2':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            ':52_0_0':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            ':52_0_1':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            ':52_0_2':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            '7.0.00_0':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            '7.0.00_1':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            '7.0.00_2':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            ':66_0_0':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            ':66_0_1':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            ':66_0_2':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            ':66_0_3':{"app":"3","cell":[str(i) for i in range(58,43,-1)],"nIntCell":3},
                            '3.0.00_0':{"app":"3","cell":[str(i) for i in range(67,64,-1)]+[str(i) for i in range(58,43,-1)],"nIntCell":0},
                            '3.0.00_1':{"app":"3","cell":[str(i) for i in range(64,61,-1)]+[str(i) for i in range(58,43,-1)],"nIntCell":0},
                            '3.0.00_2':{"app":"3","cell":[str(i) for i in range(64,61,-1)]+[str(i) for i in range(58,43,-1)],"nIntCell":0},
                            '3.0.00_3':{"app":"3","cell":[str(i) for i in range(61,58,-1)]+[str(i) for i in range(58,43,-1)],"nIntCell":0},
                            '11.0.00_0':{"app":"4","cell":[str(i) for i in range(68,86)],"nIntCell":0},
                            '11.0.00_1':{"app":"4","cell":[str(i) for i in range(68,86)],"nIntCell":0},
                            ':27_0_0':{"app":"4","cell":[str(i) for i in range(68,86)],"nIntCell":0},
                            ':27_0_1':{"app":"4","cell":[str(i) for i in range(68,86)],"nIntCell":0},
                            '18.0.00_0':{"app":"4","cell":[str(i) for i in range(68,86)],"nIntCell":0},
                            '18.0.00_1':{"app":"4","cell":[str(i) for i in range(68,86)],"nIntCell":0},
                            ':46_0_0':{"app":"4","cell":[str(i) for i in range(68,86)],"nIntCell":0},
                            ':46_0_1':{"app":"4","cell":[str(i) for i in range(68,86)],"nIntCell":0},
                            '4.0.00_0':{"app":"4","cell":[str(i) for i in range(68,86)],"nIntCell":0},
                            '4.0.00_1':{"app":"4","cell":[str(i) for i in range(68,86)],"nIntCell":0},
                            '-0.0.00_0':{"app":"5","cell":[str(i) for i in range(101,87,-1)],"nIntCell":4},
                            ':21_0_0':{"app":"5","cell":[str(i) for i in range(101,87,-1)],"nIntCell":4},
                            ':21_0_1':{"app":"5","cell":[str(i) for i in range(101,87,-1)],"nIntCell":4},
                            '-5.0.00_0':{"app":"5","cell":[str(i) for i in range(101,87,-1)],"nIntCell":4},
                            '-5.0.00_1':{"app":"5","cell":[str(i) for i in range(101,87,-1)],"nIntCell":4},
                            ':156_0_0':{"app":"5","cell":[str(i) for i in range(101,87,-1)],"nIntCell":4},
                            ':156_0_1':{"app":"5","cell":[str(i) for i in range(101,87,-1)],"nIntCell":4},
                            '-20.0.00_0':{"app":"5","cell":[str(i) for i in range(101,87,-1)],"nIntCell":4},
                            ':24_0_0':{"app":"5","cell":[str(i) for i in range(101,87,-1)],"nIntCell":4},
                            ':24_0_1':{"app":"5","cell":[str(i) for i in range(101,87,-1)],"nIntCell":4},
                            ':24_0_2':{"app":"5","cell":[str(i) for i in range(101,87,-1)],"nIntCell":4},
                            '-17.0.00_0':{"app":"5","cell":[str(i) for i in range(113,109,-1)]+[str(i) for i in range(101,87,-1)],"nIntCell":0},
                            '-17.0.00_1':{"app":"5","cell":[str(i) for i in range(109,105,-1)]+[str(i) for i in range(101,87,-1)],"nIntCell":0},
                            '-17.0.00_2':{"app":"5","cell":[str(i) for i in range(105,101,-1)]+[str(i) for i in range(101,87,-1)],"nIntCell":0},
                            '-16.0.00_0':{"app":"6","cell":[str(i) for i in range(114,132)],"nIntCell":0},
                            '-16.0.00_1':{"app":"6","cell":[str(i) for i in range(114,132)],"nIntCell":0},
                            ':73_4_0':{"app":"6","cell":[str(i) for i in range(114,132)],"nIntCell":0},
                            ':73_4_1':{"app":"6","cell":[str(i) for i in range(114,132)],"nIntCell":0},
                            '-8.0.00_0':{"app":"6","cell":[str(i) for i in range(114,132)],"nIntCell":0},
                            '-8.0.00_1':{"app":"6","cell":[str(i) for i in range(114,132)],"nIntCell":0},
                            ':140_3_0':{"app":"6","cell":[str(i) for i in range(114,132)],"nIntCell":0},
                            ':140_3_1':{"app":"6","cell":[str(i) for i in range(114,132)],"nIntCell":0},
                            ':140_3_2':{"app":"6","cell":[str(i) for i in range(114,132)],"nIntCell":0},
                            '-12.0.00_0':{"app":"6","cell":[str(i) for i in range(114,132)],"nIntCell":0},
                            '-12.0.00_1':{"app":"6","cell":[str(i) for i in range(114,132)],"nIntCell":0},
                            '-12.0.00_2':{"app":"6","cell":[str(i) for i in range(114,132)],"nIntCell":0},
                            '12.0.00_0':{"app":"7","cell":[str(i) for i in range(144,133,-1)],"nIntCell":7},
                            '12.0.00_1':{"app":"7","cell":[str(i) for i in range(144,133,-1)],"nIntCell":7},
                            ':140_0_0':{"app":"7","cell":[str(i) for i in range(144,133,-1)],"nIntCell":7},
                            ':140_0_1':{"app":"7","cell":[str(i) for i in range(144,133,-1)],"nIntCell":7},
                            ':140_0_2':{"app":"7","cell":[str(i) for i in range(144,133,-1)],"nIntCell":7},
                            '8.0.00_0':{"app":"7","cell":[str(i) for i in range(144,133,-1)],"nIntCell":7},
                            '8.0.00_1':{"app":"7","cell":[str(i) for i in range(144,133,-1)],"nIntCell":7},
                            '8.0.00_2':{"app":"7","cell":[str(i) for i in range(144,133,-1)],"nIntCell":7},
                            ':73_0_0':{"app":"7","cell":[str(i) for i in range(144,133,-1)],"nIntCell":7},
                            ':73_0_1':{"app":"7","cell":[str(i) for i in range(144,133,-1)],"nIntCell":7},
                            ':73_0_2':{"app":"7","cell":[str(i) for i in range(144,133,-1)],"nIntCell":7},
                            ':73_0_3':{"app":"7","cell":[str(i) for i in range(144,133,-1)],"nIntCell":7},
                            '16.0.00_0':{"app":"7","cell":[str(i) for i in range(165,158,-1)]+[str(i) for i in range(144,133,-1)],"nIntCell":0},
                            '16.0.00_1':{"app":"7","cell":[str(i) for i in range(158,151,-1)]+[str(i) for i in range(144,133,-1)],"nIntCell":0},
                            '16.0.00_2':{"app":"7","cell":[str(i) for i in range(151,144,-1)]+[str(i) for i in range(144,133,-1)],"nIntCell":0},
                            '16.0.00_3':{"app":"7","cell":[str(i) for i in range(151,144,-1)]+[str(i) for i in range(144,133,-1)],"nIntCell":0},
                            '17.0.00_0':{"app":"8","cell":[str(i) for i in range(166,184)],"nIntCell":0},
                            '17.0.00_1':{"app":"8","cell":[str(i) for i in range(166,184)],"nIntCell":0},
                            ':24_3_0':{"app":"8","cell":[str(i) for i in range(166,184)],"nIntCell":0},
                            '20.0.00_0':{"app":"8","cell":[str(i) for i in range(166,184)],"nIntCell":0},
                            ':156_2_0':{"app":"8","cell":[str(i) for i in range(166,184)],"nIntCell":0},
                            '5.0.00_0':{"app":"8","cell":[str(i) for i in range(166,184)],"nIntCell":0},
                            ':21_2_0':{"app":"8","cell":[str(i) for i in range(166,184)],"nIntCell":0},
                            '0.0.00_0':{"app":"8","cell":[str(i) for i in range(166,184)],"nIntCell":0}
                            }
        elif fp == "./networks/plymouth/CTM_Plymouth_122cells_4ph.csv":
            self.ctm_version = '122_4ph'
            self.sim_len = sim_len # total simulation length (s)
            self.step_tt  = np.zeros([sim_len]) # travel time
            self.step_delay = np.zeros([sim_len]) # delay

            self.state_comparison_CTM = []
            
            # Initialize Signal Controller
            P = 5 # number of signal phases + AR phase
            self.signal = np.zeros([self.sim_len, P]) # signal[row time][column phase], value 1 represent the phase is chosen
            
            network = pd.read_csv(fp, header = 0,
                        names=["cell_idx", "type", "n_pre_cell", "pr1", "pr2", "pr3",
                                "n_fol_cell","fo1","fo2","fo3", "jam_den", "capacity", 
                                "turn_l","turn_th", "turn_r","phase","demand","intersection",
                                "approach","num_lane","seg_idx","ncell2int","detect_cell","cell2lane","direction"])
            network = network.astype(str)
            network.type = network.type.astype(int)
            network.jam_den = network.jam_den.astype(float)
            network.capacity = network.capacity.astype(float)
            network.n_pre_cell = network.n_pre_cell.astype(int)
            network.n_fol_cell = network.n_fol_cell.astype(int)
            network.phase = network.phase.astype(int)
            network.turn_l = network.turn_l.astype(float)
            network.turn_th = network.turn_th.astype(float)
            network.turn_r = network.turn_r.astype(float)
            network.demand = network.demand.astype(float)
            network.approach = network.approach.astype(int)
            network.num_lane = network.num_lane.astype(int)
            network.seg_idx = network.seg_idx.astype(int)
            network.ncell2int = network.ncell2int.astype(int)
            network.detect_cell = network.detect_cell.astype(str)
            network.cell2lane = network.cell2lane.astype(str)
            network.direction = network.direction.astype(str)
            self.cell_pd = network
            self.total_demand = sum(network.demand)
            self.cell_dict = network.set_index('cell_idx').T.to_dict()

            # Intialize n, y, z as np.array to store cell status at the each time step
            self.n_cell = len(self.cell_dict) # total number of cells, including all types of cells
            self.n_i_t = np.zeros([self.sim_len,self.n_cell]) # number of vehicle at cell i at time t
            self.y_i_t = np.zeros([self.sim_len,self.n_cell]) # number of vehicle entering cell i at time [t,t+1)
            self.z_i_t = np.zeros([self.sim_len,self.n_cell]) # number of vehicle leaving cell i at time [t,t+1)
            self.avg_diff = [] # average difference between observation and CTM estimation over cells
            self.turn_ratio_dict = {}
            self.diverging_outflow = {}

            # for FD validation
            self.v_i_t = np.zeros([self.sim_len,self.n_cell]) # observed average vehicle speed at cell i at time t
            
    #         ffs=15; %free flow speed m/s
    #         t_step=2; %2s time step in ctm
    #         c_length=ffs*t_step; %cell length
    #         N=4; %Jam Density (one lane)
    #         Q=1; %Capacity (one lane)
    #         %Assuming triangle FD, calculate backward shockwave speed:
    #         kj=1000/(c_length/N);
    #         km=(Q/t_step*3600)/(ffs*3.6);
    #         w=(Q/t_step*3600)/(kj-km)/3.6;  %backwards shockwave speed m/s
    #         alpha=w/ffs;
    #         P=4;  %totally 4 phases
            
            # initialize parameters, assuming triangle FD
            self.v_f = 17.88 #10.90      # m/s
            self.delta_t = 1
            self.delta_x = self.v_f*self.delta_t
            for idx in self.cell_dict.keys():
                # here we calculate the parameter for each cell
                N = self.cell_dict[idx]['jam_den'] 
                #Q = cell_dict[idx]['capacity']
                n_lane = self.cell_dict[idx]['num_lane']
                q_max = 1800/3600*self.delta_t # 1412/3600*self.delta_t #           # maximum flow rate per lane, veh/hr/lane -> veh/s/lane
                Q = q_max*n_lane
                k_jam = 1000/(self.delta_x/(N/n_lane)) #  153 # 133 #              # jam density, veh/km
                # N = self.delta_x*n_lane*k_jam/1000 # jam number of vehicle per cell
                k_cri = (q_max*3600)/(self.v_f*3.6) # 36 #  #k_cri = (Q/self.delta_t*3600)/(self.v_f*3.6)  # veh/km
                w = (q_max*3600)/(k_jam-k_cri)/3.6   #shockwave speed, m/s
                alpha = w/self.v_f
                self.cell_dict[idx].update({
                    "N":N,
                    "Q":Q,
                    "q_max":q_max,
                    "k_jam":k_jam,
                    "k_cri":k_cri,
                    "w":w,
                    "alpha":alpha
                })
                # print("cell_dict: cell",idx,cell_dict[idx]["w"]) ###
                if self.cell_dict[idx]['type'] == 1: # diverging cell
                    self.turn_ratio_dict[idx] = {self.cell_dict[idx]['fo1']:self.cell_dict[idx]['turn_l'],
                                                self.cell_dict[idx]['fo2']:self.cell_dict[idx]['turn_th'],
                                                self.cell_dict[idx]['fo3']:self.cell_dict[idx]['turn_r']}
                    self.diverging_outflow[idx] = {self.cell_dict[idx]['fo1']:0,
                                                self.cell_dict[idx]['fo2']:0,
                                                self.cell_dict[idx]['fo3']:0}
            # specifically for plymouth, we hard code the lane-approach relationship table
            # given the edge id of the vehicle, we can get the approach idx in CTM; 
            # for veh in turning lane, we only list the intersection cells for that turning direction
            # for veh in normal/diverging/merging cell, we list cells and part of the intersection cells 
            # (based on the distance, veh will not fall into those intersection cells) 

            ################### FOR CTM_0328 WITH 4 PHASES ###################
            self.lane2cell = {"-4.0.00_0":{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            "-4.0.00_1":{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            "-4.0.00_2":{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            ':46_2_0':{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            ':46_2_1':{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            ':46_2_2':{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            "-18.0.00_0":{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            "-18.0.00_1":{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            "-18.0.00_2":{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            ':27_2_0':{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            ':27_2_1':{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            ':27_2_2':{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1},
                            ':27_2_3':{"app":"1","cell":[str(i) for i in range(12,1,-1)],"nIntCell":1}, 
                            "-11.0.00_0":{"app":"1","cell":["15"]+[str(i) for i in range(12,1,-1)],"nIntCell":0},
                            "-11.0.00_1":{"app":"1","cell":["14"]+[str(i) for i in range(12,1,-1)],"nIntCell":0},
                            "-11.0.00_2":{"app":"1","cell":["14"]+[str(i) for i in range(12,1,-1)],"nIntCell":0},
                            "-11.0.00_3":{"app":"1","cell":["13"]+[str(i) for i in range(12,1,-1)],"nIntCell":0},
                            "-3.0.00_0":{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            "-3.0.00_1":{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            "-3.0.00_2":{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            ':66_4_0':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            ':66_4_1':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            ':66_4_2':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            '-7.0.00_0':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},  
                            '-7.0.00_1':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            '-7.0.00_2':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            ':52_3_0':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            ':52_3_1':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            ':52_3_2':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            '-15.0.00_0':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            '-15.0.00_1':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0}, 
                            '-15.0.00_2':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            ':33_3_0':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            ':33_3_1':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            ':33_3_2':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            '-2.0.00_0':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            '-2.0.00_1':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            '-2.0.00_2':{"app":"2","cell":[str(i) for i in range(16,28)],"nIntCell":0},
                            '2.0.00_0':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '2.0.00_1':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '2.0.00_2':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':33_0_0':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':33_0_1':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':33_0_2':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '15.0.00_0':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '15.0.00_1':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '15.0.00_2':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':52_0_0':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':52_0_1':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':52_0_2':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '7.0.00_0':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '7.0.00_1':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '7.0.00_2':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':66_0_0':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':66_0_1':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':66_0_2':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            ':66_0_3':{"app":"3","cell":[str(i) for i in range(39,29,-1)],"nIntCell":2},
                            '3.0.00_0':{"app":"3","cell":["45","44"]+[str(i) for i in range(39,29,-1)],"nIntCell":0},
                            '3.0.00_1':{"app":"3","cell":["43","42"]+[str(i) for i in range(39,29,-1)],"nIntCell":0},
                            '3.0.00_2':{"app":"3","cell":["43","42"]+[str(i) for i in range(39,29,-1)],"nIntCell":0},
                            '3.0.00_3':{"app":"3","cell":["41","40"]+[str(i) for i in range(39,29,-1)],"nIntCell":0},
                            '11.0.00_0':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            '11.0.00_1':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            ':27_0_0':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            ':27_0_1':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            '18.0.00_0':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            '18.0.00_1':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            ':46_0_0':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            ':46_0_1':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            '4.0.00_0':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            '4.0.00_1':{"app":"4","cell":[str(i) for i in range(46,58)],"nIntCell":0},
                            '-0.0.00_0':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            ':21_0_0':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            ':21_0_1':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            '-5.0.00_0':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            '-5.0.00_1':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            ':156_0_0':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            ':156_0_1':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            '-20.0.00_0':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            ':24_0_0':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            ':24_0_1':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            ':24_0_2':{"app":"5","cell":[str(i) for i in range(69,59,-1)],"nIntCell":2},
                            '-17.0.00_0':{"app":"5","cell":["75","74"]+[str(i) for i in range(69,59,-1)],"nIntCell":0},
                            '-17.0.00_1':{"app":"5","cell":["73","72"]+[str(i) for i in range(69,59,-1)],"nIntCell":0},
                            '-17.0.00_2':{"app":"5","cell":["71","70"]+[str(i) for i in range(69,59,-1)],"nIntCell":0},
                            '-16.0.00_0':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            '-16.0.00_1':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            ':73_4_0':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            ':73_4_1':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            '-8.0.00_0':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            '-8.0.00_1':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            ':140_3_0':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            ':140_3_1':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            ':140_3_2':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            '-12.0.00_0':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            '-12.0.00_1':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            '-12.0.00_2':{"app":"6","cell":[str(i) for i in range(76,88)],"nIntCell":0},
                            '12.0.00_0':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            '12.0.00_1':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            ':140_0_0':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            ':140_0_1':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            ':140_0_2':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            '8.0.00_0':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            '8.0.00_1':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            '8.0.00_2':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            ':73_0_0':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            ':73_0_1':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            ':73_0_2':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            ':73_0_3':{"app":"7","cell":[str(i) for i in range(97,89,-1)],"nIntCell":4},
                            '16.0.00_0':{"app":"7","cell":["109","108","107","106"]+[str(i) for i in range(97,89,-1)],"nIntCell":0},
                            '16.0.00_1':{"app":"7","cell":["105","104","103","102"]+[str(i) for i in range(97,89,-1)],"nIntCell":0},
                            '16.0.00_2':{"app":"7","cell":["101","100","99","98"]+[str(i) for i in range(97,89,-1)],"nIntCell":0},
                            '16.0.00_3':{"app":"7","cell":["101","100","99","98"]+[str(i) for i in range(97,89,-1)],"nIntCell":0},
                            '17.0.00_0':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0},
                            '17.0.00_1':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0},
                            ':24_3_0':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0},
                            '20.0.00_0':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0},
                            ':156_2_0':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0},
                            '5.0.00_0':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0},
                            ':21_2_0':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0},
                            '0.0.00_0':{"app":"8","cell":[str(i) for i in range(110,122)],"nIntCell":0}
                            }

        else:
            raise NotImplementedError
        
    
    def run_CTM(self, curr_phase, curr_t, pred_horizon=10):
        '''
        INPUT:
        curr_phase = int, current signal phase, index from 0 to (#phases-1) for green phase; index as #phase (or -1) represents the AR
        curr_t = int, current time step
        pred_horizon = int, prediction horizon (number of time steps)
        
        OUTPUT:
        the predicted cell status for the pred_horizon length of time.
        the estimated total vehicle delay, travel time at each time step in the prediction horizon.
        '''
        
        if self.sim_len < pred_horizon + curr_t:
            print("WARNING: Prediction length over the simulation length! Enforce the pred horizon to be the difference between current time and simulation length")
            pred_horizon = int(self.sim_len - curr_t)
        
        # get cell status at time t
#         pred_n_i_t = np.zeros([pred_horizon,self.n_cell])
#         pred_n_i_t[0,:] = self.n_i_t[curr_t,:] 
        self.signal[curr_t:curr_t+pred_horizon, curr_phase] = 1 # current phase value be set as 1
        
        
       
        #####update cell status for pre-horizon of time based on cell types######
        
        for t in range(curr_t, curr_t+pred_horizon):
            
            ## update leaving flow
            for i in range(self.n_cell):
                if self.cell_dict[str(i+1)]["type"] in [0,2,3,5]: # ordinary, intersection, merging, source cell
                    # cell i
                    Q_i = self.cell_dict[str(i+1)]["Q"]
                    N_i = self.cell_dict[str(i+1)]["N"]
                    alpha = self.cell_dict[str(i+1)]["alpha"]
                    n_i = self.n_i_t[t,i]
                    
                    # following cells of i. for these types, # of fol is 1.
                    for f_cell in range(self.cell_dict[str(i+1)]["n_fol_cell"]):
                        fo_cell_idx = self.cell_dict[str(i+1)]["fo"+str(f_cell+1)]
                        Q_i_f = self.cell_dict[fo_cell_idx]["Q"]
                        N_i_f = self.cell_dict[fo_cell_idx]["N"]
                        n_i_f = self.n_i_t[t,int(fo_cell_idx)-1]
                        
                        # cal z_i_t: minimum of ni(t),Qi(t),Qi+1(t),a(Ni+1(t)-ni+1(t))
                        # print("cell",i,"time",t,"calculate z_i_t: min",[n_i, Q_i, Q_i_f, alpha*(N_i_f - n_i_f)]) ###
                        if self.cell_dict[str(i+1)]["type"] == 5:
                            self.z_i_t[t,i] = min([n_i, Q_i, Q_i_f])
                        else:
                            if alpha*(N_i_f - n_i_f) < 0:
                                f_cell_f = 0
                            else:
                                f_cell_f = alpha*(N_i_f - n_i_f)
                            self.z_i_t[t,i] = min([n_i, Q_i, Q_i_f, f_cell_f]) #should we use z_i_t or directly update the self.z_i_t?
#                             if i == 10:
#                                 print("cell 11: [n_i, Q_i, Q_i_f, f_cell_f] at t",t, [n_i, Q_i, Q_i_f, f_cell_f]) ###
                        if (self.cell_dict[str(i+1)]["type"] == 2) and (self.cell_dict[str(i+1)]["phase"] != 0): # intersection cell
                            phase = self.cell_dict[str(i+1)]["phase"] - 1 # ranging from 0 to #phase
                            self.z_i_t[t,i] = self.z_i_t[t,i]*self.signal[t,phase]
                            
                
                elif self.cell_dict[str(i+1)]["type"] == 1: # diverging cell
                    # following cells of i
                    Q_i = self.cell_dict[str(i+1)]["Q"]
                    N_i = self.cell_dict[str(i+1)]["N"]
                    alpha = self.cell_dict[str(i+1)]["alpha"]
#                     print("alpha",alpha,'cell',i+1)
                    n_i = self.n_i_t[t,i]
                    total_max_outflow = min([n_i,Q_i])
                    total_outflow = 0
#                     if i == 11:
#                             print("cell 12 at time",t,"total_max_out_flow = min[n_i,Q_i]", [n_i,Q_i]) ###
                    
                    
                    for f_cell in range(self.cell_dict[str(i+1)]["n_fol_cell"]):
                        fo_cell_idx = self.cell_dict[str(i+1)]["fo"+str(f_cell+1)]
                        Q_i_f = self.cell_dict[fo_cell_idx]["Q"]
                        N_i_f = self.cell_dict[fo_cell_idx]["N"]
                        n_i_f = self.n_i_t[t,int(fo_cell_idx)-1]
                        
#                         if f_cell == 0:
#                             turn_ratio = self.cell_dict[str(i+1)]["turn_l"]
#                         elif f_cell == 1:
#                             turn_ratio = self.cell_dict[str(i+1)]["turn_th"]
#                         elif f_cell == 2:
#                             turn_ratio = self.cell_dict[str(i+1)]["turn_r"]
                        #print("self.turn_ratio_dict[i][fo_cell_idx]",self.turn_ratio_dict,i,fo_cell_idx)
                        j_flow = total_max_outflow*self.turn_ratio_dict[str(i+1)][fo_cell_idx]
                        
                        
                        # cal z_i_t: minimum of ni(t),Qi(t),Qi+1(t),a(Ni+1(t)-ni+1(t))
                        if alpha*(N_i_f-n_i_f) < 0:
                            f_cell_f = 0
                        else:
                            f_cell_f = alpha*(N_i_f-n_i_f)
                        outflow = min([j_flow,Q_i_f,f_cell_f]) # note this line is different from original code
                        
#                         if i == 11:
#                             print("cell 12 at time",t,"to following cell",fo_cell_idx," min([j_flow,Q_i_f,f_cell_f])", [j_flow,Q_i_f,f_cell_f]) ###
                            
                        # print("outflow for cell",i+1,"from fol_cell",f_cell+1," at time",t,[j_flow,Q_i_f,alpha*(N_i_f-n_i_f)]) ###
                        # 0307: we would consider that even one of following cells is congested, the other cells
                        #    should be able to continue to hold traffic flows. -> then we need one array to store the outflow 
                        #    of non-congested directions. 
                        if outflow <= 0:
                            # total_outflow = 0
                            # break
                            self.diverging_outflow[str(i+1)][fo_cell_idx] = 0
                        else:
                            self.diverging_outflow[str(i+1)][fo_cell_idx] = outflow
                            total_outflow += outflow
                    
                    self.z_i_t[t,i] = total_outflow
                
                elif self.cell_dict[str(i+1)]["type"] == 4: # sink cell
                    self.z_i_t[t,i] = self.n_i_t[t,i]
                else:
                    print("Error: wrong cell type for cell "+str(i+1))
        
            ## update incoming flow
            for i in range(self.n_cell):
                if self.cell_dict[str(i+1)]["type"] in [0,1,2,4]: # ordinary, intersection, diverging, sink cell
                    
                    # previous cells
                    for p_cell in range(self.cell_dict[str(i+1)]["n_pre_cell"]):
                        pr_cell_idx = self.cell_dict[str(i+1)]["pr"+str(p_cell+1)]
                        
                        if self.cell_dict[pr_cell_idx]["type"] == 1:
                            # self.y_i_t[t,i] = self.z_i_t[t,int(pr_cell_idx)-1]*self.turn_ratio_dict[pr_cell_idx][str(i+1)]
                            self.y_i_t[t,i] = self.diverging_outflow[pr_cell_idx][str(i+1)]
#                             if pr_cell_idx == '12':
#                                 print("time",t,"cell",i+1,"recieve flow from cell",pr_cell_idx,"get",self.y_i_t[t,i]) ###
                        else:
                            self.y_i_t[t,i] = self.z_i_t[t,int(pr_cell_idx)-1]
#                         if i == 10:
#                             print("time",t,"cell 11 sending flow",self.y_i_t[t,i])
                        if self.y_i_t[t,i] < 0:
                            print("negative y!",self.y_i_t[t,i],"cell",i+1,"time",t) ###
                    
                elif self.cell_dict[str(i+1)]["type"] == 3: # merging cell
                    Q_i = self.cell_dict[str(i+1)]["Q"]
                    N_i = self.cell_dict[str(i+1)]["N"]
                    alpha = self.cell_dict[str(i+1)]["alpha"]
                    n_i = self.n_i_t[t,i]
                    total_inflow = 0
                    
                    # previous cells
                    for p_cell in range(self.cell_dict[str(i+1)]["n_pre_cell"]):
                        pr_cell_idx = self.cell_dict[str(i+1)]["pr"+str(p_cell+1)]
                        Q_i_p = self.cell_dict[pr_cell_idx]["Q"]
                        N_i_p = self.cell_dict[pr_cell_idx]["N"]
                        n_i_p = self.n_i_t[t,int(pr_cell_idx)-1]
                        total_inflow += self.z_i_t[t,int(pr_cell_idx)-1]
                    
#                     if total_inflow > min([Q_i,alpha*(N_i-n_i)]):
#                         print("WARNING: cell",i+1,"at time",t,": inflow over the capacity")
#                         self.y_i_t[t,i] = min([Q_i,alpha*(N_i-n_i)]) ### TODO: should we update z_i_t then?
#                     else:
                    self.y_i_t[t,i] = total_inflow
                        
                elif self.cell_dict[str(i+1)]["type"] == 5: # source cell
                    # in our case, we use demand. But why original code use 0?
                    # self.y_i_t[t,i] = self.cell_dict[str(i+1)]["demand"]

                    # if lp is activated, use lp info; we can use a self.lp_inc dict to store the info
                    if (self.lp_inc_prc != None) and (curr_t == t):
                        self.y_i_t[t,i] = self.lp_inc_prc[str(i+1)]
                    else:
                        self.y_i_t[t,i] = self.cell_dict[str(i+1)]["demand"]
                else:
                    print("Error: wrong cell type for cell "+str(i+1))
        
        
            ## update number of vehicle in each cell
            for i in range(self.n_cell):
                self.n_i_t[t+1,i] = round(self.n_i_t[t,i] + self.y_i_t[t,i] - self.z_i_t[t,i],3) # to avoid negative n due to float calculation
#                 if i in [11,13]: #[101,103]:
#                     print("time",t,"cell",i+1,"self.n_i_t[t,i]",self.n_i_t[t,i]," + self.y_i_t[t,i] ",self.y_i_t[t,i] ,"- self.z_i_t[t,i]",self.z_i_t[t,i]) ###
                if self.n_i_t[t+1,i] < 0:
                    print("negative n! cell",i+1,"time",t,"z",self.z_i_t[t,i],"y",self.y_i_t[t,i],"n",self.n_i_t[t,i])
                    self.n_i_t[t+1,i] = 0
                    if self.cell_dict[str(i+1)]["type"] == 1:
                        print("self.diverging_outflow[str(i+1)]",self.diverging_outflow[str(i+1)])
                        print("upstream cell",i,"time",t,"n",self.n_i_t[t,i-1],"z",self.z_i_t[t,i-1],"y",self.y_i_t[t,i-1])

                
                # update travel time and delay
                self.step_tt[t] += self.n_i_t[t,i] # vehicle inside the cell
                self.step_delay[t] += self.n_i_t[t,i] - self.z_i_t[t,i] # vehicle that are still waiting
            
        total_delay = np.sum(self.step_delay[curr_t:curr_t+pred_horizon])
        pred_net_status = self.n_i_t[curr_t:curr_t+pred_horizon,:]
            
        return total_delay, pred_net_status
    
    def run_CTM_MMITSS(self, curr_phase, curr_t, pred_horizon,n_i_t_input):
        '''
        INPUT:
        curr_phase = int, current signal phase, index from 0 to (#phases-1) for green phase; index as #phase (or -1) represents the AR
        curr_t = int, current time step
        pred_horizon = int, prediction horizon (number of time steps)
        n_i_t_input = saved n_i_t in previous planning stage
        
        OUTPUT:
        the predicted cell status for the pred_horizon length of time.
        '''
        '''
        if self.sim_len < pred_horizon + curr_t:
            print("WARNING: Prediction length over the simulation length! Enforce the pred horizon to be the difference between current time and simulation length")
            pred_horizon = int(self.sim_len - curr_t)
        '''
        # get cell status at time t
#         pred_n_i_t = np.zeros([pred_horizon,self.n_cell])
#         pred_n_i_t[0,:] = self.n_i_t[curr_t,:] 
        n_i_t=n_i_t_input.copy()
        compensate_row = np.empty((pred_horizon, 122))  # Or use np.zeros, np.ones, etc., depending on what you need
        compensate_row[:] = np.nan  # Filling the row with NaNs, or replace with the value you want
        n_i_t = np.concatenate((n_i_t, compensate_row), axis=0)
        z_i_t=n_i_t.copy()#create local z_i_t
        y_i_t=n_i_t.copy()#create local y_i_t
        signal=np.zeros((n_i_t.shape[0],4))
        signal[curr_t:curr_t+pred_horizon, curr_phase] = 1 # current phase value be set as 1
        
        
       
        #####update cell status for pre-horizon of time based on cell types######
        
        for t in range(curr_t, curr_t+pred_horizon):
            
            ## update leaving flow
            for i in range(self.n_cell):
                if self.cell_dict[str(i+1)]["type"] in [0,2,3,5]: # ordinary, intersection, merging, source cell
                    # cell i
                    Q_i = self.cell_dict[str(i+1)]["Q"]
                    N_i = self.cell_dict[str(i+1)]["N"]
                    alpha = self.cell_dict[str(i+1)]["alpha"]
                    n_i = n_i_t[t,i]
                    
                    # following cells of i. for these types, # of fol is 1.
                    for f_cell in range(self.cell_dict[str(i+1)]["n_fol_cell"]):
                        fo_cell_idx = self.cell_dict[str(i+1)]["fo"+str(f_cell+1)]
                        Q_i_f = self.cell_dict[fo_cell_idx]["Q"]
                        N_i_f = self.cell_dict[fo_cell_idx]["N"]
                        n_i_f = n_i_t[t,int(fo_cell_idx)-1]
                        
                        # cal z_i_t: minimum of ni(t),Qi(t),Qi+1(t),a(Ni+1(t)-ni+1(t))
                        # print("cell",i,"time",t,"calculate z_i_t: min",[n_i, Q_i, Q_i_f, alpha*(N_i_f - n_i_f)]) ###
                        if self.cell_dict[str(i+1)]["type"] == 5:
                            z_i_t[t,i] = min([n_i, Q_i, Q_i_f])
                        else:
                            if alpha*(N_i_f - n_i_f) < 0:
                                f_cell_f = 0
                            else:
                                f_cell_f = alpha*(N_i_f - n_i_f)
                            z_i_t[t,i] = min([n_i, Q_i, Q_i_f, f_cell_f]) #should we use z_i_t or directly update the self.z_i_t?
#                             if i == 10:
#                                 print("cell 11: [n_i, Q_i, Q_i_f, f_cell_f] at t",t, [n_i, Q_i, Q_i_f, f_cell_f]) ###
                        if (self.cell_dict[str(i+1)]["type"] == 2) and (self.cell_dict[str(i+1)]["phase"] != 0): # intersection cell
                            phase = self.cell_dict[str(i+1)]["phase"] - 1 # ranging from 0 to #phase
                            z_i_t[t,i] = z_i_t[t,i]*signal[t,phase]
                            
                
                elif self.cell_dict[str(i+1)]["type"] == 1: # diverging cell
                    # following cells of i
                    Q_i = self.cell_dict[str(i+1)]["Q"]
                    N_i = self.cell_dict[str(i+1)]["N"]
                    alpha = self.cell_dict[str(i+1)]["alpha"]
#                     print("alpha",alpha,'cell',i+1)
                    n_i = n_i_t[t,i]
                    total_max_outflow = min([n_i,Q_i])
                    total_outflow = 0
#                     if i == 11:
#                             print("cell 12 at time",t,"total_max_out_flow = min[n_i,Q_i]", [n_i,Q_i]) ###
                    
                    
                    for f_cell in range(self.cell_dict[str(i+1)]["n_fol_cell"]):
                        fo_cell_idx = self.cell_dict[str(i+1)]["fo"+str(f_cell+1)]
                        Q_i_f = self.cell_dict[fo_cell_idx]["Q"]
                        N_i_f = self.cell_dict[fo_cell_idx]["N"]
                        n_i_f = n_i_t[t,int(fo_cell_idx)-1]
                        
#                         if f_cell == 0:
#                             turn_ratio = self.cell_dict[str(i+1)]["turn_l"]
#                         elif f_cell == 1:
#                             turn_ratio = self.cell_dict[str(i+1)]["turn_th"]
#                         elif f_cell == 2:
#                             turn_ratio = self.cell_dict[str(i+1)]["turn_r"]
                        #print("self.turn_ratio_dict[i][fo_cell_idx]",self.turn_ratio_dict,i,fo_cell_idx)
                        j_flow = total_max_outflow*self.turn_ratio_dict[str(i+1)][fo_cell_idx]
                        
                        
                        # cal z_i_t: minimum of ni(t),Qi(t),Qi+1(t),a(Ni+1(t)-ni+1(t))
                        if alpha*(N_i_f-n_i_f) < 0:
                            f_cell_f = 0
                        else:
                            f_cell_f = alpha*(N_i_f-n_i_f)
                        outflow = min([j_flow,Q_i_f,f_cell_f]) # note this line is different from original code
                        
#                         if i == 11:
#                             print("cell 12 at time",t,"to following cell",fo_cell_idx," min([j_flow,Q_i_f,f_cell_f])", [j_flow,Q_i_f,f_cell_f]) ###
                            
                        # print("outflow for cell",i+1,"from fol_cell",f_cell+1," at time",t,[j_flow,Q_i_f,alpha*(N_i_f-n_i_f)]) ###
                        # 0307: we would consider that even one of following cells is congested, the other cells
                        #    should be able to continue to hold traffic flows. -> then we need one array to store the outflow 
                        #    of non-congested directions. 
                        if outflow <= 0:
                            # total_outflow = 0
                            # break
                            self.diverging_outflow[str(i+1)][fo_cell_idx] = 0
                        else:
                            self.diverging_outflow[str(i+1)][fo_cell_idx] = outflow
                            total_outflow += outflow
                    
                    z_i_t[t,i] = total_outflow
                
                elif self.cell_dict[str(i+1)]["type"] == 4: # sink cell
                    z_i_t[t,i] = n_i_t[t,i]
                else:
                    print("Error: wrong cell type for cell "+str(i+1))
        
            ## update incoming flow
            for i in range(self.n_cell):
                if self.cell_dict[str(i+1)]["type"] in [0,1,2,4]: # ordinary, intersection, diverging, sink cell
                    
                    # previous cells
                    for p_cell in range(self.cell_dict[str(i+1)]["n_pre_cell"]):
                        pr_cell_idx = self.cell_dict[str(i+1)]["pr"+str(p_cell+1)]
                        
                        if self.cell_dict[pr_cell_idx]["type"] == 1:
                            # self.y_i_t[t,i] = self.z_i_t[t,int(pr_cell_idx)-1]*self.turn_ratio_dict[pr_cell_idx][str(i+1)]
                           y_i_t[t,i] = self.diverging_outflow[pr_cell_idx][str(i+1)]
#                             if pr_cell_idx == '12':
#                                 print("time",t,"cell",i+1,"recieve flow from cell",pr_cell_idx,"get",self.y_i_t[t,i]) ###
                        else:
                            y_i_t[t,i] = z_i_t[t,int(pr_cell_idx)-1]
#                         if i == 10:
#                             print("time",t,"cell 11 sending flow",self.y_i_t[t,i])
                        if y_i_t[t,i] < 0:
                            print("negative y!",y_i_t[t,i],"cell",i+1,"time",t) ###
                    
                elif self.cell_dict[str(i+1)]["type"] == 3: # merging cell
                    Q_i = self.cell_dict[str(i+1)]["Q"]
                    N_i = self.cell_dict[str(i+1)]["N"]
                    alpha = self.cell_dict[str(i+1)]["alpha"]
                    n_i = n_i_t[t,i]
                    total_inflow = 0
                    
                    # previous cells
                    for p_cell in range(self.cell_dict[str(i+1)]["n_pre_cell"]):
                        pr_cell_idx = self.cell_dict[str(i+1)]["pr"+str(p_cell+1)]
                        Q_i_p = self.cell_dict[pr_cell_idx]["Q"]
                        N_i_p = self.cell_dict[pr_cell_idx]["N"]
                        n_i_p = n_i_t[t,int(pr_cell_idx)-1]
                        total_inflow += z_i_t[t,int(pr_cell_idx)-1]
                    
#                     if total_inflow > min([Q_i,alpha*(N_i-n_i)]):
#                         print("WARNING: cell",i+1,"at time",t,": inflow over the capacity")
#                         self.y_i_t[t,i] = min([Q_i,alpha*(N_i-n_i)]) ### TODO: should we update z_i_t then?
#                     else:
                    y_i_t[t,i] = total_inflow
                        
                elif self.cell_dict[str(i+1)]["type"] == 5: # source cell
                    # in our case, we use demand. But why original code use 0?
                    y_i_t[t,i] = self.cell_dict[str(i+1)]["demand"]
                else:
                    print("Error: wrong cell type for cell "+str(i+1))
        
        
            ## update number of vehicle in each cell
            for i in range(self.n_cell):
                n_i_t[t+1,i] = round(n_i_t[t,i] + y_i_t[t,i] - z_i_t[t,i],3) # to avoid negative n due to float calculation
#                 if i in [11,13]: #[101,103]:
#                     print("time",t,"cell",i+1,"self.n_i_t[t,i]",self.n_i_t[t,i]," + self.y_i_t[t,i] ",self.y_i_t[t,i] ,"- self.z_i_t[t,i]",self.z_i_t[t,i]) ###
                if n_i_t[t+1,i] < 0:
                    print("negative n! cell",i+1,"time",t,"z",z_i_t[t,i],"y",y_i_t[t,i],"n",n_i_t[t,i])
                    n_i_t[t+1,i] = 0
                    if self.cell_dict[str(i+1)]["type"] == 1:
                        print("self.diverging_outflow[str(i+1)]",self.diverging_outflow[str(i+1)])
                        print("upstream cell",i,"time",t,"n",n_i_t[t,i-1],"z",z_i_t[t,i-1],"y",y_i_t[t,i-1])      
        pred_net_status = n_i_t[curr_t:curr_t+pred_horizon+1,:]
            
        return pred_net_status
        
        
    
    def update_CTM(self, curr_phase, obs_t, cv_data, cav_ls, cv_ls, junc_position, transit_t, phase_time = 100, lp_inc=None, real_detect_acc = False):
        '''
        based on the observed cell status to update CTM prediction
        Logic:
            1. If get new detection result, update n_i_t at the observation time
            2. Using updated n_i_t to run CTM and predict new n_i_t within the prediction horizon

        INPUT:
            cv_data: CV data in signal_controller.py, structure: data:{lane_id:{veh_id:{"position":value,"xxx":xxx,...},...},...}
            curr_phase: current phase at the observation time, string -> need to convert to phase idx in CTM
            obs_t: observation time
            cav_ls: CAV id list
            cv_ls: CAV and augmented cv id list
            phase_time: phase time remain (in 0.1s), the length of time to generate CTM prediction
        '''
        # convert the phase time unit from 0.1s to pred_horizon with unit of 1s
        # TODO: convert the pred_horizon to be an array that containing prediction horizon for current_phase, AR, Y, and next_phase.
        
        
        if phase_time <= 0:
            pred_horizon = 1
        elif (phase_time%10 != 0):
            pred_horizon = int(phase_time//10+1)
        else:
            pred_horizon = int(phase_time//10)
        
        # convert phase in str to our idx

        # tsc program 1
        if self.ctm_version == "122_4ph":
            phase_idx_dict = {"rrrGrrrrrrrrGrrrrr":0,
                            "GGGrrrrrGGGGrrrrrr":1,
                            "rrrrGGGGrrrrrrrrrr":2,
                            "rrrrrrrrrrrrrGGGGG":3,
                            "rrrrrrrrrrrrrrrrrr":4,
                            "rrryrrrrrrrryrrrrr":4,
                            "yyyrrrrryyyyrrrrrr":4,
                            "rrrryyyyrrrrrrrrrr":4,
                            "rrrrrrrrrrrrryyyyy":4}
        else:
            # tsc program 0
            phase_idx_dict = {"GGGrrrrrGGGGrrrrrr":0,
                            "rrrGrrrrrrrrGrrrrr":1,
                            "rrrrGGGGrrrrrrrrrr":2,
                            "rrrrrrrrrrrrrGGGGG":3,
                            "rrrrrrrrrrrrrrrrrr":4,
                            "yyyrrrrryyyyrrrrrr":4,
                            "rrrryyyyrrrrrrrrrr":4,
                            "rrrrrrrrrrrrryyyyy":4,
                            "rrryrrrrrrrryrrrrr":4}
                                
        curr_phase_idx = phase_idx_dict[curr_phase]

        if curr_phase_idx != 4:
            self.latest_green_phase = curr_phase
            # if the current phase is green phase, we will use the CTM to predict traffic status till the end of MinG of next phase
            next_phase_idx = curr_phase_idx - 1
            if next_phase_idx < 0:
                next_phase_idx = 3
            if next_phase_idx == 1:
                next_phase_minG = 3
            else:
                next_phase_minG = 10

            phase_idx_group = [curr_phase_idx, 4, 4, next_phase_idx]
            phase_duration_group = [pred_horizon, 4, 1, next_phase_minG]
            start_t_group = [obs_t, obs_t+pred_horizon, obs_t+pred_horizon+4, obs_t+pred_horizon+5]
        
        else:
            # if the current phase is AR or Y
            next_phase_idx = phase_idx_dict[self.latest_green_phase] - 1
            if next_phase_idx < 0:
                next_phase_idx = 3
            if next_phase_idx == 1:
                next_phase_minG = 3
            else:
                next_phase_minG = 10

            if 'y' in curr_phase:
                phase_idx_group = [curr_phase_idx, 4, next_phase_idx]
                phase_duration_group = [pred_horizon, 1, next_phase_minG]
                start_t_group = [obs_t, obs_t+pred_horizon, obs_t+pred_horizon+1]
            else:
                phase_idx_group = [curr_phase_idx, next_phase_idx]
                phase_duration_group = [pred_horizon, next_phase_minG]
                start_t_group = [obs_t, obs_t+pred_horizon]
        
        self.t_next_Gmin_end = start_t_group[-1] + next_phase_minG
        
        obs_n_i_t = np.ones(self.n_cell)*np.nan # generate a list of NaN for each cell

        conf_i_t = np.zeros(self.n_cell) # confidence level of detection

        # stop_veh_ls = {}
        
        # 04112023 add cell speed calculation section (assuming all CAVs) for FD validation
        # initialized a global speed list for each cell at each time step, namely v_i_t
        # initialized a local speed list for each cell at the given time step (obs_t), namely obs_v_i_t
        # based on cv data, update obs_v_i_t
        #    for obs_v_i_t, each element is a list of speed for vehicles inside that cell (then, np.array dtype should be object)
        #    -> then, before updating the v_i_t, we do the np.mean for each element of obs_v_i_t  
        # at the end of simulation, save the obs_n_i_t and obs_v_i_t for FD generation
        
        # For FD
        obs_v_i_t = [] #np.ones(self.n_cell)*np.nan
        for i in range(self.n_cell):
            obs_v_i_t.append([])

        # convert the data to obs_n_i_t: from data structure to generate observation cell info -> replace the observed cell info in n_i_t 
        # step 1: decide which cell is detected based on data and cav_ls
        for lane_id in cv_data.keys():
            if lane_id in self.lane2cell.keys():
                # WZ: 03272023 currently, we would ignore vehicles inside the intersection node
                for v_id, info in cv_data[lane_id].items():
                    if v_id in cav_ls:
                        # if v is CAV, update obs_n_i and intialize detected cell as 0 if no CV inside yet
                        veh_position = np.array(info[traci.constants.VAR_POSITION])
                        veh_speed = np.array(info[traci.constants.VAR_SPEED]) # for FD

                        veh2int = max(0, np.linalg.norm(junc_position - veh_position) - 20) # subtract 20 meters to adjust the distance between junction center and stop bar        
                        # determine the cell this veh in: need the relationship between lane_id and approach. 
                        # lane_id -> approach -> get all cells in this approach 
                        # -> if turning lane, get intersection cell list -> veh2int -> idx to fetch cell id from the list
                        # -> if not turning, veh2int -> number of cells to int -> match the cell with the same ncell2int in the same approach
                    
                        cell_list = self.lane2cell[lane_id]["cell"]
                        nIntCell = self.lane2cell[lane_id]["nIntCell"]
                        fetch_idx = max(0, veh2int//self.delta_x - nIntCell) # if veh2int//self.delta_x > nIntCell else 0
                        if int(fetch_idx) >= len(cell_list):
                            fetch_idx = len(cell_list)-1
                            #print("########## int(fetch_idx)",int(fetch_idx)) ###
                        veh_cell_id = cell_list[int(fetch_idx)] # str, return the cell id where the vehicle in

                       
                        # print("DEBUG: obs_t",obs_t,"CAV veh",v_id,"veh2int",veh2int,"veh_cell_id",veh_cell_id,"lane_id",lane_id) #### -> PASS
                        if np.isnan(obs_n_i_t[int(veh_cell_id)-1]):
                            obs_n_i_t[int(veh_cell_id)-1] = 1
                            obs_v_i_t[int(veh_cell_id)-1].append(veh_speed) # for FD
                            conf_i_t[int(veh_cell_id)-1] = 0.92 # cell with CAV is fully trusted.
                        else:
                            obs_n_i_t[int(veh_cell_id)-1] += 1
                            obs_v_i_t[int(veh_cell_id)-1].append(veh_speed) # for FD
                            conf_i_t[int(veh_cell_id)-1] = 0.92

                        # veh2int -> idx of cell in this approach -> get cell id
                        # if incoming lanes, the higher cell idx, the closer to the intersection -> dedact from the largest cell idx
                        # if outgoing lanes, the higher cell idx, the farther to the intersection -> add on the smallest cell idx

                        # set its detected cells as 0 if no veh detected yet
                        # get detected cell id
                        detect_cell_ls = self.cell_dict[veh_cell_id]["detect_cell"] # a string of cell ids, "id1,id2,..."
                        detect_cell_ls = detect_cell_ls.split(",") # a list of cell idx
                        for i in detect_cell_ls:
                            # calculate dist_det_cell for each detected cell, based on which we can calculate conf_i_t
                            if (self.ctm_version == "1788") and real_detect_acc:
                                if i in self.cell_dict[veh_cell_id]["detect_cell_0"].split(","):
                                    conf_i_t[int(i)-1] = max(0.92,conf_i_t[int(i)-1])
                                elif i in self.cell_dict[veh_cell_id]["detect_cell_1"].split(","):
                                    conf_i_t[int(i)-1] = max(0.92,conf_i_t[int(i)-1])
                                elif i in self.cell_dict[veh_cell_id]["detect_cell_2"].split(","):
                                    conf_i_t[int(i)-1] = max(0.77,conf_i_t[int(i)-1])
                                elif i in self.cell_dict[veh_cell_id]["detect_cell_3"].split(","):
                                    conf_i_t[int(i)-1] = max(0.77,conf_i_t[int(i)-1]) 
                                elif i in self.cell_dict[veh_cell_id]["detect_cell_4"].split(","):
                                    conf_i_t[int(i)-1] = max(0.57,conf_i_t[int(i)-1])
                                else:
                                    print("WARNING: detected cell ",i," not within the distance-based list of CAV in cell",veh_cell_id,"!")
                            elif (self.ctm_version == "v2") and real_detect_acc:
                                # cell size: 10.9 m -> cell 0-2: 92%; cell 3-5: 77%, cell:6-7: 57%
                                if i in self.cell_dict[veh_cell_id]["detect_cell_0"].split(","):
                                    conf_i_t[int(i)-1] = max(0.92,conf_i_t[int(i)-1])
                                elif i in self.cell_dict[veh_cell_id]["detect_cell_1"].split(","):
                                    conf_i_t[int(i)-1] = max(0.92,conf_i_t[int(i)-1])
                                elif i in self.cell_dict[veh_cell_id]["detect_cell_2"].split(","):
                                    conf_i_t[int(i)-1] = max(0.92,conf_i_t[int(i)-1])
                                elif i in self.cell_dict[veh_cell_id]["detect_cell_3"].split(","):
                                    conf_i_t[int(i)-1] = max(0.77,conf_i_t[int(i)-1]) 
                                elif i in self.cell_dict[veh_cell_id]["detect_cell_4"].split(","):
                                    conf_i_t[int(i)-1] = max(0.77,conf_i_t[int(i)-1])
                                elif i in self.cell_dict[veh_cell_id]["detect_cell_5"].split(","):
                                    conf_i_t[int(i)-1] = max(0.77,conf_i_t[int(i)-1]) 
                                elif i in self.cell_dict[veh_cell_id]["detect_cell_6"].split(","):
                                    conf_i_t[int(i)-1] = max(0.57,conf_i_t[int(i)-1])
                                elif i in self.cell_dict[veh_cell_id]["detect_cell_7"].split(","):
                                    conf_i_t[int(i)-1] = max(0.57,conf_i_t[int(i)-1])  
                                else:
                                    print("WARNING: detected cell ",i," not within the distance-based list of CAV in cell",veh_cell_id,"!")
                            else:
                                conf_i_t[int(i)-1] = 1.0 
                            if np.isnan(obs_n_i_t[int(i)-1]):
                                obs_n_i_t[int(i)-1] = 0
                                obs_v_i_t[int(i)-1] = [] #.append(self.v_f)  # for FD
                                # for cell observed but no veh, should we add self.v_f? No, we should not.
                                # since we dont consider partial observation case for FD validation,
                                # we can just leave that cell as empty


                    else:
                        # if v is CV, update obs_n_i
                        veh_position = np.array(info[traci.constants.VAR_POSITION])
                        veh_speed = np.array(info[traci.constants.VAR_SPEED]) # for FD


                        veh2int = max(0, np.linalg.norm(junc_position - veh_position)-20) # add 20 meters to adjust the junction center distance
                        cell_list = self.lane2cell[lane_id]["cell"]
                        nIntCell = self.lane2cell[lane_id]["nIntCell"]
                        fetch_idx = max(0, veh2int//self.delta_x - nIntCell) # if veh2int//self.delta_x > nIntCell else 0
                        if int(fetch_idx) >= len(cell_list):
                            fetch_idx = len(cell_list)-1
                            #print("########## int(fetch_idx)",int(fetch_idx)) ###
                        veh_cell_id = cell_list[int(fetch_idx)] # str, return the cell id where the vehicle in


                       
                        # print("DEBUG: obs_t",obs_t,"AugCV veh",v_id,"veh2int",veh2int,"veh_cell_id",veh_cell_id,"lane_id",lane_id) #### -> PASS
                        if np.isnan(obs_n_i_t[int(veh_cell_id)-1]):
                            obs_n_i_t[int(veh_cell_id)-1] = 1
                            obs_v_i_t[int(veh_cell_id)-1].append(veh_speed) # for FD
                            conf_i_t[int(veh_cell_id)-1] = max(0.57,conf_i_t[int(veh_cell_id)-1])
                        else:
                            obs_n_i_t[int(veh_cell_id)-1] += 1
                            obs_v_i_t[int(veh_cell_id)-1].append(veh_speed) # for FD
                            conf_i_t[int(veh_cell_id)-1] = max(0.57,conf_i_t[int(veh_cell_id)-1])


        # step 2: get observed n_i_t, i.e. obs_n_i_t, based on data (need to match veh location and cell id)
        # step 3: replace the observed cell info
        # #DEBUG for obs_n -> PASS
        # for n in range(len(obs_n_i_t)):
        #     print("obs t",obs_t,"cell id",n+1,"obs_n_i_t",obs_n_i_t[n]) ###

######################### STOP CELL ESTIMATION START ####################
        if self.stop_cell_est == True:
            app_info = {}
            for app in range(1,9):
                app_info[app] = {"app_cell_list": self.cell_pd.loc[(self.cell_pd.approach == app)&(self.cell_pd.ncell2int!=0)].cell_idx.to_numpy(),
                                "div_cell_idx": self.cell_pd.loc[(self.cell_pd.approach == app)&(self.cell_pd.type==1)].cell_idx.to_numpy()
                                }

            stop_cell = {}
            for n in range(len(obs_v_i_t)):
                if len(obs_v_i_t[n]) > 0.9*self.cell_dict[str(n+1)]["N"]*conf_i_t[n]:
                    # the stop cells need to be almost full -> weighted by detection AP
                    avg_v = np.mean(obs_v_i_t[n]) 
                    if avg_v < 0.3:
                        # cell with stopped vehicle
                        app = self.cell_dict[str(n+1)]["approach"]
                        app_cell_list = app_info[app]["app_cell_list"]
                        idx, = np.where(app_cell_list == str(n+1))[0] # notice that, the idx here is not CTM cell index but the index of the cell in the approach cell list 
                        div_cell_idx = app_info[app]["div_cell_idx"]
                        if len(div_cell_idx) > 0:
                            # we will only consider the incoming approach with diverging cells
                            div_idx, = np.where(app_cell_list == str(div_cell_idx[0]))[0]

                            if app in stop_cell.keys():
                                # for incoming approach, the larger idx the closer to the intersection 
                                
                                if idx > div_idx: #self.cell_dict[str(n+1)]["type"] == 2:
                                    
                                    # if we simultaneously detect intersection cell and one cell before diverging cell are full, 
                                    # we can assume all downstream cell from the farest cell are full
                                    # also, since the cell idx is starting from source node to intersection to sink, when it comes to intersection, 
                                    # we have already checked the upstream normal cells
                                    if stop_cell[app]["far_app_idx"] <= div_idx:
                                        stop_cell[app]["diverging_cell_stop"] = True
                                        stop_cell[app]["close_app_idx"] = len(app_cell_list) - 1
                                        stop_cell[app]["close_dn_cell_ls"] = [app_cell_list[-1]]
                                    else:
                                        # if intersection cell and the farest cell is an intersection cell
                                        direction = str(self.cell_pd.loc[self.cell_pd.cell_idx == str(n+1)].direction.to_numpy()[0])
                                        dir_app_cell_list = self.cell_pd.loc[(self.cell_pd.approach == app)&(self.cell_pd.direction==direction)].cell_idx.to_numpy()
                                        intersection_idx, = np.where(dir_app_cell_list == str(n+1))[0]

                                        if len( stop_cell[app]["intersection_cell"]) > 0:
                                            # compare with existing intersection cell
                                            if direction in stop_cell[app]["intersection_cell"].keys():
                                                # if current direction has data already
                                                if intersection_idx > stop_cell[app]["intersection_cell"][direction]["close_cell_idx"]:
                                                    # only replace if current stop cell is closer to the stop bar
                                                    stop_cell[app]["intersection_cell"][direction]["close_cell_idx"] = intersection_idx 
                                                    stop_cell[app]["intersection_cell"][direction]["close_cell_ls"] = dir_app_cell_list[intersection_idx:]
                                                elif intersection_idx < stop_cell[app]["intersection_cell"][direction]["far_cell_idx"]:
                                                    # only replace if current stop cell is farther to the stop bar
                                                    stop_cell[app]["intersection_cell"][direction]["far_cell_idx"] = intersection_idx 
                                                    stop_cell[app]["intersection_cell"][direction]["far_cell_idx"] = dir_app_cell_list[intersection_idx:]

                                            else:
                                                # assign value
                                                stop_cell[app]["intersection_cell"][direction] = {"close_cell_idx":intersection_idx,
                                                                                                "close_cell_ls":dir_app_cell_list[intersection_idx:],
                                                                                                "far_cell_idx":intersection_idx,
                                                                                                "far_cell_ls":dir_app_cell_list[intersection_idx:]
                                                                                                }
                                        else:
                                            # assign value
                                            stop_cell[app]["intersection_cell"][direction] = {"close_cell_idx":intersection_idx,
                                                                                                "close_cell_ls":dir_app_cell_list[intersection_idx:],
                                                                                                "far_cell_idx":intersection_idx,
                                                                                                "far_cell_ls":dir_app_cell_list[intersection_idx:]
                                                                                                }
                                
                                elif self.cell_dict[str(n+1)]["type"] == 1:
                                    # if the cloest is diverging cell, since our diverging cell can keep discharge vehicle even one of the downstream is full, 
                                    #   we can arguebly say if divgerging cell is stop, all downstream intersection cells are stop cells.
                                    stop_cell[app]["diverging_cell_stop"] = True
                                    stop_cell[app]["close_app_idx"] = len(app_cell_list) - 1
                                    stop_cell[app]["close_dn_cell_ls"] = [app_cell_list[-1]] 
                                
                                else:
                                    # if closest not intersection/diverging cell
                                    if idx < stop_cell[app]["far_app_idx"]:
                                        stop_cell[app]["far_app_idx"] = idx
                                        stop_cell[app]["far_dn_cell_ls"] = app_cell_list[idx:]
                                    
                                    elif idx > stop_cell[app]["close_app_idx"]:
                                        stop_cell[app]["close_app_idx"] = idx
                                        stop_cell[app]["close_dn_cell_ls"] = app_cell_list[idx:]
                            else:
                                # initialize the approach
                                stop_cell[app] = {"far_app_idx": idx, 
                                                "far_dn_cell_ls":app_cell_list[idx:],
                                                "close_app_idx": idx, 
                                                "close_dn_cell_ls":app_cell_list[idx:],
                                                "intersection_cell":{},
                                                "diverging_cell_stop":False
                                                }
                                
                                if idx > div_idx:
                                    # if intersection cell
                                    direction = str(self.cell_pd.loc[self.cell_pd.cell_idx == str(n+1)].direction.to_numpy()[0])
                                    dir_app_cell_list = self.cell_pd.loc[(self.cell_pd.approach == app)&(self.cell_pd.direction==direction)].cell_idx.to_numpy()
                                    intersection_idx, = np.where(dir_app_cell_list == str(n+1))[0]
                                    stop_cell[app]["intersection_cell"][direction] = {"close_cell_idx":intersection_idx,
                                                                                                "close_cell_ls":dir_app_cell_list[intersection_idx:],
                                                                                                "far_cell_idx":intersection_idx,
                                                                                                "far_cell_ls":dir_app_cell_list[intersection_idx:]
                                                                                                }
                                
                                elif self.cell_dict[str(n+1)]["type"]  == 1:
                                    # if diverging cell
                                    stop_cell[app]["diverging_cell_stop"] = True
                                    stop_cell[app]["close_app_idx"] = len(app_cell_list) - 1
                                    stop_cell[app]["close_dn_cell_ls"] = [app_cell_list[-1]]
                                
                
            # a hardcode dict for green phase idx and corresponding approach idx 
            phase_2_app = {0:[1,3],
                        1:[1,3],
                        2:[5],
                        3:[7]}

            if len(stop_cell) > 0:
                print(obs_t,"time step, stop_cell:",stop_cell) ###
                if (curr_phase_idx == 4) or (obs_t <= transit_t) or (transit_t==0): 
                    # All red/yellow signal
                    for app, item in stop_cell.items():
                        div_cell_idx = app_info[app]["div_cell_idx"]
                        far_cell_id = item["far_dn_cell_ls"][0]
                        if int(far_cell_id) < int(div_cell_idx): #self.cell_dict[far_cell_id]["type"] != 2:
                            # not intersection cell
                            for dn_cell_id in item["far_dn_cell_ls"]:
                                obs_n_i_t[int(dn_cell_id)-1] = self.cell_dict[dn_cell_id]["N"]
                                obs_v_i_t[int(dn_cell_id)-1] = [0.0]
                                conf_i_t[int(dn_cell_id)-1] = 1.0
                        else:
                            # the farest cell is an intersection cell
                            for direction, intersection_item in item["intersection_cell"].items():
                                for dn_cell_id in intersection_item["far_cell_ls"]:
                                    obs_n_i_t[int(dn_cell_id)-1] = self.cell_dict[dn_cell_id]["N"]
                                    obs_v_i_t[int(dn_cell_id)-1] = [0.0]
                                    conf_i_t[int(dn_cell_id)-1] = 1.0
                else: 
                    # green phases
                    for app, item in stop_cell.items():
                        green_app = phase_2_app[curr_phase_idx]
                        far_cell_id = item["far_dn_cell_ls"][0]
                        div_cell_idx = app_info[app]["div_cell_idx"]
                        if app in green_app:
                            if len(item["intersection_cell"]) > 0:
                                # use intersection cell as cloest cell
                                for direction, intersection_item in item["intersection_cell"].items():
                                    for dn_cell_id in intersection_item["close_cell_ls"]:
                                        self.n_i_t[transit_t,:][int(dn_cell_id)-1] = self.cell_dict[dn_cell_id]["N"]
                                        self.v_i_t[transit_t,:][int(dn_cell_id)-1] = 0.0
                                
                                # for cells inbetwenn the farest cell to the intersection cell,
                                if stop_cell[app]["diverging_cell_stop"]:
                                    # if all intersection cells are assumed to be full
                                    for dn_cell_id in item["far_dn_cell_ls"]:
                                        if (dn_cell_id not in item["close_dn_cell_ls"]):
                                            self.n_i_t[transit_t,:][int(dn_cell_id)-1] = self.cell_dict[dn_cell_id]["N"]
                                            self.v_i_t[transit_t,:][int(dn_cell_id)-1] = 0.0

                                elif int(far_cell_id) > int(div_cell_idx): #self.cell_dict[far_cell_id]["type"] == 2:
                                    # the farest cell is also intersection cell
                                    for direction, intersection_item in item["intersection_cell"].items():
                                        for dn_cell_id in intersection_item["far_cell_ls"]:
                                            if (dn_cell_id not in intersection_item["close_cell_ls"]):
                                                obs_n_i_t[int(dn_cell_id)-1] = self.cell_dict[dn_cell_id]["N"]
                                                obs_v_i_t[int(dn_cell_id)-1] = [0.0]
                                                conf_i_t[int(dn_cell_id)-1] = 1.0
                                
                                else:
                                    # farest cell is not intersection cell but have intersection cells as stop cells
                                    # set from farest cell to the close intersection stop cell as stop cells
                                    intersection_close_cell_ls = ""
                                    for direction, intersection_item in item["intersection_cell"].items():
                                        intersection_close_cell_ls += intersection_item["close_cell_ls"]
                                        # TODO 0709: how to consider the directions without a stop cell?
                                    for dn_cell_id in item["far_dn_cell_ls"]: 
                                        if (dn_cell_id not in intersection_close_cell_ls):
                                            obs_n_i_t[int(dn_cell_id)-1] = self.cell_dict[dn_cell_id]["N"]
                                            obs_v_i_t[int(dn_cell_id)-1] = [0.0]
                                            conf_i_t[int(dn_cell_id)-1] = 1.0
                            else:
                                # if no intersection cell info exist
                                for dn_cell_id in item["close_dn_cell_ls"]:
                                    # print("DEBUGGGGG: transit_t",transit_t,"dn_cell_id",dn_cell_id,'item',item) ####
                                    self.n_i_t[transit_t,:][int(dn_cell_id)-1] = self.cell_dict[dn_cell_id]["N"]
                                    self.v_i_t[transit_t,:][int(dn_cell_id)-1] = 0.0
                                    
                                for dn_cell_id in item["far_dn_cell_ls"]:
                                    if dn_cell_id not in item["close_dn_cell_ls"]:
                                        # set all cells inbetween farest cell and cloeset cell as full                                    
                                        obs_n_i_t[int(dn_cell_id)-1] = self.cell_dict[dn_cell_id]["N"]
                                        obs_v_i_t[int(dn_cell_id)-1] = [0.0]
                                        conf_i_t[int(dn_cell_id)-1] = 1.0
                        else:
                            # for red phase approach
                            far_cell_id = item["far_dn_cell_ls"][0]
                            div_cell_idx = app_info[app]["div_cell_idx"]
                            
                            if int(far_cell_id) < int(div_cell_idx):  #if self.cell_dict[far_cell_id]["type"] != 2:
                                for dn_cell_id in item["far_dn_cell_ls"]:
                                    obs_n_i_t[int(dn_cell_id)-1] = self.cell_dict[dn_cell_id]["N"]
                                    obs_v_i_t[int(dn_cell_id)-1] = [0.0]
                                    conf_i_t[int(dn_cell_id)-1] = 1.0
                            else:
                                # the farest cell is an intersection cell
                                for direction, intersection_item in item["intersection_cell"].items():
                                    for dn_cell_id in intersection_item["far_cell_ls"]:
                                        obs_n_i_t[int(dn_cell_id)-1] = self.cell_dict[dn_cell_id]["N"]
                                        obs_v_i_t[int(dn_cell_id)-1] = [0.0]
                                        conf_i_t[int(dn_cell_id)-1] = 1.0

                    self.run_CTM(curr_phase_idx, transit_t, obs_t - transit_t) # update self.n_i_t based on obs_t stopped vehicle status


########################## STOP CELL ESTIMATION END ###################

        # TODO: why when we disable the stop cell est, the CTM will overestimate under the realistic detection mode? 
        # -> disable CAV_real and SCE to see if we can replicate previous result

        # use conf_i_t[n] as the weight for the linear combination of observation and CTM estimation
        obs_replace = [ min(self.cell_dict[str(n+1)]["N"], obs_n_i_t[n] + (1-conf_i_t[n])*self.n_i_t[obs_t,:][n]) if not np.isnan(obs_n_i_t[n]) else self.n_i_t[obs_t,:][n] for n in range(len(obs_n_i_t))]
        diff = obs_replace - self.n_i_t[obs_t,:]

        valid = diff[diff != 0]
        if valid.size > 0:
            self.avg_diff.append(np.mean(valid))


        # self.avg_diff.append(np.nanmean(np.where(diff!=0,diff,np.nan)))
        
        self.n_i_t[obs_t,:] =  obs_replace # the combination of observation and CTM estimation for non-observation part
        

        # 01142024: the v_i_t is combined with estimated speed.
        # Note that v_i_t will not be used for prediction in run_CTM, so we can only get v_i_t at obs_t. For other t, v_i_t will be 0 for all cells.
        self.v_i_t[obs_t,:] = [np.mean(obs_v_i_t[n]) if len(obs_v_i_t[n])>0 else self.get_cell_speed(self.n_i_t[obs_t,n]/(self.cell_dict[str(n+1)]["num_lane"]*self.delta_x/1000),n) for n in range(len(obs_v_i_t))]  # for FD
        
        self.lp_inc_prc = self.proc_lp_data(lp_inc)

        #DEBUG for n_i_t -> PASS
        # for n in range(9,15):
        #     print("EB obs t",obs_t,"cell id",n+1,"n_i_t",self.n_i_t[obs_t,n],"v_i_t",self.v_i_t[obs_t,n],"obs_v_i_t[n]",obs_v_i_t[n],"conf_i_t",conf_i_t[n]) ###
        # for n in range(37,45):
        #     print("WB obs t",obs_t,"cell id",n+1,"n_i_t",self.n_i_t[obs_t,n],"v_i_t",self.v_i_t[obs_t,n],"obs_v_i_t[n]",obs_v_i_t[n],"conf_i_t",conf_i_t[n]) ###
        # for n in range(67,75):
        #     print("NB obs t",obs_t,"cell id",n+1,"n_i_t",self.n_i_t[obs_t,n],"v_i_t",self.v_i_t[obs_t,n],"obs_v_i_t[n]",obs_v_i_t[n],"conf_i_t",conf_i_t[n]) ###
        # for n in range(96,109):
        #     print("SB obs t",obs_t,"cell id",n+1,"n_i_t",self.n_i_t[obs_t,n],"v_i_t",self.v_i_t[obs_t,n],"obs_v_i_t[n]",obs_v_i_t[n],"conf_i_t",conf_i_t[n]) ###
        # for n in [38,97,59,31,32]:
        #     if self.n_i_t[obs_t,n] > 8:
        #         print("WARNING! obs t",obs_t,
        #               "cell id",n+1,
        #               "n_i_t",self.n_i_t[obs_t,n],
        #               "y_i_t",self.y_i_t[obs_t,n],
        #               "z_i_t",self.z_i_t[obs_t,n],
        #               "obs_n_i_t[n]",obs_n_i_t[n],
        #               "conf_i_t",conf_i_t[n],
        #               "v_i_t",self.v_i_t[obs_t,n],
        #               "obs_v_i_t[n]",obs_v_i_t[n],
        #               ) ###
                
        #         if self.n_i_t[obs_t,n] > 50:
        #             raise ValueError


        # update self.n_i_t and re-run prediction
        # TODO: run loop run_CTM to keep cell status until the end of next phase minG
        total_delay_ls = []
        pred_net_status_ls = []

        for idx, _phase_idx in enumerate(phase_idx_group):
            _phase_duration = phase_duration_group[idx]
            _start_t = start_t_group[idx]

            total_delay, pred_net_status = self.run_CTM(_phase_idx, _start_t, _phase_duration)
            total_delay_ls.append(total_delay)
            pred_net_status_ls.append(pred_net_status)
        
        end_of_minG = _start_t+_phase_duration
        est_speed = [ self.get_cell_speed(self.n_i_t[end_of_minG,n]/(self.cell_dict[str(n+1)]["num_lane"]*self.delta_x/1000),n) for n in range(self.n_cell)] 

        return pred_net_status_ls[-1], est_speed # total_delay_ls[0], pred_net_status_ls[0], 
    
    def get_fake_veh_final_state(self, cell_idx, t):
        # function: return the position (distance to stop bar) and speed of the fake vehicle at time t
        final_spd = self.get_cell_speed(self.n_i_t[t,cell_idx]/(self.cell_dict[str(cell_idx)]["num_lane"]*self.delta_x/1000),cell_idx)
        distance = (self.cell_dict[str(cell_idx)]['ncell2int']-0.5)*17.88 # take the middle point of the cell as the final position
        lane_id = random.choice(self.cell_dict[str(cell_idx)]['cell2lane'].split(","))

        return distance, final_spd, lane_id
    
    def proc_lp_data(self,lp_data):
        # input: lp_data = {node_id:{lp_id:deque(last 10 step info)}}
        # output: {node_id:real-time inflow in 1 s}
        if lp_data == None:
            return None
        else:
            output_dict = {}
            for node_id, lp_info in lp_data.items():
                output_dict[node_id] = 0
                for lp_id, hist_que in lp_info.items():
                    # lp_sum = np.sum(np.asarray(hist_que))
                    lp_sum = np.sum(np.asarray(hist_que))
                    lp_prc = lp_sum//3 if (lp_sum<10) else 0 
                    # here, we hard code 3 as vehicle will take 0.3-0.4s to pass through the lp; 
                    # also, if sum==10, which is the hardcode length of hist_que, we assume one vehicle is waiting at the lp, then inflow=0
                    output_dict[node_id] += lp_prc
            # print("LP data after prc:",output_dict) ####
            return output_dict  
        
    def get_state_CTM(self,t):
        # based on n_i_t, y_i_t, and z_i_t get RL state information at time t
        # the number of segments is set to be 3
        
        veh_info_t = {app:{seg:[] for seg in range(5)} for app in range(1,9)}
        veh_len = 5 # veh length, meter
        inc_app = [[7],[5],[1,3],[1,3]] #[[1,3],[5],[7]]  #[[7],[5],[1,3],[1,3]]

        # TODO: update this function to fit the 4-phase setting. agent critic. 
        
        num_veh_inc = np.array([[0,0,0],[0,0,0],
                                [0],[0,0,0]], dtype=object) #np.zeros((len(inc_app),3))
        avg_speed_inc = np.array([[0,0,0],[0,0,0],
                                [0],[0,0,0]], dtype=object)
        num_veh_seg = np.array([[0,0,0],[0,0,0],
                                [0],[0,0,0]], dtype=object)
        
        
        # re-arrange the n_i_t info based on approach and segment 
        for i in range(self.n_cell):
            app_i = int(self.cell_dict[str(i+1)]["approach"])
            seg_i = self.cell_dict[str(i+1)]["seg_idx"]
            num_veh = self.n_i_t[t,i]
            speed_veh = self.v_i_t[t,i] if self.v_i_t[t,i] > 0 else self.get_cell_speed(self.n_i_t[t,i]/(self.cell_dict[str(i+1)]["num_lane"]*self.delta_x/1000),i) 
            # speed_veh = self.get_cell_speed(self.n_i_t[t,i]/(self.cell_dict[str(i+1)]["num_lane"]*self.delta_x/1000),i) 
            # print(t,"cell",i+1,"speed",speed_veh,"num_veh",num_veh) #########

            # step_delay = self.n_i_t[t,i] - self.z_i_t[t,i] # how to get the total delay?
            veh_info_t[app_i][seg_i].append([num_veh,speed_veh]) # veh_info_t:{app_i:{seg_i:[[num_veh,speed_veh],[,]...]}
            
        # get number of vehicle in incoming lanes
        for phase_idx in range(len(inc_app)):
            if phase_idx == 2:
                # left turning phase
                for app_idx in inc_app[phase_idx]:
                    seg_idx = 4
                    num_veh_inc[phase_idx][0] += sum(cell_info[0] for cell_info in veh_info_t[app_idx][seg_idx])
            else:
                for app_idx in inc_app[phase_idx]:
                    for seg_idx in range(1,4):
                        num_veh_inc[phase_idx][seg_idx-1] += sum(cell_info[0] for cell_info in veh_info_t[app_idx][seg_idx])
        
        norm_num_veh_in = np.linalg.norm(np.hstack(num_veh_inc)) 
        if norm_num_veh_in == 0:
            norm_num_veh_in = 1
        num_veh_inc_final = np.hstack(num_veh_inc)/norm_num_veh_in

        self.state_comparison_CTM = [np.hstack(num_veh_inc)]
        
        
        # get avg speed (for incoming lanes): the cell-based avg speed is different from the veh-based avg speed. 
        for phase_idx in range(len(inc_app)):
            if phase_idx == 2:
                for app_idx in inc_app[phase_idx]:
                    seg_idx = 4
                    avg_speed_inc[phase_idx][0] += sum(cell_info[1]*cell_info[0] for cell_info in veh_info_t[app_idx][seg_idx])
                    num_veh_seg[phase_idx][0] += sum(cell_info[0] for cell_info in veh_info_t[app_idx][seg_idx])
            else:
                for app_idx in inc_app[phase_idx]:
                    for seg_idx in range(1,4):
                        avg_speed_inc[phase_idx][seg_idx-1] += sum(cell_info[1]*cell_info[0] for cell_info in veh_info_t[app_idx][seg_idx])
                        num_veh_seg[phase_idx][seg_idx-1] += sum(cell_info[0] for cell_info in veh_info_t[app_idx][seg_idx])
        
        # replace 0 veh cell to prevent from dividing 0
        for phase_idx in range(len(inc_app)):
            if phase_idx == 2:
                if num_veh_seg[phase_idx][0] == 0:
                    avg_speed_inc[phase_idx][0] = self.v_f #force to be vf for spd diff calculation
                    num_veh_seg[phase_idx][0] = 1

            else:
                for seg_idx in range(1,4):
                    if num_veh_seg[phase_idx][seg_idx-1] == 0:
                        avg_speed_inc[phase_idx][seg_idx-1] = self.v_f #force to be vf for spd diff calculation
                        num_veh_seg[phase_idx][seg_idx-1] = 1
        avg_spd_CV = np.hstack(avg_speed_inc) / np.hstack(num_veh_seg)
        avg_spd_CV_crt = np.array([self.v_f - v if v < self.v_f else 0 for v in avg_spd_CV]) # replace result v_f with 0, where v_f usually means no observations
        
        # print("ACTOR CTM avg_spd_CV", avg_spd_CV_crt) #####
        norm_CV = np.linalg.norm(avg_spd_CV_crt)
        if norm_CV == 0:
            norm_CV = 1
        avg_speed_inc_final = avg_spd_CV_crt/norm_CV + 0.2
        
        # print("ACTOR num_veh_inc", num_veh_inc) ###
        
        return np.concatenate([avg_speed_inc_final, num_veh_inc_final])
    
    def get_state_CTM_MP(self,t,phase):
        # based on n_i_t, generate the traffic state information MP would need 
        # need to provide number of vehicle for each lane: 
        # we can build a relationship dictionary between approach/cell list and MP phase incoming/outgoing lane
        
        # given the phase, get incoming lanes and outgoing lanes cell lists
        # hard coding for Plymouth, phase: [[inc_lane cells],[out_lane cells]]
        
        if self.ctm_version == "1788":
            # tsc program 0
            phase_cell_dict = {"GGGgrrrrGGGGgrrrrr":[np.append(np.array(range(2,16)),np.array(range(30,46))),np.concatenate((np.array(range(16,28)),np.array(range(46,58)), np.array(range(76,88)), np.array(range(110,122))))],
                            "rrrrGGGGrrrrrrrrrr":[np.array(range(60,76)),np.concatenate((np.array(range(16,28)),np.array(range(46,58)),np.array(range(76,88))))],
                            "rrrrrrrrrrrrrGGGGG":[np.array(range(90,110)),np.concatenate((np.array(range(16,28)),np.array(range(46,58)),np.array(range(110,122))))]}

        elif self.ctm_version == "122_4ph":
            # tsc program 1
            phase_cell_dict = {"GGGrrrrrGGGGrrrrrr":[np.concatenate((np.array(range(2,13)),np.array(range(14,16)),np.array(range(30,40)),np.array(range(42,46)))),np.concatenate((np.array(range(16,28)),np.array(range(46,58)), np.array(range(76,88)), np.array(range(110,122))))],
                            "rrrGrrrrrrrrGrrrrr":[np.concatenate((np.array(range(11,14)),np.array(range(38,42)))),np.concatenate((np.array(range(76,88)),np.array(range(110,122))))],
                            # "rrrrGGGrrrrrrGGGrr":[np.concatenate((np.array(range(60,70)), np.array(range(72,76)),np.array(range(96,98)),np.array(range(102,110)))),np.concatenate((np.array(range(16,28)),np.array(range(46,58)),np.array(range(76,88)), np.array(range(110,122))))],
                            # "rrrrrrrGrrrrrrrrGG":[np.concatenate((np.array(range(68,72)),np.array(range(90,102)))),np.concatenate((np.array(range(46,58)),np.array(range(16,28))))]}
                            "rrrrGGGGrrrrrrrrrr":[np.array(range(60,76)),np.concatenate((np.array(range(16,28)),np.array(range(46,58)),np.array(range(76,88))))],
                            "rrrrrrrrrrrrrGGGGG":[np.array(range(90,110)),np.concatenate((np.array(range(16,28)),np.array(range(46,58)),np.array(range(110,122))))]}

        elif self.ctm_version == "v2":
            phase_cell_dict = {"GGGgrrrrGGGGgrrrrr":[np.append(np.array(range(2,24)),np.array(range(44,68))),np.concatenate((np.array(range(24,42)),np.array(range(68,86)), np.array(range(114,132)), np.array(range(166,184))))],
                        "rrrrGGGGrrrrrrrrrr":[np.array(range(88,114)),np.concatenate((np.array(range(24,42)),np.array(range(68,86)),np.array(range(114,132))))],
                        "rrrrrrrrrrrrrGGGGG":[np.array(range(134,166)),np.concatenate((np.array(range(24,42)),np.array(range(68,86)),np.array(range(166,184))))]}

        inc_cell_list = phase_cell_dict[phase][0]
        out_cell_list = phase_cell_dict[phase][1]

        # calculate pressure for the given phase at time t
        inc_pressure = 0
        out_pressure = 0
        for i in inc_cell_list:
            inc_pressure += self.n_i_t[t,int(i-1)]
        for j in out_cell_list:
            out_pressure += self.n_i_t[t,int(j-1)]

        out_pressure = 0 ## 01172024 temperally disable outpressure for isolated intersection
        # print(t,phase,"inc pressure generate by CTM",inc_pressure) ###

        pressure = inc_pressure - out_pressure

        return pressure, inc_pressure, out_pressure
    
    def get_state_CTM_OPT(self, t, phase, phase_duration):
        # FUNCTION: to generate traffic state informtion at time step t for the optimization-based TSC
        # INPUT: time step
        # OUTPUT: inter_dict_lane_based dictionary, including veh id and dist to stopbar of each veh allocated by lane and approach 

        # inter_dict_lane_based = {arm_id: {lane_id:[{'veh_id':xx, 'dist to stop bar':xxx},{...},...],...},...}, 
        # the veh info dict in the list follow an order according to the dist_to_stop_bar

        # step 1: initialize the inter_dict_lane_based based on the self.cell_dict: mapping from cell idx to lane and arm idx 
        #           -> move to inialization part of CTM -> here, we directly read such info from self.cell_dict
        # step 2: loop through cell based on its dist2int:
        #           based on number of vehicle in each cell, generate veh_id and dist_to_stop_bar, as well as the lane_id and arm_id
        # step 3: allocate veh info based on lane_id and arm_id into the inter_dict_lane_based

        inter_dict_lane_based = {str(arm_id): {str(arm_id)+"_"+str(lane_id):[] for lane_id in range(4) } for arm_id in range(1,9)}

        for idx in range(self.n_cell):
            cell_type = self.cell_dict[str(idx+1)]["type"]
            if cell_type in [0,1,2,3]:
                lane_id_ls = self.cell_dict[str(idx+1)]["cell2lane"].split(",")
                arm_id = str(self.cell_dict[str(idx+1)]["approach"])
                MinDist2int = (self.cell_dict[str(idx+1)]["ncell2int"]-1)*self.v_f # the distance between the near-end of this cell to intersection
                num_veh = int(round(self.n_i_t[t,idx])) # split at 0.5
                if num_veh > 0:
                    headway = self.v_f/num_veh
                    for v in range(num_veh):
                        veh_lane_idx = np.random.randint(int(self.cell_dict[str(idx+1)]["num_lane"])) # randomly give a idx to get lane_id of this veh
                        veh_lane = lane_id_ls[veh_lane_idx]
                        veh_id = str(veh_lane)+"_"+str(idx+1)+"_"+str(v) # veh id follows: edgeID_laneID_vehID
                        veh_dist2int = MinDist2int + v*headway # assume veh evenly distributed within each cell
                        inter_dict_lane_based[arm_id][veh_lane].append({'veh_id':veh_id,'dist2bar':veh_dist2int})
        
        self.OPT_TSC_input[t] = {"inter_dict_lane_based":inter_dict_lane_based,"phase":phase, "phase_duration":phase_duration}
        return self.OPT_TSC_input[t]
    
    def cell2direction(self,i):
        ranges_to_directions = {
            (106, 109): 0,
            (102, 105): 1,
            (89, 97): 2, (98, 101): 2,
            (44, 45): 3,
            (29, 39): 4, (42, 43): 4,
            (40, 41): 5,
            (59, 69): 6, (72, 75): 6,
            (70, 71): 7,
            (1, 12): 8, (14, 15): 8,
            (13, 13): 9
        }
        for range_tuple, direction in ranges_to_directions.items():
            if range_tuple[0] <= i <= range_tuple[1]:
                return direction
        return None

    def get_Arrival_Table(self, curr_t,plan_horizon,approaches,n_i_t,v_i_t):
        # FUNCTION: to generate arrival at time step t for the optimization-based TSC
        # INPUT: time step
        # OUTPUT: inter_dict_lane_based dictionary, including veh id and dist to stopbar of each veh allocated by lane and approach 
        Arrival_Table = np.zeros((plan_horizon, approaches))
        for i in range(122):
            if self.cell_dict[str(i+1)]['ncell2int']==0:
                continue
            else:
                direction=self.cell2direction(i+1)
                if direction!=None:
                    if v_i_t[curr_t,i]<=2:#########################need to adjust 0.9
                        Arrival_Table[0,direction]+=n_i_t[curr_t,i]
                    else:
                        ETA=round((self.cell_dict[str(i+1)]['ncell2int']*17.88-17.88)/v_i_t[curr_t,i],0)
                        if ETA<plan_horizon-1:
                            Arrival_Table[int(ETA),direction]+=n_i_t[curr_t,i]
                else:
                    continue
        return Arrival_Table
    
    def get_Arrival_Table_outflow(self, curr_t,plan_horizon,approaches):
        # FUNCTION: to generate arrival at time step t for the optimization-based TSC
        # INPUT: time step
        # OUTPUT: inter_dict_lane_based dictionary, including veh id and dist to stopbar of each veh allocated by lane and approach 
        Arrival_Table = np.zeros((plan_horizon, approaches))
        for i in range(122):
            #print("^^^^^curr_t,cell,n_i_t,v_i_t",curr_t,i,self.n_i_t[curr_t,i],self.v_i_t[curr_t,i])
            if self.cell_dict[str(i+1)]['ncell2int']==0:
                continue
            else:
                direction=self.cell2direction(i+1)
                if direction!=None:
                    if self.v_i_t[curr_t,i]<=6:#########################need to adjust 0.9
                        Arrival_Table[0,direction]+=self.n_i_t[curr_t,i]
                    else:
                        ETA=round((self.cell_dict[str(i+1)]['ncell2int']*17.88-17.88)/self.v_i_t[curr_t,i],0)
                        if ETA<plan_horizon-1:
                            Arrival_Table[int(ETA),direction]+=self.n_i_t[curr_t,i]
                else:
                    continue
        return Arrival_Table


    def get_cell_speed(self,cell_den,cell_idx):
        # generate cell average speed based on the FD
        # cell density unit veh/km
        # eps = 0.3
        # if cell_den<eps:
        #     # if no veh, to prevent vff from disturbing avg speed calculation, we set as None
        #     return 0
        if cell_den < self.cell_dict[str(cell_idx+1)]["k_cri"]:
            return self.v_f
        elif (cell_den >= self.cell_dict[str(cell_idx+1)]["k_cri"]) and (cell_den <= self.cell_dict[str(cell_idx+1)]["k_jam"]):
            return abs(self.cell_dict[str(cell_idx+1)]["w"])*(self.cell_dict[str(cell_idx+1)]["k_jam"]/cell_den - 1)
        else:
            # print("WARNING cell speed is not avaliable! cell id",cell_idx,"cell_den",cell_den)
            return 0
    
    def get_cell_speed_vectorized(self, n_i_t, cell_idx):

        # Extract relevant values from cell_dict
        k_cri = self.cell_dict["2"]["k_cri"]#all cells share the same k_cri/jam, and w
        k_jam = self.cell_dict["2"]["k_jam"]
        w = self.cell_dict["2"]["w"]
        num_lane = np.array([self.cell_dict[str(idx)]["num_lane"] for idx in cell_idx])
        delta_x = self.delta_x
        cell_den=n_i_t/(num_lane * delta_x/1000)

        # Calculate speeds
        condition1 = cell_den < k_cri
        condition2 = (cell_den >= k_cri) & (cell_den <= k_jam)
        speed = np.where(condition1, self.v_f, 0)
        speed = np.where(condition2, np.abs(w) * (k_jam / cell_den - 1), speed)

        return speed


    
    def fixed_time_CTM(self,sim_len=100):
        # this function can generate CTM under a fixed time TSC at Plymouth
        # run CTM from 0 to the sim_len
        #  mainly for warmup time simulation
        
        # total_delay = []
        if self.ctm_version in ["1788","v2"]:
            signal_time_plan = [(21,0),(5,3),(12,1),(5,3),(7,2),(5,3)] # (duration, phase idx)
        elif  self.ctm_version == "122_4ph":
            signal_time_plan = [(12,0),(5,4),(21,1),(5,4),(12,2),(5,4),(21,3),(5,4)]
        cycle_length = sum(i[0] for i in signal_time_plan)
        total_sim_len = sim_len #2100
        number_cycle = total_sim_len//cycle_length
        if total_sim_len%cycle_length != 0:
            # to finish the rest of the simulation time
            number_cycle += 1
        rest_time = sim_len
        t = 0
        #for cycle_i in range(number_cycle):
        while rest_time > 0:
            for cycle_i in range(number_cycle):
                if rest_time <= 0:
                    break
                for phase_i in range(len(signal_time_plan)):
                    delay, pred_net_status = self.run_CTM(signal_time_plan[phase_i][1],t,signal_time_plan[phase_i][0])
                    #total_delay.append([t,delay])
                    t += signal_time_plan[phase_i][0]
                    rest_time = rest_time - signal_time_plan[phase_i][0]
        
        # avg_delay = sum(i[1] for i in total_delay if i[0]>100)/self.total_demand/(t-100)


    def viz_CTM(self,start_t,end_t,legend_bar=0.1):
        '''
        viz the cell status, i.e. n_i_t, over time
        mainly for debugging, will not plot inside the CAVLight
        '''
        
        # 0: normal
        # 1: diverging
        # 2: intersection
        # 3: merging
        # 4: sink
        # 5: source
        color_set = {0:"gray",1:"green",2:"red",3:"orange",4:"blue",5:"brown"}
        
        plt.figure(figsize=(10,6))
        
        # TODO: color the line based on the cell type
        for i in range(self.n_cell):
            cell_type = self.cell_dict[str(i+1)]["type"]
            if max(self.n_i_t[:,i]) > legend_bar:
                plt.plot(range(start_t,end_t), self.n_i_t[start_t:end_t,i], color = color_set[cell_type], label="cell "+str(i+1))
            else:
                plt.plot(range(start_t,end_t), self.n_i_t[start_t:end_t,i], color = color_set[cell_type])
        plt.xlabel("time (s)")
        plt.ylabel("density (vhe/mi)")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),title="Cells with density \n over "+str(legend_bar))
        plt.title("CTM Time-density diagram with "+str(self.n_cell)+" cells and " +str(int(self.delta_t))+"s time interval")
    
        
    def viz_CTM_matrix(self,time,tlid):
        if tlid == "realtime_":
            approach = {i:[] for i in range(1,9)}
            for i in range(self.n_cell):
                app = self.cell_dict[str(i+1)]["approach"]
                approach[app].append(i)

            plt.ion()
            
            fig, axs = plt.subplots(2, 4, sharex= True, figsize=(16,8))
            
            #plt.setp(axs, xticks=range(time))
            
            for app_idx, cell_ls in approach.items():
    #             axs[int(app_idx)-1].set_yticks(cell_ls)
    #             axs[int(app_idx)-1].set_yticklabels(cell_ls)
    #             axs[int(app_idx)-1].set_ylabel(app_idx)
    #             im = axs[int(app_idx)-1].imshow(self.n_i_t[:time,cell_ls].T, cmap='hot')
                
    #             r_idx = (int(app_idx)-1)//4
    #             c_idx = (int(app_idx)-1)%4
                idx = [[0,0],[1,0],
                    [0,1],[1,1],
                    [0,2],[1,2],
                    [0,3],[1,3]]
                r_idx = idx[int(app_idx)-1][0]
                c_idx = idx[int(app_idx)-1][1]
                
                sns.heatmap(self.n_i_t[:time,cell_ls].T, 
                            yticklabels=np.array(cell_ls)+1,
                            cmap='crest',
                            ax = axs[r_idx,c_idx])
                axs[r_idx,c_idx].set_ylabel('approach '+str(app_idx))
                
                
    #         cbar_ax = fig.add_axes([1.05, 0.15, 0.05, 0.7])
    #         fig.colorbar(im, cax=cbar_ax)
                
            #plt.colorbar(fraction=0.05)
            plt.tight_layout()
            # ref: https://stackoverflow.com/questions/42024817/plotting-a-continuous-stream-of-data-with-matplotlib
            plt.pause(1)
            plt.show(block=True) 
        else: 
            approach = {i:[] for i in range(1,9)}
            for i in range(self.n_cell):
                app = self.cell_dict[str(i+1)]["approach"]
                approach[app].append(i)
            
            fig, axs = plt.subplots(2, 4, sharex= True, figsize=(16,8))
            
            #plt.setp(axs, xticks=range(time))
            
            for app_idx, cell_ls in approach.items():
    #             axs[int(app_idx)-1].set_yticks(cell_ls)
    #             axs[int(app_idx)-1].set_yticklabels(cell_ls)
    #             axs[int(app_idx)-1].set_ylabel(app_idx)
    #             im = axs[int(app_idx)-1].imshow(self.n_i_t[:time,cell_ls].T, cmap='hot')
                
    #             r_idx = (int(app_idx)-1)//4
    #             c_idx = (int(app_idx)-1)%4
                idx = [[0,0],[1,0],
                    [0,1],[1,1],
                    [0,2],[1,2],
                    [0,3],[1,3]]
                r_idx = idx[int(app_idx)-1][0]
                c_idx = idx[int(app_idx)-1][1]
                
                sns.heatmap(self.n_i_t[:time,cell_ls].T, 
                            yticklabels=np.array(cell_ls)+1,
                            cmap='crest',
                            ax = axs[r_idx,c_idx])
                axs[r_idx,c_idx].set_ylabel('approach '+str(app_idx))
                
                
    #         cbar_ax = fig.add_axes([1.05, 0.15, 0.05, 0.7])
    #         fig.colorbar(im, cax=cbar_ax)
                
            #plt.colorbar(fraction=0.05)
            plt.tight_layout()
            plt.savefig("./exp_log/CTM/"+tlid+"CTM.png")
     
    def save_nvlist(self,time,tlid):
        fp = "./exp_log/FD_validation/"+tlid+"_"
        save_data(fp+'_n_i_t.p', self.n_i_t[:time,:])
        save_data(fp+'_v_i_t.p', self.v_i_t[:time,:])
        save_data(fp+'avg_diff.p', self.avg_diff)
        print("SAVE CTM n_i_t and v_i_t successfully")

    def save_OPT_input(self,tlid):
        path = "./exp_log/OPT_input/"+tlid+"_"
        # save_data(fp+'opt_input.p', self.OPT_TSC_input)
        with open(path+'opt_input.json', 'w') as fp:
            json.dump(self.OPT_TSC_input, fp)
        print("SAVE CTM n_i_t and v_i_t successfully")