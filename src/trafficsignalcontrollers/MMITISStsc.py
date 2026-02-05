import gurobipy as gp
from gurobipy import GRB
#import sympy
import math
import json
import random
from scipy.optimize import minimize 
from sklearn.model_selection import train_test_split
from numpy.linalg import norm
import copy
import traci
from src.CTM import CTM_model 
from src.trafficsignalcontroller import TrafficSignalController
from itertools import cycle
from collections import deque

import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
import traci.constants as tc
import random
import pandas as pd
import numpy as np
import copy
import math

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(26, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 9)   # Output a single value

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        logits = self.output(x)
        probabilities = F.softmax(logits, dim=1) 
        return probabilities


class mmitissTSC(TrafficSignalController):
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, green_t, detect_r, g_max, act_ctm, args):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t, detect_r)

       
        # This opt-based TSC is required to use CTM to get state.
        # if not act_ctm:
        #     raise NotImplementedError("This MMITISS-TSC requires CTM to generate input")
        
        step_size = 0.1
        self.G_min_T= {'47':10,'15':3,'26':10,'38':10}#int(green_t*step_size) #10
        #G_min_T_turn = 10
        #G_min_T_through = 10
        self.phase_deque = deque()
        self.G_max_T = int(g_max*step_size) #40
        self.Yellow= int(yellow_t*step_size)
        self.Red= int(red_t*step_size)
        self.green_t_cnt = 10 # 1second


        # hardcoding for plymouth
        # Mapping phase from lanes
        self.depart_lane_list=['11.0.00_0','11.0.00_1','-16.0.00_0','-16.0.00_1','-1.0.00_0','-1.0.00_1','-1.0.00_2','-3.0.00_0',
                        '-3.0.00_1','-3.0.00_2','17.0.00_0','17.0.00_1','-2.0.00_0','-2.0.00_1','-2.0.00_2','4.0.00_0','4.0.00_1',
                        '18.0.00_0','18.0.00_1','-12.0.00_0','-12.0.00_1','-12.0.00_2','-7.0.00_0','-7.0.00_1','-7.0.00_2',
                        '-15.0.00_0','-15.0.00_1','-15.0.00_2','20.0.00_0','5.0.00_0','0.0.00_0']
        self.intersection_center=[522.07,435.96]
        self.plan_horizon = 120
        self.approaches = 10
        self.departure_rate=0.5
        self.phase_sequence=['47','15','26','38'] #phase name in the combination of approaches
        self.index2str={0:'rrrGrrrrrrrrGrrrrr',3:'GGGrrrrrGGGGrrrrrr',6:'rrrrGGGGrrrrrrrrrr',9:'rrrrrrrrrrrrrGGGGG'} #mapping from phase id to signal colors
        self.last_Gphase=None
        self.args = args

        self.att_action = None
        self.s = None
        self.a = None
        self.phase_to_one_hot = self.input_to_one_hot(self.green_phases+[self.all_red])
        self.g_max = g_max
        self.epsilon = args.eps
        self.eps_min = args.eps_min
        self.eps_factor = args.eps_factor
        
        self.state_action_record = []
        self.state_comparison = []
        
        self.last_green_phase = self.green_phases[0] #initialize

        # initialize I-SIG regression model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RegressionNet().to(self.device)
        self.model.load_state_dict(torch.load("/home/Documents/DRL-attacker/experiments/attacker_mmitiss/CAV_pen_rate_5.0/plymouth_bin_real_real/saved_models/ISIG_surr_class_0522.pth"))
        self.model.eval()

    def next_phase(self):
        if len(self.phase_deque) == 0:
            opt_phase_idx, self.opt_phase_duration = self.run_opt() #output signal id in sumo and its duration
            print("opt_phase_idx",opt_phase_idx,"opt_phase_duration",self.opt_phase_duration)
            

            opt_phase = self.index2str[opt_phase_idx]
            self.last_Gphase=opt_phase
            yellow_phase = ''.join([ p if p == 'r' else 'y' for p in opt_phase ])
            phases = [yellow_phase, self.all_red]
            # phases = self.get_intermediate_phases(self.phase, opt_phase)
            # if (not self.phase_cnt) and (opt_phase == 'GGGgrrrrGGGGgrrrrr'):
            #     protect_left = ["rrrGrrrrrrrrGrrrrr","rrryrrrrrrrryrrrrr"]
            # else:
            #     protect_left = []
            self.phase_deque.extend([opt_phase]+phases) # +protect_left # MOVE phases to the end
        return self.phase_deque.popleft()
    
    def next_phase_duration(self, current_phase):
        # WZ: TODO MMITISS will give the green phase duration
        if self.phase in self.green_phases:
            if self.phase == current_phase:
                return self.green_t_cnt
            else:
                return self.opt_phase_duration # if new phase, execute the given duration
        # elif self.phase == "rrrGrrrrrrrrGrrrrr":
        #         return 20 #self.green_t - 50 if self.green_t >=100 else 50 # this protected left turn should be shorter than other phases
        elif 'y' in self.phase:
            return self.yellow_t
        else:
            return self.red_t

    def update(self, data, cv_data,uv_data, mask):
        # data: include all vehicles' info, data: {lane_id:{veh_id:{VAR_LANEPOSITION, 
        #                                                           VAR_SPEED, 
        #                                                           VAR_LANE_ID, VAR_ROAD_ID, VAR_TIMELOSS, VAR_POSITION},...},...}
        # cv_data: include only CAV and detected vehicles' info
        # uv_data: include all not detected vehicles' info
        # can refer to get_subscription_data function in trafficsignalcontroller.py
        # to utilize subsciption data, e.g. self.data[l][v][traci.constants.VAR_LANE_ID]
        self.data = data
        self.cv_data = cv_data
        self.uv_data = uv_data

    def get_state_tsc(self):
        state_global, state_local = self.get_state('mmitiss', num_segments=3, act_ctm =False)
        state_global = np.concatenate([state_global,
                                        self.phase_to_one_hot[self.phase],
                                        np.array([self.phase_duration/self.g_max])]) # wz: add phase duration
        state_local = np.concatenate([state_local,
                                        self.phase_to_one_hot[self.phase],
                                        np.array([self.phase_duration/self.g_max])])
        state = [state_global,state_local]

        if self.epsilon > self.eps_min:
                self.epsilon = self.epsilon * self.eps_factor
        else:
            self.epsilon = self.eps_min

        return state
    
    def run_opt(self):
        # with open('2023-05-18-00-24-15.12682979_opt_input.json','r') as f:
        #     VehInfo=json.load(f)

        # WZ: remove the time index as we now run opt in real-time
        # data_source = "CTM"
        # if self.act_ctm:
        #Arrival_Table = self.CTM.get_Arrival_Table_outflow(int(self.t//10),self.plan_horizon+1,self.approaches) 
        #np.set_printoptions(threshold=np.inf)
        #print(Arrival_Table)
        # else:
        # Whenever the run_opt is running, we run attacker
        att_state = self.attacker.get_state(self.get_state_tsc()[1], self.att_action)

        x_torch = torch.tensor(att_state[np.newaxis, ...]).to(torch.float32).to(self.device) #torch.tensor().to(torch.float32)
        output = self.model(x_torch).squeeze()
        print(output)
        surrogate_output = max(torch.argmax(output).detach().cpu().numpy()*10*5 ,1) # add 1 to incase output as 0
        
        # TODO: Embed JSMA here. Initialize JSMA and load surrogate model into JSMA; input state for JSMA and generate features; input features and generate fake arrival table.
        # TODO: we can not apply JSMA for regression model like I-SIG surrogate. To do so, we might need 'target FGM' method; or force the regression to be classification, which is not ideal. 
        
        # adv_x_guide, feature_ids, target_action = self.attacker.get_advX_ISIG(state[1], self.curr_phase_idx)
        # self.JSMA_result.append([feature_ids, target_action,self.curr_phase_idx,adv_x_guide,state[1]])

        
        if self.att_action != None:
            self.attacker.store_experience(self.s, self.a, att_state, self.attacker.get_reward(self.att_action, self.get_reward_delay(tsc_type="cavlight")), False) 
        if self.phase in self.green_phases:
            self.last_green_phase = self.phase.copy()
        self.att_action = self.attacker.rlagent.get_action(att_state, self.epsilon, self.green_phases.index(self.last_green_phase)) # return [action, action_dist, action_dist_2, attack_scale]
        print("Attacker action:",self.att_action,'current phase',self.last_green_phase) ###
        self.s = att_state
        self.a = self.att_action.copy()

        fake_veh_info = self.attacker.get_fake_table(self.att_action, self.cv_data) # get state; get action

        Arrival_Table = self.get_Arrival_Table_obs(fake_veh_info)
        
        #print('!!!!!Arrival_Table ',Arrival_Table)
        if np.all(Arrival_Table==0):
            signal=[[self.return_signal_2(self.phase_sequence[0])[0],self.G_min_T[self.phase_sequence[0]]*10]]
        else:
            #if approaching vehicles exist, start DP
            result=self.signal_DP_outflow(Arrival_Table)##Arrival Table updated
            #print("result!!!!!!!: ",result)
            signal=self.optimal_policy_generation(result)
        
        # for i in range(3):
        #     marker+=signal[i][1]
        # if len(signal)>=4:
        #     index=self.phase_sequence.index(self.return_phase_char(signal[3][0]))
        # else:
        print("LALALALALALA",self.t,"signal",signal) ###
        index=self.phase_sequence.index(self.return_phase_char(signal[0][0])) #self.phase_sequence.index(self.return_phase_char(signal[0][0])) self.phase_sequence.index(self.return_phase_char(signal[3][0]))
        self.phase_sequence=self.phase_sequence[index+1:]+self.phase_sequence[:index+1] #self.phase_sequence[index+1:]+self.phase_sequence[:index+1] self.phase_sequence[index:]+self.phase_sequence[:index] 
        #self.phase_sequence=self.phase_sequence[1:]+self.phase_sequence[:1]

        print("signal[0][1] OPT output:",signal[0][1],"\n surrogate output:",surrogate_output)

        
        
        opt_phase = self.index2str[signal[0][0]]
        self.state_action_record.append([att_state, signal[0][1], opt_phase, self.phase, surrogate_output, self.ori_Arrival_Table])
                                        # ((self.t, 
                                        # self.delay_record, 
                                        # [opt_phase,signal[0][1]],  # tsc action: phase name, phase duration
                                        # [self.ori_Arrival_Table, Arrival_Table],
                                        # self.phase,
                                        # self.att_action # attacker action, [action, action_dist, action_dist_2, attack_scale]
                                        # ))
        
        return signal[0][0], round(surrogate_output)  # signal[0][1] #

    def get_Arrival_Table_obs(self, fake_veh_info):
        # This function will generate the arrival table for MMITISS directly from CV data rather than CTM
        veh_info={}
        veh_ids=[]
        for lane_id in self.cv_data:
            for veh_id, data in self.cv_data[lane_id].items():
                veh_info[veh_id] = data
                veh_ids.append(veh_id)
        Arrival_Table = np.zeros((self.plan_horizon+1, self.approaches))
        # get vehicle id list
        for veh_id in veh_ids:
            loc = self.return_ETA_cell(veh_id,veh_info)
            if loc:
                Arrival_Table[loc[0], loc[1]] += 1

        self.ori_Arrival_Table = Arrival_Table.copy()
        print("original arrival table",Arrival_Table) ####
        
        # DISABLE ATTACK FOR DATA COLLECTING 05162025
        # if fake_veh_info != None:
        #     for veh_id in fake_veh_info.keys():
        #         loc = self.return_ETA_cell(veh_id,fake_veh_info)
        #         if loc:
        #             Arrival_Table[loc[0], loc[1]] += 1
        
        # print("Attacked arrival table",Arrival_Table) ####

        return Arrival_Table

            
    def mapping_route2phase(self,lane_id):
        if lane_id in ['16.0.00_0','8.0.00_0','12.0.00_0',':140_0_0',':73_0_0']:# south bound start
            return 0
        elif lane_id in ['16.0.00_1',':73_0_1',':140_0_1']:
            return 1
        elif lane_id in ['16.0.00_2','16.0.00_3','8.0.00_2',':140_0_2',':73_0_2',':73_0_3']:
            return 2
        elif lane_id in ['8.0.00_1','12.0.00_1']:
            if random.random()<0.17345:
                return 1
            else:
                return 2
        elif lane_id in ['3.0.00_0',':66_0_0','7.0.00_0',':52_0_0','15.0.00_0',':33_0_0','2.0.00_0']:# west bound start
            return 3
        elif lane_id in ['3.0.00_1','3.0.00_2',':66_0_1',':66_0_2','7.0.00_1',':52_0_1','15.0.00_1',':33_0_1','2.0.00_1']:
            return 4
        elif lane_id in ['3.0.00_3',':66_0_3','7.0.00_2',':52_0_2','15.0.00_2',':33_0_2','2.0.00_2']:
            return 5
        elif lane_id in ['-17.0.00_0','-17.0.00_1',':24_0_1',':24_0_0']:
            return 6
        elif lane_id in ['-17.0.00_2',':24_0_2']:
            return 7
        elif lane_id in ['-20.0.00_0','-5.0.00_0','-5.0.00_1','-0.0.00_0',':156_0_0',':156_0_1']:
            if random.random()<0.22604:
                return 7
            else:
                return 6
        elif lane_id in ['-11.0.00_0','-11.0.00_1','-11.0.00_2',':27_2_0',':27_2_1',':27_2_2','-18.0.00_0','-18.0.00_1',':46_2_0',':46_2_1','-4.0.00_0','-4.0.00_1']:
            return 8
        elif lane_id in ['-11.0.00_3',':27_2_3','-18.0.00_2',':46_2_2','-4.0.00_2']:
            return 9

    def phase2ETA(self,veh_id,veh_info):
        veh_speed=veh_info[veh_id][traci.constants.VAR_SPEED]
        x2,y2=self.intersection_center
        if veh_speed <= 2:
            return 0
        else:
            x1,y1 = veh_info[veh_id][traci.constants.VAR_POSITION]
            return round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)/veh_speed)
        
    def return_ETA_cell(self, veh_id,veh_info):
        phase_id = -1  # False bug ?idk
        lane_id=veh_info[veh_id][traci.constants.VAR_LANE_ID]
        phase_id =self.mapping_route2phase(lane_id)
        ETA = self.phase2ETA(veh_id,veh_info)
        if ETA >= self.plan_horizon-1:
            return False
        else:
            if phase_id!=None:
                return [ETA, phase_id]

    #DP-related functions!!!!!!!!!!!!!!!!!
    def phaseid2signalid(self, phaseid):#phaseid to signalid on sumo
        if phaseid == 1:
            return [5]
        elif phaseid == 5:
            return [9]
        elif phaseid == 2:
            return [8]
        elif phaseid == 6:
            return [3,4]
        elif phaseid == 3:
            return [7]
        elif phaseid == 4:
            return [0,1]
        elif phaseid == 7:
            return [2]
        elif phaseid == 8:
            return [6]
    def phaseid2directionid(self, phaseid):#phaseid to direction on sumo
        if phaseid == 1:
            return [7]
        elif phaseid == 5:
            return [14]
        elif phaseid == 2:
            return [11,12,13]
        elif phaseid == 6:
            return [4,5,6]
        elif phaseid == 3:
            return [10]
        elif phaseid == 4:
            return [0,1]
        elif phaseid == 7:
            return [2,3]
        elif phaseid == 8:
            return [8,9]

    def return_signal_2(self, phase):  # input '16','15'...
        if phase == '15':
            return [0,1,2] #phase_id in SUMO
        elif phase == '26':
            return [3,4,5]
        elif phase == "38":
            return [6,7,8]
        elif phase == '47':
            return [9,10,11]
    
    '''no use for 4 phases
    def return_signal_4(self, phase):  # input '1256','3847'...
        if phase=='1256':
            return [0,1,2]
    '''

    def return_phase_loc(self,phase):
        if phase == '15':
            phase_loc=[5,9]
        elif phase == '26':
            phase_loc=[3,4,8]
        elif phase == '38':
            phase_loc=[6,7]
        elif phase == '47':
            phase_loc=[0,1,2]
        return phase_loc #direction column id in arrival table

    def return_phase_loc_spec(self,phase):
        if phase=='15':
            phase_loc_1=[5,9] #the direction with one lane
            phase_loc_2=[] #the direction with two lanes
            phase_loc_3=[] #the direction with three lanes
        elif phase=='26':
            phase_loc_1=[3] 
            phase_loc_2=[4]
            phase_loc_3=[8]
        elif phase=='38':
            phase_loc_1=[7]
            phase_loc_2=[6]
            phase_loc_3=[]
        elif phase=='47':
            phase_loc_1=[0,1]
            phase_loc_2=[2]
            phase_loc_3=[]
        return phase_loc_1,phase_loc_2,phase_loc_3 #departure rate*1, departure rate*2, departure rate*3

    def return_phase_char(self, index):
        if index==0:
            return '15'
        elif index==3:
            return '26'
        elif index==6:
            return '38'
        elif index==9:
            return '47'
    
    def signal_DP_sup_2_outflow(self,s,x,phase,prev_stage_calculation,Arrival_Table,offset):
        prev_stage_bu=prev_stage_calculation.copy()[s-x-(self.Yellow+self.Red)*2-offset]###zth20240207:prev_stage_calculation
        phase_loc_1,phase_loc_2,phase_loc_3=self.return_phase_loc_spec(phase)
        # clearance
        for i in range(s-5-x+1, s-5+1):
            prev_stage_bu=np.vstack((prev_stage_bu, prev_stage_bu[i-1,]+np.append(Arrival_Table[i,:], [0, 0])))
            prev_stage_bu[i, phase_loc_1]=np.maximum(prev_stage_bu[i, phase_loc_1] - self.departure_rate, 0)
            prev_stage_bu[i, phase_loc_2]=np.maximum(prev_stage_bu[i, phase_loc_2] - self.departure_rate*2, 0)
            prev_stage_bu[i, phase_loc_3]=np.maximum(prev_stage_bu[i, phase_loc_3] - self.departure_rate*3, 0)
            prev_stage_bu[i, -1] = prev_stage_bu[i-1, -1] +np.sum(prev_stage_bu[i, :self.approaches])
        for i in range(s-5+1, s+1):
            prev_stage_bu=np.vstack((prev_stage_bu, prev_stage_bu[i-1,]+np.append(Arrival_Table[i,:], [0, 0])))
            prev_stage_bu[i, -1] =prev_stage_bu[i-1, -1]+ np.sum(prev_stage_bu[i, :self.approaches])
        return x, prev_stage_bu[-1, -1], prev_stage_bu

    def signal_DP_sup_1_outflow(self,s,phase,prev_stage_calculation,Arrival_Table,s_start,offset):#s=row_index+1
        '''
        s: current second
        s_start: first second of this planning horizon
        phase: current phase
        '''
        if s-(self.G_min_T[phase]+self.Yellow+self.Red)<s_start+1:
            flag=True #replace former value function
        else:
            flag=False #append the value function directly
        upload=[-1,np.inf]
        X=[]
        for i in range(self.G_min_T[phase],self.G_max_T+1):
            if s-5-i>=s_start-self.Yellow-self.Red-self.G_min_T[phase]:
                X.append(i)
        for x_minor in X:
            x,value,stage=self.signal_DP_sup_2_outflow(s,x_minor,phase,prev_stage_calculation,Arrival_Table,offset)###zth20240207:prev_stage_calculation
            if value<upload[1]:#######zth20240123:delete equal
                upload=[x,value,stage,flag] 
        return upload

    def signal_DP_outflow(self, Arrival_Table):
        #not deciding skip or not
        #stage1 calculation
        phase_sequence_index = 0
        flag = True
        while flag:
            phase=self.phase_sequence[phase_sequence_index%len(self.phase_sequence)]
            phase_loc=self.return_phase_loc(phase) #corresponding direction in arrival table
            all_zero_columns = np.all(Arrival_Table == 0, axis=0) #check if every direction has incoming vehicles,if no, the corresponding column is 1
            if all_zero_columns[phase_loc].all():
                phase_sequence_index += 1
                continue
            flag = False
            phase_loc_1,phase_loc_2,phase_loc_3=self.return_phase_loc_spec(phase) #the direction with 1,2,3 lanes
            prev_stage_calculation=[]###20240123 record previous traffic flow
            for i in range(self.G_min_T[phase]+self.Yellow+self.Red+1,self.G_max_T+self.Yellow+self.Red+1+1):#first +1 because we need one more +1, the second +1 because we need to obtain the value
                s1c_temp=np.zeros((i, self.approaches+2))
                s1c_temp[self.G_min_T[phase]+self.Yellow+self.Red:i, -2] = np.arange(self.G_min_T[phase], i-5)
                s1c_temp[0, :self.approaches]=Arrival_Table[0, :self.approaches]
                s1c_temp[0, -1] =np.sum(s1c_temp[0, :self.approaches])
                for j in range(1, i-5):###zth20240207:i-4 to i-5
                    s1c_temp[j, :self.approaches]=s1c_temp[j-1,  :self.approaches]+Arrival_Table[j, :self.approaches]
                    s1c_temp[j, phase_loc_1]=np.maximum(s1c_temp[j, phase_loc_1] - self.departure_rate, 0)
                    s1c_temp[j, phase_loc_2]=np.maximum(s1c_temp[j, phase_loc_2] - self.departure_rate*2, 0)
                    s1c_temp[j, phase_loc_3]=np.maximum(s1c_temp[j, phase_loc_3] - self.departure_rate*3, 0)
                    s1c_temp[j, -1]=s1c_temp[j-1, -1]+np.sum(s1c_temp[j, :self.approaches])
                for j in range(i-5,len(s1c_temp)):###zth20240207:i-4 to i-5
                    s1c_temp[j, :self.approaches] = s1c_temp[j - 1, :self.approaches] + Arrival_Table[j, :self.approaches]
                    s1c_temp[j, -1] = s1c_temp[j-1, -1] +np.sum(s1c_temp[j, :self.approaches])
                prev_stage_calculation.append(s1c_temp)
        # Build decision table
        historical_decision_table=np.zeros((self.plan_horizon+1, 1))
        historical_decision_table[0:len(prev_stage_calculation[-1])]=prev_stage_calculation[-1][:,-2].reshape(-1, 1)
        # Build value function table
        historical_value_table=np.zeros((self.plan_horizon+1, 1))
        historical_value_table[0:len(prev_stage_calculation[-1])]=prev_stage_calculation[-1][:,-1].reshape(-1, 1)
        # Signal
        historical_signals = [phase]
        #stage>=1
        flag=True#flag for stop
        s_start=self.G_min_T[phase] + self.Yellow + self.Red
        phase_sequence_index+=1
        offset = self.G_min_T[phase]
        while flag:
            phase=self.phase_sequence[phase_sequence_index%len(self.phase_sequence)]
            phase_loc=self.return_phase_loc(phase)

            #check phase to skip
            all_zero_columns = np.all(Arrival_Table == 0, axis=0)
            if all_zero_columns[phase_loc].all():
                phase_sequence_index+=1
                continue 

            s_start+=self.G_min_T[phase]+self.Yellow+self.Red 
            #Start
            historical_signals.append(phase)
            #build decision table
            decision_table = np.zeros((self.plan_horizon+1, 1))
            # Build value function table as a NumPy array
            value_table = np.zeros((self.plan_horizon+1, 1))
            
            prev_stage_calculation_previous=prev_stage_calculation.copy()#avoid the impact of the update of prev_stage_calculation
            for s in range(s_start, min(s_start+self.G_max_T-offset+1, 121)):
                x,v,stage,replacement=self.signal_DP_sup_1_outflow(s,phase,prev_stage_calculation_previous,Arrival_Table,s_start,offset)
                decision_table[s]=x
                value_table[s]=v
                if len(stage) <= len(prev_stage_calculation[-1]):
                    prev_stage_calculation[s-(offset + self.Yellow + self.Red)]=stage.copy()
                else:
                    prev_stage_calculation.append(stage.copy())
            historical_decision_table = np.concatenate((historical_decision_table, decision_table), axis=1)
            historical_value_table = np.concatenate((historical_value_table, value_table), axis=1)
            
            #finish one stage

            #check if stop
            if s == 120:
                flag = False

            phase_sequence_index+=1
        return historical_decision_table,historical_value_table,historical_signals
        

    def optimal_policy_generation(self,result):
        stage_index=len(result[2])-1
        time_index=self.plan_horizon#####20240204 delete -1
        time_all=[]
        for i in range(len(result[2])):
            time_temp=result[0][int(time_index),int(stage_index)]
            time_all.append(time_temp)
            stage_index-=1
            if time_temp==0:
                time_index=max(time_index-time_temp,0)
            else:
                time_index=max(time_index-time_temp-5,0)
        time_all.reverse()
        signal_time=[[a, b*10] for a, b in zip(result[2], time_all)]
        #test20240202
        # flag=True
        # while flag:
        #     if signal_time[0][0]!=signal_time[1][0]:
        #         flag=False
        #     else:
        #         signal_time[1][1]+=signal_time[0][1]
        #         signal_time.pop(0)
        #         if signal_time[0][1]>=self.G_max_T*10:
        #             signal_time[0][1]=self.G_max_T*10
        #             flag=False
        #delete 0
        new_list=[]
        for i in range(len(signal_time)):
            if signal_time[i][1]!=0 and len(signal_time[i][0])==4:
                signal_return=self.return_signal_4(signal_time[i][0])
                new_list.append([signal_return[0],signal_time[i][1]])
                new_list.append([signal_return[1],40])
                new_list.append([signal_return[2],10])
            elif signal_time[i][1]!=0 and len(signal_time[i][0])==2:
                signal_return=self.return_signal_2(signal_time[i][0])
                new_list.append([signal_return[0],signal_time[i][1]])
                new_list.append([signal_return[1],40])
                new_list.append([signal_return[2],10])
        return new_list


