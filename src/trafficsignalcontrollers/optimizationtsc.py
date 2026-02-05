import gurobipy as gp
from gurobipy import GRB
import numpy as np
#import sympy
import math
import json
import random
from scipy.optimize import minimize 
from sklearn.model_selection import train_test_split
from numpy.linalg import norm
import copy
import traci
import pandas as pd
from src.CTM import CTM_model 
from src.trafficsignalcontroller import TrafficSignalController
from itertools import cycle
from collections import deque



class OptimizationTSC(TrafficSignalController):
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, green_t, detect_r, g_max, act_ctm):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t, detect_r)

        # '''
        # <route id="rightstrit" edges="5 8"/>
        # <route id="leftstrit" edges="1 4"/>
        # <route id="downstrit" edges="3 6"/>
        # <route id="upstrit" edges="7 2"/> 
        # <route id="leftup" edges="1 6"/>
        # <route id="leftdown" edges="1 2"/>
        # <route id="rightup" edges="5 6"/>
        # <route id="rightdown" edges="5 2"/>
        # <route id="upleft" edges="7 8"/>
        # <route id="upright" edges="7 4"/>
        # <route id="downleft" edges="3 8"/>
        # <route id="downright" edges="3 4"/>
        # '''    

        # This opt-based TSC is required to use CTM to get state.
        # if not act_ctm:
        #     raise NotImplementedError("This OPT-TSC requires CTM to generate input")

        self.cycle = cycle(self.green_phases)
        self.green_t = green_t
        self.green_t_cnt = 10 # unit 0.1s
        self.g_max = g_max
        self.t = 0
        self.phase_deque = deque()
        self.phase_type = "4ph" #"3ph"

        self.route_dict={
            "rightstrit":{"edges":[5,8],"lanes":[0,1]},
            "leftstrit":{"edges":[1,4],"lanes":[0,1]},
            "downstrit":{"edges":[3,6],"lanes":[0,1]},
            "upstrit":{"edges":[7,2],"lanes":[0,1]},
            "leftup":{"edges":[1,6],"lanes":[2]},
            "leftdown":{"edges":[1,2],"lanes":[0]},
            "rightup":{"edges":[5,6],"lanes":[0]},
            "rightdown":{"edges":[5,2],"lanes":[2]},
            "upleft":{"edges":[7,8],"lanes":[0]},
            "upright":{"edges":[7,4],"lanes":[2]},
            "downleft":{"edges":[3,8],"lanes":[2]},
            "downright":{"edges":[3,4],"lanes":[0]}
            }
        
        # '''
        # lane_to_phase_dict={
        #         1:{0:3,1:3,2:8},
        #         3:{0:5,1:5,2:2},
        #         5:{0:7,1:7,2:4},
        #         7:{0:1,1:1,2:6}
        #         }
        # '''
        if self.phase_type == "3ph":
            # 3 phases
            self.lane_to_phase_dict={
                    1:{0:1,1:1,2:1,3:1},
                    3:{0:1,1:1,2:1,3:1},
                    5:{0:2,1:2,2:2},
                    7:{0:3,1:3,2:3,3:3}
                    }
        else:
            # 4 phases
            # self.lane_to_phase_dict={
            #     1:{0:5,1:2,2:2,3:2},
            #     3:{0:1,1:6,2:6,3:6},
            #     5:{0:8,1:3,2:8},
            #     7:{0:7,1:7,2:4,3:4}
            #     }

            self.lane_to_phase_dict={
            1:{0:5,1:2,2:2,3:2},
            3:{0:1,1:6,2:6,3:6},
            5:{0:3,1:3,2:3},
            7:{0:4,1:4,2:4,3:4}
            }
        
        self.lane2arm = {"-4.0.00_0":{"app":"1","lane":"1_2"},
                            "-4.0.00_1":{"app":"1","lane":"1_1"},
                            "-4.0.00_2":{"app":"1","lane":"1_0"},
                            ':46_2_0':{"app":"1","lane":"1_2"},
                            ':46_2_1':{"app":"1","lane":"1_1"},
                            ':46_2_2':{"app":"1","lane":"1_0"},
                            "-18.0.00_0":{"app":"1","lane":"1_2"},
                            "-18.0.00_1":{"app":"1","lane":"1_1"},
                            "-18.0.00_2":{"app":"1","lane":"1_0"},
                            ':27_2_0':{"app":"1","lane":"1_3"},
                            ':27_2_1':{"app":"1","lane":"1_2"},
                            ':27_2_2':{"app":"1","lane":"1_1"},
                            ':27_2_3':{"app":"1","lane":"1_0"},
                            "-11.0.00_0":{"app":"1","lane":"1_3"},
                            "-11.0.00_1":{"app":"1","lane":"1_2"},
                            "-11.0.00_2":{"app":"1","lane":"1_1"},
                            "-11.0.00_3":{"app":"1","lane":"1_0"},
                            "-3.0.00_0":{"app":"2","lane":"2_2"}, 
                            "-3.0.00_1":{"app":"2","lane":"2_1"}, 
                            "-3.0.00_2":{"app":"2","lane":"2_0"}, 
                            ':66_4_0':{"app":"2","lane":"2_2"},  
                            ':66_4_1':{"app":"2","lane":"2_1"},  
                            ':66_4_2':{"app":"2","lane":"2_0"},  
                            '-7.0.00_0':{"app":"2","lane":"2_2"},   
                            '-7.0.00_1':{"app":"2","lane":"2_1"},  
                            '-7.0.00_2':{"app":"2","lane":"2_0"}, 
                            ':52_3_0':{"app":"2","lane":"2_2"}, 
                            ':52_3_1':{"app":"2","lane":"2_1"},  
                            ':52_3_2':{"app":"2","lane":"2_0"}, 
                            '-15.0.00_0':{"app":"2","lane":"2_2"}, 
                            '-15.0.00_1':{"app":"2","lane":"2_1"},  
                            '-15.0.00_2':{"app":"2","lane":"2_0"}, 
                            ':33_3_0':{"app":"2","lane":"2_2"}, 
                            ':33_3_1':{"app":"2","lane":"2_1"}, 
                            ':33_3_2':{"app":"2","lane":"2_0"}, 
                            '-2.0.00_0':{"app":"2","lane":"2_2"}, 
                            '-2.0.00_1':{"app":"2","lane":"2_1"}, 
                            '-2.0.00_2':{"app":"2","lane":"2_0"}, 
                            '2.0.00_0':{"app":"3","lane":"3_2"}, 
                            '2.0.00_1':{"app":"3","lane":"3_1"}, 
                            '2.0.00_2':{"app":"3","lane":"3_0"}, 
                            ':33_0_0':{"app":"3","lane":"3_2"}, 
                            ':33_0_1':{"app":"3","lane":"3_1"}, 
                            ':33_0_2':{"app":"3","lane":"3_0"}, 
                            '15.0.00_0':{"app":"3","lane":"3_2"}, 
                            '15.0.00_1':{"app":"3","lane":"3_1"}, 
                            '15.0.00_2':{"app":"3","lane":"3_0"}, 
                            ':52_0_0':{"app":"3","lane":"3_2"}, 
                            ':52_0_1':{"app":"3","lane":"3_1"}, 
                            ':52_0_2':{"app":"3","lane":"3_0"}, 
                            '7.0.00_0':{"app":"3","lane":"3_2"}, 
                            '7.0.00_1':{"app":"3","lane":"3_1"}, 
                            '7.0.00_2':{"app":"3","lane":"3_0"}, 
                            ':66_0_0':{"app":"3","lane":"3_3"}, 
                            ':66_0_1':{"app":"3","lane":"3_2"}, 
                            ':66_0_2':{"app":"3","lane":"3_1"}, 
                            ':66_0_3':{"app":"3","lane":"3_0"}, 
                            '3.0.00_0':{"app":"3","lane":"3_3"}, 
                            '3.0.00_1':{"app":"3","lane":"3_2"}, 
                            '3.0.00_2':{"app":"3","lane":"3_1"}, 
                            '3.0.00_3':{"app":"3","lane":"3_0"}, 
                            '11.0.00_0':{"app":"4","lane":"4_1"}, 
                            '11.0.00_1':{"app":"4","lane":"4_0"}, 
                            ':27_0_0':{"app":"4","lane":"4_1"}, 
                            ':27_0_1':{"app":"4","lane":"4_0"}, 
                            '18.0.00_0':{"app":"4","lane":"4_1"}, 
                            '18.0.00_1':{"app":"4","lane":"4_0"}, 
                            ':46_0_0':{"app":"4","lane":"4_1"}, 
                            ':46_0_1':{"app":"4","lane":"4_0"}, 
                            '4.0.00_0':{"app":"4","lane":"4_1"}, 
                            '4.0.00_1':{"app":"4","lane":"4_0"}, 
                            '-0.0.00_0':{"app":"5","lane":"5_2"}, 
                            ':21_0_0':{"app":"5","lane":"5_2"}, 
                            ':21_0_1':{"app":"5","lane":"5_1"}, 
                            '-5.0.00_0':{"app":"5","lane":"5_2"}, 
                            '-5.0.00_1':{"app":"5","lane":"5_1"}, 
                            ':156_0_0':{"app":"5","lane":"5_2"}, 
                            ':156_0_1':{"app":"5","lane":"5_1"}, 
                            '-20.0.00_0':{"app":"5","lane":"5_2"}, 
                            ':24_0_0':{"app":"5","lane":"5_2"}, 
                            ':24_0_1':{"app":"5","lane":"5_1"}, 
                            ':24_0_2':{"app":"5","lane":"5_0"}, 
                            '-17.0.00_0':{"app":"5","lane":"5_2"}, 
                            '-17.0.00_1':{"app":"5","lane":"5_1"}, 
                            '-17.0.00_2':{"app":"5","lane":"5_0"}, 
                            '-16.0.00_0':{"app":"6","lane":"6_1"}, 
                            '-16.0.00_1':{"app":"6","lane":"6_0"}, 
                            ':73_4_0':{"app":"6","lane":"6_1"}, 
                            ':73_4_1':{"app":"6","lane":"6_0"}, 
                            '-8.0.00_0':{"app":"6","lane":"6_1"}, 
                            '-8.0.00_1':{"app":"6","lane":"6_0"}, 
                            ':140_3_0':{"app":"6","lane":"6_2"}, 
                            ':140_3_1':{"app":"6","lane":"6_1"}, 
                            ':140_3_2':{"app":"6","lane":"6_0"}, 
                            '-12.0.00_0':{"app":"6","lane":"6_2"}, 
                            '-12.0.00_1':{"app":"6","lane":"6_1"}, 
                            '-12.0.00_2':{"app":"6","lane":"6_0"}, 
                            '12.0.00_0':{"app":"7","lane":"7_1"}, 
                            '12.0.00_1':{"app":"7","lane":"7_0"},
                            ':140_0_0':{"app":"7","lane":"7_2"},
                            ':140_0_1':{"app":"7","lane":"7_1"},
                            ':140_0_2':{"app":"7","lane":"7_0"},
                            '8.0.00_0':{"app":"7","lane":"7_2"},
                            '8.0.00_1':{"app":"7","lane":"7_1"},
                            '8.0.00_2':{"app":"7","lane":"7_0"},
                            ':73_0_0':{"app":"7","lane":"7_3"},
                            ':73_0_1':{"app":"7","lane":"7_2"},
                            ':73_0_2':{"app":"7","lane":"7_1"},
                            ':73_0_3':{"app":"7","lane":"7_0"},
                            '16.0.00_0':{"app":"7","lane":"7_3"},
                            '16.0.00_1':{"app":"7","lane":"7_2"},
                            '16.0.00_2':{"app":"7","lane":"7_1"},
                            '16.0.00_3':{"app":"7","lane":"7_0"},
                            '17.0.00_0':{"app":"8","lane":"8_1"},
                            '17.0.00_1':{"app":"8","lane":"8_0"},
                            ':24_3_0':{"app":"8","lane":"8_0"},
                            '20.0.00_0':{"app":"8","lane":"8_0"},
                            ':156_2_0':{"app":"8","lane":"8_0"},
                            '5.0.00_0':{"app":"8","lane":"8_0"},
                            ':21_2_0':{"app":"8","lane":"8_0"},
                            '0.0.00_0':{"app":"8","lane":"8_0"},
                            }

    def next_phase(self):
        if len(self.phase_deque) == 0:
            opt_phase = self.run_opt()
            phases = self.get_intermediate_phases(self.phase, opt_phase)
            # if (not self.phase_cnt) and (opt_phase == 'GGGgrrrrGGGGgrrrrr'):
            #     protect_left = ["rrrGrrrrrrrrGrrrrr","rrryrrrrrrrryrrrrr"]
            # else:
            #     protect_left = []
            self.phase_deque.extend(phases+[opt_phase]) # +protect_left
        return self.phase_deque.popleft()
    
    def next_phase_duration(self, current_phase):
        # WZ: the logic of combing OPT-TSC into current framework: 
        # if next phase is in green phases, the phase duration 
        #  is determined by rem_g in run_opt
        #  however, the return value of this function would determine
        #  the frequency of executing increment_controller function
        #  and we want to run the optimization process every second
        #  therefore, we continue to set the next phase duration
        #  as green_t_cnt rathther than determined green phase duration
        #  from OPT. 
        # Or, if we want to directly use the OPT output phase duration, 
        #  we can directly return the remaining green time
        if self.phase in self.green_phases:
            if self.phase == current_phase:
                return self.green_t_cnt
            else:
                if self.phase == "rrrGrrrrrrrrGrrrrr":
                    return 20
                else:
                    return self.green_t
        # elif self.phase == "rrrGrrrrrrrrGrrrrr":
        #         return 20 #self.green_t - 50 if self.green_t >=100 else 50 # this protected left turn should be shorter than other phases
        elif 'y' in self.phase:
            return self.yellow_t
        else:
            return self.red_t

    def update(self, data, cv_data,uv_data, mask):
        if mask:
            self.data = cv_data
            self.uv_data = uv_data
        else:
            self.data = data
        # self.data = cv_data
    
    def run_opt(self):
        # with open('2023-05-18-00-24-15.12682979_opt_input.json','r') as f:
        #     VehInfo=json.load(f)

        # WZ: remove the time index as we now run opt in real-time
        # data_source = "CTM"

        if self.act_ctm:
            VehInfo = self.CTM.get_state_CTM_OPT(int(self.t//10),self.phase,self.phase_duration)
        else:
            VehInfo = self.get_state_OPT_CAV(self.phase,self.phase_duration)
                
        self.GetPhaseResult(VehInfo)
        #for time_step in VehInfo.keys():
            #only retrieve remaining green value when current phase is green(not amber or all red)
        current_phase_str=VehInfo['phase']
        current_phase_num=self.sumo_phase_to_num(current_phase_str)
        if current_phase_num!=10000:
            remaining_green_time=VehInfo['remaining_green_time']
        else:
            remaining_green_time = []

        print("\n Time:",self.t,"phase:",self.phase,"remain green:",remaining_green_time)####
        if (len(remaining_green_time) == 0) or (self.phase_duration >= self.g_max):
            return next(self.cycle)
        else:
            return self.phase

    def get_state_OPT_CAV(self,phase,phase_duration):
        # This function will generate the input for OPT directly from CV data rather than CTM
        # step 1: identify the arm_id and lane_id the vehicle belongs to -> directly hard code a lane2arm dict
        # step 2: calculate dist2int for each vehicle and assign {vehicle id, dist2int} to {armID: laneID: {}}
        # step 3: sort cumulate list of veh info dictionary based on dist2bar

        inter_dict_lane_based = {str(arm_id): {str(arm_id)+"_"+str(lane_id):[] for lane_id in range(4) } for arm_id in range(1,9)}
        # note that we use lane number 0 to represent left-turning lane, which is different from SUMO which start with right side as 0 

        for lane_id in self.data.keys():
            if lane_id in self.lane2arm.keys():
                arm_id = self.lane2arm[lane_id]["app"]
                veh_lane = self.lane2arm[lane_id]["lane"]
                for v_id, info in self.data[lane_id].items():
                    veh_position = np.array(info[traci.constants.VAR_POSITION])
                    veh2int = abs(np.linalg.norm(self.junc_position - veh_position) - 20) 
                    inter_dict_lane_based[arm_id][veh_lane].append({'veh_id':v_id,'dist2bar':veh2int})
        
        # for arm_id in inter_dict_lane_based.keys():
        #     for lane_id in inter_dict_lane_based[arm_id].keys():
        #         inter_dict_lane_based[arm_id][lane_id] = [ i for i in pd.DataFrame(inter_dict_lane_based[arm_id][lane_id]).sort_values(by=['dist2bar']).T.to_dict().values()]

        return {"inter_dict_lane_based":inter_dict_lane_based,"phase":phase, "phase_duration":phase_duration}

    def GetPhaseResult(self,VehInfo):
        #VehInfo: VehInfo dict, including vehicle information, current green phase and elapsed green time.
        for direction in VehInfo['inter_dict_lane_based'].keys():
            for lane_id in VehInfo['inter_dict_lane_based'][direction].keys():
                VehList=copy.deepcopy(VehInfo['inter_dict_lane_based'][direction][lane_id])
                Dist2Stopbar=[]
                VehListSorted=[]
                for i in range(len(VehList)):
                    Dist2Stopbar.append(VehInfo['inter_dict_lane_based'][direction][lane_id][i]['dist2bar'])
                    Dist2Stopbar.sort()
                for SortDis in Dist2Stopbar:
                    for i in range(len(VehList)):
                        if VehList[i]['dist2bar']==SortDis:
                            VehListSorted.append(VehList[i])
                VehInfo['inter_dict_lane_based'][direction][lane_id]=VehListSorted
                print("direction",direction,"lane",lane_id,"\n veh_list_sorted:",VehInfo['inter_dict_lane_based'][direction][lane_id],"\n num of veh:",len(VehInfo['inter_dict_lane_based'][direction][lane_id])) ####


        current_phase_str=VehInfo['phase']
        current_phase_num=self.sumo_phase_to_num(current_phase_str)
        if current_phase_num!=10000:
            phase_result_value,west_result,east_result,north_result,south_result,g_rem_value,r_rem_value,non_empty_solution,I_phase_G=\
                self.co_optimization(VehInfo,self.lane_to_phase_dict,current_phase_num)
            #optimization result, for debug use
            VehInfo['phase_result']=phase_result_value
            VehInfo['west_result']=west_result
            VehInfo['east_result']=east_result
            VehInfo['north_result']=north_result
            VehInfo['south_result']=south_result
            VehInfo['g_rem_value']=g_rem_value
            VehInfo['r_rem_value']=r_rem_value
            VehInfo['I_phase_G']=I_phase_G
            remaining_green_time=[]

            ####
            print("VehInfo['phase_result']:",VehInfo['phase_result'],"\n",
                    "VehInfo['west_result']",VehInfo['west_result'],"\n",
                    "VehInfo['east_result']",VehInfo['east_result'],"\n",
                    "VehInfo['north_result']",VehInfo['north_result'],"\n",
                    "VehInfo['south_result']",VehInfo['south_result'],"\n")
            ####

            for i in range(len(g_rem_value)):
                if g_rem_value[i]!=0:
                    remaining_green_time.append(g_rem_value[i])
                    break
            #the remaining green duration for current green phase (float)
            VehInfo['remaining_green_time']=remaining_green_time
    
        return VehInfo
                

    def sort_lane_veh(self,inter_dict):
        inter_dict_sorted={}
        for time_step in inter_dict.keys():
            inter_dict_current=inter_dict[time_step]
            direction_dict={}
            for veh in inter_dict_current:
                if int(veh['lane_id'][0]) not in direction_dict.keys():
                    direction_dict[int(veh['lane_id'][0])]=[]
                    direction_dict[int(veh['lane_id'][0])].append(veh)
                else:
                    direction_dict[int(veh['lane_id'][0])].append(veh)
            inter_dict_sorted[time_step]=direction_dict
        
        inter_dict_lane_based={}
        for time_step in inter_dict_sorted.keys():
            inter_dict_current=inter_dict_sorted[time_step]
            inter_dict_current_lane={}
            for direction in inter_dict_current.keys():
                lane_dict={}
                for veh in inter_dict_current[direction]:
                    if int(veh['lane_id'][-1]) not in lane_dict.keys():
                        lane_dict[int(veh['lane_id'][-1])]=[]
                        lane_dict[int(veh['lane_id'][-1])].append(veh)
                    else:
                        lane_dict[int(veh['lane_id'][-1])].append(veh)
                inter_dict_current_lane[direction]=lane_dict
            inter_dict_lane_based[time_step]=inter_dict_current_lane
        #remove wrong surrounding vehicle data

        
        inter_dict_lane_based_new={}
        for time_step in inter_dict_lane_based.keys():
            inter_dict_current=inter_dict_lane_based[time_step]
            inter_dict_current_lane={}
            for direction in inter_dict_current.keys():
                lane_dict={}
                for lane_id in inter_dict_current[direction].keys():
                    dis_list=[]
                    veh_list=[]
                    for veh in inter_dict_current[direction][lane_id]:
                        dis_list.append(veh['position_in_lane'])
                    dis_list.sort(reverse=True)
                    #print(dis_list)
                    for veh in inter_dict_current[direction][lane_id]:
                        for i in range(len(dis_list)):
                            if veh['position_in_lane']==dis_list[i]:
                                veh['index_in_lane']=i
                    for i in range(len(dis_list)):
                        for veh in inter_dict_current[direction][lane_id]:
                            if veh['index_in_lane']==i:
                                veh_list.append(veh)
                    lane_dict[lane_id]=veh_list
                    
                inter_dict_current_lane[direction]=lane_dict
            inter_dict_lane_based_new[time_step]=inter_dict_current_lane
        
        return inter_dict_sorted,inter_dict_lane_based_new
        

    #inter_dict_lane_based=lane_to_phase(inter_dict_lane_based)
                
    def get_current_green_phase(self,current_phase,elapsed_time):
        if self.phase_type == "3ph":
            sumo_to_dual_ring={
                0:[1],
                1:[],
                2:[],
                3:[2],
                4:[],
                5:[],
                6:[3],
                7:[],
                8:[]
                }
            I_phase_G=list(np.zeros(3))
            for i in range(len(I_phase_G)):
                if i+1 in sumo_to_dual_ring[current_phase]:
                    I_phase_G[i]=1
                else:
                    I_phase_G[i]=0
            return I_phase_G
        else:
            sumo_to_dual_ring={
                # 0:[1,5],
                # 1:[],
                # 2:[],
                # 3:[2,6],
                # 4:[],
                # 5:[],
                # 6:[3,7],
                # 7:[],
                # 8:[],
                # 9:[4,8],
                # 10:[],
                # 11:[]}
                0:[1,5],
                1:[],
                2:[],
                3:[2,6],
                4:[],
                5:[],
                6:[3,7],
                7:[],
                8:[],
                9:[4,8],
                10:[],
                11:[]}
            #sumo_phase_length=[23,4,2,10,4,2,23,4,2,10,4,2]
            #phase_lost=list(np.ones(8)*6)
            I_phase_G=list(np.zeros(8))
            for i in range(len(I_phase_G)):
                if i+1 in sumo_to_dual_ring[current_phase]:
                    I_phase_G[i]=1
                else:
                    I_phase_G[i]=0
            return I_phase_G

    def sumo_phase_to_num(self,current_phase_str):
        if self.phase_type == "3ph":
            str2num={
            'GGGgrrrrGGGGgrrrrr':0,
            'rrrrGGGGrrrrrrrrrr':3,
            'rrrrrrrrrrrrrGGGGG':6
            }
        else:
            # str2num={
            # 'rrrGrrrrrrrrGrrrrr':0,
            # 'GGGrrrrrGGGGrrrrrr':3,
            # 'rrrrrrrGrrrrrrrrGG':6,
            # 'rrrrGGGrrrrrrGGGrr':9
            # }
            str2num={
            'rrrGrrrrrrrrGrrrrr':0,
            'GGGrrrrrGGGGrrrrrr':3,
            'rrrrGGGGrrrrrrrrrr':6,
            'rrrrrrrrrrrrrGGGGG':9
            }
                
        if current_phase_str in str2num.keys():
            current_phase_num=str2num[current_phase_str]
        else:
            current_phase_num=10000
        return current_phase_num

    def add_vars(self,m,veh_dict,direction,g_rem,r_rem,C,phase,I_phase_G,lane_to_phase_dict):
        if self.phase_type == "3ph":
            M=10000
            t_headway=1.8
            v_lim=16.7
            current_green_lost=5
            left_turn_headway=1.8
            
            numLane=len(veh_dict)
            vehNumLane=[]
            for lane_id in veh_dict.keys():
                vehNumLane.append(len(veh_dict[lane_id]))
            maxVehNum=max(vehNumLane)
            #print("total lane num: "+str(numLane))
            #print("max veh num: "+str(maxVehNum))
            #print(vehNumLane)
            
            t_arr=0
            t_arr=m.addVars(numLane,maxVehNum,vtype=GRB.CONTINUOUS,name="t_arr")
            v_front=0
            v_front=m.addVars(numLane,maxVehNum,vtype=GRB.CONTINUOUS,name="v_front")
            u_jk_g=0
            u_jk_g=m.addVars(numLane,maxVehNum,vtype=GRB.BINARY,name="u_jk_g")
            u_jk_r=0
            u_jk_r=m.addVars(numLane,maxVehNum,vtype=GRB.BINARY,name="u_jk_r")
            v_delay=0
            v_delay=m.addVars(numLane,maxVehNum,vtype=GRB.CONTINUOUS,name="v_delay")
            v_delay_sum=0
            v_delay_sum=m.addVar(vtype=GRB.CONTINUOUS,name="v_delay_sum")
            
            for i in range(len(list(veh_dict.keys()))):
                lane_id=list(veh_dict.keys())[i]
                #print("idx:"+str(i))
                #print('Lane_id: '+str(lane_id))
                #print("veh num in lane"+str(len(veh_dict[lane_id])))
                for j in range(len(veh_dict[lane_id])):  #add constraints for single vehicle
                    #print("vehicle idx "+str(j))
                    veh=veh_dict[lane_id][j]
                    veh_target_lane_phase=lane_to_phase_dict[direction][int(lane_id[-1])]
                    
                    ego_dist2bar=veh['dist2bar']
                    v_front_before_ego=0
                    for num in range(len(veh_dict[lane_id])):
                        if veh_dict[lane_id][num]['dist2bar']<ego_dist2bar:
                            v_front_before_ego=v_front_before_ego+1
                    #print(v_front_before_ego)
                    #calculate_total_front_vehicle_num
                    if v_front_before_ego!=0:
                        if direction==1 or direction==3:
                            if int(lane_id[-1])==0:
                                m.addConstr(v_front[i,j]==sum((1-u_jk_g[i,num_front]) for num_front in range(v_front_before_ego))*left_turn_headway*I_phase_G[veh_target_lane_phase-1]+\
                                                sum(u_jk_r[i,num_front] for num_front in range(v_front_before_ego))*left_turn_headway*(1-I_phase_G[veh_target_lane_phase-1]), name="left turn front vehicle dissipate time")
                        else:
                            m.addConstr(v_front[i,j]==sum((1-u_jk_g[i,num_front]) for num_front in range(v_front_before_ego))*t_headway*I_phase_G[veh_target_lane_phase-1]+\
                                            sum(u_jk_r[i,num_front] for num_front in range(v_front_before_ego))*t_headway*(1-I_phase_G[veh_target_lane_phase-1]), name="front vehicle dissipate time")
                    if j==0:
                        m.addConstr(v_front[i,j]==0)
                    if I_phase_G[veh_target_lane_phase-1]==1:
                        m.addConstr(M*(1-u_jk_g[i,j])>=I_phase_G[veh_target_lane_phase-1]*(t_arr[i,j]-g_rem[veh_target_lane_phase-1]),name="g_rem_constr1")
                        m.addConstr(I_phase_G[veh_target_lane_phase-1]*(t_arr[i,j]-g_rem[veh_target_lane_phase-1])>=-M*u_jk_g[i,j],name="g_rem_constr2")
                        m.addConstr(M*(u_jk_g[i,j])>=I_phase_G[veh_target_lane_phase-1]*(g_rem[veh_target_lane_phase-1]+current_green_lost+C-phase[veh_target_lane_phase-1]+v_front[i,j]-t_arr[i,j]),name="g_rem_constr3")
                        m.addConstr(I_phase_G[veh_target_lane_phase-1]*(g_rem[veh_target_lane_phase-1]+current_green_lost+C-phase[veh_target_lane_phase-1]+v_front[i,j]-t_arr[i,j])>=-M*(1-u_jk_g[i,j]),name="g_rem_constr4")
                    if I_phase_G[veh_target_lane_phase-1]==0:
                        m.addConstr(u_jk_r[i,j]==1)
                        m.addConstr(M*(1-u_jk_r[i,j])>=(1-I_phase_G[veh_target_lane_phase-1])*(t_arr[i,j]-r_rem[veh_target_lane_phase-1]-C),name="r_rem_constr1")
                        m.addConstr((1-I_phase_G[veh_target_lane_phase-1])*(t_arr[i,j]-r_rem[veh_target_lane_phase-1]-C)>=-M*u_jk_r[i,j],name="r_rem_constr2")
                        m.addConstr(M*u_jk_r[i,j]>=(1-I_phase_G[veh_target_lane_phase-1])*(r_rem[veh_target_lane_phase-1]+phase[veh_target_lane_phase-1]-current_green_lost-t_arr[i,j]),name="r_rem_constr3")
                        m.addConstr((1-I_phase_G[veh_target_lane_phase-1])*(r_rem[veh_target_lane_phase-1]+phase[veh_target_lane_phase-1]-current_green_lost-t_arr[i,j])>=-M*(1-u_jk_r[i,j]),name="r_rem_constr4")
                        m.addConstr(t_arr[i,j]>=(1-I_phase_G[veh_target_lane_phase-1])*(r_rem[veh_target_lane_phase-1]+v_front[i,j]),name="r_rem_constr5")   ##############!!!!!should multiply by 1.8
                
                    m.addConstr(t_arr[i,j]>=ego_dist2bar/v_lim,name="min_t_arr")   #need to be change to real distance
                    # m.addConstr(v_delay[i,j]==t_arr[i,j]-ego_dist2bar/v_lim, name="v_delay_calculate")
                    if direction!=1 and direction!=3: 
                        m.addConstr(v_delay[i,j]==t_arr[i,j]-ego_dist2bar/v_lim, name="v_delay_calculate")
                    else:
                        if direction==1 or direction==3:
                            if int(lane_id[-1])!=0:
                                m.addConstr(v_delay[i,j]==t_arr[i,j]-ego_dist2bar/v_lim, name="v_delay_calculate")
                            else:
                                m.addConstr(v_delay[i,j]==0, name="v_delay_calculate")
                    
            
            for i in range(len(list(veh_dict.keys()))):
                lane_id=list(veh_dict.keys())[i]
                #print("veh num in lane: ")
                #print(len(veh_dict[lane_id]))
        
                if len(veh_dict[lane_id])>=2:
                    for j in range(len(veh_dict[lane_id])-1):
                        m.addConstr(t_arr[i,j+1]>=t_arr[i,j]+1.8)
                        
                if direction==1 or direction==3:
                    if int(lane_id[-1])==0:
                        if len(veh_dict[lane_id])>=2:
                            for j in range(len(veh_dict[lane_id])-1):
                                m.addConstr(t_arr[i,j+1]>=t_arr[i,j]+left_turn_headway)
                    
            #for i in range(len(list(veh_dict.keys()))):
            #    lane_id=list(veh_dict.keys())[i]
            #    for j in range(len(veh_dict[lane_id])):
            #        m.addConstr(v_delay_sum==v_delay_sum+v_delay[i,j],name="v_delay")
            m.addConstr(v_delay_sum==sum(v_delay[i,j] for i in range(len(list(veh_dict.keys()))) for j in range(len(veh_dict[list(veh_dict.keys())[i]]))), name="delay sum")
            return t_arr,v_front,u_jk_g,u_jk_r,v_delay,v_delay_sum
        else:
            M=10000
            t_headway=1 #.8
            v_lim=17.88
            current_green_lost=0
            
            numLane=len(veh_dict)
            vehNumLane=[]
            for lane_id in veh_dict.keys():
                vehNumLane.append(len(veh_dict[lane_id]))
            maxVehNum=max(vehNumLane)
            #print("total lane num: "+str(numLane))
            #print("max veh num: "+str(maxVehNum))
            #print(vehNumLane)
            
            t_arr=0
            t_arr=m.addVars(numLane,maxVehNum,vtype=GRB.CONTINUOUS,lb=0,name="t_arr")
            v_front=0
            v_front=m.addVars(numLane,maxVehNum,vtype=GRB.CONTINUOUS,lb=0,name="v_front")
            u_jk_g=0
            u_jk_g=m.addVars(numLane,maxVehNum,vtype=GRB.BINARY,lb=0,name="u_jk_g")
            u_jk_r=0
            u_jk_r=m.addVars(numLane,maxVehNum,vtype=GRB.BINARY,lb=0,name="u_jk_r")
            v_delay=0
            v_delay=m.addVars(numLane,maxVehNum,vtype=GRB.CONTINUOUS,lb=0,name="v_delay")
            v_delay_sum=0
            v_delay_sum=m.addVar(vtype=GRB.CONTINUOUS,lb=0,name="v_delay_sum")
            
            for i in range(len(list(veh_dict.keys()))):
                lane_id=list(veh_dict.keys())[i]
                #print("idx:"+str(i))
                #print('Lane_id: '+str(lane_id))
                #print("veh num in lane"+str(len(veh_dict[lane_id])))
                for j in range(len(veh_dict[lane_id])):  #add constraints for single vehicle
                    #print("vehicle idx "+str(j))
                    veh=veh_dict[lane_id][j]
                    veh_target_lane_phase=self.lane_to_phase_dict[direction][int(lane_id[-1])]
                    
                    ego_dist2bar=veh['dist2bar']
                    v_front_before_ego=0
                    for num in range(len(veh_dict[lane_id])):
                        if veh_dict[lane_id][num]['dist2bar']<ego_dist2bar:
                            v_front_before_ego=v_front_before_ego+1
                    #print(v_front_before_ego)
                    #calculate_total_front_vehicle_num
                    if v_front_before_ego!=0:
                        m.addConstr(v_front[i,j]==sum((1-u_jk_g[i,num_front]) for num_front in range(v_front_before_ego))*t_headway*I_phase_G[veh_target_lane_phase-1]+\
                                            sum(u_jk_r[i,num_front] for num_front in range(v_front_before_ego))*t_headway*(1-I_phase_G[veh_target_lane_phase-1]), name="front vehicle dissipate time")
                    if j==0:
                        m.addConstr(v_front[i,j]==0)
                    if I_phase_G[veh_target_lane_phase-1]==1:
                        m.addConstr(M*(1-u_jk_g[i,j])>=I_phase_G[veh_target_lane_phase-1]*(t_arr[i,j]-g_rem[veh_target_lane_phase-1]),name="g_rem_constr1")
                        m.addConstr(I_phase_G[veh_target_lane_phase-1]*(t_arr[i,j]-g_rem[veh_target_lane_phase-1])>=-M*u_jk_g[i,j],name="g_rem_constr2")
                        m.addConstr(M*(u_jk_g[i,j])>=I_phase_G[veh_target_lane_phase-1]*(g_rem[veh_target_lane_phase-1]+current_green_lost+C-phase[veh_target_lane_phase-1]+v_front[i,j]-t_arr[i,j]),name="g_rem_constr3")
                        m.addConstr(I_phase_G[veh_target_lane_phase-1]*(g_rem[veh_target_lane_phase-1]+C-phase[veh_target_lane_phase-1]+v_front[i,j]-t_arr[i,j])>=-M*(1-u_jk_g[i,j]),name="g_rem_constr4")
                    if I_phase_G[veh_target_lane_phase-1]==0:
                        m.addConstr(u_jk_r[i,j]==1)
                        m.addConstr(M*(1-u_jk_r[i,j])>=(1-I_phase_G[veh_target_lane_phase-1])*(t_arr[i,j]-r_rem[veh_target_lane_phase-1]-C),name="r_rem_constr1")
                        m.addConstr((1-I_phase_G[veh_target_lane_phase-1])*(t_arr[i,j]-r_rem[veh_target_lane_phase-1]-C)>=-M*u_jk_r[i,j],name="r_rem_constr2")
                        m.addConstr(M*u_jk_r[i,j]>=(1-I_phase_G[veh_target_lane_phase-1])*(r_rem[veh_target_lane_phase-1]+phase[veh_target_lane_phase-1]-t_arr[i,j]),name="r_rem_constr3")
                        m.addConstr((1-I_phase_G[veh_target_lane_phase-1])*(r_rem[veh_target_lane_phase-1]+phase[veh_target_lane_phase-1]-t_arr[i,j])>=-M*(1-u_jk_r[i,j]),name="r_rem_constr4")
                        m.addConstr(t_arr[i,j]>=(1-I_phase_G[veh_target_lane_phase-1])*(r_rem[veh_target_lane_phase-1]+v_front[i,j]),name="r_rem_constr5")   ##############!!!!!should multiply by 1.8
                
                    m.addConstr(t_arr[i,j]>=ego_dist2bar/v_lim,name="min_t_arr")   #need to be change to real distance
                    m.addConstr(v_delay[i,j]==t_arr[i,j]-ego_dist2bar/v_lim, name="v_delay_calculate")
            
            for i in range(len(list(veh_dict.keys()))):
                lane_id=list(veh_dict.keys())[i]
                #print("veh num in lane: ")
                #print(len(veh_dict[lane_id]))
                if len(veh_dict[lane_id])>=2:
                    for j in range(len(veh_dict[lane_id])-1):
                        m.addConstr(t_arr[i,j+1]>=t_arr[i,j]+1.8)
                    
            #for i in range(len(list(veh_dict.keys()))):
            #    lane_id=list(veh_dict.keys())[i]
            #    for j in range(len(veh_dict[lane_id])):
            #        m.addConstr(v_delay_sum==v_delay_sum+v_delay[i,j],name="v_delay")
            m.addConstr(v_delay_sum==sum(v_delay[i,j] for i in range(len(list(veh_dict.keys()))) for j in range(len(veh_dict[list(veh_dict.keys())[i]]))), name="delay sum")

            
            
            return t_arr,v_front,u_jk_g,u_jk_r,v_delay,v_delay_sum

    
    def co_optimization(self,VehInfo,lane_to_phase_dict,current_phase_num):
    #(elapsed_time,inter_dict_current_time,route_dict,lane_to_phase_dict,time_step,current_green):
        # elapsed_time: elapsed time of current phase
        # inter_dict_lane_based: {arm_id:{lane_id:{[{veh1},{veh2},{veh3}.....]}}} vehicles listed according to distance to stop bar, from downstream to upstream
        # lane_to_phase_dict: relationship between lane id and green phase idx
        # veh1:{'distance to stop bar': float, \
        #       'traveled distance': float,\
        #       'veh id': str}
        
        if self.phase_type == "3ph":
            PhaseLength=3
            m=gp.Model("opt")
                        
            elapsed_time=VehInfo['phase_duration']/10
            #current_green_str=VehInfo[time_step]['phase']
            #current_green=sumo_phase_to_num(current_phase_str)
            current_green=current_phase_num
            
            inter_dict_current_time=VehInfo['inter_dict_lane_based']
            
            I_phase_G = self.get_current_green_phase(current_green,elapsed_time)
            #print(elapsed_time)
            current_green_lost=5 #changed from 6 to 5
            green_min=6+5 #changed from 6 to 5
            green_max=80+5 #changed from 6 to 5
            #print(I_phase_G)
            
            current_green_dual_ring=[]
            for i in range(PhaseLength):
                if I_phase_G[i]==1:
                    current_green_dual_ring.append(i)
                    
            #print("current_green_dual_ring: "+str(current_green_dual_ring))
                
            
            phase=m.addVars(PhaseLength,lb=green_min,ub=green_max,vtype=GRB.CONTINUOUS,name="sig")
            g_rem=m.addVars(PhaseLength,vtype=GRB.CONTINUOUS,name="g_rem")
            r_rem=m.addVars(PhaseLength,vtype=GRB.CONTINUOUS,name="r_rem")
            C=m.addVar(vtype=GRB.CONTINUOUS,name="C")
            #m.addConstr(phase[0]+phase[1]+phase[2]+phase[3]==phase[4]+phase[5]+phase[6]+phase[7])
            #m.addConstr(phase[0]+phase[1]==phase[4]+phase[5])
            #m.addConstr(phase[2]+phase[3]==phase[6]+phase[7])
            m.addConstr(C==phase[0]+phase[1]+phase[2])
            #m.addConstr(phase[0]==phase[4])
            #m.addConstr(phase[1]==phase[5])
            #m.addConstr(phase[2]==phase[6])
            #m.addConstr(phase[3]==phase[7])
            current_green_remain=m.addVar(lb=0,vtype=GRB.CONTINUOUS,name="current_green_remain")
            #for i in range(8):
            #    if I_phase_G[i]==1:
            #        m.addConstr(current_green_remain==phase[i]-elapsed_time)
            #        break
                
            for i in range(PhaseLength):
                #print("iter: "+str(i))
                if I_phase_G[i]==1:
                    m.addConstr(g_rem[i]==phase[i]-(elapsed_time+current_green_lost))
                    m.addConstr(r_rem[i]==0)
                else:
                    current_green_temp=current_green_dual_ring[0]
                    green_list=[0,1,2,0,1,2]
                    if i<current_green_temp:
                        green_before=green_list[current_green_temp+1:i+3]
                    else:
                        green_before=green_list[current_green_temp+1:i]
                    m.addConstr(current_green_remain==phase[current_green_dual_ring[0]]-elapsed_time)
                    m.addConstr(r_rem[i]==current_green_remain+sum(phase[num] for num in green_before))
                    m.addConstr(g_rem[i]==0)
            
            for i in range(PhaseLength):
                if I_phase_G[i]==1:
                    m.addConstr(phase[i]>=elapsed_time+5) #changed from 6 to 5
                    
            #TotalVehNum=len(veh_dict[0])+len(veh_dict[1])+len(veh_dict[2])
            west_t_arr,west_v_front,west_u_jk_g,west_u_jk_r,west_v_delay,west_v_delay_sum=\
            self.add_vars(m,inter_dict_current_time['1'],1,g_rem,r_rem,C,phase,I_phase_G,self.lane_to_phase_dict)
            east_t_arr,east_v_front,east_u_jk_g,east_u_jk_r,east_v_delay,east_v_delay_sum=\
            self.add_vars(m,inter_dict_current_time['3'],3,g_rem,r_rem,C,phase,I_phase_G,self.lane_to_phase_dict)
            north_t_arr,north_v_front,north_u_jk_g,north_u_jk_r,north_v_delay,north_v_delay_sum=\
            self.add_vars(m,inter_dict_current_time['7'],7,g_rem,r_rem,C,phase,I_phase_G,self.lane_to_phase_dict)
            south_t_arr,south_v_front,south_u_jk_g,south_u_jk_r,south_v_delay,south_v_delay_sum=\
            self.add_vars(m,inter_dict_current_time['5'],5,g_rem,r_rem,C,phase,I_phase_G,self.lane_to_phase_dict)
            m.update()
            #m.display()
            m.setObjective(west_v_delay_sum+east_v_delay_sum+north_v_delay_sum+south_v_delay_sum, GRB.MINIMIZE)
            
            m.setParam(GRB.Param.Threads, 8)
            m.params.NonConvex=2
            m.Params.IntFeasTol = 1e-7
            softlimit = 5
            hardlimit = 30

            def softtime(model, where):
                if where == GRB.Callback.MIP:
                    runtime = model.cbGet(GRB.Callback.RUNTIME)
                    objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
                    objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                    gap = abs((objbst - objbnd) / objbst)

                    if runtime > softlimit:
                        if gap<0.01:
                            model.terminate()
                            
        
            m.setParam('TimeLimit', hardlimit)
            m.optimize(softtime)
            
            #m.computeIIS()
            #m.write("model.ilp")
            if m.getAttr('SolCount')!=0:
                phase_result=m.getAttr('X', phase)
                
                west_time_arrival=m.getAttr('X',west_t_arr)
                west_v_front=m.getAttr('X',west_v_front)
                west_u_jk_g=m.getAttr('X',west_u_jk_g)
                west_u_jk_r=m.getAttr('X',west_u_jk_r)
                west_v_delay=m.getAttr('X',west_v_delay)
                #west_v_delay_sum=m.getAttr('X',west_v_delay_sum)
                
                west_result=[west_time_arrival.values(),west_v_front.values(),west_u_jk_g.values(),west_u_jk_r.values(),west_v_delay.values(),west_v_delay_sum.x]
                
                east_time_arrival=m.getAttr('X',east_t_arr)
                east_v_front=m.getAttr('X',east_v_front)
                east_u_jk_g=m.getAttr('X',east_u_jk_g)
                east_u_jk_r=m.getAttr('X',east_u_jk_r)
                east_v_delay=m.getAttr('X',east_v_delay)
                #east_v_delay_sum=m.getAttr('X',east_v_delay_sum)
                east_result=[east_time_arrival.values(),east_v_front.values(),east_u_jk_g.values(),east_u_jk_r.values(),east_v_delay.values(),east_v_delay_sum.x]
                
                north_time_arrival=m.getAttr('X',north_t_arr)
                north_v_front=m.getAttr('X',north_v_front)
                north_u_jk_g=m.getAttr('X',north_u_jk_g)
                north_u_jk_r=m.getAttr('X',north_u_jk_r)
                north_v_delay=m.getAttr('X',north_v_delay)
                #north_v_delay_sum=m.getAttr('X',north_v_delay_sum)
                north_result=[north_time_arrival.values(),north_v_front.values(),north_u_jk_g.values(),north_u_jk_r.values(),north_v_delay.values(),north_v_delay_sum.x]
                
                south_time_arrival=m.getAttr('X',south_t_arr)
                south_v_front=m.getAttr('X',south_v_front)
                south_u_jk_g=m.getAttr('X',south_u_jk_g)
                south_u_jk_r=m.getAttr('X',south_u_jk_r)
                south_v_delay=m.getAttr('X',south_v_delay)
                #south_v_delay_sum=m.getAttr('X',south_v_delay_sum)
                south_result=[south_time_arrival.values(),south_v_front.values(),south_u_jk_g.values(),south_u_jk_r.values(),south_v_delay.values(),south_v_delay_sum.x]
                
                
                g_rem=m.getAttr('X',g_rem)
                r_rem=m.getAttr('X',r_rem)
                g_rem_value=g_rem.values()
                r_rem_value=r_rem.values()
                phase_result_value=phase_result.values()
                non_empty_solution=1
                
            else:
                m.computeIIS()
                m.write("model.ilp")
                phase_result_value=[]
                west_time_arrival=[]
                east_time_arrival=[]
                north_time_arrival=[]
                south_time_arrival=[]
                g_rem_value=[]
                r_rem_value=[]
                print(str(time_step)+" no solution")
                non_empty_solution=0
                
                
            return phase_result_value,west_result,east_result,north_result,south_result,g_rem_value,r_rem_value,non_empty_solution,I_phase_G
        else:
            m=gp.Model("opt")
            '''
            for direction in inter_dict_current_time.keys():
                for lane in inter_dict_current_time[direction].keys():
                    for veh in inter_dict_current_time[direction][lane]:
                        current_green=veh['current_phase']
                        #elapsed_time=round(veh['elapsed_time'],1)
                        break
                    break
                break
            '''
            
                            
                        
            elapsed_time=VehInfo['phase_duration']/10
            #current_green_str=VehInfo[time_step]['phase']
            #current_green=sumo_phase_to_num(current_phase_str)
            current_green=current_phase_num
            
            inter_dict_current_time=VehInfo['inter_dict_lane_based']
            
            I_phase_G=self.get_current_green_phase(current_green,elapsed_time)
            #print(elapsed_time)
            current_green_lost=5 #changed from 6 to 5
            green_min=[7,15,15,15,7,15,15,15] #changed from 6 to 5
            green_max=40+5 #changed from 6 to 5
            #print(I_phase_G)
            
            current_green_dual_ring=[]
            for i in range(8):
                if I_phase_G[i]==1:
                    current_green_dual_ring.append(i)
                    
            #print("current_green_dual_ring: "+str(current_green_dual_ring))
            phase=m.addVars(8,lb=5,ub=green_max,vtype=GRB.CONTINUOUS,name="sig")
            g_rem=m.addVars(8,vtype=GRB.CONTINUOUS,name="g_rem")
            r_rem=m.addVars(8,vtype=GRB.CONTINUOUS,name="r_rem")
            C=m.addVar(vtype=GRB.CONTINUOUS,name="C")
            m.addConstr(phase[0]+phase[1]+phase[2]+phase[3]==phase[4]+phase[5]+phase[6]+phase[7])
            m.addConstr(phase[0]+phase[1]==phase[4]+phase[5])
            m.addConstr(phase[2]+phase[3]==phase[6]+phase[7])
            m.addConstr(C==phase[0]+phase[1]+phase[2]+phase[3])
            # m.addConstr(phase[0]==phase[4])
            # m.addConstr(phase[1]==phase[5])
            m.addConstr(phase[2]==phase[6])
            m.addConstr(phase[3]==phase[7])
            current_green_remain=m.addVar(lb=0,vtype=GRB.CONTINUOUS,name="current_green_remain")
            #for i in range(8):
            #    if I_phase_G[i]==1:
            #        m.addConstr(current_green_remain==phase[i]-elapsed_time)
            #        break
            for i in range(8):
                m.addConstr(phase[i]>=green_min[i],name="minimum green constraints")

            for i in range(8):
                #print("iter: "+str(i))
                if I_phase_G[i]==1:
                    m.addConstr(g_rem[i]==phase[i]-(elapsed_time+current_green_lost))
                    m.addConstr(r_rem[i]==0)
                else:
                        
                    if i<4:
                        current_green_temp=current_green_dual_ring[0]
                        green_list=[0,1,2,3,0,1,2,3]
                        if i<current_green_temp:
                            green_before=green_list[current_green_temp+1:i+4]
                        else:
                            green_before=green_list[current_green_temp+1:i]
                        #print('green_before: '+str(green_before))
                    else:
                        current_green_temp=current_green_dual_ring[1]
                        green_list=[4,5,6,7,4,5,6,7]
                        if i<current_green_temp:
                            green_before=green_list[current_green_temp-4+1:i]
                        else:
                            green_before=green_list[current_green_temp-4+1:i-4]
                        #print('green_before: '+str(green_before))
                    
                    if i<4:
                        m.addConstr(current_green_remain==phase[current_green_dual_ring[0]]-elapsed_time)
                    else:
                        m.addConstr(current_green_remain==phase[current_green_dual_ring[1]]-elapsed_time)
                    m.addConstr(r_rem[i]==current_green_remain+sum(phase[num] for num in green_before))
                    m.addConstr(g_rem[i]==0)
            
            for i in range(8):
                if I_phase_G[i]==1:
                    m.addConstr(phase[i]>=elapsed_time+5) #changed from 6 to 5
                    
            #TotalVehNum=len(veh_dict[0])+len(veh_dict[1])+len(veh_dict[2])
                    
                    
        
            west_t_arr,west_v_front,west_u_jk_g,west_u_jk_r,west_v_delay,west_v_delay_sum=\
            self.add_vars(m,inter_dict_current_time['1'],1,g_rem,r_rem,C,phase,I_phase_G,self.lane_to_phase_dict)
            east_t_arr,east_v_front,east_u_jk_g,east_u_jk_r,east_v_delay,east_v_delay_sum=\
            self.add_vars(m,inter_dict_current_time['3'],3,g_rem,r_rem,C,phase,I_phase_G,self.lane_to_phase_dict)
            north_t_arr,north_v_front,north_u_jk_g,north_u_jk_r,north_v_delay,north_v_delay_sum=\
            self.add_vars(m,inter_dict_current_time['7'],7,g_rem,r_rem,C,phase,I_phase_G,self.lane_to_phase_dict)
            south_t_arr,south_v_front,south_u_jk_g,south_u_jk_r,south_v_delay,south_v_delay_sum=\
            self.add_vars(m,inter_dict_current_time['5'],5,g_rem,r_rem,C,phase,I_phase_G,self.lane_to_phase_dict)
            m.update()
            #m.display()
            m.setObjective(west_v_delay_sum+east_v_delay_sum+north_v_delay_sum+south_v_delay_sum+C, GRB.MINIMIZE)
            
            m.setParam(GRB.Param.Threads, 8)
            m.params.NonConvex=2
            m.setParam("Method", 2)
            m.Params.IntFeasTol = 1e-7
            softlimit = 10
            hardlimit = 100
        
            m.setParam('MIPFocus', 1)
            m.setParam('FeasibilityTol', 1e-6)
            
        

            def softtime(model, where):
                if where == GRB.Callback.MIP:
                    runtime = model.cbGet(GRB.Callback.RUNTIME)
                    objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
                    objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                    gap = abs((objbst - objbnd) / objbst)#

                    if runtime > softlimit:
                        if gap<0.001:
                            model.terminate()
                            
        
            m.setParam('TimeLimit', hardlimit)
            m.Params.MIPGap = 0.001
            m.optimize()

            # Get the solution status
            status = m.Status

            # Print the solution status
            print("Solution status:", status)
            
            #m.computeIIS()
            #m.write("model.ilp")
            if m.getAttr('SolCount')!=0:
                
            
                phase_result=m.getAttr('X', phase)
                
                west_time_arrival=m.getAttr('X',west_t_arr)
                west_v_front=m.getAttr('X',west_v_front)
                west_u_jk_g=m.getAttr('X',west_u_jk_g)
                west_u_jk_r=m.getAttr('X',west_u_jk_r)
                west_v_delay=m.getAttr('X',west_v_delay)
                #west_v_delay_sum=m.getAttr('X',west_v_delay_sum)
                
                west_result=[west_time_arrival.values(),west_v_front.values(),west_u_jk_g.values(),west_u_jk_r.values(),west_v_delay.values(),west_v_delay_sum.x]
                
                east_time_arrival=m.getAttr('X',east_t_arr)
                east_v_front=m.getAttr('X',east_v_front)
                east_u_jk_g=m.getAttr('X',east_u_jk_g)
                east_u_jk_r=m.getAttr('X',east_u_jk_r)
                east_v_delay=m.getAttr('X',east_v_delay)
                #east_v_delay_sum=m.getAttr('X',east_v_delay_sum)
                east_result=[east_time_arrival.values(),east_v_front.values(),east_u_jk_g.values(),east_u_jk_r.values(),east_v_delay.values(),east_v_delay_sum.x]
                
                north_time_arrival=m.getAttr('X',north_t_arr)
                north_v_front=m.getAttr('X',north_v_front)
                north_u_jk_g=m.getAttr('X',north_u_jk_g)
                north_u_jk_r=m.getAttr('X',north_u_jk_r)
                north_v_delay=m.getAttr('X',north_v_delay)
                #north_v_delay_sum=m.getAttr('X',north_v_delay_sum)
                north_result=[north_time_arrival.values(),north_v_front.values(),north_u_jk_g.values(),north_u_jk_r.values(),north_v_delay.values(),north_v_delay_sum.x]
                
                south_time_arrival=m.getAttr('X',south_t_arr)
                south_v_front=m.getAttr('X',south_v_front)
                south_u_jk_g=m.getAttr('X',south_u_jk_g)
                south_u_jk_r=m.getAttr('X',south_u_jk_r)
                south_v_delay=m.getAttr('X',south_v_delay)
                #south_v_delay_sum=m.getAttr('X',south_v_delay_sum)
                south_result=[south_time_arrival.values(),south_v_front.values(),south_u_jk_g.values(),south_u_jk_r.values(),south_v_delay.values(),south_v_delay_sum.x]
                
                
                g_rem=m.getAttr('X',g_rem)
                r_rem=m.getAttr('X',r_rem)
                g_rem_value=g_rem.values()
                r_rem_value=r_rem.values()
                phase_result_value=phase_result.values()
                non_empty_solution=1
                
            else:
                m.computeIIS()
                m.write("model.ilp")
                phase_result_value=[]
                west_time_arrival=[]
                east_time_arrival=[]
                north_time_arrival=[]
                south_time_arrival=[]
                g_rem_value=[]
                r_rem_value=[]
                #print(str(time_step)+" no solution")
                non_empty_solution=0
                
                
            return phase_result_value,west_result,east_result,north_result,south_result,g_rem_value,r_rem_value,non_empty_solution,I_phase_G


    '''
    I_phase_G=[1,0,0,0,1,0,0,0]
    current_green_dual_ring=[]
    for i in range(8):
        if I_phase_G[i]==1:
            current_green_dual_ring.append(i)
    print("current green dual ring: "+str(current_green_dual_ring))
                
    for i in range(8):
        print("iter: "+str(i))
        if I_phase_G[i]==1:
            print("current phase is green!")
        else:         
            if i<4:
                current_green_temp=current_green_dual_ring[0]
                green_list=[0,1,2,3,0,1,2,3]
                if i<current_green_temp:
                    green_before=green_list[current_green_temp+1:i+4]
                else:
                    green_before=green_list[current_green_temp+1:i]
                print('green_before: '+str(green_before))
            else:
                current_green_temp=current_green_dual_ring[1]
                green_list=[4,5,6,7,4,5,6,7]
                if i<current_green_temp:
                    green_before=green_list[current_green_temp-4+1:i]
                else:
                    green_before=green_list[current_green_temp-4+1:i-4]
                print('green_before: '+str(green_before))
    '''

