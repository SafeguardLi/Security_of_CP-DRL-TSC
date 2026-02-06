'''
wz:
The TrafficSignalController class define the basic functions of each tsc and serve as the parent class of them.
This parent class define the basic function of a tsc, including updating traffic information by talking to SUMO and update or continue traffic signal phases
'''

import os, time, sys, copy

import numpy as np
import pandas as pd
import math


from collections import deque

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

from src.trafficmetrics import TrafficMetrics
from collections import defaultdict
from src.helper_funcs import phi, estimate_velocity, newell_franklin
from src.CTM import CTM_model 

class TrafficSignalAttacker:
    """Abstract base class for all traffic signal controller.

    Build your own traffic signal controller by implementing the follow methods.
    """
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, detect_r):
        self.netdata = netdata
        self.cell_size = 10 # 10 m cell size
        #print('initilize###########\n\n')
        self.historic_data = defaultdict(lambda:defaultdict(list))
        self.conn = conn
        self.mode = mode
        self.id = tsc_id
        self.red_t = red_t
        self.yellow_t = yellow_t
        self.green_phases = self.get_tl_green_phases()
        #print("green phase: \n:", self.green_phases) ###
        self.phase_time = 0
        self.all_red = len((self.green_phases[0]))*'r'
        self.phase = self.all_red
        self.phase_cnt = False ### yongjie
        self.phase_lanes = self.phase_lanes(self.green_phases)
        #create subscription for this traffic signal junction to gather
        #vehicle information efficiently
        self.detect_radius = detect_r
        self.conn.junction.subscribeContext(tsc_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, self.detect_radius,
                                        [traci.constants.VAR_LANEPOSITION, 
                                        traci.constants.VAR_SPEED, 
                                        traci.constants.VAR_LANE_ID,
                                        traci.constants.VAR_ROAD_ID,
                                        traci.constants.VAR_TIMELOSS,
                                        traci.constants.VAR_POSITION])
        #get all incoming lanes to intersection
        self.incoming_lanes = set()
        self.outgoing_lanes = set() # wz: add outgoing lanes
        for p in self.phase_lanes:
            for l in self.phase_lanes[p]:
                self.incoming_lanes.add(l)
                for ol in self.netdata['lane'][l]['outgoing']: # wz: count number of outgoing lanes -> nn input
                    self.outgoing_lanes.add(ol)

        self.incoming_lanes = sorted(list(self.incoming_lanes))
        self.outgoing_lanes = sorted(list(self.outgoing_lanes)) # wz


        ## 02132023 connected lane section
        # WZ 10232022: modify the incoming lane and outgoing lane data structure
        self.get_SR_lanes(self.id) #,map_name) # WZ 0215 mainly for Plymouth for now.
        self.conn_incoming_lane = {inc_lane: [inc_lane] for inc_lane in self.incoming_lanes}
        self.conn_outgoing_lane = {out_lane: [out_lane] for out_lane in self.outgoing_lanes}
        # consider when the number of inc or out lanes is greater than ONE -> we assign all connected lanes to a certain lane (w. go straight phase) per phase
            # find the edge of the lane -> find the connected edge -> list and add the lanes on that edge to the go-straight incoming lane -> find the next edge
            # how to determine which lane is for the go-stright and right-turning? -> just hard code
        # add lane length into consideration -> this will be consider when we build junc_data rather than now
        
        for inc_lane in self.SR_lanes:
            lane_list = self.get_conn_lanes(inc_lane, "incoming")
            while lane_list[0] not in ['origin','destination','intersection']:
                self.conn_incoming_lane[inc_lane].extend(lane_list)
                lane_list = self.get_conn_lanes(lane_list[0], "incoming")
            self.conn_incoming_lane[inc_lane] = list(sorted(self.conn_incoming_lane[inc_lane]))
            

        edge_id_temp = []
        for out_lane in self.conn_outgoing_lane.keys():
            edge_id = self.netdata['lane'][out_lane]['edge']
            if edge_id not in edge_id_temp:
                # to avoid duplicated assignment of outgoing lanes
                edge_id_temp.append(edge_id)

                lane_list = self.get_conn_lanes(out_lane, "outgoing")
                while lane_list[0] not in ['origin','destination','intersection']:
                    self.conn_outgoing_lane[out_lane].extend(lane_list)
                    lane_list = self.get_conn_lanes(lane_list[0], "outgoing")

                self.conn_outgoing_lane[out_lane] = list(sorted(self.conn_outgoing_lane[out_lane]))
        
        #print("self.conn_incoming_lane", self.conn_incoming_lane,
        #        "self.conn_outgoing_lane", self.conn_outgoing_lane) #####
        self.inc_lanes_list = []
        self.out_lanes_list = []

        for key in self.conn_incoming_lane.keys():
            self.inc_lanes_list.extend(self.conn_incoming_lane[key])

        for key in self.conn_outgoing_lane.keys():
            self.out_lanes_list.extend(self.conn_outgoing_lane[key])

        # print("\n self.inc_lanes_list",self.inc_lanes_list) ###
        # print("\n self.out_lanes_list",self.out_lanes_list) ###

        #lane capacity is the lane length divided by the average vehicle length+stopped headway
        self.lane_capacity = np.array(
            [float(self.netdata['lane'][lane]['length'])/7.5 for lane in self.incoming_lanes])
        self.lane_capacity_out = np.array(
            [float(self.netdata['lane'][lane]['length']) / 7.5 for lane in self.outgoing_lanes]) # wz: outgoing pressure
        #for collecting various traffic metrics at the intersection
        #can be extended in trafficmetric.py class to collect new metrics
        if mode == 'train':
            self.metric_args = [] #['delay']
        if mode == 'test':
            # self.metric_args = ['queue', 'delay']
            # self.metric_args = ['queue', 'delay', 'vehicle', 'intersec'] # wz
            self.metric_args = ['delay', 'main_delay','side_delay']
        self.trafficmetrics = TrafficMetrics(tsc_id, self.incoming_lanes, netdata, self.metric_args, mode)

        self.ep_rewards = []
        self.phase_record = []
        self.state_record = [] #WZ, July 19, to viz the possible congestion
        self.action_dist_record = None

        # wz: to store historical velocity_dict and rho_dict of incoming lanes
        # self.historic_velocity = defaultdict()
        # self.historic_rho = defaultdict()
        # self.historic_EQ = defaultdict()
        # self.historic_TQ = defaultdict()

        # wz: congestion indicator
        self.cong_ind = 0
        # self.cong_thresh = cong_thresh
        self.phase_duration = 0

        self.stored = False

        self.max_pressure_lanes = self.max_pressure_lanes() # borrow from MP

        self.tot_inc_lane_len = {key: {} for key in self.max_pressure_lanes.keys()} #dict.fromkeys(self.max_pressure_lanes.keys(),{}) # for density calculation in reward, Jan 17 2023
        self.tot_out_lane_len = {key: {} for key in self.max_pressure_lanes.keys()} #dict.fromkeys(self.max_pressure_lanes.keys(),{})
        self.delay_record = [] # initialization

        # initialize CTM. 032023 WZ: since we now run CTM only for one intersection; 
        # if we extend our work to multiple intersection and they share the same CTM, we should move 
        # to the sumosim.py to initialize CTM
        self.junc_position = np.array(self.conn.junction.getPosition(self.id))

        CTM_fp="./networks/plymouth/CTM_plymouth_1788_4ph.csv"
        # CTM_fp="./networks/plymouth/CTM_Plymouth_122cells_4ph.csv"
        # CTM_fp="./networks/plymouth/CTM_plymouth_2407_1090.csv"

        if CTM_fp =="./networks/plymouth/CTM_plymouth_1788_4ph.csv":
            self.lane2souceNode = {"-4.0.00_0":"1","-4.0.00_1":"1","-4.0.00_2":"1",
                                 "15.0.00_0":"29","15.0.00_1":"29","15.0.00_2":"29",
                                 "-0.0.00_0":"59",
                                 "12.0.00_0":"89","12.0.00_1":"89"} #HARD CODING FOR CTM Plymouth 0328 -> CTM_plymouth_2407_1788
        elif CTM_fp =="./networks/plymouth/CTM_plymouth_2407_1090.csv":
            self.lane2souceNode = {"-4.0.00_0":"1","-4.0.00_1":"1","-4.0.00_2":"1",
                                 "15.0.00_0":"43","15.0.00_1":"43","15.0.00_2":"43",
                                 "-0.0.00_0":"87",
                                 "12.0.00_0":"133","12.0.00_1":"133"}
        else:
            print("WARNING: no source node info provided for Loop detectors!")
        
        CTM_sim_len = 2400 # longer than actual simulation length to avoid out of index
        self.CTM = CTM_model(CTM_fp,CTM_sim_len) # use default input
        # run for the warmup time
        self.CTM.fixed_time_CTM(100) # warmup time
        self.net_status = None
        self.est_spd = None

        self.init_loop(10) # lp will record 10 steps data

        self.transit_t = 0
    
    def init_loop(self, hist_len=10):
        # get all loop detector ID list
        lp_id_list = self.conn.inductionloop.getIDList()
        self.lp_hist = {}
        # lp_hist will be lp_inc inside CTM function.
        # lp_hist = {source_node_id:{lp_id:nvehContri of lane 1}, ...]}
        for lp_id in lp_id_list:
            # select loop detectors on the stop bars
            if self.conn.inductionloop.getLaneID(lp_id) in self.lane2souceNode.keys():
                node_id = self.lane2souceNode[self.conn.inductionloop.getLaneID(lp_id)]
                if node_id in self.lp_hist.keys():
                    self.lp_hist[node_id][lp_id] = deque(maxlen=hist_len)
                else:
                    self.lp_hist[node_id] = {lp_id: deque(maxlen=hist_len)}
    
    def update_loop_info(self):
        # update the deque with new loop detector info
        for node_id in self.lp_hist.keys():
            for lp_id in self.lp_hist[node_id].keys():
                self.lp_hist[node_id][lp_id].append(self.conn.inductionloop.getLastStepVehicleNumber(lp_id))

    def get_conn_lanes(self, lane_id, direction):
        # return a list of lanes that connect to the input lane_id
        # direction: the direction we do search, e.g. for "incoming", we will search the incoming lanes to the given lane
        # notify if the upstream or downstream of the given lane is an intersection node or origin/destination
        
        edge_id = self.netdata['lane'][lane_id]['edge']

        if direction == "incoming":
            if edge_id in self.netdata['origin']:
                print(lane_id," is on the origin edge and no incoming lanes")
                return ['origin']
            else:
                inc_node = self.netdata['edge'][edge_id]['outnode'] # WZ: note that the 'outnode' here is the node from upstream, dk why :(
                if inc_node in self.netdata['inter'].keys():
                    print(lane_id,": the upstream is a signalized intersection") # WZ TODO: how to avoid signal-free intersection? e.g. town04
                    return ['intersection']
                # TODO: elif inc_node self.netdata['node'][inc_node]['viaLaneID'] returns a dictionary {viaLaneID: getToLane.getID}
                else:
                    if lane_id in self.netdata['node'][inc_node]['viaLaneID'].values():
                        # if the toLane of the connection is the laneID, get the viaLaneID as the internal lane ID; 
                        # then add that lane ID to the end of connected edge lanes.
                        internal_ID = [i for i in self.netdata['node'][inc_node]['viaLaneID'] if self.netdata['node'][inc_node]['viaLaneID'][i][:-2]==lane_id[:-2]]
                    else:
                        internal_ID = []
                    temp_inc_lane = self.netdata['lane'][lane_id]['incoming']
                    inc_edge_id = self.netdata['lane'][temp_inc_lane[0]]['edge']
                    lane_list = self.netdata['edge'][inc_edge_id]['lanes']
                    lane_list.extend(internal_ID)
                    #print("lane_list",lane_list)
                    # lane_list = self.netdata['lane'][lane_id]['incoming']
                    return lane_list
       
        elif direction == "outgoing":
            if edge_id in self.netdata['destination']:
                print(lane_id," is on the destination edge and no outgoing lanes")
                return ['destination']
            else:
                out_node = self.netdata['edge'][edge_id]['incnode']
                if out_node in self.netdata['inter'].keys():
                    print(lane_id,": the downstream is a signalized intersection") # WZ TODO: how to avoid signal-free intersection?
                    return ['intersection']
                else:
                    if lane_id in self.netdata['node'][out_node]['viaLaneID_out'].values():
                        # if the toLane of the connection is the laneID, get the viaLaneID as the internal lane ID; 
                        # then add that lane ID to the end of connected edge lanes.
                        internal_ID = [i for i in self.netdata['node'][out_node]['viaLaneID_out'] if self.netdata['node'][out_node]['viaLaneID_out'][i][:-2]==lane_id[:-2]]
                    else:
                        internal_ID = []
                    #lane_list = list(self.netdata['lane'][lane_id]['outgoing'].keys())
                    temp_out_lane = list(self.netdata['lane'][lane_id]['outgoing'].keys())
                    out_edge_id = self.netdata['lane'][temp_out_lane[0]]['edge']
                    lane_list = self.netdata['edge'][out_edge_id]['lanes']
                    lane_list.extend(internal_ID)
                    #print("lane_list_out",lane_list)
                    return lane_list
        else:
            print("ERROR: wrong input of direction",direction)

    def max_pressure_lanes(self):
        """for each green phase, get all incoming
        and outgoing lanes for that phase, store
        in dict for max pressure calculation
        """
        max_pressure_lanes = {}
        for g in self.green_phases:
            inc_lanes = set()
            out_lanes = set()
            for l in self.phase_lanes[g]:
                inc_lanes.update(self.conn_incoming_lane[l])
                for ol in self.netdata['lane'][l]['outgoing']:
                    out_lanes.update(self.conn_outgoing_lane[ol])

            max_pressure_lanes[g] = {'inc':inc_lanes, 'out':out_lanes}
            # print("phase",g,"max_pressure_lanes[g]",max_pressure_lanes[g]) #####
        return max_pressure_lanes

    def get_SR_lanes(self,tlid, map_name="plymouth"):
        # WZ: we can also directly get the right-turning lane by reading each incoming lane with index 0.
        if map_name in ["plymoth",'plymouth']:
            if tlid == "79":
                # first version: with all SR lanes
                #self.SR_lanes = ['3.0.00_0','3.0.00_1','3.0.00_2',
                #            '-11.0.00_0','-11.0.00_1','-11.0.00_2',
                #            '-17.0.00_0','-17.0.00_1',
                #            '16.0.00_0','16.0.00_1'] # hard code go straight and right-turn lane id; lane id starting with zero at the right-most lane

                # version 2: only use one go straight lane for each direction so as to avoid duplicated assignment of connecting lanes
                self.SR_lanes = ['3.0.00_1','-11.0.00_1','-17.0.00_1','16.0.00_1']
        elif map_name == "simp_plymouth":
            if tlid == "J1":
                self.SR_lanes = ['-E6_1','-E4_1','E5_1','E3_1']
        else:
            print("ERROR please provide a map with pre-define SR lanes")

    def update_historic_data(self, data, time): # Maintain historic CV data for Queue Estimation
        
        all_lanes = self.incoming_lanes + self.outgoing_lanes
        for lane in all_lanes:
            # if len(lane.split('_'))<3: ## to avoid the lanes inside the intersection
            if lane in data.keys():
                for veh_id in data[lane]:
                    speed = data[lane][veh_id][64] # VAR_SPEED
                    position = data[lane][veh_id][86] # VAR_POSITION
                    self.historic_data[lane][time].append((float(speed), float(position)))#, float(veh_id))) # ,time)
                if len(data[lane])<1:
                    number=0
                else:  
                    number = len(self.historic_data[lane][time])
                # print(self.netdata['lane'].keys())
                if self.netdata['lane'][lane]['length']/5.0 > number:
                    n = self.netdata['lane'][lane]['length']/5.0 - number
                    self.historic_data[lane][time].extend(int(n)*[(0, math.inf)])
            else:
                n = self.netdata['lane'][lane]['length']/5.0
                self.historic_data[lane][time].extend(int(n)*[(0, math.inf)])



    # def display_queue(self, data, time):
    #     self.historic_TQ[time] = [len(data[lane]) for lane in self.incoming_lanes]


    # def run(self, con_veh_ls, unequipped_veh_ls, cav_ls, mask, time, estimate_queue, act_ctm, act_lp):
    #     self.act_ctm = act_ctm
    #     self.act_lp = act_lp

    #     data, cv_data, uv_data = self.get_subscription_data(con_veh_ls,unequipped_veh_ls,mask)
    #     self.t = time


    #     self.trafficmetrics.update(data, cv_data) # 

    #     if act_lp:
    #         self.update_loop_info()
    #         lp_info = self.lp_hist
    #     else:
    #         lp_info = None

    #     self.update(data,cv_data,uv_data,mask)

    #     # update CTM here
    #     if act_ctm and (self.t%10 == 0):
    #         # CTM is running every 1 s, will estimate the traffic status until the end of current phase 
    #         if self.args.detect_mode == "CAV_real":
    #             real_detect_acc = True
    #         else:
    #             real_detect_acc = False
    #         self.net_status, self.est_spd = self.CTM.update_CTM(self.phase,int(self.t//10),cv_data, cav_ls, con_veh_ls, self.junc_position, int(self.transit_t//10), self.phase_time, lp_info, real_detect_acc) 

    #     self.increment_controller()

    def get_metrics(self):
        for m in self.metric_args:
            metric = self.trafficmetrics.get_metric(m)

    def get_traffic_metrics_history(self):
        return {m: self.trafficmetrics.get_history(m) for m in self.metric_args}

    def increment_controller(self):
        if self.phase_time == 0:
            ###get new phase and duration
            # time_pre = time.time()
            next_phase = self.next_phase()
            # print("### phase generation time:",time.time() - time_pre,"(s)") ##
            current_phase = self.phase
            if next_phase is not None:
                self.conn.trafficlight.setRedYellowGreenState( self.id, next_phase )
            self.phase = next_phase
            self.phase_time = self.next_phase_duration(current_phase)
            if current_phase == next_phase:
                self.phase_duration += self.phase_time
            else:
                self.phase_duration = self.phase_time
            # print("phase_duration", self.phase_duration)
        self.phase_time -= 1 # phase time unit is the step length, now 0.1s each time step
        self.phase_record.append(self.phase)


    def get_intermediate_phases(self, phase, next_phase):
        if phase == next_phase:
            self.phase_cnt = True
            return []
        elif phase == self.all_red:
            self.phase_cnt = False
            return []
        else:
            self.phase_cnt = False
            yellow_phase = ''.join([ p if p == 'r' else 'y' for p in phase ])
            self.transit_t = self.t + self.red_t + self.yellow_t # the time when the green phase start, for CTM stop vehicle estimation
            # print([yellow_phase, self.all_red])
            return [yellow_phase, self.all_red]

    def next_phase(self):
        raise NotImplementedError("Subclasses should implement this!")
        
    def next_phase_duration(self,current_phase):
        raise NotImplementedError("Subclasses should implement this!")

    def update(self, data, cv_data, uv_data, mask): #
        """
            Implement this function to perform any
           traffic signal class specific control/updates 
        """
        raise NotImplementedError("Subclasses should implement this!")

    def get_subscription_data(self,con_veh_ls,unequipped_veh_ls,mask = False): # ): # original
        #use SUMO subscription to retrieve vehicle info in batches
        #around the traffic signal controller
        tl_data = self.conn.junction.getContextSubscriptionResults(self.id)                          
        #create empty incoming lanes for use else where
        lane_vehicles = {l:{} for l in self.incoming_lanes}
        lane_vehicles_cv = {l: {} for l in self.incoming_lanes} # 
        lane_vehicles_uv = {l: {} for l in self.incoming_lanes} #
        # out_lane_vehicles = {l:{} for l in self.outgoing_lanes} # wz: presslight

        if tl_data is not None:
            for v in tl_data:
                lane = tl_data[v][traci.constants.VAR_LANE_ID]
                # wz: note, here the lane could be the outgoing lane. So we will store vehicle info on outgoing lanes
                if lane not in lane_vehicles:
                    lane_vehicles[lane] = {}
                lane_vehicles[lane][v] = tl_data[v] # wz: this basically reorganizes vehicles info by lanes

            if mask:
                for cv in con_veh_ls:
                    if cv in tl_data:
                        lane = tl_data[cv][traci.constants.VAR_LANE_ID]
                        if lane not in lane_vehicles_cv:
                            lane_vehicles_cv[lane] = {}
                        lane_vehicles_cv[lane][cv] = tl_data[cv]
                for uv in unequipped_veh_ls:
                    if uv in tl_data:
                        lane = tl_data[uv][traci.constants.VAR_LANE_ID]
                        if lane not in lane_vehicles_uv:
                            lane_vehicles_uv[lane] = {}
                        lane_vehicles_uv[lane][uv] = tl_data[uv]

        return lane_vehicles, lane_vehicles_cv, lane_vehicles_uv #, out_lane_vehicles

    def get_tl_green_phases(self):
        program_id = self.conn.trafficlight.getProgram(self.id)
        for logic in self.conn.trafficlight.getAllProgramLogics(self.id):
            if logic.programID == program_id:
                break
        #get only the green phases
        green_phases = [ p.state for p in logic.getPhases() 
                         if 'y' not in p.state 
                         and ('G' in p.state or 'g' in p.state) ]

        #sort to ensure parity between sims (for RL actions)
        return sorted(green_phases)
    
    def phase_lanes(self, actions):
        phase_lanes = {a:[] for a in actions}
        for a in actions:
            green_lanes = set()
            red_lanes = set()
            for s in range(len(a)):
                if a[s] == 'g' or a[s] == 'G':
                    green_lanes.add(self.netdata['inter'][self.id]['tlsindex'][s])
                elif a[s] == 'r':
                    red_lanes.add(self.netdata['inter'][self.id]['tlsindex'][s])

            ###some movements are on the same lane, removes duplicate lanes
            pure_green = [l for l in green_lanes if l not in red_lanes]
            if len(pure_green) == 0:
                phase_lanes[a] = list(set(green_lanes))
            else:
                phase_lanes[a] = list(set(pure_green))
        return phase_lanes

    #helper functions for rl controllers
    # wz: below are all helper functions
    def input_to_one_hot(self, phases):
        identity = np.identity(len(phases))                                 
        one_hots = { phases[i]:identity[i,:]  for i in range(len(phases)) }
        return one_hots

    def int_to_input(self, phases):
        return { p:phases[p] for p in range(len(phases)) }

    def get_state(self): 
        pass

        
    def get_num_vehicle_cav(self,  num_segments=1, agent=None, lane_dir=None):
        # INPUT:
        #   num_segment: number of road segments for incoming lanes
        #   agent: for critic or for actor
        #   lane_dir: "inc" for incoming or "out" for outgoing
        if (num_segments == 1) or (lane_dir == "out"):
            if agent == "critic":
                # if not self.act_ctm:
                num_cv = []
                num_uv = []
                for phase in self.max_pressure_lanes.keys():
                    cv_count = 0
                    uv_count = 0
                    for lane in self.max_pressure_lanes[phase][lane_dir]:
                        if lane in self.cv_data.keys():
                            cv_count += len(self.cv_data[lane])
                        if lane in self.uv_data.keys():
                            uv_count += len(self.uv_data[lane])
                    num_cv.append(cv_count)
                    num_uv.append(uv_count)

                #return np.concatenate([np.array(num_cv),np.array(num_uv)])
                norm_CV = np.linalg.norm(np.array(num_cv))
                norm_UV = np.linalg.norm(np.array(num_uv))
                if norm_CV == 0:
                    norm_CV = 1
                if norm_UV == 0:
                    norm_UV = 1

                return np.concatenate([np.array(num_cv)/norm_CV,
                                    np.array(num_uv)/norm_UV])
                # else:
                #     num_cv = []
                #     for phase in self.max_pressure_lanes.keys():
                #         cv_count = 0
                #         for lane in self.max_pressure_lanes[phase][lane_dir]:
                #             if lane in self.data.keys():
                #                 cv_count += len(self.data[lane])
                #         num_cv.append(cv_count)

                #     #return np.concatenate([np.array(num_cv),np.array(num_uv)])
                #     norm_CV = np.linalg.norm(np.array(num_cv))
                #     if norm_CV == 0:
                #         norm_CV = 1

                #     return np.array(num_cv)/norm_CV

            elif agent == "actor":
                num_cv = []
                for phase in self.max_pressure_lanes.keys():
                    cv_count = 0
                    for lane in self.max_pressure_lanes[phase][lane_dir]:
                        if lane in self.cv_data.keys():
                            cv_count += len(self.cv_data[lane])
                    num_cv.append(cv_count)

                #return np.array(num_cv)
                norm_CV = np.linalg.norm(np.array(num_cv))
                if norm_CV == 0:
                    norm_CV = 1
                return np.array(num_cv)/norm_CV
            else:
                raise NotImplementedError('get_num_vehicle wrong input:', agent)
        else:
            # Segementation logic: calculate the distance from vehicle to intersection
            #                   and then assign into different segments.

            #junc_position = np.array(self.conn.junction.getPosition(self.id))

            if agent == 'critic':
                # if not self.act_ctm:
                n_phase = len(self.max_pressure_lanes)
                cv_state = np.zeros((n_phase, num_segments))
                uv_state = np.zeros((n_phase, num_segments))
                for i, phase in enumerate(self.max_pressure_lanes):
                    for lane in self.max_pressure_lanes[phase][lane_dir]:
                        if lane in self.cv_data.keys():
                            for car, info in self.cv_data[lane].items():
                                veh_position = np.array(info[traci.constants.VAR_POSITION]) ### TODO: might have bug 02142023
                                pos = max(0,np.linalg.norm(self.junc_position - veh_position)-20)
                                if pos <= self.detect_radius:
                                    ind = min(num_segments, int(pos // (
                                                self.detect_radius / num_segments)))
                                    cv_state[i][ind] += 1

                        if lane in self.uv_data.keys():
                            for car, info in self.uv_data[lane].items():
                                veh_position = np.array(info[traci.constants.VAR_POSITION]) ###
                                pos = max(0,np.linalg.norm(self.junc_position - veh_position)-20)
                                if pos <= self.detect_radius:
                                    ind = min(num_segments, int(pos // (
                                                self.detect_radius / num_segments)))
                                    uv_state[i][ind] += 1
                
                norm_CV = np.linalg.norm(np.ravel(cv_state))
                norm_UV = np.linalg.norm(np.ravel(uv_state))
                if norm_CV == 0:
                    norm_CV = 1
                if norm_UV == 0:
                    norm_UV = 1

                self.state_comparison = [np.ravel(cv_state),np.ravel(uv_state)] # store num of veh info

                print("Critic num_veh before norm: CV", cv_state, "\n UV:", uv_state) ###

                return np.concatenate([np.ravel(cv_state)/norm_CV,
                                    np.ravel(uv_state)/norm_UV])
                # else:
                #     n_phase = len(self.max_pressure_lanes)
                #     cv_state = np.zeros((n_phase, num_segments))
                #     for i, phase in enumerate(self.max_pressure_lanes):
                #         for lane in self.max_pressure_lanes[phase][lane_dir]:
                #             if lane in self.data.keys():
                #                 for car, info in self.data[lane].items():
                #                     veh_position = np.array(info[traci.constants.VAR_POSITION]) ### TODO: might have bug 02142023
                #                     pos = np.linalg.norm(self.junc_position - veh_position)
                #                     if pos <= self.detect_radius:
                #                         ind = min(num_segments, int(pos // (
                #                                     self.detect_radius / num_segments)))
                #                         cv_state[i][ind] += 1

                    
                #     norm_CV = np.linalg.norm(np.ravel(cv_state))
                #     if norm_CV == 0:
                #         norm_CV = 1

                #     self.state_comparison = [np.ravel(cv_state)] # store num of veh info

                #     return np.ravel(cv_state)/norm_CV
                                        
            elif agent == 'actor':
                n_phase = len(self.max_pressure_lanes)
                cv_state = np.zeros((n_phase, num_segments))
                for i, phase in enumerate(self.max_pressure_lanes):
                    for lane in self.max_pressure_lanes[phase][lane_dir]:
                        if lane in self.cv_data.keys():
                            for car, info in self.cv_data[lane].items():
                                veh_position = np.array(info[traci.constants.VAR_POSITION])
                                pos = max(0,np.linalg.norm(self.junc_position - veh_position)-20)
                                if pos <= self.detect_radius:
                                    ind = min(num_segments, int(pos // (
                                                self.detect_radius / num_segments)))
                                    cv_state[i][ind] += 1
                #return np.ravel(cv_state)
                norm_CV = np.linalg.norm(np.ravel(cv_state))
                if norm_CV == 0:
                    norm_CV = 1
                return np.ravel(cv_state)/norm_CV
            else:
                raise NotImplementedError('get_num_vehicle wrong input:', agent)
    
    def get_avg_delay(self,agent = None):
        if agent == "critic":
            avg_delay_cv = []
            avg_delay_uv = []
            for phase in self.max_pressure_lanes.keys():
                cv_delay = np.array([])
                uv_delay = np.array([])
                for lane in self.max_pressure_lanes[phase]['inc']:
                    if (lane in self.cv_data.keys()) and (len(self.cv_data[lane]) != 0):
                        # cv_df = pd.DataFrame.from_dict(self.cv_data[lane]).T
                        # cv_df.columns=["lane_position","speed","land_id","road_id",
                        #                "delay","position"]
                        # cv_delay = np.append(cv_delay, np.array(cv_df.delay))
                        for v in self.cv_data[lane].keys():
                             cv_delay = np.append(cv_delay, np.array(self.cv_data[lane][v][traci.constants.VAR_TIMELOSS]))
                    else:
                        cv_delay = np.append(cv_delay, 0)
                    if (lane in self.uv_data.keys()) and (len(self.uv_data[lane]) != 0):
                        # uv_df = pd.DataFrame.from_dict(self.uv_data[lane]).T
                        # uv_df.columns = ["lane_position", "speed", "land_id", "road_id",
                        #                  "delay", "position"]
                        # uv_delay = np.append(uv_delay, np.array(uv_df.delay))
                        for v in self.uv_data[lane].keys():
                             uv_delay = np.append(uv_delay, np.array(self.uv_data[lane][v][traci.constants.VAR_TIMELOSS]))
                    else:
                        uv_delay = np.append(uv_delay, 0)

                avg_delay_cv.append(np.mean(cv_delay))
                avg_delay_uv.append(np.mean(uv_delay))

            #return np.concatenate([np.array(avg_speed_cv), np.array(avg_speed_uv)])
            # norm_CV = np.linalg.norm(np.array(avg_delay_cv))
            # norm_UV = np.linalg.norm(np.array(avg_delay_uv))
            # if norm_CV == 0:
            #     norm_CV = 1
            # if norm_UV == 0:
            #     norm_UV = 1
            # if self.mode == 'test':
            #     print("\n Delay:",np.concatenate([np.array(avg_delay_cv), np.array(avg_delay_uv)])) ###
            return np.concatenate([np.array(avg_delay_cv)/100,
                                    np.array(avg_delay_uv)/100])

        elif agent == "actor":
            avg_delay_cv = []
            for phase in self.max_pressure_lanes.keys():
                cv_delay = np.array([])
                for lane in self.max_pressure_lanes[phase]['inc']:
                    if (lane in self.cv_data.keys()) and (len(self.cv_data[lane]) != 0):
                        for v in self.cv_data[lane].keys():
                             cv_delay = np.append(cv_delay, np.array(self.cv_data[lane][v][traci.constants.VAR_TIMELOSS]))
                        # cv_df = pd.DataFrame.from_dict(self.cv_data[lane]).T
                        # print("####DEBUG cv_df",cv_df.head()) ####
                        # cv_df.columns = ["lane_position", "speed", "land_id", "road_id",
                        #                  "delay", "position"]
                        # cv_delay = np.append(cv_delay, np.array(cv_df.speed))
                    else:
                        cv_delay = np.append(cv_delay, 0)
                avg_delay_cv.append(np.mean(cv_delay))

            #return np.array(avg_speed_cv)
            # norm_CV = np.linalg.norm(np.array(avg_delay_cv))
            # if norm_CV == 0:
            #     norm_CV = 1
            return np.array(avg_delay_cv)/100
    
    def get_avg_speed(self, num_segments=1, agent=None):
        # Function: return average speed per road segment per green phase
        # if num_segments == 1:
        #     if agent == "critic":
        #         # the speed of each vehicle is stored in the self.data, cv_data, and uv_data
        #         # current cv_data: {lane:{veh:{"speed":[],"lane_position":[],...]}},}
        #         avg_speed_cv = []
        #         avg_speed_uv = []
        #         for phase in self.max_pressure_lanes.keys():
        #             cv_speed = np.array([])
        #             uv_speed = np.array([])
        #             for lane in self.max_pressure_lanes[phase]['inc']:
        #                 if lane in self.cv_data.keys():
        #                     cv_df = pd.DataFrame.from_dict(self.cv_data[lane]).T
        #                     cv_speed = np.append(cv_speed, np.array(cv_df.speed))
        #                 else:
        #                     speed_limit = traci.lane.getMaxSpeed(lane)
        #                     cv_speed = np.append(cv_speed, speed_limit)
        #                 if lane in self.uv_data.keys():
        #                     uv_df = pd.DataFrame.from_dict(self.uv_data[lane]).T
        #                     uv_speed = np.append(uv_speed, np.array(uv_df.speed))
        #                 else:
        #                     speed_limit = traci.lane.getMaxSpeed(lane)
        #                     uv_speed = np.append(uv_speed, speed_limit)
        #
        #             avg_speed_cv.append(np.mean(cv_speed))
        #             avg_speed_uv.append(np.mean(uv_speed))
        #
        #         #return np.concatenate([np.array(avg_speed_cv), np.array(avg_speed_uv)])
        #         norm_CV = np.linalg.norm(np.array(avg_speed_cv))
        #         norm_UV = np.linalg.norm(np.array(avg_speed_uv))
        #         if norm_CV == 0:
        #             norm_CV = 1
        #         if norm_UV == 0:
        #             norm_UV = 1
        #         return np.concatenate([np.array(avg_speed_cv)/norm_CV,
        #                                np.array(avg_speed_uv)/norm_UV])
        #     elif agent == "actor":
        #         avg_speed_cv = []
        #         for phase in self.max_pressure_lanes.keys():
        #             cv_speed = np.array([])
        #             for lane in self.max_pressure_lanes[phase]['inc']:
        #                 if lane in self.cv_data.keys():
        #                     cv_df = pd.DataFrame.from_dict(self.cv_data[lane]).T
        #                     cv_speed = np.append(cv_speed, np.array(cv_df.speed))
        #                 else:
        #                     speed_limit = traci.lane.getMaxSpeed(lane)
        #                     cv_speed = np.append(cv_speed, speed_limit)
        #             avg_speed_cv.append(np.mean(cv_speed))
        #
        #         #return np.array(avg_speed_cv)
        #         norm_CV = np.linalg.norm(np.array(avg_speed_cv))
        #         if norm_CV == 0:
        #             norm_CV = 1
        #         return np.array(avg_speed_cv)/norm_CV
        #
        #     else:
        #         raise NotImplementedError('get_avg_speed wrong input:', agent)
        # else:
        # junc_position = np.array(self.conn.junction.getPosition(self.id))
        if agent == "critic":
            # 1. decide which lane and segment the vehicle is in
            # 2. store the speed into the matrix of speed in lane and segment
            # 3. calculate average value in each element of the matrix
            # 4. ravel the np matrix into 1-D array
            # if self.act_ctm:
            #     n_phase = len(self.max_pressure_lanes)
            #     # THE CV here is actually all vehicles
            #     cv_cnt = np.zeros((n_phase, num_segments))
            #     cv_speed = np.zeros((n_phase, num_segments))
            #     for i, phase in enumerate(self.max_pressure_lanes):
            #         empty_cv_lane = 0
            #         num_lane_phase = len(self.max_pressure_lanes[phase]['inc'])
            #         for lane in self.max_pressure_lanes[phase]['inc']:
            #             if lane in self.data.keys():
            #                 for car, info in self.data[lane].items():
            #                     veh_position = np.array(info[traci.constants.VAR_POSITION])
            #                     pos = np.linalg.norm(self.junc_position - veh_position)
            #                     if pos <= self.detect_radius:
            #                         ind = min(num_segments, int(pos // (
            #                                 self.detect_radius / num_segments)))
            #                         cv_cnt[i][ind] += 1
            #                         cv_speed[i][ind] += info[traci.constants.VAR_SPEED]
            #             else:
            #                 empty_cv_lane += 1
            #                 if empty_cv_lane == num_lane_phase:
            #                     for ind in range(num_segments):
            #                         cv_speed[i][ind] == traci.lane.getMaxSpeed(lane)

            #     # replace all 0s in cnt matrix with 1 so as to avoid NaN
            #     cv_cnt[cv_cnt == 0] = 1
            #     #return np.concatenate([np.ravel(cv_speed/cv_cnt), np.ravel(uv_speed/uv_cnt)])
            #     avg_spd_CV = np.ravel(cv_speed / cv_cnt)
            #     norm_CV = np.linalg.norm(avg_spd_CV)
            #     if norm_CV == 0:
            #         norm_CV = 1

            #     return avg_spd_CV/norm_CV
            # else:
            n_phase = len(self.max_pressure_lanes)
            cv_cnt = np.zeros((n_phase, num_segments))
            cv_speed = np.zeros((n_phase, num_segments))
            uv_cnt = np.zeros((n_phase, num_segments))
            uv_speed = np.zeros((n_phase, num_segments))
            for i, phase in enumerate(self.max_pressure_lanes):
                empty_cv_lane = 0
                empty_uv_lane = 0
                num_lane_phase = len(self.max_pressure_lanes[phase]['inc'])
                for lane in self.max_pressure_lanes[phase]['inc']:
                    if lane in self.cv_data.keys():
                        for car, info in self.cv_data[lane].items():
                            veh_position = np.array(info[traci.constants.VAR_POSITION])
                            pos = max(0,np.linalg.norm(self.junc_position - veh_position)-20)
                            # max(0, np.linalg.norm(junc_position - veh_position) - 20)
                            if pos <= self.detect_radius:
                                ind = min(num_segments, int(pos // (
                                        self.detect_radius / num_segments)))
                                cv_cnt[i][ind] += 1
                                cv_speed[i][ind] += info[traci.constants.VAR_SPEED]
                    else:
                        empty_cv_lane += 1
                        if empty_cv_lane == num_lane_phase:
                            for ind in range(num_segments):
                                cv_speed[i][ind] = traci.lane.getMaxSpeed(lane)
                        
                    if lane in self.uv_data.keys():
                        for car, info in self.uv_data[lane].items():
                            veh_position = np.array(info[traci.constants.VAR_POSITION])
                            pos = np.linalg.norm(self.junc_position - veh_position)
                            if pos <= self.detect_radius:
                                ind = min(num_segments, int(pos // (
                                            self.detect_radius / num_segments)))
                                uv_cnt[i][ind] += 1
                                uv_speed[i][ind] += info[traci.constants.VAR_SPEED]
                    else:
                        empty_uv_lane += 1
                        if empty_uv_lane == num_lane_phase:
                            for ind in range(num_segments):
                                uv_speed[i][ind] = traci.lane.getMaxSpeed(lane)

                # if no vehicles, set the corresponding speed as vff for later speed diff calculation
                for ind in range(num_segments):
                    if uv_cnt[i][ind] == 0:
                        uv_speed[i][ind] = 17.88 #hard code!!! vff
                    if cv_cnt[i][ind] == 0:
                        cv_speed[i][ind] = 17.88

                

            # replace all 0s in cnt matrix with 1 so as to avoid NaN
            cv_cnt[cv_cnt == 0] = 1
            uv_cnt[uv_cnt == 0] = 1
            #return np.concatenate([np.ravel(cv_speed/cv_cnt), np.ravel(uv_speed/uv_cnt)])

            avg_spd_CV = 17.88 - np.ravel(cv_speed / cv_cnt)
            avg_spd_UV = 17.88 - np.ravel(uv_speed / uv_cnt)
            avg_spd_CV = np.array([max(0,v) for v in avg_spd_CV])
            avg_spd_UV = np.array([max(0,v) for v in avg_spd_UV])

            print("CRITIC avg_spd_CV",avg_spd_CV) #,"avg_spd_UV",avg_spd_UV) ###
            norm_CV = np.linalg.norm(avg_spd_CV)
            norm_UV = np.linalg.norm(avg_spd_UV)
            if norm_CV == 0:
                norm_CV = 1
            if norm_UV == 0:
                norm_UV = 1

            # print("CV SPEED",cv_speed) ###
            # print("UV SPEED",uv_speed) ###
            # print("CV number",cv_cnt) ###
            # print("UV number",uv_cnt) ###

            return np.concatenate([avg_spd_CV/norm_CV + 0.2,
                                avg_spd_UV/norm_UV + 0.2])


        elif agent == "actor":
            n_phase = len(self.max_pressure_lanes)
            cv_cnt = np.zeros((n_phase, num_segments))
            cv_speed = np.zeros((n_phase, num_segments))
            for i, phase in enumerate(self.max_pressure_lanes):
                empty_cv_lane = 0
                num_lane_phase = len(self.max_pressure_lanes[phase]['inc'])
                for lane in self.max_pressure_lanes[phase]['inc']:
                    if lane in self.cv_data.keys():
                        for car, info in self.cv_data[lane].items():
                            veh_position = np.array([info[traci.constants.VAR_POSITION][0],
                                                     info[traci.constants.VAR_POSITION][1]])
                            pos = max(0,np.linalg.norm(self.junc_position - veh_position)-20)
                            if pos <= self.detect_radius:
                                ind = min(num_segments, int(pos // (
                                        self.detect_radius / num_segments)))
                                cv_cnt[i][ind] += 1
                                cv_speed[i][ind] += info[traci.constants.VAR_SPEED]
                    else:
                        empty_cv_lane += 1
                        if empty_cv_lane == num_lane_phase:
                            for ind in range(num_segments):
                                cv_speed[i][ind] = traci.lane.getMaxSpeed(lane)
                
                for ind in range(num_segments):
                    if cv_cnt[i][ind] == 0:
                        cv_speed[i][ind] = 17.88

            # replace all 0s in cnt matrix with 1 so as to avoid NaN
            cv_cnt[cv_cnt == 0] = 1
            #return np.ravel(cv_speed/cv_cnt)
            avg_spd_CV = 17.88 - np.ravel(cv_speed / cv_cnt) # hard code free flow speed!!!
            avg_spd_CV = np.array([max(0,v) for v in avg_spd_CV])
            norm_CV = np.linalg.norm(avg_spd_CV)
            if norm_CV == 0:
                norm_CV = 1
            return avg_spd_CV/norm_CV + 0.2

        else:
            raise NotImplementedError('get_avg_speed wrong input:', agent)


    def get_delay_state(self):
        # return average delay per lane
        avg_delay_lane, avg_delay_cv_lane, avg_delay_uv_lane = self.trafficmetrics.get_metric_lane('delay')

        delay_state_v = np.array(list(avg_delay_lane.values()))
        delay_state_cv = np.array(list(avg_delay_cv_lane.values()))
        delay_state_uv = np.array(list(avg_delay_uv_lane.values()))

        # print('delay_state:',delay_state)
        return delay_state_v, delay_state_cv, delay_state_uv
    

    # def get_estimated_queue(self,inc_lanes):
    #     # this function will return the estimated queue length on each incoming lane based on Prof. Saif's work
    #     # t0 = time.time()
    #     center_list = {lane: np.arange(self.cell_size/2, self.netdata['lane'][lane]['length'], self.cell_size) for lane in inc_lanes} #self.incoming_lanes}
    #     # print('center_list_time::',time.time()-t0)###

    #     # print('center_list',center_list,'\n\n\n')
    #     # t0 = time.time()
    #     velocity_dict = {lane: [estimate_velocity(x_i, self.historic_data[lane], self.netdata['lane'][lane]['speed'], self.t) for x_i in center_list[lane]] for lane in inc_lanes} #self.incoming_lanes}
    #     # print('velocity_time::',time.time()-t0)###
    #     # print('velocity_dict',velocity_dict,'\n\n\n')

    #     self.historic_velocity[self.t] = velocity_dict
        
    #     # t0 = time.time()
    #     rho_dict = {lane: [newell_franklin(velocity,self.netdata['lane'][lane]['speed']) for velocity in velocity_dict[lane]] for lane in inc_lanes} #self.incoming_lanes }
    #     # print('rho_dict_time::',time.time()-t0)###
    #     # print('rho_dict',rho_dict,'\n\n\n')
    #     # TODO: we need value for each cell, here we only get the rho for each lane
    #     self.historic_rho[self.t] = rho_dict
    #     # t0 = time.time()
    #     queue_length = [sum([rho*self.cell_size for rho in rho_dict[lane]]) for lane in inc_lanes] #self.incoming_lanes]
    #     # print('queue_length_time::',time.time()-t0)###
    #     # print('queue_length',queue_length,'\n\n\n\n\n\n')
    #     self.historic_EQ[self.t] = queue_length

    #     return queue_length

    # def get_estimated_queue_out(self, out_lanes):
    #     # this function will return the estimated queue length on each outgoing lane
    #     center_list = {lane: np.arange(self.cell_size / 2, self.netdata['lane'][lane]['length'], self.cell_size) for
    #                    lane in out_lanes} # self.outgoing_lanes}

    #     velocity_dict = {lane: [estimate_velocity(x_i, self.historic_data[lane],self.netdata['lane'][lane]['speed'], self.t) for x_i in center_list[lane]]
    #                      for lane in out_lanes} #self.outgoing_lanes}

    #     rho_dict = {lane: [newell_franklin(velocity,self.netdata['lane'][lane]['speed']) for velocity in velocity_dict[lane]] for lane in out_lanes}
    #                 #self.outgoing_lanes}

    #     queue_length = [sum([rho * self.cell_size for rho in rho_dict[lane]]) for lane in out_lanes] #self.outgoing_lanes]

    #     return queue_length


    
    def get_num_vehicle(self, global_critic='none', num_segments=1):
        #number of vehicles in each incoming lane divided by the lane's capacity
        if num_segments==1: #

            if global_critic == 'total':
                return np.array([len(self.data[lane]) for lane in self.incoming_lanes])
            elif global_critic == 'sep':
                return np.concatenate([np.array([len(self.cv_data[lane]) for lane in self.incoming_lanes]), np.array([len(self.uv_data[lane]) for lane in self.incoming_lanes])])
            elif global_critic == 'none':
                return np.array([len(self.cv_data[lane]) for lane in self.incoming_lanes])
            else:
                raise NotImplementedError('global_critic is getting a weird value :',global_critic)

        else:
            n_incoming_lanes = len(self.incoming_lanes)
            if global_critic == 'total':
                state = np.zeros((n_incoming_lanes,num_segments))
                for i,lane in enumerate(self.incoming_lanes):
                    for car,info in self.data[lane].items():
                        pos = info[traci.constants.VAR_LANEPOSITION]
                        ind = min(num_segments,int(pos//(self.detect_radius/num_segments))) # by default, 150 is the detect_radius
                        state[i][ind]+=1
                return np.ravel(state)
            if global_critic == 'sep':
                cv_state = np.zeros((n_incoming_lanes,num_segments))
                for i,lane in enumerate(self.incoming_lanes):
                    for car,info in self.cv_data[lane].items():
                        pos = info[traci.constants.VAR_LANEPOSITION]
                        ind = min(num_segments,int(pos//(self.detect_radius/num_segments)))
                        cv_state[i][ind]+=1
                uv_state = np.zeros((n_incoming_lanes,num_segments))
                for i,lane in enumerate(self.incoming_lanes):
                    for car,info in self.uv_data[lane].items():
                        pos = info[traci.constants.VAR_LANEPOSITION]
                        ind = min(num_segments,int(pos//(self.detect_radius/num_segments)))
                        uv_state[i][ind]+=1
                    return np.concatenate([np.ravel(cv_state),np.ravel(uv_state)])
            elif global_critic == 'none':
                cv_state = np.zeros((n_incoming_lanes,num_segments))
                for i,lane in enumerate(self.incoming_lanes):
                    for car,info in self.cv_data[lane].items():
                        pos = info[traci.constants.VAR_LANEPOSITION]
                        ind = min(num_segments,int(pos//(self.detect_radius/num_segments)))
                        cv_state[i][ind]+=1
                return np.ravel(cv_state)
            else:
                raise NotImplementedError('global_critic is getting a weird value :',global_critic)

    def get_normalized_density(self, global_critic='none'):
        #number of vehicles in each incoming lane divided by the lane's capacity
        if global_critic == 'total':
            return np.array([len(self.data[lane]) for lane in self.incoming_lanes])/self.lane_capacity
        elif global_critic == 'sep':
            return np.concatenate([np.array([len(self.cv_data[lane]) for lane in self.incoming_lanes])/self.lane_capacity, \
                np.array([len(self.uv_data[lane]) for lane in self.incoming_lanes])/self.lane_capacity])
        elif global_critic=='none':
            return np.array([len(self.cv_data[lane]) for lane in self.incoming_lanes])/self.lane_capacity
        else:
            raise NotImplementedError('global_critic is getting a weird value :',global_critic)


    def get_normalized_density_out(self,global_critic='none'):
        #number of vehicles in each outgoing lane divided by the lane's capacity
        if global_critic == 'total':
            return np.array([len(self.data[lane]) if lane in self.data else 0 for lane in self.outgoing_lanes])/self.lane_capacity_out
        elif global_critic == 'sep':
            return np.concatenate([np.array([len(self.cv_data[lane]) if lane in self.cv_data else 0 for lane in self.outgoing_lanes])/self.lane_capacity_out,\
             np.array([len(self.uv_data[lane]) if lane in self.uv_data else 0 for lane in self.outgoing_lanes])/self.lane_capacity_out])
        elif global_critic=='none':
            return np.array([len(self.cv_data[lane]) if lane in self.cv_data else 0 for lane in self.outgoing_lanes])/self.lane_capacity_out
        else:
            raise NotImplementedError('global_critic is getting a weird value',global_critic)
        # wz: TODO: we can calculate the density of UVs here by subtraction

    def get_normalized_queue(self):
        lane_queues = []
        for lane in self.incoming_lanes:
            q = 0
            # wz: here is changed from self.data to be self.cv_data, which indicates that we assume agent can only observe CVs and mask is always true
            for v in self.cv_data[lane]:
                if self.cv_data[lane][v][traci.constants.VAR_SPEED] < 0.3:
                    q += 1
            lane_queues.append(q)
        return np.array(lane_queues)/self.lane_capacity

    def empty_intersection(self):
        # for lane in self.incoming_lanes:
        #     if len(self.cv_data[lane]) > 0:
        #         return False
        # return True
        return False

    # def get_reward(self, tsc_type = None, global_critic = 'none',all_veh_r = False):
    #     #return negative delay as reward

    #     # wz: if global critic, the reward for critic should come from all vehicles
    #     if global_critic != 'none' or all_veh_r:
    #         data = self.data
    #     else:
    #         data = self.cv_data

    #     if tsc_type in ['presslight',  'presslight_a2c', 'a2c_ps','a2c']: #,'presslight_ddpg', ':
    #         W = []
    #         #pressure_max = 0
    #         # TODO: Add a congestion indicator here when we calculate density -> store it in a global variable
    #         # for each green phase
    #         # congest_ind = 0
    #         for g in self.green_phases:
    #             # for each lane that is green in that phase
    #             for l in self.phase_lanes[g]:
    #                 # num of veh/ capacity of lane;
    #                 # we assume that veh length + mini gap is 7.5 meters (default value in sumo)
    #                 inc_dense = 7.5*len(data[l])/float(self.netdata['lane'][l]['length'])
    #                 #if inc_dense > self.cong_thresh: # TODO: meed to be tuned; add an argument
    #                 #    congest_ind = 1
    #                 for ol in self.netdata['lane'][l]['outgoing']:
    #                     if ol in data:
    #                         out_dense = 7.5*len(data[ol])/float(self.netdata['lane'][ol]['length'])
    #                     else:
    #                         out_dense = 0
    #                     W.append(inc_dense - out_dense)
    #                     #pressure_max += 1

    #         #self.cong_ind = congest_ind
    #         P = abs(sum(W))
    #         r = - P

    #         '''#print("r",r) ###
    #         if reward == 'speed': # TODO: add a new arg for reward of PressLight-A2C
    #             # add average speed of vehicles at the intersection. This would encourage agent to increase speed of vehicle
    #             pressure = - P/pressure_max # to normalized P with the maximum possible pressure
    #             speed = # TODO: how to get average speed of the intersection; with a higher speed, agent will get higher reward (can be defined as -max.speed/avg.speed)
    #             r = pressure + speed * alpha # TODO: add alpha ( weight) into args
    #         if reward == 'delay':
    #             pressure = - P/pressure_max
    #             all_delay, cv_delay = self.trafficmetrics.get_metric('delay')
    #             # TODO: normalize delay
    #             if global_critic == 'total' or global_critic == 'sep':
    #                 if int(all_delay) > max_delay: # TODO: add a global value max_delay
    #                     max_delay = int(all_delay)
    #                 delay = -int(all_delay)/max_delay
    #             else:
    #                 if int(cv_delay) > max_delay:
    #                     max_delay = int(cv_delay)
    #                 delay = -int(cv_delay)/max_delay
    #             r = delay * alpha + pressure'''
        
    #     elif tsc_type == 'cavlight':
    #         alpha = 0.7
    #         beta = 0.2
    #         reward_delay = self.get_reward_delay(self.tsc_type) # self.get_reward(self.tsc_type) # 
    #         reward_density = self.get_reward_density(self.tsc_type)
    #         # add penalty for frequent phase changing
    #         if self.phase_cnt:
    #             reward_phase_switching = 0
    #         else:
    #             reward_phase_switching = -2
            
    #         r = alpha*reward_delay + beta*reward_density + (1-alpha-beta)*reward_phase_switching
    #         print("reward_delay:",reward_delay,"reward_density:",reward_density,"Total reward:",r) ###

    #     else:
    #         all_delay, cv_delay, all_delay_avg, cv_delay_avg = self.trafficmetrics.get_metric('delay') # wz: we use cv_delay as reward for agent
    #         if (global_critic != 'none') or all_veh_r:
    #             delay = int(all_delay)
    #         else:
    #             delay = int(cv_delay)

    #         if delay == 0:
    #             r = 0
    #         else:
    #             r = -delay

    #     self.ep_rewards.append(r) # store reward
    #     return r
    
    def get_reward_density(self, tsc_type=None):
        data = self.data

        # junc_position = np.array(self.conn.junction.getPosition(self.id))
        if tsc_type in ['cavlight']:
            W = []
            
            for g in self.max_pressure_lanes.keys():
                inc_lanes = self.max_pressure_lanes[g]['inc']
                out_lanes = self.max_pressure_lanes[g]['out']
                # pressure is defined as the number of vehicles in a lane
                # inc_lane_length = {}
                # out_lane_length = {}
                # for l in inc_lanes:
                #     if not l.startswith(":"):
                #         inc_lane_length[l] = float(self.netdata['lane'][l]['length'])
                #     else:
                #         inc_lane_length[l] = 23.0 # use a fixed length, which can store 3 vehicles.
                # for l in out_lanes:
                #     if not l.startswith(":"):
                #         out_lane_length[l] = float(self.netdata['lane'][l]['length'])
                #     else:
                #         out_lane_length[l] = 23.0 # use a fixed length

                # Jan 17, 2023 WZ: get position-weighted density
                     #sum(out_lane_length.values())
                inc_veh_position = []
                out_veh_position = []

                for l in inc_lanes:
                    if l in self.data:
                        # self.tot_inc_lane_len[g][l] = inc_lane_length[l]
                        for v in self.data[l].keys():
                            veh_position = np.array(self.data[l][v][traci.constants.VAR_POSITION])
                            pos = max(0,np.linalg.norm(self.junc_position - veh_position)-20)
                            inc_veh_position.append(200 - pos) 
                for l in out_lanes:
                    if l in self.data:
                        # self.tot_out_lane_len[g][l] = out_lane_length[l]
                        for v in self.data[l].keys():
                            veh_position = np.array(self.data[l][v][traci.constants.VAR_POSITION]) 
                            pos = max(0,np.linalg.norm(self.junc_position - veh_position)-20)
                            out_veh_position.append(200 - pos)
                inc_pressure = (sum(inc_veh_position)/200.0) #*7.5/sum(self.tot_inc_lane_len[g].values())
                out_pressure = (sum(out_veh_position)/200.0) #*7.5/sum(self.tot_out_lane_len[g].values())
                #inc_pressure = sum(inc_veh_position)/200
                #out_pressure = sum(out_veh_position)/200


                #inc_pressure = sum([sum([(200 - self.data[l][v]["dist2int"])/200 for v in self.data[l]])*7.5/inc_lane_length[l] if l in self.data else 0 for l in inc_lanes])/len(inc_lanes)
                #out_pressure = sum([sum([(200 - self.data[l][v]["dist2int"])/200 for v in self.data[l]])*7.5/out_lane_length[l] if l in self.data else 0 for l in out_lanes])/len(out_lanes)

                #print(g,"inc_pressure",inc_pressure) ### 

                # # density-based pressure: 
                # inc_pressure = sum([len(self.data[l])*7.5/inc_lane_length[l] if l in self.data else 0 for l in inc_lanes])/len(inc_lanes)
                # out_pressure = sum([len(self.data[l])*7.5/out_lane_length[l] if l in self.data else 0 for l in out_lanes])/len(out_lanes)

                # speed_list = np.array([])
                
                # for lane in inc_lanes:
                #     if lane in data.keys():
                #         speed_df = pd.DataFrame.from_dict(data[lane]).T
                #         speed_list = np.append(speed_list, np.array(speed_df.speed))
                #         tmp_speed_limit = traci.lane.getMaxSpeed(lane)
                #         if tmp_speed_limit > speed_limit:
                #             speed_limit = tmp_speed_limit
                # if len(speed_list) != 0:
                #     avg_speed = np.mean(speed_list)
                # else:
                #     avg_speed = speed_limit

                #print("#### Phase:",g,"inc_pressure:",inc_pressure,"out_pressure",out_pressure) ####
                #print("#### Phase:",g,"avg_speed:",inc_pressure,"speed_limit",speed_limit) ####
                W.append((inc_pressure - out_pressure)) #*(1+(speed_limit-avg_speed)/speed_limit))
                print(g,"pressure:",inc_pressure - out_pressure) ###

            P = abs(sum(W))/10
            r = - P
            print("density reward:",r) ###
        # self.ep_rewards.append(r)  # store reward
        return r

    def get_reward_delay(self, tsc_type=None):

        data = self.data

        if tsc_type in ['cavlight']:
            W = []
            for g in self.max_pressure_lanes.keys():
                inc_lanes = self.max_pressure_lanes[g]['inc']
                # out_lanes = self.max_pressure_lanes[g]['out']
                # # pressure is defined as the number of vehicles in a lane
                # inc_lane_length = {}
                # out_lane_length = {}
                # for l in inc_lanes:
                #     if not l.startswith(":"):
                #         inc_lane_length[l] = float(self.netdata['lane'][l]['length'])
                #     else:
                #         inc_lane_length[l] = 23.0 # use a fixed length, which can store 3 vehicles.
                # for l in out_lanes:
                #     if not l.startswith(":"):
                #         out_lane_length[l] = float(self.netdata['lane'][l]['length'])
                #     else:
                #         out_lane_length[l] = 23.0 # use a fixed length
                inc_veh_delay = []
                #out_veh_delay = []

                for l in inc_lanes:
                    if l in self.data:
                        # self.tot_inc_lane_len[g][l] = inc_lane_length[l]
                        for v in self.data[l]:
                            if not np.isnan(self.data[l][v][traci.constants.VAR_TIMELOSS]):
                                inc_veh_delay.append(self.data[l][v][traci.constants.VAR_TIMELOSS])
                # for l in out_lanes:
                #     if l in self.data:
                #         # self.tot_out_lane_len[g][l] = out_lane_length[l]
                #         for v in self.data[l]:
                #             if not np.isnan(self.data[l][v]["delay"]):
                #                 out_veh_delay.append(self.data[l][v]["delay"])
                
                if len(inc_veh_delay) > 0:
                    inc_pressure = np.sum(inc_veh_delay)/10 #np.mean(inc_veh_delay) #(sum(inc_veh_position)/200.0)*7.5/sum(self.tot_inc_lane_len[g].values())
                else:
                    inc_pressure = 0
                # if len(out_veh_delay) >0:
                #     out_pressure = np.mean(out_veh_delay) #(sum(out_veh_position)/200.0)*7.5/sum(self.tot_out_lane_len[g].values())
                # else:
                #     out_pressure = 0
                # W.append(inc_pressure - out_pressure)
                W.append(inc_pressure)

            P = (abs(sum(W)))/100 # directly scale the delay
            r = - P #*10
            # print("delay reward:",r) ###
        # self.ep_rewards.append(r)  # store reward
        return r

    def empty_dtse(n_lanes, dist, cell_size):
        return np.zeros((n_lanes, int(dist/cell_size)+3 ))
    
    def phase_dtse(phase_lanes, lane_to_int, dtse):
        phase_dtse = {}
        for phase in phase_lanes:
            copy_dtse = np.copy(dtse)
            for lane in phase_lanes[phase]:
                copy_dtse[lane_to_int[lane],:] = 1.0
            phase_dtse[phase] = copy_dtse
        return phase_dtse

    def get_dtse(self):
        dtse = np.copy(self._dtse)
        for lane,i in zip(self.incoming_lanes, range(len(self.incoming_lanes))):
            for v in self.cv_data[lane]:
                pos = self.cv_data[lane][v][traci.constants.VAR_LANEPOSITION]
                dtse[i, pos:pos+1] = 1.0

        return dtse

'''
right_on_red_phases = []
for phase in green_phases:
    new_phase = []
    for idx in range(len(phase)):
        if self.netdata['inter'][self.id]['tlsindexdir'][idx] == 'r' and phase[idx] == 'r':
            new_phase.append('s')
        else:
            new_phase.append(phase[idx])
    right_on_red_phases.append(''.join(new_phase))
'''

'''
n_g = len(green_phases)
                                                                                            
right_on_red_phases = []
for phase in green_phases:
    new_phase = []
    for idx in range(len(phase)):
        if self.netdata['inter'][self.id]['tlsindexdir'][idx] == 'r' and phase[idx] == 'r':
            new_phase.append('s')
        else:
            new_phase.append(phase[idx])
    right_on_red_phases.append(''.join(new_phase))
                                                                                            
green_phases = [ p for p in right_on_red_phases 
                 if 'y' not in p
                 and ('G' in p or 'g' in p) ]
'''
'''
n_ror = len(ror_phases)
if n_ror != n_g:
    print('==========')
    print(self.id)
    print(green_phases)
    print(ror_phases)
'''
