'''
This file provides the functions to
1) generate sumo simulation, including network and vehicles
2) control traffic based on algorithms in tsc_factory and communicate with sumo through traci
'''

import os, sys, subprocess

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import numpy as np
import random # wz
import os
import sumolib

from src.trafficsignalcontroller import TrafficSignalController
from src.tsc_factory import tsc_factory
from src.att_factory import att_factory
from src.vehiclegen import VehicleGen
from src.helper_funcs import write_line_to_file, check_and_make_dir, get_time_now, write_to_log, get_fp
from src.picklefuncs import save_data
from shutil import copy2
from collections import defaultdict
from src.trafficmetrics import GlobalTrafficMetrics

class SumoSim:
    def __init__(self, cfg_fp, sim_len, tsc, nogui, netdata, args, idx):
        self.cfg_fp = cfg_fp
        self.sim_len = sim_len
        self.tsc = tsc
        self.sumo_cmd = 'sumo' if nogui else 'sumo-gui' 
        self.netdata = netdata
        self.args = args
        self.idx = idx

        if self.args.no_random_flow:
            np.random.seed(self.args.seed + self.idx)
            random.seed(self.args.seed + self.idx)

        # wz: record the phase history of each tsc
        self.phase_hist = defaultdict()
        # self.action_hist = defaultdict()

        # # wz: for Queue Estimation
        # self.velo_hist_all = defaultdict()
        # self.rho_hist_all = defaultdict()
        # self.EQ_hist_all = defaultdict()
        # self.TQ_hist_all = defaultdict()

        # CVs
        self.con_veh = set()
        self.unequipped_veh = set()
        self.pen_rate = args.pen_rate #40
        self.succ_detect_rate = args.succ_detect_rate #10
        self.mask = args.mask #True

        if args.mode == 'train':
            self.global_metric_args = []
        else:
            if args.record_position:
                #self.global_metric_args = ['flow', 'position']
                #self.global_metric_args = ['route']
                self.global_metric_args = ['flow', 'position', 'route']
            else:
                # self.global_metric_args = ['flow']
                self.global_metric_args = []
        self.global_traffic_metrics = GlobalTrafficMetrics(netdata, self.global_metric_args, self.args.mode)

        print(idx) # wz: idx is the index of thread
        check_and_make_dir('./runtime')
        self.cwd = './runtime/' + str(idx) + '/'
        check_and_make_dir(self.cwd)

        # if self.args.sim == 'single':
        
        if 'plymouth' not in self.args.sim:
            copy2(f'./networks/{self.args.sim}/{self.args.sim}.net.xml', self.cwd)
            copy2(f'./networks/{self.args.sim}/{self.args.sim}_{self.args.flow_type}_{self.args.mode}.flow.xml', self.cwd)
            copy2(f'./networks/{self.args.sim}/{self.args.sim}_{self.args.turn_type}_{self.args.mode}.turn.xml', self.cwd)
            copy2(f'./networks/{self.args.sim}/{self.args.sim}.sumocfg', self.cwd)

            self.cfg_fp = self.cwd + f'{self.args.sim}.sumocfg'
        else:
            copy2(f'./networks/{self.args.sim}/{self.args.sim}.net.xml', self.cwd)
            # copy2(f'./networks/{self.args.sim}/{self.args.sim}_{self.args.flow_type}_{self.args.mode}.flow.xml', self.cwd)
            # copy2(f'./networks/{self.args.sim}/{self.args.sim}_{self.args.turn_type}_{self.args.mode}.turn.xml', self.cwd)
            copy2(f'./networks/{self.args.sim}/{self.args.sim}_{self.args.mode}.rou.xml', self.cwd)
            copy2(f'./networks/{self.args.sim}/{self.args.sim}_{self.args.mode}.sumocfg', self.cwd)
            copy2(f'./networks/{self.args.sim}/{self.args.sim}_sourcenode.additional.xml', self.cwd)

            self.cfg_fp = self.cwd + f'{self.args.sim}_{self.args.mode}.sumocfg'
        
        

        # wz: read network for marl
        net_fp = 'networks/' + args.sim + '/' + args.sim + '.net.xml'
        self.net = sumolib.net.readNet(net_fp)

        # CAVLight
        self.warm_start_len = 1000 #1000, 100 seconds
        self.detec_range = args.detec_range #80.0 # meters for CAV/infrastructure detetction (not communication) #TODO set it as an argument for tuning

    def gen_con_veh(self):
        '''
        get the subscription of all vehicles and filter a portion of them as CVs
        v_data: id of all newly departed vehicles
        :return: a list of ids of CVs
        '''
        # get subscription of all new departed vehicles
        for veh_id in self.conn.simulation.getDepartedIDList():
            if self.args.mode == 'train':
                self.conn.vehicle.subscribe(veh_id, [traci.constants.VAR_LANE_ID])
            else:
                # record vehicle details
                # self.conn.vehicle.subscribe(veh_id, [traci.constants.VAR_LANE_ID])
                self.conn.vehicle.subscribe(veh_id, [traci.constants.VAR_LANE_ID,
                                                     traci.constants.VAR_POSITION,
                                                     traci.constants.VAR_EDGES]) 
                # WZ Jan 18, 2023. we dont need to keep the same vehicle info dict as 
                # long as the state and reward consistent with co-sim
            
            # wz: mask CVs
            # use a function of time to control the pen_rate, we can refer to the vehiclegen.py
            # if random.randint(1, 100) <= self.pen_rate:
            if self.args.dynamic_pen_rate:
                penetration_rate = self.dynamic_pen_rate(self.t, self.args.sim_len, self.pen_rate)
            else:
                penetration_rate = self.pen_rate

            if random.randint(1, 100) <= penetration_rate: #self.pen_rate:
                self.con_veh.add(veh_id)
                self.conn.vehicle.setColor(veh_id, (255, 0, 0)) # red
            else:
                self.unequipped_veh.add(veh_id)

        # update con_veh to eliminate arrived CVs
        self.veh = set(self.conn.vehicle.getAllSubscriptionResults().keys())
        self.con_veh = self.get_ID_list(self.con_veh,True)
        self.unequipped_veh = self.get_ID_list(self.unequipped_veh,True)
    
    def get_detect_veh(self, detect_mode):
        #start_time = time.time()
        detec_veh = set()

        if detect_mode == "CAV_w_intersection":
            for uv_id in self.unequipped_veh:
                uv_position = np.array(self.conn.vehicle.getPosition(uv_id))
                #uv_raod = traci.vehicle.getRoadID(uv_id)
                self.conn.vehicle.setColor(uv_id, (255, 255, 0)) # initialize as yellow

                # loop through junctions
                for junc_id in self.tl_junc:
                    junc_position =  np.array(self.conn.junction.getPosition(junc_id))
                    dist_junc = np.linalg.norm(junc_position-uv_position)

                    if dist_junc <= self.detec_range:
                        detec_veh.add(uv_id)
                        self.conn.vehicle.setColor(uv_id, (0, 0, 255)) # turn blue

                    else: # self.args.sumo_detect control both TSC and CAVs
                    #elif self.args.sumo_detect: # self.args.sumo_detect: only control CAVs
                        # loop through con_veh
                        for cv_id in self.con_veh:
                            #if uv_raod == traci.vehicle.getRoadID(cv_id): # to save some computations
                            cv_position = np.array(self.conn.vehicle.getPosition(cv_id))
                            dist_veh = np.linalg.norm(cv_position-uv_position)
                            if dist_veh <= self.detec_range:
                                detec_veh.add(uv_id)
                                self.conn.vehicle.setColor(uv_id, (0, 0, 255)) # turn blue
                                break # once the uv is detected, we do not record

        elif detect_mode == "CAV_real":
            # Dec 07, 2023: Update this part to consider occlusion
            # Goal: a certain percentage of UV will not be detected even within the detection range.
            # Then, instead of drop a certain percentage of vehicle, can we just add one more condition to add such UV?
            # say, 10% of removal rate = 90% of successfully detected rate.
            for uv_id in self.unequipped_veh:
                uv_position = np.array(self.conn.vehicle.getPosition(uv_id))
                self.conn.vehicle.setColor(uv_id, (255, 255, 0)) # initialize as yellow

                # loop through con_veh
                for cv_id in self.con_veh:
                    cv_position = np.array(self.conn.vehicle.getPosition(cv_id))
                    dist_veh = np.linalg.norm(cv_position-uv_position)

                    # generate detection based on Waymo's detection average percision data
                    if (dist_veh > 50) and (dist_veh <= self.detec_range) and (random.randint(1, 100) <= 57.5):
                        detec_veh.add(uv_id)
                        self.conn.vehicle.setColor(uv_id, (0, 0, 255)) # turn blue
                        break # once the uv is detected, we do not record
                    # Even though one UV can be detected by multiple CAVs, once it is detected, we dont need to consider which CAV detects it.
                    # For now, we dont consider misclassified vehicles (i.e. items not vehicles but classified as vehicle)
                    # therefore, as long as one vehicle is detcted, we can record it and skip all rest CAVs
                    elif (dist_veh > 30) and (dist_veh <= 50) and (random.randint(1, 100) <= 77):
                        detec_veh.add(uv_id)
                        self.conn.vehicle.setColor(uv_id, (0, 0, 255)) # turn blue
                        break 
                    elif (dist_veh <= 30) and (random.randint(1, 100) <= 92):
                        detec_veh.add(uv_id)
                        self.conn.vehicle.setColor(uv_id, (0, 0, 255)) # turn blue
                        break 
        
        elif detect_mode == "CAV":
            # Dec 07, 2023: Update this part to consider occlusion
            # Goal: a certain percentage of UV will not be detected even within the detection range.
            # Then, instead of drop a certain percentage of vehicle, can we just add one more condition to add such UV?
            # say, 10% of removal rate = 90% of successfully detected rate.
            for uv_id in self.unequipped_veh:
                uv_position = np.array(self.conn.vehicle.getPosition(uv_id))
                self.conn.vehicle.setColor(uv_id, (255, 255, 0)) # initialize as yellow

                # loop through con_veh
                for cv_id in self.con_veh:
                    cv_position = np.array(self.conn.vehicle.getPosition(cv_id))
                    dist_veh = np.linalg.norm(cv_position-uv_position)

                    # Assume detection average percision is the same within the detection range
                    if (dist_veh <= self.detec_range) and (random.randint(1, 100) <= self.succ_detect_rate):
                        detec_veh.add(uv_id)
                        self.conn.vehicle.setColor(uv_id, (0, 0, 255)) # turn blue
                        break # once the uv is detected, we do not record
                    

        elif detect_mode == "intersection":
            for uv_id in self.unequipped_veh:
                uv_position = np.array(self.conn.vehicle.getPosition(uv_id))
                self.conn.vehicle.setColor(uv_id, (255, 255, 0)) # initialize as yellow

                # loop through junctions
                for junc_id in self.tl_junc:
                    junc_position =  np.array(self.conn.junction.getPosition(junc_id))
                    dist_junc = np.linalg.norm(junc_position-uv_position)

                    if dist_junc <= self.detec_range:
                        detec_veh.add(uv_id)
                        self.conn.vehicle.setColor(uv_id, (0, 0, 255)) # turn blue

        else:
            print("ERROR: wrong input for argument detect_mode!!!")

        detec_veh = self.get_ID_list(detec_veh,True)

        return detec_veh

    def dynamic_pen_rate(self, t, simlen, pen_rate):
        t_sin = np.linspace(0 * np.pi, 1 * np.pi, simlen)
        return np.sin(t_sin[t])*pen_rate


    def get_subscription_data(self,mask = False): # ): # original

        # TODO: Jan 20, 2023. Adapat this section to the Plymouth map.

        #use SUMO subscription to retrieve vehicle info in batches
        #around the traffic signal controller
        v_data = self.conn.vehicle.getAllSubscriptionResults()
        lane_vehicles = {}
        lane_vehicles_cv = {} #
        lane_vehicles_uv = {} # 
        for v in v_data:
            lane = v_data[v][traci.constants.VAR_LANE_ID]
            # wz: note, here the lane could be the outgoing lane. So we will store vehicle info on outgoing lanes
            if lane not in lane_vehicles:
                lane_vehicles[lane] = {}
            lane_vehicles[lane][v] = v_data[v] # wz: this basically reorganizes vehicles info by lanes

            # wz: for CVs
            if mask:
                for cv in self.con_veh:
                    if cv in v_data:
                        lane = v_data[cv][traci.constants.VAR_LANE_ID]
                        if lane not in lane_vehicles_cv:
                            lane_vehicles_cv[lane] = {}
                        lane_vehicles_cv[lane][cv] = v_data[cv]
                for uv in self.unequipped_veh:
                    if uv in v_data:
                        lane = v_data[uv][traci.constants.VAR_LANE_ID]
                        if lane not in lane_vehicles_uv:
                            lane_vehicles_uv[lane] = {}
                        lane_vehicles_uv[lane][uv] = v_data[uv]

        return lane_vehicles, lane_vehicles_cv, lane_vehicles_uv #, out_lane_vehicles

    def get_ID_list(self,veh_set, mask=False):
        return veh_set & self.veh \
            if mask else set(
            self.conn.vehicle.getIDList())  # wz: take intersection so to eliminate IDs of normal vehicles and CVs that have already arrived

#############

    def gen_sim(self):
        #create sim stuff and intersections
        #serverless_connect()
        #self.conn, self.sumo_process = self.server_connect()

        if self.args.no_random_flow:
            np.random.seed(self.args.seed + self.idx)
            seed = np.random.randint(100)
            random.seed(self.args.seed + self.idx)
            print("\n No Random. Simulation seed",seed)
        else:
            seed = np.random.randint(100)
            print("\n Random. Simulation seed",seed)

        if 'plymouth' not in self.args.sim:
            net_xml = self.cwd + f'{self.args.sim}.net.xml'
            flow_xml = self.cwd + f'{self.args.sim}_{self.args.flow_type}_{self.args.mode}.flow.xml'
            turn_xml = self.cwd + f'{self.args.sim}_{self.args.turn_type}_{self.args.mode}.turn.xml'
            rou_xml = self.cwd + f'{self.args.sim}.rou.xml'
            # os.system(
            #         'duarouter -n ' + net_xml + ' -r ' + flow_xml + ' -o ' + rou_xml + ' --randomize-flows --seed ' + str(
            #             seed))

            random_flow = '' if self.args.no_random_flow else ' --randomize-flows'
            os.system('jtrrouter -n ' + net_xml +
                    ' -r ' + flow_xml +
                    ' -t ' + turn_xml +
                    ' -o ' + rou_xml +
                    random_flow +
                    ' --seed ' + str(seed))

            print("CMD for Sim Generation: ", 'jtrrouter -n ' + net_xml +
                ' -r ' + flow_xml +
                ' -t ' + turn_xml +
                ' -o ' + rou_xml +
                random_flow +
                ' --seed ' + str(seed))

        # random_sim = '' if self.args.no_random_flow else '--random' 
        sumoBinary = checkBinary(self.sumo_cmd)
        port = self.args.port+self.idx
        self.sumo_process = subprocess.Popen([sumoBinary, "-c",
                                         self.cfg_fp, "--remote-port",
                                         str(port), "--no-warnings",
                                         "--no-step-log",'--random',"--seed",str(seed)],
                                         stdout=None, stderr=None)

        self.conn = traci.connect(port)

        self.t = 0
        self.v_start_times = {}
        self.v_travel_times = {}
        # wz: use vehicle file now
        # self.vehiclegen = None
        # if self.args.sim == 'double' or self.args.sim == 'single' or self.args.sim == 'four':
        #    self.vehiclegen = VehicleGen(self.netdata,
        #                                 self.args.sim_len,
        #                                 self.args.demand,
        #                                 self.args.scale,
        #                                 self.args.mode, self.conn)

        # wz: generate vehicles before simulation based on the flow file

        # yx: update tsc program
        for tsc_id in self.conn.trafficlight.getIDList():
            self.conn.trafficlight.setProgram(tsc_id, self.args.tsc_program)



    def serverless_connect(self):
        traci.start([self.sumo_cmd, 
                     "-c", self.cwd + self.cfg_fp,
                     "--no-step-log", 
                     "--no-warnings",
                     "--random"])

    def server_connect(self):
        sumoBinary = checkBinary(self.sumo_cmd)
        port = self.args.port+self.idx
        sumo_process = subprocess.Popen([sumoBinary, "-c",
                                         self.cfg_fp, "--remote-port",
                                         str(port), "--no-warnings",
                                         "--no-step-log", "--random"],
                                         stdout=None, stderr=None)

        return traci.connect(port), sumo_process

    def get_traffic_lights(self):
        '''
        wz: get set of IDs of traffic light intersections with green phase
        '''
        #find all the junctions with traffic lights
        trafficlights = self.conn.trafficlight.getIDList()
        junctions = self.conn.junction.getIDList()
        tl_juncs = set(trafficlights).intersection( set(junctions) )
        tls = []
     
        #only keep traffic lights with more than 1 green phase
        for tl in tl_juncs:
            #subscription to get traffic light phases
            self.conn.trafficlight.subscribe(tl, [traci.constants.TL_COMPLETE_DEFINITION_RYG])
            tldata = self.conn.trafficlight.getAllSubscriptionResults()
            for logic in tldata[tl][traci.constants.TL_COMPLETE_DEFINITION_RYG]:
                if logic.programID == self.args.tsc_program:
                    break
            #for some reason this throws errors for me in SUMO 1.2
            #have to do subscription based above
            '''
            logic = self.conn.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0] 
            '''
            #get only the green phases
            green_phases = [ p.state for p in logic.getPhases()
                             if 'y' not in p.state
                             and ('G' in p.state or 'g' in p.state) ]
            if len(green_phases) > 1:
                tls.append(tl)

        #for some reason these intersections cause problems with tensorflow
        #I have no idea why, it doesn't make any sense, if you don't believe me 
        #then comment this and try it, if you an fix it you are the real MVP
        if self.args.sim == 'lust':
            lust_remove = ['-12', '-78', '-2060']
            for r in lust_remove:
                if r in tls:
                    tls.remove(r)
        return set(tls) 


    def create_tsc(self, eps, neural_networks = None):
        self.tl_junc = self.get_traffic_lights() 
        if not neural_networks:
            neural_networks = {tl:None for tl in self.tl_junc}
        #create traffic signal controllers for the junctions with lights
        self.tsc = { tl:tsc_factory(self.args.tsc, tl, self.args, self.netdata, neural_networks[tl], eps, self.conn)  
                     for tl in self.tl_junc }
        
    def create_attacker(self, rl_stats, exp_replays, eps, neural_networks = None):
        self.tl_junc = self.get_traffic_lights() 
        if not neural_networks:
            neural_networks = {tl+'_att':None for tl in self.tl_junc}
        # WZ: hard code ppo_att as attacker for now, self.args.att_type
        self.attacker = { tl+'_att':att_factory("ppo_att", tl, self.args, self.netdata, rl_stats[tl+'_att'], exp_replays[tl+'_att'], neural_networks[tl+'_att'], eps, self.conn)  
                     for tl in self.tl_junc }

    def update_netdata(self):
        tl_junc = self.get_traffic_lights()
        tsc = { tl:TrafficSignalController(self.conn, tl, self.args.mode, self.netdata, 2, 3, self.args.detect_r)
                     for tl in tl_junc }

        for t in tsc:
            self.netdata['inter'][t]['incoming_lanes'] = tsc[t].incoming_lanes
            self.netdata['inter'][t]['green_phases'] = tsc[t].green_phases
            self.netdata['inter'][t]['outgoing_lanes'] = tsc[t].outgoing_lanes # wz: add outgoing lanes

        all_intersections = set(self.netdata['inter'].keys())
        #only keep intersections that we want to control
        for i in all_intersections - tl_junc:
            del self.netdata['inter'][i]

        return self.netdata

    def sim_step(self):
        # to count time step
        self.conn.simulationStep()
        self.t += 1
        # if self.t % 100 == 0:
        #     print(self.idx, self.t)

    def run_offset(self, offset):
        while self.t < offset:
            #create vehicles if vehiclegen class exists
            # wz: comment out the following two lines since there is no more vehiclegen
            #if self.vehiclegen:
            #    self.vehiclegen.run()

            self.gen_con_veh()
            self.update_travel_times()
            self.sim_step()

    def run(self):
        #execute simulation for desired length
        #TODO: Oct 22, 2024. add attacker function; replace tsc training-related features
        # the TSC takes actions and get state from self.tsc -> we need to combine our attacker into the self.tsc


        while self.t < self.sim_len:

            if self.t <= self.warm_start_len:
                self.gen_con_veh()
                self.update_travel_times()
                self.sim_step()
            else: 
                self.gen_con_veh()

                if self.args.sumo_detect:
                    self.detect_veh = self.get_detect_veh(self.args.detect_mode)###Julia: self.detect_veh is the list of ID for vehicles detected
                    aug_con_veh = set().union(self.detect_veh,self.con_veh)
                    no_detect_veh = self.unequipped_veh - aug_con_veh
                    #self.con_veh.update(self.detect_veh)
                else:
                    aug_con_veh = self.con_veh.copy()
                    no_detect_veh = self.unequipped_veh.copy()
                    

                self.update_travel_times()
                v_data, cv_data, uv_data = self.get_subscription_data(self.mask) # this is for global traffic metrics not for signal control
                self.global_traffic_metrics.update(v_data, cv_data)
                #run all traffic signal controllers in network, wz: note the t is not time here
                # ALSO, the self.tsc here is defined in the create_tsc, rather the original tsc attribute passed in sumosim
                # there, each t in the self.tsc is the tsc agent at each intersection
                for t in self.tsc:
                    # self.tsc[t].run() # wz: orginal
                    self.tsc[t].run(aug_con_veh, no_detect_veh, self.con_veh, self.mask, self.t, self.args.estimate_queue, self.args.act_ctm, self.args.act_lp, self.attacker[t+'_att']) # wz: pass the list of CVs to tsc


                self.sim_step()

        # wz: get phase and reward record
        for t in self.tsc:
            self.phase_hist[t] = self.tsc[t].phase_record
            # if self.tsc[t].tsc_type in [ 'presslight_a2c', 'a2c','a2c_r','a2c_ps','a2c_psr','a2c_sr','cavlight']:
            #     self.action_hist[t] = self.tsc[t].action_record

        if self.args.mode == "test":
            if self.args.tsc in ["cavlight","mmitiss"]:
                for tl_id in self.tl_junc:
                    fp = '/'.join(['state_action_record']+[str(tl_id)]) + '/'
                    SA_fp = get_fp(self.args, fp)
                    check_and_make_dir(SA_fp)
                    save_data(SA_fp +str(get_time_now())+ '_state_action_record.p', self.tsc[tl_id].state_action_record) # TODO: output info. refer to traffic metrics
                    save_data(SA_fp +str(get_time_now())+ '_JSMA.p', self.tsc[tl_id].JSMA_result)
                    save_data(SA_fp +str(get_time_now())+ '_attack_phase_dist.p', self.tsc[tl_id].attack_phase_dist)
                    save_data(SA_fp +str(get_time_now())+ '_fake_traj_input.p', self.tsc[tl_id].fake_veh_traj_input)
                    save_data(SA_fp +str(get_time_now())+ '_CTM_cmp.p', self.tsc[tl_id].CTM_state_cmp)
                    # save_data(SA_fp +str(get_time_now())+ '_features_cmp.p', self.tsc[tl_id].features_cmp)
                    # save_data(SA_fp +str(get_time_now())+ '_CTM_state_collect_debug_cmp.p', self.tsc[tl_id].CTM_state_collect_debug)
                    print("Successfully store state and action for visualization")
        
            if self.args.act_ctm:
                for tl_id in self.tl_junc:
                    self.tsc[tl_id].CTM.viz_CTM_matrix(int(self.t//10),str(get_time_now())+tl_id)
                    self.tsc[tl_id].CTM.save_nvlist(int(self.t//10),str(get_time_now())+tl_id) # for CTM FD validation
                    # self.tsc[tl_id].CTM.save_OPT_input(str(get_time_now())+tl_id)
            # self.velo_hist_all[t] = self.tsc[t].historic_velocity
            # self.rho_hist_all[t] = self.tsc[t].historic_rho
            # self.EQ_hist_all[t] = self.tsc[t].historic_EQ
            # self.TQ_hist_all[t] = self.tsc[t].historic_TQ
            # print('ep_reward',t,self.tsc[t].ep_rewards) ###

    def get_d(self, id1, id2):
        # wz: we hard code the length of lane here; also, the get_d only works for a manhattan grid
        Node1 = self.net.getNode(id1)
        Node2 = self.net.getNode(id2)
        coord1 = Node1.getCoord()
        coord2 = Node2.getCoord()
        return int(abs(coord1[0] - coord2[0]) / 200 + abs(coord1[1] - coord2[1]) / 200)

    def update_travel_times(self):
        for v in self.conn.simulation.getDepartedIDList():
            self.v_start_times[v] = self.t

        for v in self.conn.simulation.getArrivedIDList():
            self.v_travel_times[v] = self.t - self.v_start_times[v]
            del self.v_start_times[v]

    def get_intersection_subscription(self):
        tl_data = {}
        lane_vehicles = { l:{} for l in self.lanes}
        for tl in self.tl_junc:
            tl_data[tl] = self.conn.junction.getContextSubscriptionResults(tl)
            if tl_data[tl] is not None:
                for v in tl_data[tl]:
                    lane_vehicles[ tl_data[tl][v][traci.constants.VAR_LANE_ID] ][v] = tl_data[tl][v]
        return lane_vehicles

    def sim_stats(self):
        tt = self.get_travel_times()
        if len(tt) > 0 :
            #print( '----------\ntravel time (mean, std) ('+str(np.mean(tt))+', '+str(np.std(tt))+')\n' )
            return [str(int(np.mean(tt))), str(int(np.std(tt)))]
        else:
            return [str(int(0.0)), str(int(0.0))]

    def get_travel_times(self):
        return [self.v_travel_times[v] for v in self.v_travel_times]

    def get_global_metrics(self):
        return {m: self.global_traffic_metrics.get_history(m) for m in self.global_metric_args}

    def get_tsc_metrics(self):
        tsc_metrics = {}
        for tsc in self.tsc:
            tsc_metrics[tsc] = self.tsc[tsc].get_traffic_metrics_history()
        return tsc_metrics

    def close(self):
        #self.conn.close()
        self.conn.close()
        self.sumo_process.terminate()
