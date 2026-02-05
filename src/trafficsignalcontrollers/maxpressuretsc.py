import random
from itertools import cycle
from collections import deque
from src.CTM import CTM_model 

from src.trafficsignalcontroller import TrafficSignalController

class MaxPressureTSC(TrafficSignalController):
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, green_t, detect_r, estimate_queue,pen_rate, pen_rate_est, args):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t, detect_r)
        self.green_t = green_t
        self.t = 0
        #for keeping track of vehicle counts for websters calc
        self.phase_deque = deque()
        self.max_pressure_lanes = self.get_max_pressure_lanes()
        self.data = None
        self.uv_data = None
        #store how many green movements each phase has
        #for breaking ties in max pressure
        self.phase_g_count = {}
        for p in self.green_phases:
            self.phase_g_count[p] = sum([1 for m in p if m == 'g' or m == 'G'])
        self.tsc_type = 'maxpressure'
        self.estimate_queue = estimate_queue
        self.pen_rate_est = pen_rate_est
        self.pen_rate = pen_rate/100.0
        self.green_t_cnt = 10
        self.select_phase = None 
        self.args = args

    def next_phase(self):
        ###need to do deque here
        if len(self.phase_deque) == 0:
            max_pressure_phase = self.max_pressure()
            phases = self.get_intermediate_phases(self.phase, max_pressure_phase)
            self.phase_deque.extend(phases+[max_pressure_phase])
        return self.phase_deque.popleft()

    def get_max_pressure_lanes(self):
        """for each green phase, get all incoming
        and outgoing lanes for that phase, store
        in dict for max pressure calculation
        """
        max_pressure_lanes = {}
        for g in self.green_phases:
            inc_lanes = set()
            out_lanes = set()
            # TODO 10232022: update this section to enable phase lanes considers the connected lanes -> to make it consistent with self.data 
            #for l in self.phase_lanes[g]:
            #    inc_lanes.add(l)
            #    for ol in self.netdata['lane'][l]['outgoing']:
            #        out_lanes.add(ol)
            for l in self.phase_lanes[g]:
                inc_lanes.update(self.conn_incoming_lane[l])
                for ol in self.netdata['lane'][l]['outgoing']:
                    out_lanes.update(self.conn_outgoing_lane[ol])

            max_pressure_lanes[g] = {'inc':inc_lanes, 'out':out_lanes}
            # print("phase",g,"max_pressure_lanes[g]",max_pressure_lanes[g])#####
        return max_pressure_lanes

    def max_pressure(self):
        phase_pressure = {}
        no_vehicle_phases = []
        #compute pressure for all green movements
        if self.act_ctm:
            for g in self.green_phases:
                phase_pressure[g], inc_pressure, out_pressure = self.CTM.get_state_CTM_MP(int(self.t//10),g)
                print("CTM-based pressure: phase",g,"pressure:",phase_pressure[g],"inc_pre",inc_pressure) ####
                if inc_pressure == 0 and out_pressure == 0:
                    no_vehicle_phases.append(g)
            veh_dict_OPT = self.CTM.get_state_CTM_OPT(int(self.t//10),self.phase,self.phase_duration)

        else:
            for g in self.green_phases:
                inc_lanes = self.max_pressure_lanes[g]['inc']
                out_lanes = self.max_pressure_lanes[g]['out']
                #pressure is defined as the number of vehicles in a lane
                inc_pressure = sum([ len(self.data[l]) if l in self.data else 0 for l in inc_lanes])
                out_pressure = sum([ len(self.data[l]) if l in self.data else 0 for l in out_lanes])
                out_pressure = 0 ## 01172024 temperally disable outpressure for isolated intersection
                phase_pressure[g] = inc_pressure - out_pressure
                print("phase",g,"pressure:",phase_pressure[g],"inc_pre",inc_pressure) ####
                if inc_pressure == 0 and out_pressure == 0:
                    no_vehicle_phases.append(g)

        
        if len(no_vehicle_phases) == len(self.green_phases):
            ###if no vehicles -> randomly select a phase 
            #return random.choice(self.green_phases)

            ### -> 10/10/2022 WZ: change the logic to choose the last action rather than a random one
            if self.select_phase != None:
                return self.select_phase
            else:
                return random.choice(self.green_phases)
            
        else:
            #choose phase with max pressure
            #if two phases have equivalent pressure
            #select one with more green movements
            #return max(phase_pressure, key=lambda p:phase_pressure[p])
            phase_pressure = [ (p, phase_pressure[p]) for p in phase_pressure]
            phase_pressure = sorted(phase_pressure, key=lambda p:p[1], reverse=True)
            phase_pressure = [ p for p in phase_pressure if p[1] == phase_pressure[0][1] ]
            # if more than one phase available, randomly select one
            self.select_phase = random.choice(phase_pressure)[0]
            return self.select_phase

            '''
            if len(phase_pressure) == 1:
                return phase_pressure[0][0]
            else:
                #if two phases have same pressure and same number of green movements
                green_count = [ (p[0], self.phase_g_count[p[0]]) for p in phase_pressure ]
                green_count = sorted(green_count, key=lambda p:p[1], reverse=True)
                green_count = [ p for p in green_count if p[1] == green_count[0][1] ]
                if len(green_count) == 1:
                    return green_count[0][0]
                else:
                    return random.choice(green_count)[0]
            '''


    def next_phase_duration(self, current_phase):
        if self.phase in self.green_phases:
            if self.phase == current_phase:
                return self.green_t_cnt
            else:
                return self.green_t
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
