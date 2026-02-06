import numpy as np
from itertools import cycle
from collections import deque

from src.trafficsignalcontroller import TrafficSignalController

class NextDurationRLTSC(TrafficSignalController):
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, gmin, gmax, rlagent, tsc_type, detect_r):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t, detect_r)
        self.cycle = cycle(self.green_phases)
        self.phase_deque = deque()
        self.data = None
        self.rlagent = rlagent
        self.phase_to_one_hot = self.input_to_one_hot(self.green_phases+[self.all_red])
        #help convert tanh rl action to constrained
        #next phase duration
        self.mid = ((gmax - gmin)/2.0) + gmin
        self.interval = gmax - self.mid
        #experience dict
        self.acting = False
        self.s = None
        self.a = None
        self.tsc_type = tsc_type # wz: to store the tsc info

    def next_phase(self):
        if len(self.phase_deque) == 0:
            next_phase = self.get_next_phase()
            phases = self.get_intermediate_phases(self.phase, next_phase)
            self.phase_deque.extend(phases+[next_phase])
        return self.phase_deque.popleft()

    def get_next_phase(self):
        #find the next green phase
        #with vehicles in approaching lanes
        i = 0
        while i <= len(self.green_phases):
            phase = next(self.cycle)
            if not self.phase_lanes_empty(phase):
                if self.acting:
                    if self.tsc_type == 'presslight_ddpg':
                #if self.rlagent.global_critic != None:
                    # wz: Jan10, once we use presslight_a2c, the state will always contain state_global and state_local
                    # wz: but when global_critic == None, these two are the same in terms of values
                        state_global, state_local = self.get_state(self.tsc_type, estimate_queue = False, global_critic = self.rlagent.global_critic, num_segments=1)
                        state_global = np.concatenate([state_global,self.phase_to_one_hot[self.phase],np.array([self.phase_duration])])
                        state_local = np.concatenate([state_local,self.phase_to_one_hot[self.phase],np.array([self.phase_duration])])
                        state = (state_global,state_local)
                    else:
                        state = np.concatenate( [self.get_state(), self.phase_to_one_hot[phase]] )
                    terminal = False
                    self.store_experience(state, terminal)
                if not self.acting:
                    if self.tsc_type == 'presslight_ddpg':
                #if self.rlagent.global_critic != None:
                    # wz: Jan10, once we use presslight_a2c, the state will always contain state_global and state_local
                    # wz: but when global_critic == None, these two are the same in terms of values
                        state_global, state_local = self.get_state(self.tsc_type, estimate_queue = False, global_critic = self.rlagent.global_critic, num_segments=1)
                        state_global = np.concatenate([state_global,self.phase_to_one_hot[self.phase],np.array([self.phase_duration])])
                        state_local = np.concatenate([state_local,self.phase_to_one_hot[self.phase],np.array([self.phase_duration])])
                        state = (state_global,state_local)
                    else:
                        state = np.concatenate( [self.get_state(), self.phase_to_one_hot[phase]] ) 
                self.s = state
                if self.tsc_type == 'presslight_ddpg':                                                                         
                    action = self.rlagent.get_action(state[1])  
                else:                                                       
                    action = self.rlagent.get_action(state)                                                       
                self.a = action                                                                        
                self.acting = True
                return phase
            i += 1
        ##if no vehicles approaching intersection
        #default to all red
        phase = self.all_red
        if self.acting:
            #print('-------TERMINAL---------')
            if self.tsc_type == 'presslight_ddpg':
                #if self.rlagent.global_critic != None:
                    # wz: Jan10, once we use presslight_a2c, the state will always contain state_global and state_local
                    # wz: but when global_critic == None, these two are the same in terms of values
                state_global, state_local = self.get_state(self.tsc_type, estimate_queue = False, global_critic = self.rlagent.global_critic, num_segments=1)
                state_global = np.concatenate([state_global,self.phase_to_one_hot[self.phase],np.array([self.phase_duration])])
                state_local = np.concatenate([state_local,self.phase_to_one_hot[self.phase],np.array([self.phase_duration])])
                state = (state_global,state_local)
            else:
                state = np.concatenate( [self.get_state(), self.phase_to_one_hot[phase]] )
            terminal = True
            self.store_experience(state, terminal)
            self.acting = False
        return phase 

    def next_phase_duration(self):
        if self.phase in self.green_phases:
            t = self.convert_action(self.a)
            #print(' phase time '+str(t))
            return t
        elif 'y' in self.phase:
            return self.yellow_t
        else:
            return self.red_t

    def store_experience(self, next_state, terminal):
        if self.tsc_type == 'presslight_ddpg':
            self.rlagent.store_experience_presslight_ddpg(self.s, self.a, next_state, self.get_reward(), terminal)
        else:
            self.rlagent.store_experience(self.s, self.a, next_state, self.get_reward(), terminal)
        #put experience in buffer
        #update rl stats

    def convert_action(self, action):
        #convert rl action to green
        #phase time, respecting constraints
        # wz: this function convert the output action (a value between -1 and 1)
        # to the next phase duration (mid+-interval*fraction) [action determines the sign and fraction]
        return int((action*self.interval)+self.mid)

    def update(self, data, cv_data,uv_data, mask):
        self.data = data
        self.cv_data = cv_data
        self.uv_data = uv_data

    def phase_lanes_empty(self, phase):
        for l in self.phase_lanes[phase]:
            if len(self.data[l]) > 0:
                return False
        return True
