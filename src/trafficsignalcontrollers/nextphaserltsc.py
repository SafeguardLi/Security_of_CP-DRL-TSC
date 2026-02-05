import time  # <--- Added Import
import random
import numpy as np
from itertools import cycle
from collections import deque
import pandas as pd
import collections
import traci

from src.trafficsignalcontroller import TrafficSignalController
from src.fake_veh_traj_gen import optimization_process

class NextPhaseRLTSC(TrafficSignalController):
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, green_t,g_max, rlagent, tsc_type, epsilon, eps_min, eps_factor, estimate_queue, num_segments, cong_thresh, detect_r, sync, all_veh_r, act_ctm, args):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t, detect_r)
        self.num_segments = num_segments 
        self.green_t = green_t 
        self.g_max = g_max
        self.t = 0
        self.estimate_queue = estimate_queue
        
        self.phase_deque = deque()
        self.data = None
        self.delay_green = False
        self.phase_to_one_hot = self.input_to_one_hot(self.green_phases+[self.all_red])
        self.int_to_phase = self.int_to_input(self.green_phases)
        self.rlagent = rlagent
        self.tsc_type = tsc_type 
        self.acting = False
        self.action_first=True
        self.s = None
        self.a = None
        self.a_dist = None
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_factor = eps_factor
        self.sync = sync
        assert self.epsilon >= self.eps_min, 'epsilon can not be smaller than eps_min'
        self.cong_thresh = cong_thresh
        
        if self.tsc_type in ['cavlight']:
            self.state_action_record = []
            self.state_comparison = []
            self.JSMA_result = []
            self.CTM_state_cmp = []
            self.features_cmp = []
            self.attack_phase_dist = []
            self.fake_veh_traj_input = []
        else:
            self.state_action_record = None

        self.all_veh_r = all_veh_r

        self.green_t_cnt = 10 
        self.act_ctm = act_ctm

        self.args = args

        self.max_attack_scale = self.args.max_attack_scale

        self.last_tsc_action = None
        self.att_action = None
        self.curr_phase_idx = 0
        self.exceed_gmax = False

        self.feature2cells = {  10:[i for i in range(98,110)],11:[i for i in range(94,98)],12:[i for i in range(90,94)],
                                13:[i for i in range(68,76)], 14:[i for i in range(64,68)], 15:[i for i in range(60,64)],
                                16:[13,45],
                                17:[i for i in range(10,16) if i!=13 ]+[i for i in range(38,50) if i!=45 ],
                                18:[i for i in range(6,10)]+[i for i in range(34,38)],19:[i for i in range(2,6)]+[i for i in range(30, 34)]
                                }
        self.feature_ids = None
        self.instant_feature_ids = None
        self.CTM_est_state = []
        self.end_of_Gmin = False
        self.att_success_idx = 0
        self.att_success_prob = 0.0 
        self.CTM_state_collect_debug = {'real':[],'CTM':[],'time':[],'phase':[]}

        self.fake_vehicle_num = 0
        self.failed_JSMA_cnt = 0
        self.num_JSMA_cnt = 0
        self.num_flip_cnt = 0

        self.last4phase_delay = collections.deque(maxlen=4)
        
        # --- REWARD & TIME TRACKING INITIALIZATION ---
        self.last_delay = 0
        self.last_action_time = 0
        self.accumulated_delay_reward = 0.0
        
        # New Buffers for Delayed Experience Storage
        self.pending_experience = None    # Holds Exp from (T-1)
        self.current_cycle_exp = None     # Holds Exp from (T)
        self.last_reward_time = 0         # Tracks time for dt calculation
        
        self.fake_traj_dict_allT = {} 
        self.s_eff = 0
        
        self.fake_veh_gen_rate = 0.0

        self.interval_delay_energy = 0
        self.last_tsc_dist = None

        self.incoming_lanes_set = set(self.incoming_lanes)
        self.lane_speed_limits = {l: self.netdata['lane'][l].get('speed', 17.88) for l in self.incoming_lanes}

        self.cumulative_impact = 0.0
        self.impact_steps = 0
        self.phase_attack_successful = False

        self.red_phase_status = {}
        self.initial_attack_time = 0
        self.num_opt_cnt = 0 

        self.t_fakeTraj_duration = 0
    
    def feature_to_phase(self, feature):
        range_dict = {(10, 12): 0,
                    (13, 15): 1,
                    (16, 16): 2,
                    (17, 19): 3
                }
        for (start, end), value in range_dict.items():
            if start <= feature <= end:
                return value
        return None # Or raise an error
        

    def next_phase(self):
        if len(self.phase_deque) == 0:
            next_phase = self.get_next_phase()
            phases = self.get_intermediate_phases(self.phase, next_phase)
            self.phase_deque.extend(phases+[next_phase])
        return self.phase_deque.popleft()

    def next_phase_duration(self,current_phase):
        if self.phase in self.green_phases:
            if self.phase == current_phase:
                return self.green_t_cnt
            else:
                if self.phase == 'rrrGrrrrrrrrGrrrrr':
                    return 30 
                else:
                    return self.green_t
        elif 'y' in self.phase:
            return self.yellow_t
        else:
            return self.red_t

    def proceed_phase(self):
        self.curr_phase_idx += 1
        if self.curr_phase_idx >= len(self.green_phases):
            self.curr_phase_idx = 0    

    # --- INSTRUMENTED UPDATE FUNCTION ---
    def update(self, data, cv_data, uv_data, mask): 
        # t_start = time.time()
        self.data = data
        self.cv_data = cv_data
        self.uv_data = uv_data

        if int(self.t) % 10 == 0:
            instant_delay_sum = 0.0
            
            # Iterate only through lanes we care about (Intersection Incoming)
            # Using items() directly avoids repeated dictionary lookups
            for lane_id, vehicles in self.data.items():
                if lane_id not in self.incoming_lanes_set:
                    continue
                
                # Use cached speed limit
                speed_limit = self.lane_speed_limits.get(lane_id, 17.88)
                if speed_limit <= 0: continue

                # Inner loop optimization
                for veh_info in vehicles.values():
                    speed = veh_info[64] #64: traci.constants.VAR_SPEED
                    # Direct math is faster than max(0, ...) function call
                    if speed < speed_limit:
                        instant_delay_sum += (1.0 - (speed / speed_limit))
            
            self.interval_delay_energy += (instant_delay_sum * 10)
        
        # t_end = time.time()
        # Only warn if update takes > 5ms (it should be microseconds)
        # if (t_end - t_start) > 0.005:
        #     print(f"DEBUG_TIMER: update() SLOW at step {self.t}: {t_end - t_start:.6f}s")


    def get_next_phase(self):
        if self.empty_intersection():
            pass
        else:
            if self.phase == self.all_red and not self.delay_green:
                self.delay_green = True
                self.acting = False
                return self.all_red
            self.delay_green = False
            
            # --- OPTIMIZATION START: Lazy Init for Cache ---
            if not hasattr(self, 'green_history_cache'):
                self.green_history_cache = {}
                self.green_history_last_t = {}
            if not hasattr(self, 'last_traj_update_idx'):
                self.last_traj_update_idx = -1

            curr_t_idx = self.t // 10
            
            # --- CHECK: Only update if the integer second has changed ---
            if curr_t_idx != self.last_traj_update_idx:
                
                # Reset the main dictionary for the new time step
                self.fake_traj_dict = {}
                
                if self.feature_ids is not None:
                    for idx in self.feature_ids:
                        if idx not in self.fake_traj_dict_allT: continue
                            
                        is_red = self.red_phase_status.get(idx, False)
                        
                        if is_red:
                            # [RED PHASE]
                            # Clear Green cache for this feature as we are now Red
                            if idx in self.green_history_cache:
                                del self.green_history_cache[idx]
                                del self.green_history_last_t[idx]
                                
                            # Fetch ONLY current snapshot
                            if curr_t_idx in self.fake_traj_dict_allT[idx]:
                                current_data = self.fake_traj_dict_allT[idx][curr_t_idx]
                                for lane, vehs in current_data.items():
                                    if lane not in self.fake_traj_dict: self.fake_traj_dict[lane] = {}
                                    self.fake_traj_dict[lane].update(vehs)
                                    
                        else:
                            # [GREEN PHASE] - Incremental Update
                            if idx not in self.green_history_cache:
                                self.green_history_cache[idx] = {}
                                self.green_history_last_t[idx] = -1

                            last_t = self.green_history_last_t[idx]
                            
                            # Determine fetch range
                            if last_t == curr_t_idx - 1:
                                steps_to_fetch = [curr_t_idx] # Just the new second
                            else:
                                steps_to_fetch = range(self.initial_attack_time, curr_t_idx + 1) # Full rebuild
                                self.green_history_cache[idx] = {} 

                            # Update Cache
                            for t in steps_to_fetch:
                                if t in self.fake_traj_dict_allT[idx]:
                                    step_data = self.fake_traj_dict_allT[idx][t]
                                    for lane, vehs in step_data.items():
                                        if lane not in self.green_history_cache[idx]: 
                                            self.green_history_cache[idx][lane] = {}
                                        self.green_history_cache[idx][lane].update(vehs)
                            
                            self.green_history_last_t[idx] = curr_t_idx
                            
                            # Merge Cache to Result
                            for lane, vehs in self.green_history_cache[idx].items():
                                if lane not in self.fake_traj_dict: self.fake_traj_dict[lane] = {}
                                self.fake_traj_dict[lane].update(vehs)

                # Update the timestamp so we skip this block for the next ~9 steps (0.9s)
                self.last_traj_update_idx = curr_t_idx
            
            else:
                # [SKIP] Reuse existing self.fake_traj_dict from previous call
                pass

            # print("self.fake_traj_dict:", self.fake_traj_dict)

            # ---------------------------------------------------------
            # 1. STATE GENERATION
            # ---------------------------------------------------------
            if self.tsc_type in ['cavlight','mmitiss']:
                state_global, state_local, state_real_cv = self.get_state(self.tsc_type, num_segments=self.num_segments, act_ctm =self.act_ctm)
                state_global = np.concatenate([state_global,
                                               self.phase_to_one_hot[self.phase],
                                               np.array([self.phase_duration/self.g_max])]) 
                state_local = np.concatenate([state_local,
                                              self.phase_to_one_hot[self.phase],
                                              np.array([self.phase_duration/self.g_max])])
                state_real_cv = np.concatenate([state_real_cv,
                                              self.phase_to_one_hot[self.phase],
                                              np.array([self.phase_duration/self.g_max])])
                state = [state_global,state_local]
            else:
                state = np.concatenate( [self.get_state(self.tsc_type, num_segments=self.num_segments), self.phase_to_one_hot[self.phase]] )
            
            # ---------------------------------------------------------
            # 2. PROBABILITY MONITORING (COUNTERFACTUAL)
            # ---------------------------------------------------------
            # Query the victim with the CLEAN state (No Attack)
            # Use epsilon=1e-5 for deterministic output of the victim's preference
            # use surrogate model to calculate clean action
            self.rlagent.init_attacker()
            if self.mode == 'test':
                clean_ret = self.rlagent.get_action(state_real_cv, 1e-5, False) # Use Real model for TSC decsion
            elif self.mode == 'train':
                clean_ret = self.rlagent.get_action(state_real_cv, 1e-5, True) # True to use surrogate model for TSC decsion
            else:
                raise ValueError("Wrong mode parameter!")

            clean_idx = clean_ret[0]
            clean_dist = clean_ret[1] # Probability distribution [P_stay, P_switch]

            # action_ret_clean_real = self.rlagent.get_action(state[1], 1e-5, False)
            # clean_dist_real = action_ret_clean_real[1]

            # ---------------------------------------------------------
            # 3. ACTUAL VICTIM RESPONSE (WITH ATTACK)
            # ---------------------------------------------------------
            if self.acting:
                # Apply the attack vector to generate the perturbed state
                input_state = self.attacker.get_attacked_state(
                    state[1], 
                    self.att_action, 
                    self.norm_CV, 
                    self.norm_V_spd, 
                    self.fake_vehicle_num, 
                    jsma_features = self.feature_ids
                ) 
            else:
                input_state = state[1].copy()

            # input_state = state[1].copy()

            # # Attack impact calculation. mkeep this part to see if we want to use surrogate model for flip in testing.
            # if self.mode == 'test':
            #     action_ret_surr = self.rlagent.get_action(input_state, 1e-5, False) # False to use real model for TSC decsion
            # elif self.mode == 'train':
            #     action_ret_surr = self.rlagent.get_action(input_state, 1e-5, True) # True to use surrogate model for TSC decsion
            # else:
            #     raise ValueError("Wrong mode parameter!")

            # action_idx_surr = action_ret_surr[0]
            # action_dist_surr = action_ret_surr[1]

            # Query the victim with the ATTACKED state
            if self.mode == 'test':
                action_ret = self.rlagent.get_action(input_state, 1e-5, False) # Should be False to call the real model 
                action_idx = action_ret[0]
                action_dist = action_ret[1]
                self.a_dist = action_dist
            elif self.mode == 'train':
                action_ret = self.rlagent.get_action(input_state, 1e-5, True) 
                action_idx = action_ret[0]
                action_dist = action_ret[1]
                self.a_dist = action_dist
            else:
                raise ValueError("Wrong mode parameter!")

            # print("surrogate clean:",clean_dist,"surrogate attacked:",action_dist_surr,
            #       "\n real clean:",clean_dist_real,"real attacked:",action_dist) ###            
            # ---------------------------------------------------------
            # 4. REWARD CALCULATION & STORAGE
            # ---------------------------------------------------------
            if self.phase == 'rrrGrrrrrrrrGrrrrr':
                minG = 30
            else:
                minG = 100
                
            if (self.phase_duration >= minG) and ( not self.end_of_Gmin):

                self.CTM_state_cmp.append([self.state_comparison,self.CTM.state_comparison_CTM])

                if self.att_action is not None:
                    self.att_success_idx = 1 if action_idx == self.att_action[0] else 0
                    target_act = self.att_action[0]
                else:
                    self.att_success_idx = 0
                    target_act = None
                
                if self.last_tsc_dist is not None:
                    self.att_success_prob = self.last_tsc_dist[target_act]
                else:
                    self.att_success_prob = 0.0

                if self.pending_experience is not None:
                    old_exp = self.pending_experience
                    current_time = self.t 
                    dt = current_time - self.last_reward_time
                    if dt <= 0: dt = 1.0 
                    
                    avg_delay = self.accumulated_delay_reward / dt
                    
                    # === COMPUTE AGGREGATES ===
                    # Calculate Average Impact over the phase
                    if self.impact_steps > 0:
                        phase_avg_impact = self.cumulative_impact / self.impact_steps
                    else:
                        phase_avg_impact = 0.0
                    
                    # Check if we ever flipped during the phase
                    phase_was_flip = self.phase_attack_successful
                    # ==========================

                    # Send to Reward Function
                    # We pass 'phase_was_flip' as 'att_success_idx' to enable the Gate/Bonus
                    r_delay, r_jsma = self.attacker.get_reward(
                        old_exp['a'], 
                        avg_delay, 
                        old_exp['fake_veh_gen_rate'], 
                        att_success_idx = 1 if phase_was_flip else 0, # <--- KEY CHANGE
                        s_eff = old_exp['s_eff'],
                        success_prob = 0.0,
                        impact_factor = phase_avg_impact # <--- KEY CHANGE
                    )

                    if self.current_cycle_exp is not None:
                         self.attacker.store_experience(
                            old_exp['s'],        
                            old_exp['a'],        
                            self.current_cycle_exp['s'],          
                            r_delay,        
                            r_jsma,
                            False,               
                            old_exp['s_eff']  
                        )
                
                # RESET ACCUMULATORS FOR NEXT PHASE
                self.cumulative_impact = 0.0
                self.impact_steps = 0
                self.phase_attack_successful = False
                self.accumulated_delay_reward = 0.0
                self.last_reward_time = self.t
                
                if self.current_cycle_exp is not None:
                    self.current_cycle_exp['att_success_idx'] = self.att_success_idx
                    self.current_cycle_exp['att_success_prob'] = self.att_success_prob
                    self.pending_experience = self.current_cycle_exp.copy()

                self.end_of_Gmin = True

            
            # Accumulate interval energy (delay)
            self.accumulated_delay_reward += self.interval_delay_energy 
            self.interval_delay_energy = 0.0


            # ---------------------------------------------------------
            # 4. CALCULATE METRICS (Flip & Impact)
            # ---------------------------------------------------------
            if self.acting:
                # Metric A: Did we flip the switch? (Binary)
                # True if the Attack forced a different action than the Baseline
                current_step_is_flip = (clean_idx != action_idx)
            
                # LATCH: If we flipped it now, mark the whole phase as successful.
                # Once True, it stays True until reset.
                if current_step_is_flip:
                    if not self.phase_attack_successful:
                        self.num_flip_cnt += 1 
                    self.phase_attack_successful = True
                # if self.phase_attack_successful and (len(self.feature_ids)<=0):
                #     print("np.array(state_real_cv) - np.array(state[1])",np.array(state_real_cv) - np.array(state[1]))
                #     print(f"--- TSC Policy Analysis (Step {self.t}) ---")
                #     print(f"Clean Dist (P_stay, P_switch): {clean_dist}")
                #     print(f"Attacked Dist (P_stay, P_switch): {action_dist_surr}")
                #     print(f"Confidence Drop: {clean_dist[clean_idx] - action_dist_surr[clean_idx]:.4f}")
                #     print(f"Action Flipped: {clean_idx != action_idx_surr}")
                #     raise ValueError("NO JSMA BUT ATTACK SUCCESS!")

                # Metric B: How much did we damage confidence in the ORIGINAL choice? (Continuous)
                # We track the probability of the CLEAN action.
                prob_clean_original = clean_dist[clean_idx]
                prob_clean_attacked = action_dist[clean_idx]
                
                # Drop = Damage. Max(0) ensures we don't reward making the victim MORE confident.
                step_impact = max(0.0, prob_clean_original - prob_clean_attacked)

                # debug
                # print(f"--- TSC Policy Analysis (Step {self.t}) ---")
                # print(f"Clean Dist (P_stay, P_switch): {clean_dist}")
                # print(f"Attacked Dist (P_stay, P_switch): {action_dist_surr}")
                # print(f"Confidence Drop: {clean_dist[clean_idx] - action_dist_surr[clean_idx]:.4f}")
                # print(f"Action Flipped: {clean_idx != action_idx_surr}")
            else:
                step_impact = 0.0
            
            # Accumulate metrics for the current phase duration
            self.cumulative_impact += step_impact
            self.impact_steps += 1
            
            # ---------------------------------------------------------
            # 6. TSC DECISION EXECUTION
            # ---------------------------------------------------------
            
            if (self.phase_duration > self.g_max) and (action_idx == 1):
                self.proceed_phase()
                next_phase = self.int_to_phase[self.curr_phase_idx]
                self.exceed_gmax = True
            else:
                self.exceed_gmax = False
                if action_idx == 1:
                    next_phase = self.int_to_phase[self.curr_phase_idx]
                else:
                    self.proceed_phase()
                    next_phase = self.int_to_phase[self.curr_phase_idx]

            # ---------------------------------------------------------
            # 7. ATTACK GENERATION (New Cycle)
            # ---------------------------------------------------------
            if (action_idx == 0) or self.exceed_gmax:
                self.end_of_Gmin = False

                self.last_tsc_dist = action_dist
                self.attack_phase_dist.append([self.phase, self.phase_duration])

                if next_phase == 'rrrGrrrrrrrrGrrrrr':
                    next_phase_minG = 3
                else:
                    next_phase_minG = 10

                next_phase_idx = self.curr_phase_idx 
                self.CTM_est_state = np.concatenate( [self.CTM.get_state_CTM(int(self.CTM.t_next_Gmin_end)),  self.phase_to_one_hot[next_phase], np.array([next_phase_minG*10/self.g_max])])

                # wo CTM: replace self.CTM_est_state with state[1]
                att_state = self.attacker.get_state(state[1], self.CTM_est_state) 
                # att_state = self.attacker.get_state(state[1], state[1]) 
                
                self.att_action = self.attacker.rlagent.get_action(att_state, self.epsilon, self.curr_phase_idx) 

                self.s = att_state
                self.a = self.att_action.copy()
                
                self.current_cycle_exp = {
                    "s": self.s,
                    "a": self.a,
                    "fake_veh_gen_rate": 0.0, 
                    "att_success_idx": 0,
                    "s_eff": 1.0,               
                    "att_success_prob": 0.0
                }

                delay_record = self.get_reward_delay(tsc_type="cavlight")
                self.state_action_record.append((self.t, delay_record, action_idx, self.a_dist, [state[1], input_state], self.phase, self.att_action, next_phase))
                # wo CTM: replace CTM_est_state with state[1]
                self.adv_x_guide, self.feature_ids, target_action = self.rlagent.get_advX(self.CTM_est_state, next_phase_idx, self.att_action)
                # self.adv_x_guide, self.feature_ids, target_action = self.rlagent.get_advX(state[1], next_phase_idx, self.att_action)

                print("JSMA select feature",self.feature_ids) 
                # self.feature_ids = [] ## benign case

                self.num_JSMA_cnt += 1
                
                if len(self.feature_ids) <= 0:
                    # print("No JSMA feature found") <--- COMMENTED OUT
                    self.failed_JSMA_cnt += 1
                    self.s_eff = 0
                else:
                    self.s_eff = 1
                
                self.current_cycle_exp["s_eff"] = self.s_eff
                fail_JSMA_rate = round(self.failed_JSMA_cnt/self.num_JSMA_cnt,2)
                flip_rate = round(self.num_flip_cnt/self.num_JSMA_cnt,2) 
                opt_succ_rate = round(self.num_opt_cnt/self.num_JSMA_cnt,2)
                print("Failed JSMA rate",  fail_JSMA_rate, "total flip rate", flip_rate,"Opt success rate",opt_succ_rate)
                
                

                # (Fake Vehicle Generation Loop - Omitted for brevity, logic remains same)
                fake_vehicle_dict = {}
                filled_spot_record = {} 
                initial_dist_ls = []
                self.red_phase_status = {}
                self.fake_traj_dict_allT = {}

                # TODO: update this section to fit the 2-features framework. for each feature, we should run one round of optimization
                # combine the two feature's fake veh trajectories given each time step
                self.t_fakeTraj_duration = []
                for idx in self.feature_ids:
                    if self.feature_to_phase(idx) != self.curr_phase_idx:
                        # if next phase is not the phase where fake vehicle is inserted, here the curr_phase is already the next phase
                        red_phase_flag = True
                    else:
                        red_phase_flag = False


                    for _fake_veh in range(15): 
                        fake_veh_id = 'fake_'+str(idx)+str(_fake_veh)+str(self.t)
                        cell_ls = self.feature2cells[idx]

                        final_dist2bar, final_spd, final_lane = self.fake_final_state_gen(cell_ls, filled_spot_record)
                        
                        if final_dist2bar == False:
                            pass 
                        
                        else:
                            initial_dist2bar, initial_spd = self.get_initial_fake_state(final_dist2bar, final_spd, int(self.CTM.t_next_Gmin_end)-int(self.t/10)) 
                            initial_dist2bar = self.adjust_ini_pos(initial_dist2bar, initial_dist_ls)
                            initial_dist_ls.append(initial_dist2bar)

                            fake_vehicle_dict[fake_veh_id] = [initial_dist2bar, initial_spd, final_dist2bar, final_spd, final_lane]

                    fake_vehicle_dict = self.align_initial_with_final(fake_vehicle_dict)

                    opt_start = time.time()
                    self.fake_traj_dict_allT[idx] = optimization_process(fake_vehicle_dict, 
                                                attack_time_gap = min(15,int(self.CTM.t_next_Gmin_end)-self.t//10), 
                                                extend_time_gap = 20, initial_attack_time = self.t//10, red_phase=red_phase_flag)
                    # self.fake_traj_dict_allT[idx] = {} 
                    opt_time = time.time() - opt_start
                    # print("opt_time",opt_time)
                    
                    self.initial_attack_time = self.t//10
                    self.red_phase_status[idx] = red_phase_flag
                    self.fake_veh_traj_input.append([fake_vehicle_dict,red_phase_flag]) 
                
                    self.t_fakeTraj_duration.append(opt_time)
                
                for idx in self.feature_ids:
                    if len(self.fake_traj_dict_allT[idx])>0:
                        self.num_opt_cnt += 1
                        break

                self.fake_veh_gen_rate = len(fake_vehicle_dict.keys())/self.max_attack_scale 
                self.current_cycle_exp["fake_veh_gen_rate"] = self.fake_veh_gen_rate
                self.fake_vehicle_num = len(fake_vehicle_dict.keys())  
                print("fake vehicle num:",self.fake_vehicle_num)

                self.JSMA_result.append([self.feature_ids, self.att_action, self.curr_phase_idx,self.adv_x_guide,self.CTM_est_state,self.instant_feature_ids, fail_JSMA_rate, flip_rate, opt_succ_rate, self.fake_veh_gen_rate,self.phase_attack_successful, self.t_fakeTraj_duration])

                

            self.last_tsc_action = action_idx
            self.acting = True

            return next_phase

    
    def align_initial_with_final(self, fake_vehicle_dict):
        df = pd.DataFrame.from_dict(
            fake_vehicle_dict,
            orient="index",
            columns=["initial_dist2bar", "initial_spd", "final_dist2bar", "final_spd", "final_lane"]
        )

        result = df.copy()

        for lane, sub in df.groupby("final_lane"):
            initial_pairs_sorted = sub.sort_values("initial_dist2bar", ascending=False)[["initial_dist2bar", "initial_spd"]].to_numpy()
            sub_sorted_final = sub.sort_values("final_dist2bar", ascending=False).copy()
            sub_sorted_final[["initial_dist2bar", "initial_spd"]] = initial_pairs_sorted
            result.loc[sub_sorted_final.index, ["initial_dist2bar", "initial_spd"]] = \
                sub_sorted_final[["initial_dist2bar", "initial_spd"]].values

        out_dict = {
            vid: [
                float(row.initial_dist2bar),
                float(row.initial_spd),
                float(row.final_dist2bar),
                float(row.final_spd),
                row.final_lane
            ]
            for vid, row in result.iterrows()
        }

        return out_dict
    
    def adjust_ini_pos(self, initial_dist2bar, initial_dist_ls):
        if not initial_dist_ls: 
            return initial_dist2bar

        closest = min(initial_dist_ls, key=lambda x: abs(x - initial_dist2bar))
        diff = abs(initial_dist2bar - closest)

        if diff >= 6:
            return initial_dist2bar  

        candidate = initial_dist2bar
        while any(abs(candidate - x) < 6 for x in initial_dist_ls):
            candidate -= 6  

        return candidate

    
    def get_initial_fake_state(self,final_dist2bar, final_spd, t_duration):
        initial_spd = 17.88
        t_duration = max(15, t_duration) 
        initial_dist2bar = (initial_spd+final_spd)/2*t_duration+final_dist2bar 

        return initial_dist2bar, initial_spd

    
    def fake_final_state_gen(self, cell_ls, filled_spot_record):
        available_cells = cell_ls[:]

        while available_cells:
            select_cell_idx = random.choice(available_cells)

            final_dist2bar, final_spd, final_lane = self.CTM.get_fake_veh_final_state(
                select_cell_idx, int(self.CTM.t_next_Gmin_end)
            )
            assigned_dist = self.assign_spot(
                filled_spot_record, select_cell_idx, final_lane, final_dist2bar
            )

            if assigned_dist is not False:
                return assigned_dist, final_spd, final_lane
            else:
                available_cells.remove(select_cell_idx)

        return False, None, None

    
    def assign_spot(self, record, cell, lane, dist):
        lane_spots = record.setdefault(cell, {}).setdefault(lane, [])

        for offset in [0, -5, +5]:
            candidate = dist + offset
            if candidate not in lane_spots:
                lane_spots.append(candidate)
                return candidate

        return False  
    
    def get_att_state(self, att_state, ori_state, norm_list, feature_ids, theta=10):
        if len(feature_ids[0]) == 0:
            return ori_state
        
        else:
            denorm_state_spd = (ori_state[:9]-0.2)*norm_list[0]
            denorm_state_Nveh_in = ori_state[9:18]*norm_list[1]
            denorm_state_Nveh_out = ori_state[18:21]*norm_list[2]

            for idx in np.array(feature_ids).reshape(-1):
                if idx<9:
                    denorm_state_spd[idx] = denorm_state_spd[idx] + theta
                elif idx<18:
                    denorm_state_Nveh_in[idx-9] = denorm_state_Nveh_in[idx-9] + theta
                elif idx<21:
                    denorm_state_Nveh_out[idx-18] = denorm_state_Nveh_out[idx-18] + theta

            norm_spd = np.linalg.norm(denorm_state_spd) if np.linalg.norm(denorm_state_spd) > 0 else 1
            norm_Nveh_in = np.linalg.norm(denorm_state_Nveh_in) if np.linalg.norm(denorm_state_Nveh_in) > 0  else 1
            norm_Nveh_out = np.linalg.norm(denorm_state_Nveh_out) if np.linalg.norm(denorm_state_Nveh_out) > 0  else 1

            return np.concatenate([denorm_state_spd/norm_spd+0.2, 
                                denorm_state_Nveh_in/norm_Nveh_in, 
                                denorm_state_Nveh_out/norm_Nveh_out,
                                ori_state[21:]])