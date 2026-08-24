import os
import time  # <--- Added Import
import random
import numpy as np
from itertools import cycle
from collections import deque
import pandas as pd
import collections
import traci

from src.trafficsignalcontroller import TrafficSignalController, LEFT_TURN_PHASES
from src.fake_veh_traj_gen import optimization_process
from src.sdsm_defense import SDSMDefense
from src.occupancy_map import OccupancyMapViz

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
        
        # Attack-record buffers: needed by any attack-capable controller. PressLight's
        # attack subclass (NextPhasePressLightAttackTSC) also runs this __init__ with
        # tsc_type='presslight'; the benign PressLight uses a different class entirely.
        if self.tsc_type in ['cavlight', 'presslight']:
            self.state_action_record = []
            self.state_comparison = []
            self.JSMA_result = []
            self.CTM_state_cmp = []
            self.ctm_pred_log = []   # CTM prediction-accuracy log: per decision, the forward
                                     # prediction for t_pred (=t_next_Gmin_end) + the real-fit
                                     # current counts, so pred@t_pred can be matched to real@t_now.
            self.attack_deviation = []  # per attack cycle: JSMA-DESIGNED state (adv_x_guide) vs the
                                        # REALIZED attacked state[1] at the injected features. Compare
                                        # static (realizes ~= design) vs traj_gen (deviates) offline.
            self.flip_sequence = []     # per attack decision: (clean, target, realized) sequence for
                                        # run-length / clustering analysis (static=long runs of
                                        # sustained flips; traj_gen=short isolated flips).
            self.features_cmp = []
            self.attack_phase_dist = []
            self.fake_veh_traj_input = []
            # FROZEN benign baseline for the marginal-delay reward: mean avg_delay over the
            # first WARMUP benign-equivalent (non-flipped) phases, then FROZEN (never updated).
            # Freezing is the fix — a perpetual EMA drifts up with attack-inflated delay and
            # collapses the marginal to ~0, starving the policy.
            self._benign_delay_sum = 0.0
            self._benign_delay_n = 0
            self._benign_delay_baseline = None
        else:
            self.state_action_record = None

        self.all_veh_r = all_veh_r

        self.green_t_cnt = 10 
        self.act_ctm = act_ctm

        self.args = args

        # -ctm_vf: recalibrate CTM free-flow propagation to the real network speed. self.CTM was
        # built in super().__init__; scale its free-flow SENDING term by (ctm_vf / v_f) (<=1) so the
        # CTM stops discharging vehicles ~14% too fast (v_f=17.88 vs real ~15.65 -> under-prediction).
        _ctm_vf = float(getattr(self.args, 'ctm_vf', 0.0) or 0.0)
        if _ctm_vf > 0 and hasattr(self, 'CTM') and getattr(self.CTM, 'v_f', 0):
            self.CTM.v_ff_ratio = min(1.0, _ctm_vf / self.CTM.v_f)
            print("[ctm_vf] %s: free-flow sending scaled by %.3f (target v_f=%.2f / built-in %.2f)" % (
                tsc_id, self.CTM.v_ff_ratio, _ctm_vf, self.CTM.v_f))

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
        # For corr3 intersections, the Plymouth cell indices above don't exist — build dynamically.
        if getattr(self.CTM, 'ctm_version', '').startswith('corr3_'):
            self.feature2cells = self._build_corr3_feature2cells()
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
        # Surrogate (blackbox) training data: (state[1], action_dist, action_idx) collected
        # during BENIGN test when -collect_sa is set. Saved to _surrogate_data.p by sumosim.
        self.sa_collect = []
        # Gate-2 realization diagnostic: of decisions where the attacker REQUESTS a
        # flip (target != victim's clean action), how often the realized injection
        # actually drives the victim to that target. Printed per decision so a run
        # tee'd to a log yields the JSMA+injection realized-flip rate directly.
        self.req_cnt = 0        # decisions with target != clean_idx
        self.req_succ_cnt = 0   # of those, realized action_idx == target
        # Gate-2b Q-margin probe: mean margin_to_target (Q[target]-Q[clean]) clean vs attacked
        self._qm_clean_sum = 0.0
        self._qm_att_sum = 0.0
        self._qm_n = 0

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

        # --- SDSM DEFENSE ---
        junc_pos = (netdata['node'][tsc_id]['x'], netdata['node'][tsc_id]['y'])
        self.sdsm_defense = SDSMDefense(
            junction_pos=junc_pos,
            detect_range=getattr(args, 'detec_range', 80.0))  # note: dest='detec_range' in argparse
        self.fake_veh_weight = getattr(self.args, 'fake_veh_scale', 1.0)  # multiplier applied to fake vehicle counts
        self.fake_cav_id = None
        self.omap_viz = OccupancyMapViz(tsc_id=tsc_id, out_dir='exp_log/defense')
        self._defense_log = []  # trust score trajectory: list of dicts

        # Build CTM-lane-ID → SUMO-lane-ID mapping.
        # CTM_phase_lane and self.phase_lanes share the same phase string keys.
        # Both are sorted so index i of CTM list corresponds to index i of SUMO list.
        self._ctm_to_sumo_lane = {}
        for phase in self.CTM_phase_lane:
            if phase in self.phase_lanes:
                ctm_ls = sorted(self.CTM_phase_lane[phase])
                sumo_ls = sorted(self.phase_lanes[phase])
                if len(ctm_ls) != len(sumo_ls):
                    print(f'[CTM→SUMO] WARNING phase "{phase}": count mismatch '
                          f'ctm={ctm_ls} sumo={sumo_ls} — mapping all to first SUMO lane')
                    for cl in ctm_ls:
                        self._ctm_to_sumo_lane[cl] = sumo_ls[0]
                else:
                    for cl, sl in zip(ctm_ls, sumo_ls):
                        self._ctm_to_sumo_lane[cl] = sl
        # --- END SDSM DEFENSE ---

    def feature_to_phase(self, feature):
        # corr3: use the per-intersection map built alongside feature2cells so the
        # mapping is correct for both 4-phase (offset 10) and 2-phase (offset 4).
        corr3_map = getattr(self, 'corr3_feature_to_phase', None)
        if corr3_map is not None:
            return corr3_map.get(feature, None)
        range_dict = {(10, 12): 0,
                    (13, 15): 1,
                    (16, 16): 2,
                    (17, 19): 3
                }
        for (start, end), value in range_dict.items():
            if start <= feature <= end:
                return value
        return None # Or raise an error

    def _build_corr3_feature2cells(self):
        """Build state[1] feature-index → CTM cell list for corr3 intersections.

        Mirrors _build_phase_array phase ordering (list(max_pressure_lanes.keys())):
          - left-turn phase → 1 feature slot (all approach segs combined)
          - through phases  → 3 feature slots (one per seg 1,2,3)
        Feature indices start at the CV-count offset of state[1], which is the
        width of the avg_speed block = (n_phases-1)*num_segments + 1. For the
        4-phase intersections this is 10 (matching the old Plymouth layout); for
        the 2-phase 62532012 it is 4. Also populates self.corr3_feature_to_phase
        so feature_to_phase() stays consistent with this per-intersection offset.

        Uses lane→approach lookup via ALL cells (including intersection cells at
        seg=0) so that CTM_phase_lane lanes that only appear on seg=0 cells still
        correctly resolve to the right approach (and thus to its approach cells).
        """
        phases = list(self.max_pressure_lanes.keys())
        left_turn = LEFT_TURN_PHASES.get(self.id)
        if left_turn not in phases:
            left_turn = phases[0]

        # Offset of the attackable CV/inc block within state[1]. CAVLight puts it right
        # after the avg_speed block: (n_phases-1)*num_segments+1. PressLight's phase-based
        # state starts with the inc block, so its subclass sets self._cv_block_offset=0.
        cv_start = getattr(self, '_cv_block_offset', (len(phases) - 1) * self.num_segments + 1)

        # lane → approach_id (built from ALL cells, incl. seg=0)
        lane_to_approach = {}
        for k, info in self.CTM.cell_dict.items():
            app = int(info['approach'])
            for lane in str(info.get('cell2lane', '')).split(','):
                lane = lane.strip()
                if lane and lane != 'nan':
                    lane_to_approach[lane] = app

        # approach_id → seg_idx → [1-indexed cell numbers]
        app_seg_cells = {}
        for k, info in self.CTM.cell_dict.items():
            cell_num = int(k)
            app = int(info['approach'])
            seg = int(info['seg_idx'])
            if app > 0 and seg in (1, 2, 3):
                app_seg_cells.setdefault(app, {}).setdefault(seg, []).append(cell_num)

        feature2cells = {}
        self.corr3_feature_to_phase = {}
        feature_idx = cv_start  # CV-count features start right after avg_speed block

        for phase_idx, phase in enumerate(phases):
            ctm_lanes = self.CTM_phase_lane.get(phase, [])
            approach_ids = set()
            for lane in ctm_lanes:
                app = lane_to_approach.get(lane, 0)
                if app > 0:
                    approach_ids.add(app)

            is_left_turn = (phase == left_turn)
            if is_left_turn:
                # 1 feature slot: aggregate approach cells across all segs
                cells = []
                for app in approach_ids:
                    for seg in (1, 2, 3):
                        cells.extend(app_seg_cells.get(app, {}).get(seg, []))
                if cells:
                    feature2cells[feature_idx] = sorted(set(cells))
                self.corr3_feature_to_phase[feature_idx] = phase_idx
                feature_idx += 1
            else:
                # 3 feature slots: one per segment (seg 1=closest, seg 3=farthest)
                for seg in (1, 2, 3):
                    cells = []
                    for app in approach_ids:
                        cells.extend(app_seg_cells.get(app, {}).get(seg, []))
                    if cells:
                        feature2cells[feature_idx] = sorted(set(cells))
                    self.corr3_feature_to_phase[feature_idx] = phase_idx
                    feature_idx += 1

        # PRESSLIGHT out-injection: the out block sits right after the inc block in state[1]
        # ([inc(A), out(P), ...]). Map each out feature -> that phase's OUTGOING lanes so a
        # JSMA-selected out feature becomes fake DOWNSTREAM congestion (raises out[p] ->
        # lowers pressure[p]). Gated to the phase-based presslight attacker (_cv_block_offset=0);
        # CAVLight has no out block in state[1] so this stays empty for it.
        self.out_feature2lanes = {}
        if getattr(self, '_cv_block_offset', None) == 0:
            out_feat_start = feature_idx   # inc block occupies [0, feature_idx); out follows
            for q, phase in enumerate(phases):
                self.out_feature2lanes[out_feat_start + q] = list(self.max_pressure_lanes[phase].get('out', []))
                self.corr3_feature_to_phase[out_feat_start + q] = q

        return feature2cells

    def _real_current_by_approach(self):
        """Ground-truth CURRENT vehicle counts (cv_data + uv_data = ALL vehicles) per approach,
        within detect range, in the SAME approach order as CTM.get_approach_seg_counts. Lets us
        compare the CTM's CURRENT-state estimate vs reality (isolates the CTM's own calibration from
        its forward-propagation, unlike the pred_log which is CTM-vs-CTM). Returns None on any issue.
        """
        try:
            cd = self.CTM.cell_dict
            app_ids = sorted(set(int(cd[k]['approach']) for k in cd if int(cd[k]['approach']) > 0))
            if not hasattr(self, '_lane_to_app'):
                m = {}
                for k, info in cd.items():
                    a = int(info['approach'])
                    for ln in str(info.get('cell2lane', '')).split(','):
                        ln = ln.strip()
                        if ln and ln != 'nan':
                            m[ln] = a
                self._lane_to_app = m
            cnt = {a: 0 for a in app_ids}
            for data in (getattr(self, 'cv_data', {}) or {}, getattr(self, 'uv_data', {}) or {}):
                for lane, vehs in data.items():
                    a = self._lane_to_app.get(lane)
                    if a is None or a not in cnt:
                        continue
                    for car in vehs:
                        try:
                            d = self.conn.vehicle.getNextTLS(car)
                            pos = d[0][2] if d else 0.0
                        except Exception:
                            pos = 0.0
                        if pos <= self.detect_radius:
                            cnt[a] += 1
            return np.array([cnt[a] for a in app_ids], dtype=float)
        except Exception:
            return None

    def build_future_state1(self, t_future, state1):
        """Attack-timeline fix (-future_jsma): project the CURRENT victim state[1] forward to the
        next decision time `t_future` (= t_next_Gmin_end) using the CTM, so JSMA targets the state
        the victim will ACTUALLY decide on — the same timeline as e9bd9b9 (CTM estimates the future
        state, JSMA targets it, the trajectory realizes fakes to arrive by that decision).

        The old design ran JSMA on `CTM_est_state` directly because the victim's actor consumed the
        CTM-state format. The corr3 victim's actor consumes state[1] = [inc(segmented), out(perphase)]
        (a different shape), so we must project INTO that shape instead of feeding raw CTM_est_state.

        Method (multiplicative, shape/scale/index-robust): for each INC feature slot, apply the CTM's
        predicted per-slot GROWTH ratio (sum of n_i_t over the slot's cells at t_future vs now) to the
        current normalized inc value, then L2-renormalize the inc block (matching get_num_vehicle_cav).
        The ratio is dimensionless, so it is invariant to the CTM-total-vs-CV scale and cancels the
        normalization constant. The OUT block is kept as-is (out lanes are downstream; the CTM does
        not project them here) — documented limitation. Never raises: on any error returns state1.

        LIMITATION: a slot that is EMPTY now (inc≈0) but fills by t_future is not captured (0*ratio=0);
        queues that GROW from a non-empty base (the usual red-phase accumulation the attack targets)
        are captured. Refine to an additive model if the A/B shows this matters.
        """
        try:
            s = np.asarray(state1, dtype=float).copy()
            f2c = getattr(self, 'feature2cells', None)
            n_i_t = getattr(self.CTM, 'n_i_t', None)
            if not f2c or n_i_t is None:
                return s
            # inc block = feature slots before the out block; out_feature2lanes keys start the out block
            out_keys = getattr(self, 'out_feature2lanes', {}) or {}
            n_inc = min(out_keys.keys()) if out_keys else len(s)
            n_inc = int(min(n_inc, len(s)))
            T = n_i_t.shape[0]; C = n_i_t.shape[1]
            t_now = int(self.t // 10)
            t_fut = int(t_future)
            # clamp times into the projected CTM horizon
            if not (0 <= t_now < T) or not (0 <= t_fut < T):
                return s
            inc = s[:n_inc]
            out = s[n_inc:]
            ratio = np.ones(n_inc, dtype=float)
            for idx in range(n_inc):
                cells = f2c.get(idx)
                if not cells:
                    continue
                cols = [c - 1 for c in cells if 0 <= (c - 1) < C]   # cell_num (1-idx) -> n_i_t col
                if not cols:
                    continue
                now = float(np.sum(n_i_t[t_now, cols]))
                fut = float(np.sum(n_i_t[t_fut, cols]))
                if now > 1e-6:
                    ratio[idx] = fut / now
                # else: empty-now slot -> leave ratio 1.0 (inc≈0 stays ≈0; documented limitation)
            inc_fut = inc * ratio
            nrm = np.linalg.norm(inc_fut)
            if nrm > 0:
                inc_fut = inc_fut / nrm
            future = np.concatenate([inc_fut, out])
            if future.shape != s.shape:
                return s
            return future
        except Exception as _e:
            print("[future_jsma] build_future_state1 failed, using current state[1]:", _e)
            return np.asarray(state1, dtype=float)

    def build_future_state1_cavlight(self, t_future, state1):
        """CAVLight attack-timeline fix (-future_jsma): reconstruct the victim actor state[1] at the
        next decision time `t_future` from the CTM future projection, so JSMA targets the state the
        victim will decide on (e9bd9b9 timeline).

        CAVLight state[1] = concat([ get_avg_speed(actor) , get_num_vehicle_cav(actor,'inc') ]),
        each of width A = len//2 (verified lengths: 4-phase=20/A=10, 2-phase=12/A=6). avg slot j and
        cv slot A+j are the SAME phase-segment, whose CTM cells are feature2cells[A+j]. Transforms
        (from trafficsignalcontroller.get_avg_speed / get_num_vehicle_cav) reproduced exactly:
          - cv_count : sum cell counts over the slot's cells, L2-normalized.
          - avg_speed: mean FD cell speed -> deficit max(0, 17.88 - spd), L2-normalized, + 0.2
                       (the +0.2 hard-coded speed offset). Empty/no-cell slot => free-flow => 0.2.
        Sourced from n_i_t[t_future] = clean, FAKE-FREE baseline (JSMA solves the fake target on it),
        matching how the injection realizes fakes to ARRIVE by t_future. Never raises.

        NOTE: validated against the CURRENT code by construction (an earlier confusion came from
        comparing to PRE-+0.2 stale pickles). Confirm empirically with the -future_jsma A/B.
        """
        try:
            s = np.asarray(state1, dtype=float).copy()
            f2c = getattr(self, 'feature2cells', None)
            n_i_t = getattr(self.CTM, 'n_i_t', None)
            delta_x = getattr(self.CTM, 'delta_x', None)
            if not f2c or n_i_t is None or not delta_x or len(s) % 2 != 0:
                return s
            A = len(s) // 2                     # avg block width == cv block width
            T, C = n_i_t.shape
            t_fut = int(t_future)
            if not (0 <= t_fut < T):
                return s
            cd = self.CTM.cell_dict
            VF = 17.88                          # free-flow ref used by get_avg_speed
            cnt = np.zeros(A); diff = np.zeros(A)
            for j in range(A):
                cells = f2c.get(A + j)          # cv slot A+j == avg slot j (same phase-segment)
                if not cells:
                    continue                    # no cells -> count 0, deficit 0 (=> avg 0.2 after +0.2)
                cols = [c - 1 for c in cells if 0 <= (c - 1) < C]
                if not cols:
                    continue
                cnt[j] = float(np.sum(n_i_t[t_fut, cols]))
                spds = []
                for c in cols:
                    nl = float(cd[str(c + 1)]['num_lane'])
                    if nl <= 0:
                        continue
                    den = n_i_t[t_fut, c] / (nl * delta_x / 1000.0)   # veh/km
                    spds.append(self.CTM.get_cell_speed(den, c))
                if spds:
                    diff[j] = max(0.0, VF - float(np.mean(spds)))
            cv_block = cnt / (np.linalg.norm(cnt) or 1.0)
            avg_block = diff / (np.linalg.norm(diff) or 1.0) + 0.2     # +0.2 hard-coded speed offset
            future = np.concatenate([avg_block, cv_block])
            if future.shape != s.shape:
                return s
            return future
        except Exception as _e:
            print("[future_jsma] build_future_state1_cavlight failed, using current state[1]:", _e)
            return np.asarray(state1, dtype=float)

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


    # ---- victim-specific attack hooks (default = CAVLight / A2C) ----------------
    # Overridden by NextPhasePressLightAttackTSC for the phase-based DQN victim so the
    # shared get_next_phase attack loop works unchanged across victim types.
    def _gen_attack_state(self):
        """Return [state_for_critic, state_local]; state[1] is the attackable local
        state fed to the victim. CAVLight: A2C 3-tuple + phase one-hot + duration."""
        if self.tsc_type in ['cavlight', 'mmitiss']:
            state_global, state_local, _ = self.get_state(self.tsc_type,
                                                          num_segments=self.num_segments,
                                                          act_ctm=self.act_ctm)
            tail = [self.phase_to_one_hot[self.phase], np.array([self.phase_duration / self.g_max])]
            state_global = np.concatenate([state_global] + tail)
            state_local = np.concatenate([state_local] + tail)
            return [state_global, state_local]
        return np.concatenate([self.get_state(self.tsc_type, num_segments=self.num_segments),
                               self.phase_to_one_hot[self.phase]])

    def _clean_action(self):
        """Victim response with fake_veh_weight=0 (unperturbed reference for the
        confidence-drop metric). Returns (clean_idx, clean_dist). CAVLight default."""
        old_wt = self.fake_veh_weight
        self.fake_veh_weight = 0.0
        _, clean_local, _ = self.get_state(self.tsc_type, num_segments=self.num_segments,
                                           act_ctm=self.act_ctm)
        self._clean_state1 = np.asarray(clean_local, dtype=float).copy()  # RAW clean [inc,out] (no fakes)
        clean_local = np.concatenate([clean_local, self.phase_to_one_hot[self.phase],
                                      np.array([self.phase_duration / self.g_max])])
        self.fake_veh_weight = old_wt
        clean_ret = self.rlagent.get_action(clean_local, 1e-5, False)
        return clean_ret[0], clean_ret[1]

    def _attacker_obs(self, state):
        """Attacker's observation of the victim state[1] (+CTM). CAVLight: CTM block
        included (canonical dim 46 when shared)."""
        if getattr(self.args, 'shared_att', False):
            return self.attacker.canonicalize_state(state[1], self.CTM_est_state)
        return self.attacker.get_state(state[1], self.CTM_est_state)

    def _step_impact(self, clean_out, attacked_out, clean_idx):
        """Per-step attack impact = how much the attack changed the victim's benign
        output toward flipping. CAVLight (A2C, softmax policy): drop in the clean
        action's probability. Both outputs are probability vectors here."""
        return max(0.0, float(clean_out[clean_idx]) - float(attacked_out[clean_idx]))

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
                                
                            # Fetch ONLY current snapshot (RED path = STATIC out-injection: one-time,
                            # bounded). TRAJGEN out-injection is routed to the GREEN accumulate/repeat
                            # path instead (see red_phase_status assignment in the out-injection block).
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
            # SDSM DEFENSE: cross-check fake vs real CAV reports
            # ---------------------------------------------------------
            if self.args.sdsm_defense:
                _t0   = time.time()
                _ftd  = getattr(self, 'fake_traj_dict', {})
                _cavs = getattr(self, '_cav_ls', set())
                if not _ftd or not _cavs:
                    self.fake_veh_weight = getattr(self.args, 'fake_veh_scale', 1.0)
                else:
                    # Build live CAV position dict (query TraCI once per step)
                    _t1 = time.time()
                    cav_positions = {}
                    for cav_id in _cavs:
                        try:
                            cav_positions[cav_id] = self.conn.vehicle.getPosition(cav_id)
                        except Exception:
                            pass  # vehicle departed
                    _dt_traci = time.time() - _t1

                    if not cav_positions:
                        self.fake_veh_weight = getattr(self.args, 'fake_veh_scale', 1.0)
                    else:
                        _t2 = time.time()
                        real_sdsm = self.sdsm_defense.build_real_sdsm(
                            self.cv_data, cav_positions)
                        _dt_real_sdsm = time.time() - _t2

                        _t3 = time.time()
                        fake_sdsm, fake_cav_id, fake_cav_pos = \
                            self.sdsm_defense.build_fake_sdsm(
                                _ftd, self.conn, self._ctm_to_sumo_lane)
                        _dt_fake_sdsm = time.time() - _t3

                        if not fake_sdsm:
                            self.fake_veh_weight = getattr(self.args, 'fake_veh_scale', 1.0)
                        else:
                            self.fake_cav_id = fake_cav_id
                            # Include the fake CAV's claimed position so the
                            # defense evaluates it like any other broadcaster.
                            # Type 1 now fires only if fake vehicles fall outside
                            # its stated detect_range; Type 2 catches it when
                            # nearby real CAVs fail to corroborate its reports.
                            cav_positions_with_fake = {
                                **cav_positions, fake_cav_id: fake_cav_pos}
                            merged_sdsm = {**real_sdsm, **fake_sdsm}

                            _t4 = time.time()
                            trust_scores = self.sdsm_defense.run_defense(
                                merged_sdsm, cav_positions_with_fake)
                            _dt_run_defense = time.time() - _t4

                            _t5 = time.time()
                            self.fake_veh_weight = self.sdsm_defense.get_fake_weight(
                                self.fake_cav_id)
                            # Only consider trust scores of verified real CAVs
                            # (those in cav_positions).  trust_scores contains ALL
                            # ever-seen IDs including stale old fake_cav_ids whose
                            # trust is 0 — including them pollutes real_min_trust.
                            real_min = min(
                                (trust_scores.get(k, 1.0) for k in cav_positions),
                                default=1.0)
                            stats = self.sdsm_defense.last_step_stats
                            _dt_viz = 0.0
                            if hasattr(self, 'omap_viz'):
                                _t6 = time.time()
                                self.omap_viz.render(
                                    omap=self.sdsm_defense.omap,
                                    cav_positions=cav_positions_with_fake,
                                    trust_scores=trust_scores,
                                    fake_cav_id=self.fake_cav_id,
                                    fake_veh_weight=self.fake_veh_weight,
                                    timestep=self.t)
                                _dt_viz = time.time() - _t6

                            _dt_total = time.time() - _t0
                            print(f'[SDSMDefense t={self.t}] '
                                  f'traci={_dt_traci*1e3:.1f}ms '
                                  f'real_sdsm={_dt_real_sdsm*1e3:.1f}ms '
                                  f'fake_sdsm={_dt_fake_sdsm*1e3:.1f}ms '
                                  f'run_defense={_dt_run_defense*1e3:.1f}ms '
                                  f'viz={_dt_viz*1e3:.1f}ms '
                                  f'total={_dt_total*1e3:.1f}ms')
                            self._defense_log.append({
                                't': self.t,
                                'fake_trust': trust_scores.get(self.fake_cav_id, 1.0),
                                'real_min_trust': real_min,
                                'fake_weight': self.fake_veh_weight,
                                'n_cav': len(cav_positions),
                                'total_claims': stats['total_claims'],
                                'occupied_cells': stats['occupied_cells'],
                                'n_type1': stats['n_type1'],
                                'n_type2': stats['n_type2'],
                                'penalized': str(stats['penalized']),
                                'all_cav_trust': str({k: round(v, 3)
                                                      for k, v in sorted(trust_scores.items())}),
                            })
            else:
                self.fake_veh_weight = getattr(self.args, 'fake_veh_scale', 1.0)
            # ---------------------------------------------------------
            # END SDSM DEFENSE
            # ---------------------------------------------------------

            # ---------------------------------------------------------
            # 1. STATE GENERATION
            # ---------------------------------------------------------
            # Victim-specific: returns [state_for_critic, state_local] where state[1] is
            # the attackable local state fed to the victim. Overridden per victim type
            # (CAVLight A2C 3-tuple vs PressLight DQN single vector).
            state = self._gen_attack_state()
            
            # ---------------------------------------------------------
            # 2. PROBABILITY MONITORING (COUNTERFACTUAL)
            # ---------------------------------------------------------
            # Only needed when an attacker is present (requires JSMA surrogate model)
            if self.attacker is not None:
                # Clean reference: victim response WITHOUT fake vehicles (fake_veh_weight=0),
                # for the confidence-drop metric. Victim-specific state build is delegated.
                clean_idx, clean_dist = self._clean_action()
            else:
                # benign run: no attacker, no surrogate model needed
                clean_idx = None
                clean_dist = None

            # action_ret_clean_real = self.rlagent.get_action(state[1], 1e-5, False)
            # clean_dist_real = action_ret_clean_real[1]

            # ---------------------------------------------------------
            # 3. ACTUAL VICTIM RESPONSE (WITH ATTACK)
            # ---------------------------------------------------------
            if self.acting and self.attacker is not None:
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
                # -tsc_surrogate: drive the TSC with the SURROGATE (controller-fidelity test);
                # otherwise False = real victim model (default, incl. the blackbox ATTACK test where
                # only JSMA uses the surrogate but the real TSC must decide).
                _use_surr = getattr(self.args, 'tsc_surrogate', False)
                action_ret = self.rlagent.get_action(input_state, 1e-5, _use_surr)
                action_idx = action_ret[0]
                action_dist = action_ret[1]
                self.a_dist = action_dist
            elif self.mode == 'train':
                # False = use real actor model (white-box); surrogate has wrong input dim for corr3
                action_ret = self.rlagent.get_action(input_state, 1e-5, False)
                action_idx = action_ret[0]
                action_dist = action_ret[1]
                self.a_dist = action_dist
            else:
                raise ValueError("Wrong mode parameter!")

            # Surrogate data collection (blackbox prep): log the CLEAN (state -> action) pair the
            # victim produces during BENIGN operation (attacker is None -> input_state == state[1]).
            if getattr(self.args, 'collect_sa', False) and self.attacker is None:
                # Format matches surrogate_model_train convention: (state, action, phase, dist)
                # i[0]=state, i[1]=action_idx, i[2]=phase (as before); i[3]=action_dist (extra,
                # enables soft distillation so the surrogate's decision MARGIN matches the victim).
                self.sa_collect.append((np.asarray(input_state, dtype=np.float32),
                                        int(action_idx),
                                        int(self.curr_phase_idx),
                                        np.asarray(action_dist, dtype=np.float32)))

            # FORCE-FLIP diagnostic (-force_flip): bypass JSMA + injection and directly force
            # the victim to the attacker's REQUESTED target action (self.att_action from the
            # previous cycle). This isolates the delay-reward + policy from the JSMA/injection
            # pipeline: if the attacker converges here, the reward design is sound and JSMA is
            # the bottleneck; if not, the delay signal itself is the problem.
            self._forced_flip = False
            if getattr(self.args, 'force_flip', False) and self.att_action is not None:
                _tgt = int(self.att_action[0])
                self._forced_flip = (_tgt != action_idx)
                action_idx = _tgt
                self.a_dist = action_dist

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
                    # FLIP-CLUSTERING log: per attack decision, the sequence needed to test the
                    # TEMPORAL mechanism. Static should produce LONG RUNS of consecutive on-target
                    # (sustained misallocation); traj_gen should produce SHORT/ISOLATED flips (the
                    # perturbation decays between decisions -> victim recovers). Offline: run-lengths
                    # of consecutive (realized==target) and (realized!=clean).
                    try:
                        # TRAFFIC-TRAJECTORY probe: real per-approach queue (ground-truth cv+uv within
                        # detect range) at THIS decision, so we can track how the arterial queue evolves
                        # over the run and find where static (starves -> queue grows) diverges from
                        # trajgen (recovers -> queue flat). Real traffic, not perceived.
                        _ra = self._real_current_by_approach()
                        self.flip_sequence.append({
                            't': int(self.t),
                            'clean': int(clean_idx) if clean_idx is not None else -1,
                            'target': int(target_act),
                            'realized': int(action_idx),
                            'phase': int(self.curr_phase_idx),
                            'traj_gen': bool(getattr(self.args, 'traj_gen', False)),
                            'real_app': [float(x) for x in _ra] if _ra is not None else None,
                        })
                    except Exception:
                        pass
                    # Gate-2 realized-flip diagnostic (only meaningful when a real flip
                    # is REQUESTED, i.e. target != victim's clean action). action_idx is
                    # the victim's response to the actually-injected state, so this is the
                    # true JSMA+injection realization rate, decoupled from the trajectory.
                    if clean_idx is not None and target_act != clean_idx:
                        self.req_cnt += 1
                        if action_idx == target_act:
                            self.req_succ_cnt += 1
                        # Gate-2b probe: does the fake-vehicle injection actually move the
                        # victim's Q toward the target? margin_to_target = Q[target]-Q[clean].
                        # clean is <0 (clean_idx is argmax); a working attack raises it toward
                        # >0 (flip). att~=clean => injection has ~0 effect (magnitude/plumbing);
                        # att up but still <0 => injection moves Q but too weak to cross margin.
                        try:
                            if clean_dist is not None and action_dist is not None:
                                cq = np.asarray(clean_dist, float).ravel()
                                aq = np.asarray(action_dist, float).ravel()
                                self._qm_clean_sum += float(cq[target_act] - cq[clean_idx])
                                self._qm_att_sum += float(aq[target_act] - aq[clean_idx])
                                self._qm_n += 1
                        except Exception:
                            pass
                        _rr = round(self.req_succ_cnt / max(1, self.req_cnt), 3)
                        print("REALIZE target!=clean:", self.req_succ_cnt, "/", self.req_cnt,
                              "=", _rr, "| clean=", clean_idx, "target=", target_act,
                              "realized=", action_idx)
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

                    # --- marginal-delay reward (opt-in via -marginal_delay) ------------
                    # Score the attack against a benign counterfactual instead of absolute
                    # delay, so flips that REDUCE delay earn NEGATIVE reward. A non-flipped
                    # phase is benign-equivalent (victim's decision == its clean decision;
                    # fakes are perception-only), so those phases define a running benign
                    # baseline (EMA). A flipped phase is scored by its delay ABOVE that
                    # baseline (signed). Removes the perverse incentive where a
                    # delay-reducing attack still scores high on absolute delay.
                    if getattr(self.args, 'marginal_delay', False):
                        WARMUP = 30   # non-flipped phases to average, then FREEZE the baseline
                        if not self.phase_attack_successful:
                            # benign-equivalent phase: accumulate the baseline ONLY during warmup
                            if self._benign_delay_n < WARMUP:
                                self._benign_delay_sum += avg_delay
                                self._benign_delay_n += 1
                                self._benign_delay_baseline = self._benign_delay_sum / self._benign_delay_n
                            delay_for_reward = 0.0
                        else:
                            base = self._benign_delay_baseline if self._benign_delay_baseline is not None else avg_delay
                            delay_for_reward = avg_delay - base   # signed marginal vs FROZEN baseline
                    else:
                        delay_for_reward = avg_delay

                    # === COMPUTE AGGREGATES ===
                    # Calculate Average Impact over the phase
                    if self.impact_steps > 0:
                        phase_avg_impact = self.cumulative_impact / self.impact_steps
                    else:
                        phase_avg_impact = 0.0
                    
                    # Check if we ever flipped during the phase
                    phase_was_flip = self.phase_attack_successful
                    # ==========================

                    # Send to Reward Function (DRL attacker only)
                    # We pass 'phase_was_flip' as 'att_success_idx' to enable the Gate/Bonus
                    if self.attacker is not None:
                        r_delay, r_jsma = self.attacker.get_reward(
                            old_exp['a'],
                            delay_for_reward,
                            old_exp['fake_veh_gen_rate'],
                            att_success_idx = 1 if phase_was_flip else 0,
                            s_eff = old_exp['s_eff'],
                            success_prob = 0.0,
                            impact_factor = phase_avg_impact,
                            marginal = getattr(self.args, 'marginal_delay', False)
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
            if self.acting and self.attacker is not None and clean_idx is not None:
                # Metric A: Did we flip the switch? (Binary)
                current_step_is_flip = (clean_idx != action_idx)
                if current_step_is_flip:
                    if not self.phase_attack_successful:
                        self.num_flip_cnt += 1
                    self.phase_attack_successful = True

                # Metric B: how much the attack changed the victim's benign output toward
                # a flip. Victim-specific measure (A2C = clean-action probability drop;
                # PressLight/DQN = normalized Q-margin erosion — softmax(Q) is not used).
                # FORCE-FLIP: no injection means attacked==clean so _step_impact would be 0;
                # a forced flip IS a full change of output, so credit impact=1.
                if getattr(self, '_forced_flip', False):
                    step_impact = 1.0
                else:
                    step_impact = self._step_impact(clean_dist, action_dist, clean_idx)
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
            _att_model = getattr(self.args, 'att_model', None)
            _rule_based = _att_model in ('minPressure', 'random')
            # Deploy-to-one: rule-based attackers (minPressure/random) don't own an attacker object,
            # so the att_target gating (which sets non-target attackers to None) doesn't reach them —
            # they must honor att_target explicitly, else a "per-target" run attacks EVERY intersection
            # (the bug that made minPressure per-target == simultaneous ~+130%). Empty att_target =
            # attack all (simultaneous), matching the DRL path.
            _target = getattr(self.args, 'att_target', '')
            if _rule_based and _target and str(self.id) != str(_target):
                self.last_tsc_action = action_idx
                return next_phase
            if self.attacker is None and not _rule_based:
                # No attacker instantiated and not using rule-based attack; skip attack entirely
                self.last_tsc_action = action_idx
                return next_phase

            if (action_idx == 0) or self.exceed_gmax:
                self.end_of_Gmin = False

                self.last_tsc_dist = action_dist
                self.attack_phase_dist.append([self.phase, self.phase_duration])

                if next_phase == 'rrrGrrrrrrrrGrrrrr':
                    next_phase_minG = 3
                else:
                    next_phase_minG = 10

                next_phase_idx = self.curr_phase_idx 
                if self.CTM.ctm_version.startswith('corr3_'):
                    ctm_features = self.CTM.get_state_CTM_corr3(int(self.CTM.t_next_Gmin_end))
                else:
                    ctm_features = self.CTM.get_state_CTM(int(self.CTM.t_next_Gmin_end))
                self.CTM_est_state = np.concatenate([ctm_features, self.phase_to_one_hot[next_phase], np.array([next_phase_minG*10/self.g_max])])

                # --- CTM prediction-accuracy log (corr3 only) --------------------------------
                # Per decision, record the CTM's FORWARD prediction for t_pred (=t_next_Gmin_end)
                # AND its real-fit CURRENT counts, in the SAME approach x segment layout (raw
                # counts, no side effects). Offline: match a record's t_pred to a LATER record's
                # t_now to compare predicted-vs-realized -> the CTM forward-prediction accuracy that
                # -future_jsma depends on. Both come from the SAME solve just done above.
                if self.CTM.ctm_version.startswith('corr3_') and hasattr(self, 'ctm_pred_log'):
                    try:
                        _tnow = int(self.t // 10)
                        _tpred = int(self.CTM.t_next_Gmin_end)
                        _ctm_now = self.CTM.get_approach_seg_counts(_tnow)
                        self.ctm_pred_log.append({
                            't_now': _tnow,
                            't_pred': _tpred,
                            'real_now': _ctm_now,                                  # CTM estimate @ now (per app x seg)
                            'pred': self.CTM.get_approach_seg_counts(_tpred),      # CTM forward prediction @ t_pred
                            'phase': int(self.curr_phase_idx),
                            # CTM current-state accuracy (isolates calibration from propagation):
                            'ctm_now_app': _ctm_now.reshape(-1, 3).sum(axis=1),    # CTM current per-approach total
                            'real_truth_app': self._real_current_by_approach(),    # ground truth (cv+uv) per-approach
                        })
                    except Exception:
                        pass

                if getattr(self.args, 'att_model', None) == 'minPressure':
                    # Rule-based attack: per-phase pressure from the VICTIM's actor state[1] CV-count
                    # block (the counts the victim actually perceives), aggregated by the per-
                    # intersection feature->phase map. This is geometry-agnostic: corr3_feature_to_phase
                    # already encodes both the 4-phase layout (CV offset 10) and the 2-phase one (offset
                    # 4), so it works at 62477148/62500824 (P=4) AND 62532012 (P=2) with no hardcoded
                    # slices. (The old [(10,13)..(17,20)] slices were for the ISOLATED get_state_CTM
                    # speed+count state; on corr3 CTM_est_state is counts-only 12-dim -> they indexed the
                    # phase one-hot / out of bounds -> garbage. Use state[1] with the real map instead.)
                    victim_state = state[1]
                    n_phase = len(self.green_phases)
                    # INCOMING CV-count block only: state[1] = avg_speed(A_MAX) + cv_count(A_MAX) + ...,
                    # so counts live at [A_MAX, 2*A_MAX). A_MAX = (P-1)*s+1. Bounding to this range
                    # excludes the out features (also in corr3_feature_to_phase) and the phase one-hot.
                    _s = int(getattr(self, 'num_segments', 3))
                    A_MAX = (n_phase - 1) * _s + 1
                    inc_lo, inc_hi = A_MAX, 2 * A_MAX
                    phase_pressure = [0.0] * n_phase
                    f2p = getattr(self, 'corr3_feature_to_phase', None)
                    if f2p is not None:
                        _items = f2p.items()
                    else:
                        # isolated fallback: original CV slots on the speed+count state
                        _items = ((feat, self.feature_to_phase(feat)) for feat in range(inc_lo, inc_hi))
                    for feat, ph in _items:
                        if ph is not None and 0 <= ph < n_phase and inc_lo <= feat < inc_hi and feat < len(victim_state):
                            phase_pressure[ph] += victim_state[feat]
                    cur = self.curr_phase_idx
                    current_pressure = phase_pressure[cur] if cur < n_phase else 0.0
                    max_other_pressure = max(
                        (p for i, p in enumerate(phase_pressure) if i != cur), default=0.0
                    )
                    # adversarial: switch AWAY from the highest-pressure phase (opposite of max-pressure)
                    mp_action = 0 if current_pressure >= max_other_pressure else 1
                    print(f"[minPressure] phase={cur}/{n_phase} curr={current_pressure:.2f} max_other={max_other_pressure:.2f} action={mp_action}")
                    self.att_action = [mp_action, None, None, mp_action, False]
                    self.s = None
                elif getattr(self.args, 'att_model', None) == 'random':
                    # Rule-based RANDOM baseline: pick the target phase action uniformly at random.
                    # Same JSMA + injection pipeline realizes it (isolates "does a coherent target
                    # selector matter?" vs minPressure/DRL). No training needed.
                    rand_action = random.randint(0, 1)
                    self.att_action = [rand_action, None, None, rand_action, False]
                    self.s = None
                else:
                    # DRL attacker: query the PPO agent for the target phase action
                    # wo CTM: replace self.CTM_est_state with state[1]
                    att_state = self._attacker_obs(state)
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
                # White-box JSMA attacks the cavlight actor, whose input IS state[1].
                # Always pass state[1] for corr3 (26-dim for 4-phase, 12-dim for the
                # 2-phase 62532012); CTM_est_state (13-dim) no longer matches the actor
                # input and would raise a shape error. feature_ids land in the CV-count
                # block, whose offset feature2cells is now built to match per-intersection.
                _is_corr3_jsma = getattr(self.CTM, 'ctm_version', '').startswith('corr3_')
                if getattr(self.args, 'force_flip', False):
                    # FORCE-FLIP: no JSMA, no injection — victim already forced to att_action.
                    self.adv_x_guide, self.feature_ids, target_action = None, np.array([], dtype=int), int(self.att_action[0])
                elif _is_corr3_jsma:
                    # -future_jsma: target the CTM-projected FUTURE victim state at the next decision
                    # time (t_next_Gmin_end) instead of the current state[1] — restores the e9bd9b9
                    # attack timeline. Off by default -> current behavior. Dispatch by victim state
                    # layout: PressLight (_cv_block_offset==0, state[1]=[inc,out]) uses the inc-growth
                    # projection; CAVLight (state[1]=[avg_speed,cv_count]) uses the 2-block CTM
                    # reconstruction. Both fall back to current state[1] on any issue.
                    if getattr(self.args, 'future_jsma', False):
                        t_fut = int(self.CTM.t_next_Gmin_end)
                        if getattr(self, '_cv_block_offset', None) == 0:
                            jsma_state = self.build_future_state1(t_fut, state[1])          # PressLight
                        else:
                            jsma_state = self.build_future_state1_cavlight(t_fut, state[1]) # CAVLight
                    else:
                        jsma_state = state[1]
                    self.adv_x_guide, self.feature_ids, target_action = self.rlagent.get_advX(jsma_state, next_phase_idx, self.att_action)
                else:
                    # non-corr3 (e.g. ISOLATED plymouth): dispatch by victim type. The PressLight victim
                    # (_cv_block_offset==0) is a DQN whose input is its 20-dim state[1] ([inc,out,phase,
                    # time]) — NOT CTM_est_state (26-dim, which mis-sizes the white-box JSMA: P=round(26/5)
                    # =5 instead of 4). CAVLight's isolated actor uses the CTM state.
                    jsma_state = state[1] if getattr(self, '_cv_block_offset', None) == 0 else self.CTM_est_state
                    self.adv_x_guide, self.feature_ids, target_action = self.rlagent.get_advX(jsma_state, next_phase_idx, self.att_action)

                print("JSMA select feature",self.feature_ids)
                # self.feature_ids = [] ## benign case

                # ATTACK-STATE DEVIATION LOG: at the JSMA-selected features, the DESIGNED value
                # (adv_x_guide, what JSMA wants the attacked state to be) vs the REALIZED value
                # (state[1], the attacked state the victim actually decides on -- fakes injected in
                # get_state). Static realizes ~= design; -traj_gen deviates (cannot reach the design).
                # Offline: compare mean |designed - realized| at feature_ids, static vs traj_gen.
                try:
                    if self.adv_x_guide is not None and self.feature_ids is not None and len(self.feature_ids) > 0:
                        _adv = np.asarray(self.adv_x_guide, float).ravel()
                        _st = np.asarray(state[1], float).ravel()
                        _fids = [int(x) for x in self.feature_ids if int(x) < min(len(_adv), len(_st))]
                        # PARADOX PROBE: full per-slot PERCEIVED state (clean vs attacked) so we can see
                        # the DISTRIBUTION across phase slots (per-phase pressure is what the victim
                        # decides on), not just the block sums. Recompute clean here (no fakes) reliably.
                        try:
                            _ow = self.fake_veh_weight; self.fake_veh_weight = 0.0
                            if hasattr(self, '_presslight_local_state'):     # presslight: single-vector build
                                _cl = self._presslight_local_state()
                            else:                                            # cavlight: 3-tuple, state[1]=actor
                                _, _cl, _ = self.get_state(self.tsc_type, num_segments=self.num_segments, act_ctm=self.act_ctm)
                            self.fake_veh_weight = _ow
                            _clean = list(np.asarray(_cl, float))
                        except Exception as _e:
                            self.fake_veh_weight = _ow
                            print("[probe] clean recompute failed:", _e); _clean = None
                        _outk = sorted(k for k in (getattr(self, 'out_feature2lanes', {}) or {}) if k < len(_st))
                        self.attack_deviation.append({
                            't': int(self.t),
                            'feature_ids': _fids,
                            'designed': [float(_adv[i]) for i in _fids],
                            'realized': [float(_st[i]) for i in _fids],
                            'traj_gen': bool(getattr(self.args, 'traj_gen', False)),
                            'att_state': [float(x) for x in _st],       # full attacked state[1]
                            'clean_state': _clean,                       # full clean state[1] (no fakes)
                            'out_idx': _outk,                            # which slots are the OUT block
                        })
                except Exception:
                    pass

                self.num_JSMA_cnt += 1
                
                if getattr(self.args, 'force_flip', False):
                    # FORCE-FLIP: the "attack" succeeds by construction (victim forced), so
                    # s_eff=1 and no injection — isolates the delay reward from JSMA.
                    self.s_eff = 1
                elif len(self.feature_ids) <= 0:
                    # print("No JSMA feature found") <--- COMMENTED OUT
                    self.failed_JSMA_cnt += 1
                    self.s_eff = 0
                    if getattr(self.args, 'jsma_no_fallback', False):
                        # NO heuristic rescue: JSMA found nothing -> skip injection this cycle.
                        # feature_ids stays empty (no fake vehicles) and s_eff stays 0, so the
                        # agent is actually penalized (jsma_penalty=-1, reward_delay gated to 0).
                        # Forces the attacker to learn states/targets where JSMA finds salient
                        # features instead of silently drifting to a blind phase heuristic.
                        self.feature_ids = np.array([], dtype=int)
                    elif _is_corr3_jsma:
                        # For corr3: random-weight JSMA classifier may already predict the
                        # target class → active_indices empty → no features selected.
                        # Fall back using clean_idx to pick injection direction that causes
                        # actual confidence drop in the TSC's real actor:
                        #   clean_idx=0 (stay) → inject into COMPETING approaches so the TSC
                        #     becomes less confident about staying (action_dist[0] drops).
                        #   clean_idx=1 (switch) → inject into CURRENT approach so the TSC
                        #     becomes less confident about switching (action_dist[1] drops).
                        # This ensures step_impact > 0 regardless of what JSMA returns.
                        if clean_idx == 0:
                            # TSC would naturally stay → boost competing to erode that preference
                            phase_feats = [f for f in self.feature2cells
                                           if self.feature_to_phase(f) != self.curr_phase_idx]
                        else:
                            # TSC would naturally switch → boost current to erode that preference
                            phase_feats = [f for f in self.feature2cells
                                           if self.feature_to_phase(f) == self.curr_phase_idx]
                        if not phase_feats:
                            phase_feats = list(self.feature2cells.keys())
                        self.feature_ids = np.array(phase_feats[:2])
                        if len(self.feature_ids) > 0:
                            self.s_eff = 1  # Fallback injection still earns reward
                else:
                    self.s_eff = 1
                
                self.current_cycle_exp["s_eff"] = self.s_eff
                fail_JSMA_rate = round(self.failed_JSMA_cnt/self.num_JSMA_cnt,2)
                flip_rate = round(self.num_flip_cnt/self.num_JSMA_cnt,2) 
                opt_succ_rate = round(self.num_opt_cnt/self.num_JSMA_cnt,2)
                print("Failed JSMA rate",  fail_JSMA_rate, "total flip rate", flip_rate,"Opt success rate",opt_succ_rate)
                # Persist the realization funnel to a file (tee-independent) so we can read
                # the diagnostic after any run. Overwritten each cycle -> final file holds the
                # full-episode cumulative aggregate. Distinguishes the two failure modes:
                #   REQ low            -> attacker rarely requests a flip (policy/off-distribution)
                #   REQ high, realized low -> injection can't realize the requested flip (Gate 2b)
                #   fail_rate high     -> JSMA finds no feature (Gate 2a)
                try:
                    import time as _time
                    with open("exp_log/realize_%s.txt" % self.id, "w") as _rf:
                        _rf.write("id=%s pid=%d updated=%s t_sim=%s\n" % (
                            self.id, os.getpid(),
                            _time.strftime("%Y-%m-%d %H:%M:%S"), getattr(self, 't', '?')))
                        _rf.write("num_JSMA=%d failed_JSMA=%d fail_rate=%.3f\n" % (
                            self.num_JSMA_cnt, self.failed_JSMA_cnt,
                            self.failed_JSMA_cnt / max(1, self.num_JSMA_cnt)))
                        _rf.write("total_flip(clean!=realized)=%d flip_rate=%.3f\n" % (
                            self.num_flip_cnt, self.num_flip_cnt / max(1, self.num_JSMA_cnt)))
                        _rf.write("REQ(target!=clean)=%d REQ_realized(action==target)=%d realized_rate=%.3f\n" % (
                            self.req_cnt, self.req_succ_cnt,
                            self.req_succ_cnt / max(1, self.req_cnt)))
                        _rf.write("Qmargin_to_target clean=%.4f attacked=%.4f (need >0 to flip; n=%d)\n" % (
                            self._qm_clean_sum / max(1, self._qm_n),
                            self._qm_att_sum / max(1, self._qm_n), self._qm_n))
                        _oit = getattr(self, '_opt_in_total', 0); _oif = getattr(self, '_opt_in_fail', 0)
                        _rf.write("INCOMING optimizer(gurobi): total=%d failed=%d fail_rate=%.3f (out-density model has no optimizer)\n" % (
                            _oit, _oif, _oif / max(1, _oit)))
                except Exception:
                    pass
                
                

                # (Fake Vehicle Generation Loop - Omitted for brevity, logic remains same)
                fake_vehicle_dict = {}
                filled_spot_record = {} 
                initial_dist_ls = []
                self.red_phase_status = {}
                self.fake_traj_dict_allT = {}
                # BUGFIX (one-time attack): also clear the green-phase accumulation cache each decision
                # so fake vehicles from PRIOR attack decisions don't carry over (within a decision the
                # trajectory is still rebuilt from fake_traj_dict_allT; across decisions it starts fresh).
                self.green_history_cache = {}
                self.green_history_last_t = {}

                # TODO: update this section to fit the 2-features framework. for each feature, we should run one round of optimization
                # combine the two feature's fake veh trajectories given each time step
                self.t_fakeTraj_duration = []
                _is_corr3 = getattr(self, 'CTM', None) is not None and getattr(self.CTM, 'ctm_version', '').startswith('corr3_')
                for idx in self.feature_ids:
                    # OUT-INJECTION (PressLight): an out-block feature -> fake CVs spoofed onto
                    # the phase's OUTGOING lanes. get_num_vehicle_cav('out') is single-segment
                    # (presence-based), so position is nominal; placement raises out[p] and thus
                    # lowers that phase's pressure. Bypasses the CTM-cell path used for 'inc'.
                    out_lanes = getattr(self, 'out_feature2lanes', {}).get(idx)
                    if out_lanes is not None:
                        if out_lanes:
                            t_start = self.t // 10
                            t_end = t_start + min(15, int(self.CTM.t_next_Gmin_end) - t_start) + 20
                            traj = {}
                            if getattr(self.args, 'traj_gen', False):
                                # TRAJGEN (outgoing): realistic FREE-FLOW stream. Fake CVs enter the
                                # outgoing lane and COAST at the lane's free-flow speed (assume no
                                # downstream obstruction -> they keep moving), lane_pos advancing each
                                # second, dropped once past the lane end. Realistic alternative to the
                                # static/frozen block; the sustained presence (=out count) is whatever
                                # a free-flow stream supports within the detection range.
                                # -out_leave_speed: leaving speed (m/s); 0 => lane free-flow. Model the
                                # outgoing lane as a STANDING QUEUE whose DENSITY rises as speed drops
                                # (car-following: spacing = max(MIN_GAP, v*TIME_GAP)). In a corridor the
                                # downstream signal makes departing vehicles queue, so a low leaving
                                # speed => a dense packed queue filling the detection range => high
                                # out-count; free-flow => sparse. The queue crawls forward at v.
                                v_set = float(getattr(self.args, 'out_leave_speed', 0.0))
                                MIN_GAP = 7.5    # veh length + min standstill gap (m)
                                TIME_GAP = 2.0   # desired time headway (s) at speed
                                det = float(getattr(self, 'detect_radius', 80.0))
                                for ol in out_lanes:
                                    # outgoing lanes may be INTERNAL junction lanes (':...') absent
                                    # from netdata['lane'] -> fall back to free-flow defaults.
                                    _li = self.netdata['lane'].get(ol, {})
                                    v = v_set if v_set > 0 else float(_li.get('speed', 17.88))
                                    v = max(v, 0.5)                          # avoid div-by-0 / frozen
                                    L = float(_li.get('length', 100.0))
                                    span = min(L, det)
                                    spacing = max(MIN_GAP, v * TIME_GAP)     # slower -> denser queue
                                    N = max(1, int(span / spacing))          # CVs present on the lane
                                    for t in range(t_start, t_end):
                                        shift = (v * (t - t_start)) % spacing  # whole queue crawls at v
                                        for i in range(N):
                                            pos = shift + i * spacing
                                            if pos > span:
                                                continue
                                            # consistent id (bounded standing queue) by default; with
                                            # -out_trail append the timestep -> unique id per step so the
                                            # green cache accumulates a DENSE TRAIL of visited positions.
                                            if getattr(self.args, 'out_trail', False):
                                                vid = 'fakeout_q_%s_%d_%d_%d' % (idx, i, self.t, t)
                                            else:
                                                vid = 'fakeout_q_%s_%d_%d' % (idx, i, self.t)
                                            traj.setdefault(t, {}).setdefault(ol, {})[vid] = {"speed": v, "lane_pos": pos}
                            else:
                                # STATIC injection: fake CVs frozen near the lane start (max presence).
                                for _fake_veh in range(15):
                                    ol = out_lanes[_fake_veh % len(out_lanes)]
                                    vid = 'fakeout_' + str(idx) + '_' + str(_fake_veh) + '_' + str(self.t)
                                    for t in range(t_start, t_end):
                                        traj.setdefault(t, {}).setdefault(ol, {})[vid + '_' + str(t)] = {"speed": 5.0, "lane_pos": 10.0}
                            self.fake_traj_dict_allT[idx] = traj
                        self.initial_attack_time = self.t // 10
                        # Out-injection routing:
                        #  - STATIC (per-timestep ids '..._{t}') -> SNAPSHOT path (True): the green cache
                        #    would pile up a NEW id every step at a fixed pos -> stacked ghosts. Snapshot
                        #    keeps it bounded/one-time.
                        #  - TRAJGEN density (CONSISTENT ids '...q_{idx}_{i}_{gen_t}') -> GREEN accumulate/
                        #    repeat path (False): .update() REPLACES same ids (bounded) AND the cache is
                        #    not reset per-step, so the out queue SUSTAINS past t_end instead of vanishing
                        #    (restores the standing-presence persistence the corridor trajgen lost).
                        self.red_phase_status[idx] = not getattr(self.args, 'traj_gen', False)
                        continue

                    cell_ls = self.feature2cells.get(idx)
                    if cell_ls is None:
                        continue

                    if self.feature_to_phase(idx) != self.curr_phase_idx:
                        # if next phase is not the phase where fake vehicle is inserted, here the curr_phase is already the next phase
                        red_phase_flag = True
                    else:
                        red_phase_flag = False

                    # PER-FEATURE RESET: each feature's optimization must solve only its OWN ~15 fake
                    # vehicles. Previously fake_vehicle_dict accumulated across features (init'd once
                    # before the loop) -> later features optimized ~22-30 vehicles, which cannot fit
                    # with headway in the corridor's short (~68 m) segments -> Gurobi infeasible (the
                    # 70%->41% success drop). filled_spot_record / initial_dist_ls stay accumulated so
                    # cross-feature placements still don't physically overlap.
                    fake_vehicle_dict = {}

                    for _fake_veh in range(15):
                        fake_veh_id = 'fake_'+str(idx)+str(_fake_veh)+str(self.t)

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
                    if _is_corr3 and not getattr(self.args, 'traj_gen', False):
                        # DEFAULT (static): bypass the Gurobi optimizer for corr3 — place fake
                        # vehicles at their target position for every attack timestep. Validates the
                        # reward signal without the solver. Enable the realistic optimizer with
                        # -traj_gen (goes to the else branch: optimization_process).
                        t_start = self.t // 10
                        t_end = t_start + min(15, int(self.CTM.t_next_Gmin_end) - t_start) + 20
                        traj = {}
                        for veh_id, (_, _, fin_d, fin_s, fin_lane) in fake_vehicle_dict.items():
                            if fin_d >= 0 and fin_lane:
                                for t in range(t_start, t_end):
                                    traj.setdefault(t, {}).setdefault(fin_lane, {})[f"{veh_id}_{t}"] = {"speed": fin_s, "lane_pos": fin_d}
                        self.fake_traj_dict_allT[idx] = traj
                    else:
                        # -traj_gen: realistic optimizer. If infeasible it returns {} and NO fake
                        # vehicles are injected this cycle (no static fallback, by design).
                        _atg = min(15, int(self.CTM.t_next_Gmin_end) - self.t // 10)
                        _nveh = len(fake_vehicle_dict)
                        _dists = [v[2] for v in fake_vehicle_dict.values() if v[2] is not None and v[2] >= 0]
                        _fspds = [v[3] for v in fake_vehicle_dict.values() if v[3] is not None]
                        # -opt_dynamic_n: on infeasibility ({}), retry with fewer fakes (drop one from the
                        # densest same-lane group each round). GREEN fails (~0.09) because fast (~free-flow)
                        # heterogeneous-speed fakes packed into ~5 m headway slots can't hold the fixed-order
                        # gap while moving; thinning the tightest lane relieves it -> PARTIAL injection > none.
                        _kept = _nveh
                        _acc = getattr(self.args, 'opt_acc_low', -3.5)   # #2: emergency-braking bound
                        if getattr(self.args, 'opt_dynamic_n', False):
                            _work = dict(fake_vehicle_dict)
                            _res = optimization_process(_work, attack_time_gap=_atg, extend_time_gap=20,
                                                        initial_attack_time=self.t // 10, red_phase=red_phase_flag,
                                                        acc_low=_acc)
                            while (not _res) and len(_work) > 3:
                                # drop one vehicle from the lane that currently has the most fakes
                                _bylane = {}
                                for _vid, _v in _work.items():
                                    _bylane.setdefault(_v[4], []).append(_vid)
                                _densest = max(_bylane.values(), key=len)
                                del _work[_densest[-1]]
                                _res = optimization_process(_work, attack_time_gap=_atg, extend_time_gap=20,
                                                            initial_attack_time=self.t // 10, red_phase=red_phase_flag,
                                                            acc_low=_acc)
                            self.fake_traj_dict_allT[idx] = _res
                            _kept = len(_work) if _res else 0
                        else:
                            self.fake_traj_dict_allT[idx] = optimization_process(fake_vehicle_dict,
                                                        attack_time_gap = _atg,
                                                        extend_time_gap = 20, initial_attack_time = self.t//10, red_phase=red_phase_flag,
                                                        acc_low=_acc)
                        # TRUE incoming-optimizer success rate (separate from the misleading opt_succ_rate,
                        # which the always-succeeding out-density model inflates). Gurobi returns {} on
                        # infeasibility -> empty result = a real incoming-injection failure.
                        self._opt_in_total = getattr(self, '_opt_in_total', 0) + 1
                        _ok = bool(self.fake_traj_dict_allT[idx])
                        # DIAGNOSTIC: what inputs drive infeasibility? (attack_time_gap window, N vehicles
                        # that must fit with headway, target-distance span, mean target speed). corr3 fail
                        # rate (~55%) >> isolated (~30%); GREEN (red=False) fails far more than RED -> mean
                        # final-speed (fspd) is the discriminator (free-flow discharge vs ~0 queue).
                        if not hasattr(self, 'opt_diag'): self.opt_diag = []
                        # SPATIAL attribution: which approach/geometry does this optimization serve?
                        # Tests whether failures cluster on specific physical structure (fewer lanes =
                        # tighter headway; shorter cells; high-demand arterial through-movements that
                        # saturate -> low fspd). num_lane = approach capacity; ntgt = distinct target
                        # lanes actually used (nveh/ntgt = vehicles-per-lane, the binding headway load).
                        _cd = getattr(self.CTM, 'cell_dict', {})
                        _cells = [str(c) for c in (cell_ls or []) if str(c) in _cd]
                        _apps = sorted(set(int(_cd[c]['approach']) for c in _cells)) if _cells else []
                        _nlanes = [int(_cd[c]['num_lane']) for c in _cells]
                        _ncell2int = [int(_cd[c].get('ncell2int', -1)) for c in _cells]
                        _ntgt = len(set(v[4] for v in fake_vehicle_dict.values())) if fake_vehicle_dict else 0
                        # FINAL number of fake vehicles actually generated (distinct base ids in the output
                        # trajectory; the optimizer/replay can drop vehicles that end past the bar). With
                        # -opt_dynamic_n this is <= kept <= nveh; it is the real injected count.
                        _traj_out = self.fake_traj_dict_allT[idx]
                        _gen_ids = set()
                        if _traj_out:
                            for _lanes in _traj_out.values():
                                for _vehs in _lanes.values():
                                    for _vid in _vehs:
                                        _gen_ids.add(str(_vid).rsplit('_', 1)[0])
                        _ngen = len(_gen_ids)
                        self.opt_diag.append({'atg': int(_atg), 'nveh': int(_nveh), 'red': bool(red_phase_flag),
                                              'dmin': float(min(_dists)) if _dists else -1.0,
                                              'dmax': float(max(_dists)) if _dists else -1.0,
                                              'fspd': float(np.mean(_fspds)) if _fspds else -1.0,
                                              'kept': int(_kept), 'ngen': int(_ngen), 'ok': _ok,
                                              'feat': int(idx), 'app': _apps[0] if _apps else -1,
                                              'apps': _apps, 'nlane': min(_nlanes) if _nlanes else -1,
                                              'ntgt': int(_ntgt),
                                              'ncell2int': max(_ncell2int) if _ncell2int else -1})
                        if not _ok:
                            self._opt_in_fail = getattr(self, '_opt_in_fail', 0) + 1
                            if getattr(self.args, 'opt_fallback', False):
                                # DIAGNOSTIC: Gurobi failed -> place fakes at TARGET positions (frozen,
                                # static-style) so trajgen injects on EVERY JSMA-success decision. Tests
                                # if injection FREQUENCY (the ~54% failures) is the trajgen weakness.
                                _ts = self.t // 10; _te = _ts + min(15, int(self.CTM.t_next_Gmin_end) - _ts) + 20
                                _tj = {}
                                for veh_id, (_, _, fin_d, fin_s, fin_lane) in fake_vehicle_dict.items():
                                    if fin_d >= 0 and fin_lane:
                                        for t in range(_ts, _te):
                                            _tj.setdefault(t, {}).setdefault(fin_lane, {})[f"{veh_id}_{t}"] = {"speed": fin_s, "lane_pos": fin_d}
                                self.fake_traj_dict_allT[idx] = _tj
                    opt_time = time.time() - opt_start
                    # print("opt_time",opt_time)
                    
                    self.initial_attack_time = self.t//10
                    self.red_phase_status[idx] = red_phase_flag
                    self.fake_veh_traj_input.append([fake_vehicle_dict,red_phase_flag]) 
                
                    self.t_fakeTraj_duration.append(opt_time)

                # DIAGNOSTIC (-inject_drop_rate R): randomly DROP a fraction R of feature injections
                # (set to {}) to match trajgen's Gurobi failure frequency. Used to test STATIC at 46%
                # frequency: if static still lands high impact -> PERSISTENCE (few frozen injections
                # suffice); if it collapses to ~+2% -> FREQUENCY was the driver. Isolates the fallback.
                _dr = float(getattr(self.args, 'inject_drop_rate', 0.0) or 0.0)
                if _dr > 0:
                    import random as _rnd
                    for idx in list(self.feature_ids):
                        if _rnd.random() < _dr:
                            self.fake_traj_dict_allT[idx] = {}

                for idx in self.feature_ids:
                    if len(self.fake_traj_dict_allT.get(idx, {})) > 0:
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

        _gap = getattr(self.args, 'fake_spacing', 5.0) + 1  # initial-position min gap (default 6)
        closest = min(initial_dist_ls, key=lambda x: abs(x - initial_dist2bar))
        diff = abs(initial_dist2bar - closest)

        if diff >= _gap:
            return initial_dist2bar

        candidate = initial_dist2bar
        while any(abs(candidate - x) < _gap for x in initial_dist_ls):
            candidate -= _gap

        return candidate

    
    def get_initial_fake_state(self,final_dist2bar, final_spd, t_duration):
        # #1: match initial speed to the target (a fake entering a queue is already slowed) so it need
        # not decelerate 17.88->0 -> removes the rear-catches-stopped-front headway collapse. Default 17.88.
        if getattr(self.args, 'opt_init_spd_match', False) and final_spd is not None:
            initial_spd = float(final_spd)
        else:
            initial_spd = 17.88
        # Match the trajectory optimizer's window = min(15, t_next_Gmin_end - t) so the fake vehicle
        # is placed only as far back as it can actually reach the target within the speed limit.
        # (Old max(15,.) placed it for >=15s of travel while the optimizer allowed <15s -> the
        # required average speed exceeded speed_limit -> Gurobi infeasibleOrUnbounded.)
        t_duration = max(1, min(15, t_duration))
        initial_dist2bar = (initial_spd+final_spd)/2*t_duration+final_dist2bar

        return initial_dist2bar, initial_spd

    
    def fake_final_state_gen(self, cell_ls, filled_spot_record):
        available_cells = cell_ls[:]

        while available_cells:
            select_cell_idx = random.choice(available_cells)

            final_dist2bar, final_spd, final_lane = self.CTM.get_fake_veh_final_state(
                select_cell_idx, int(self.CTM.t_next_Gmin_end)
            )
            # -fake_spd_cap: on a served (green) approach get_fake_veh_final_state returns ~free-flow
            # 17.88; capping the target speed injects gentler slowing/queue fakes whose trajectories are
            # more parallel -> satisfy fixed-order headway -> higher green-phase feasibility.
            _cap = getattr(self.args, 'fake_spd_cap', 0.0)
            if _cap and final_spd is not None and final_spd > _cap:
                final_spd = _cap
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

        _sp = getattr(self.args, 'fake_spacing', 5.0)  # target spacing between same-lane fakes (m)
        for offset in [0, -_sp, +_sp]:
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