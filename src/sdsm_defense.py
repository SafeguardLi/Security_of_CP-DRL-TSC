#!/usr/bin/env python3
"""
SDSM consistency defense for CAVLight.

SDSMDefense cross-checks claimed CAV detections against occupancy geometry
to identify fake SDSMs injected by the attacker.

Call run_defense() once per decision step (every time get_next_phase() is
about to build state).  The returned trust scores are used to downweight
fake vehicle counts in the TSC state.
"""

import numpy as np
import traci.constants as tc

from src.occupancy_map import OccupancyMap, TrustScoreManager


class SDSMDefense:
    """
    Cross-checks claimed CAV detections against occupancy geometry.

    Typical usage (inside NextPhaseRLTSC.get_next_phase):

        real_sdsm = self.sdsm_defense.build_real_sdsm(self.cv_data, cav_positions)
        fake_sdsm = self.sdsm_defense.build_fake_sdsm(
                        self.fake_traj_dict, self.conn, self.netdata)
        if fake_sdsm:
            merged  = {**real_sdsm, **fake_sdsm}
            scores  = self.sdsm_defense.run_defense(merged, cav_positions)
            weight  = self.sdsm_defense.get_fake_weight(fake_cav_id)
    """

    def __init__(self,
                 junction_pos: tuple,
                 detect_range: float = 80.0,
                 map_size: float = 400.0,
                 cell_size: float = 2.0,
                 coverage_threshold: float = 0.3,
                 min_expected: int = 3):
        """
        junction_pos       : (x, y) world coordinates of the protected intersection
        detect_range       : CAV-to-UV detection radius in metres (must match sim)
        map_size           : side length of the occupancy grid in metres
        cell_size          : cell side length in metres
        coverage_threshold : min fraction of expected detectors that must
                             corroborate a report before it is suspicious.
                             Lowered from 0.5 → 0.3 so that 1-of-3 corroboration
                             is sufficient (avoids penalising CAVs on cross-approaches
                             that legitimately miss vehicles on other arms).
        min_expected       : minimum number of in-range CAVs required before any
                             Type 2 conflict can be raised.  Set to 3 (was 2) to
                             eliminate the ambiguous 2-CAV case that was the main
                             source of real-CAV false positives at low MPR.
        """
        self.jx, self.jy = junction_pos
        self.detect_range = detect_range
        self.coverage_threshold = coverage_threshold
        self.min_expected = min_expected
        self.omap = OccupancyMap(self.jx, self.jy, map_size, cell_size)
        self.trust = TrustScoreManager()

    # ------------------------------------------------------------------
    # SDSM construction
    # ------------------------------------------------------------------

    def build_real_sdsm(self, cv_data: dict, cav_positions: dict) -> dict:
        """
        Reconstruct per-CAV SDSM from detected-vehicle subscription data.

        For each real CAV, the SDSM contains every vehicle in cv_data whose
        reported position is within detect_range of that CAV.

        Args:
            cv_data       : {lane_id: {veh_id: {VAR_POSITION, VAR_SPEED, ...}}}
            cav_positions : {cav_id: (x, y)} — positions of all real CAVs

        Returns:
            {cav_id: {veh_id: {'pos': (x, y), 'speed': float}}}
        """
        sdsm = {cav_id: {} for cav_id in cav_positions}
        for lane_vehs in cv_data.values():
            for veh_id, vdata in lane_vehs.items():
                try:
                    vx, vy = vdata[tc.VAR_POSITION]
                except (KeyError, TypeError):
                    continue
                spd = vdata.get(tc.VAR_SPEED, 0.0)
                for cav_id, (cx, cy) in cav_positions.items():
                    if np.hypot(vx - cx, vy - cy) <= self.detect_range:
                        sdsm[cav_id][veh_id] = {'pos': (vx, vy), 'speed': spd}
        return sdsm

    def build_fake_sdsm(self, fake_traj_dict: dict, conn,
                        ctm_to_sumo_lane: dict) -> dict:
        """
        Convert fake_traj_dict into an SDSM as if broadcast by a fake CAV.

        fake_traj_dict format:
            {ctm_lane_id: {fake_veh_id: {'speed': float, 'lane_pos': float}}}
        where lane_pos = distance from stop bar in metres (positive = behind bar).
        ctm_lane_id values (e.g. '5_0') are CTM-internal IDs; ctm_to_sumo_lane maps
        them to the corresponding SUMO lane ID for geometry queries.

        Strategy:
          - Convert every (ctm_lane_id, lane_pos) to world (x, y) via SUMO lane shape.
          - Pick the vehicle at the MEDIAN lane_pos (middle of the group)
            as the "fake CAV" — it can plausibly broadcast about vehicles both
            ahead and behind.
          - All other fake vehicles become its reported detections.

        Returns:
            {fake_cav_id: {other_fake_veh_id: {'pos': (x, y), 'speed': float}}}
            or {} if fake_traj_dict is empty.
        """
        all_fake = []
        for lane_id, vehs in fake_traj_dict.items():
            sumo_lane_id = ctm_to_sumo_lane.get(lane_id)
            if sumo_lane_id is None:
                continue
            try:
                lane_len = conn.lane.getLength(sumo_lane_id)
                shape = conn.lane.getShape(sumo_lane_id)
            except Exception:
                continue
            for veh_id, info in vehs.items():
                pos_from_start = lane_len - info['lane_pos']
                xy = self._interpolate_shape(shape, pos_from_start)
                all_fake.append({
                    'veh_id': veh_id,
                    'lane_pos': info['lane_pos'],
                    'speed': info['speed'],
                    'pos': xy,
                })

        if not all_fake:
            return {}, None, None

        # Fake CAV = vehicle at the median lane_pos (middle of the group)
        sorted_fake = sorted(all_fake, key=lambda e: e['lane_pos'])
        fake_cav_entry = sorted_fake[len(sorted_fake) // 2]
        fake_cav_id = fake_cav_entry['veh_id']
        fake_cav_pos = fake_cav_entry['pos']   # world (x, y) of the fake CAV

        detections = {
            e['veh_id']: {'pos': e['pos'], 'speed': e['speed']}
            for e in all_fake
            if e['veh_id'] != fake_cav_id
        }
        # Return: sdsm dict, fake CAV id, fake CAV claimed position.
        # The caller should add (fake_cav_id → fake_cav_pos) to cav_positions so
        # that expected_detectors treats the fake CAV like any other broadcaster.
        # Type 1 then fires only if the fake CAV claims detections outside its own
        # stated range; Type 2 fires when real CAVs nearby don't corroborate it.
        return {fake_cav_id: detections}, fake_cav_id, fake_cav_pos

    @staticmethod
    def _interpolate_shape(shape: list, dist_from_start: float) -> tuple:
        """
        Return (x, y) at dist_from_start metres along the polyline shape.
        shape is a list of (x, y) tuples from traci lane.getShape().
        Clamps to the end point if dist_from_start exceeds total length.
        """
        accum = 0.0
        for i in range(len(shape) - 1):
            x0, y0 = shape[i]
            x1, y1 = shape[i + 1]
            seg = np.hypot(x1 - x0, y1 - y0)
            if accum + seg >= dist_from_start:
                frac = (dist_from_start - accum) / max(seg, 1e-9)
                return (x0 + frac * (x1 - x0), y0 + frac * (y1 - y0))
            accum += seg
        return shape[-1]

    # ------------------------------------------------------------------
    # Defense pass
    # ------------------------------------------------------------------

    def run_defense(self, sdsm_dict: dict, cav_positions: dict) -> dict:
        """
        Main consistency check.

        Steps:
          1. Populate occupancy map from all claims in sdsm_dict.
          2. For each occupied cell compute expected_detectors from geometry.
          3. Two conflict types per cell:
               Type 1 — out-of-range reporter: claims detection beyond geometric range.
               Type 2 — minority corroboration: fewer in-range CAVs report a vehicle
                        than coverage_threshold * n_expected.
          4. Attribution:
               - n_expected < 2  : skip (can't adjudicate).
               - n_expected == 2 : penalize BOTH expected CAVs (ambiguous).
               - n_expected >= 3 : majority vote — penalize the minority side only
                                   (i.e. the in-range reporters if they are fewer
                                    than the non-reporters; do nothing otherwise).
          5. Recover scores only for CAVs NOT penalized in this step.

        Args:
            sdsm_dict     : merged {cav_id: {veh_id: {'pos', 'speed'}}}
            cav_positions : {cav_id: (x, y)}

        Returns:
            Current trust scores {cav_id: float}.
        Side-effect:
            self.last_step_stats is set with per-step diagnostics dict.
        """
        self.omap.reset()

        total_claims = 0
        for cav_id, detections in sdsm_dict.items():
            self.trust.ensure(cav_id)
            for veh_id, info in detections.items():
                x, y = info['pos']
                self.omap.mark(veh_id, x, y, cav_id)
                total_claims += 1

        penalized_this_step: set = set()
        n_type1 = 0
        n_type2 = 0

        for cell, actual_reporters in self.omap.detected_by.items():
            expected = self.omap.expected_detectors(
                cell, cav_positions, self.detect_range)
            n_expected = len(expected)

            # Type 1: out-of-range reporter — penalize regardless of n_expected.
            # A CAV claiming a detection from outside its geometric range is always
            # suspicious.
            for cav_id in (actual_reporters - expected):
                self.trust.penalize(cav_id, factor=1.0)
                penalized_this_step.add(cav_id)
                n_type1 += 1

            # Type 2: minority-corroboration — only adjudicate when enough CAVs
            # are in range.  The 2-CAV case is dropped (was the dominant source of
            # real-CAV false positives at low MPR: one CAV on a cross-approach
            # legitimately misses a vehicle on another arm, yet both got penalised).
            if n_expected < self.min_expected:
                continue

            corroborated = actual_reporters & expected
            n_c = len(corroborated)
            coverage = n_c / n_expected

            if coverage >= self.coverage_threshold:
                continue  # sufficient corroboration — no Type 2 conflict

            # Majority vote: the minority reporters are the suspicious fabricators.
            n_nc = n_expected - n_c
            if n_c < n_nc:
                for cav_id in corroborated:
                    self.trust.penalize(cav_id, factor=1.0)
                    penalized_this_step.add(cav_id)
                    n_type2 += 1

        self.trust.recover_all(exclude=penalized_this_step)
        scores = dict(self.trust.scores)

        # Store per-step diagnostics for the caller to log
        self.last_step_stats = {
            'total_claims': total_claims,
            'occupied_cells': len(self.omap.detected_by),
            'n_type1': n_type1,
            'n_type2': n_type2,
            'penalized': sorted(penalized_this_step),
        }
        return scores

    # ------------------------------------------------------------------
    # Weight query
    # ------------------------------------------------------------------

    def get_fake_weight(self, fake_cav_id: str,
                        threshold: float = 0.5) -> float:
        """
        Return a multiplicative weight in [0, 1] for fake vehicle counts.

        If fake_cav_id's trust score falls below threshold the weight is 0
        (fake vehicles fully suppressed).  Otherwise the weight equals the
        trust score, linearly reducing the contribution.
        """
        score = self.trust.get(fake_cav_id)
        return 0.0 if score < threshold else score
