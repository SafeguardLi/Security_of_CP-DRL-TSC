"""Sybil-attacker state / action-injection / reward for the Arabi et al. (CSR 2021)
benchmark, adapted to plymouth's 4-phase CAVLight victim.

CAVLight actor count block = state[10:20] (num_segments=3):
    [10,11,12] Southbound (SB) seg 1,2,3     [16]        Main-L
    [13,14,15] Northbound (NB) seg 1,2,3     [17,18,19]  Main-THR seg 1,2,3
(matches nextphaseattacker.get_fake_veh). The paper's four roads map onto plymouth's four
phases; EW and WE are the same Main-THR movement, so the four "approaches" are the four
phases: Main-THR, Main-L, NB, SB (plus the paper's two combos: both-main, both-side).

STATE (6-dim, mirrors paper's (n0,n1,n2,n3,tl,T)):
    [ n_THR, n_L, n_NB, n_SB, phase_idx_norm, remaining_green_norm ]

ACTION (LOCATION-based, PAIRED): the paper injects only 1-2 Sybil vehicles, far below the
CAVLight attacker's budget. The CAVLight JSMA attacker perturbs TWO features (~30 vehicles),
so for a FAIR comparison each Sybil action floods TWO aligned segments with `max_veh` (=15)
vehicles each (30 total). The DQN chooses a street (side / main) and a distance band
(far / mid / near); the count stays fixed.

Pairing — every action injects 30 Sybil vehicles total (= 2 * max_veh, matching CAVLight's
2-feature / 30-vehicle budget):
    side = NB + SB  : 15 into the NB cell AND 15 into the SB cell at that distance band
    main = THR only : the main-through merges both approaches (EB+WB) into ONE count cell,
                      so inject BOTH approaches' worth (2*15 = 30) into that THR band cell;
                      Main-Left (cell 16) is left untouched.

Action space (6 actions) — there is NO "none": the attacker ALWAYS injects, so it cannot
collapse to the trivial do-nothing optimum. It only chooses WHERE:
    0 = side-far  (NBf 13 +15, SBf 10 +15)   3 = main-far  (THRf 17 +30)
    1 = side-mid  (NBm 14 +15, SBm 11 +15)   4 = main-mid  (THRm 18 +30)
    2 = side-near (NBn 15 +15, SBn 12 +15)   5 = main-near (THRn 19 +30)
(`scale_levels` is retained in the signatures for call-site compatibility but is ignored.)
"""
import numpy as np

CV_LO, CV_HI = 10, 20                     # CAVLight count block state[10:20]
_PHASE_CELLS = {'THR': [17, 18, 19], 'L': [16], 'NB': [13, 14, 15], 'SB': [10, 11, 12]}
_ENTRY = {'THR': 19, 'L': 16, 'NB': 15, 'SB': 12}          # (legacy) nearest-stopline cell per phase

# per-cell labels: by CTM geometry cell 10/13/17 are the FARTHEST-upstream segments,
# 12/15/19 the NEAREST stopline (see feature2cells in nextphaserltsc.py).
_CELL_LABELS = {10: 'SBf', 11: 'SBm', 12: 'SBn',
                13: 'NBf', 14: 'NBm', 15: 'NBn',
                16: 'L',
                17: 'THRf', 18: 'THRm', 19: 'THRn'}

# action (1-indexed after 'none') -> (cells to flood, veh-per-cell as a multiple of max_veh).
# side actions hit 2 cells at 1x (15+15); main actions hit 1 THR cell at 2x (30) since both
# main-through approaches merge into that single count cell.
ACTION_SPEC = [
    ([13, 10], 1),   # 1 side-far  : NB far + SB far
    ([14, 11], 1),   # 2 side-mid  : NB mid + SB mid
    ([15, 12], 1),   # 3 side-near : NB near + SB near
    ([17],     2),   # 4 main-far  : THR far  (both approaches, Main-Left ignored)
    ([18],     2),   # 5 main-mid  : THR mid
    ([19],     2),   # 6 main-near : THR near
]
ACTION_LABELS = ['sideF', 'sideM', 'sideN', 'mainF', 'mainM', 'mainN']

# kept so plot_sybil_training / old references don't break
ACTION_CELLS = [spec[0] for spec in ACTION_SPEC]
APPROACHES = list(ACTION_CELLS)
APPROACH_LABELS = list(ACTION_LABELS)


def n_actions(scale_levels=None):
    """6 paired (street x distance) actions; NO 'none' (always inject). `scale_levels` ignored."""
    return len(ACTION_SPEC)


def _decode(action, scale_levels, max_veh):
    """action (0..5) -> (list_of_cells, veh_per_cell). ALWAYS injects (no none).
    Each action injects 2*max_veh total: side = max_veh into each of 2 cells;
    main = 2*max_veh into the single THR band cell. Out-of-range is clamped."""
    a = int(action)
    a = max(0, min(a, len(ACTION_SPEC) - 1))
    cells, mult = ACTION_SPEC[a]
    return list(cells), max(1, int(max_veh) * mult)


def n_added(action, scale_levels, max_veh):
    """total Sybil vehicles injected by this action (for the -d*n cost term)."""
    cells, veh = _decode(action, scale_levels, max_veh)
    return veh * len(cells)


def action_label(action, scale_levels, max_veh):
    a = int(action)
    a = max(0, min(a, len(ACTION_LABELS) - 1))
    cells, veh = _decode(a, scale_levels, max_veh)
    return f'{ACTION_LABELS[a]}x{veh}'


def build_state(victim_state, curr_phase_idx, phase_time, g_max, n_phases):
    """6-dim Sybil-DQN state from the victim's (clean) actor state (normalized cell sums)."""
    s = np.asarray(victim_state, np.float32)
    n_thr = float(s[_PHASE_CELLS['THR']].sum()); n_l = float(s[_PHASE_CELLS['L']].sum())
    n_nb = float(s[_PHASE_CELLS['NB']].sum());   n_sb = float(s[_PHASE_CELLS['SB']].sum())
    phase_idx_norm = curr_phase_idx / max(n_phases - 1, 1)
    remaining_green_norm = max(0.0, 1.0 - float(phase_time) / max(g_max, 1))
    return np.array([n_thr, n_l, n_nb, n_sb, phase_idx_norm, remaining_green_norm], np.float32)


def inject(victim_state, action, norm_CV, scale_levels, max_veh):
    """Add the action's Sybil vehicles to the victim count block and re-normalize (L2),
    mirroring nextphaseattacker.get_fake_veh."""
    cells, veh = _decode(action, scale_levels, max_veh)
    s = np.asarray(victim_state, np.float32).copy()
    if not cells or veh <= 0:
        return s
    for i in range(CV_LO, CV_HI):                 # un-normalize to raw counts
        s[i] = s[i] * norm_CV
    for c in cells:                               # inject `veh` Sybil vehicles at each entry cell
        s[c] = s[c] + veh
    nrm = np.linalg.norm(s[CV_LO:CV_HI])
    if nrm > 0:
        for i in range(CV_LO, CV_HI):
            s[i] = s[i] / nrm
    return s


def congestion_reward(conn, d, n_add):
    """Paper reward:  Nhalt - Nmove - d*n   over REAL (legitimate) vehicles.
    Sybils are fake *detected* vehicles (not in SUMO), so TraCI counts are legitimate-only."""
    try:
        veh_ids = conn.vehicle.getIDList()
        n_halt = sum(1 for v in veh_ids if conn.vehicle.getSpeed(v) < 0.1)
        n_move = len(veh_ids) - n_halt
    except Exception:
        n_halt = n_move = 0
    return float(n_halt - n_move - d * n_add), n_halt, n_move
