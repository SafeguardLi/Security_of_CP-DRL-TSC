import numpy as np

from src.trafficsignalcontrollers.nextphaserltsc import NextPhaseRLTSC
from src.nn_factory import _CORR3_N_APP


class NextPhasePressLightAttackTSC(NextPhaseRLTSC):
    """Attack TSC for a PHASE-BASED PressLight (binary switch/stay DQN) victim.

    PressLight's redesigned state mirrors CAVLight's per-phase structure:
        state[1] = inc(A) + out(P) + phase_one_hot(P+1) + time(1),  A = (P-1)*num_segments+1
    so the entire fake-vehicle injection / CTM / reward pipeline in NextPhaseRLTSC is
    reused verbatim. Only three victim-specific hooks differ from the CAVLight default:

      * _gen_attack_state / _clean_action: PressLight get_state returns a single vector
        (no A2C 3-tuple, no CTM block), and the attackable inc block leads the state.
      * _attacker_obs: the shared attacker observes the no-CTM canonical padding
        (include_ctm=False, dim 22) instead of the CAVLight 46-dim CTM canonical.

    The attackable inc block sits at offset 0 (vs CAVLight's avg_speed offset), so
    _cv_block_offset=0 is set before super().__init__ builds feature2cells, and the
    JSMA feature_range (0, A) is handled in PressLightAgent.init_attacker.
    """

    def __init__(self, *args, **kwargs):
        # inc (attackable) block starts at index 0 of the phase-based state -> feature2cells
        # must anchor at 0 (read by _build_corr3_feature2cells in super().__init__).
        self._cv_block_offset = 0
        super().__init__(*args, **kwargs)
        # #approaches for the geometry indicator in the no-CTM canonical state
        self._n_app = _CORR3_N_APP.get(self.id, None)
        # PressLight has no avg_speed block, so get_avg_speed (which sets norm_V_spd) is
        # never called. get_attacked_state is a passthrough that ignores these norms, but
        # they must exist. norm_CV is (re)set by get_state('presslight') each step.
        self.norm_V_spd = 1.0
        self.norm_CV = 1.0

    def _presslight_local_state(self):
        """Build the 20/10-dim phase-based victim state[1] = [inc, out, phase, time].
        Uses the current self.fake_veh_weight (set by the caller for clean vs attacked)."""
        raw = self.get_state('presslight', num_segments=self.num_segments)  # [inc(A), out(P)]
        return np.concatenate([raw,
                               self.phase_to_one_hot[self.phase],
                               np.array([self.phase_duration / self.g_max])])

    def _gen_attack_state(self):
        local = self._presslight_local_state()   # includes fakes (fake_veh_weight from caller)
        return [local, local]                    # state[1] is the attackable local state

    def _clean_action(self):
        old_wt = self.fake_veh_weight
        self.fake_veh_weight = 0.0
        clean_local = self._presslight_local_state()
        self.fake_veh_weight = old_wt
        clean_ret = self.rlagent.get_action(clean_local, 1e-5, False)
        return clean_ret[0], clean_ret[1]

    def _attacker_obs(self, state):
        # PressLight has no CTM block: canonical = [inc, out, phase, time] + geometry (22-dim)
        return self.attacker.canonicalize_state(state[1], None,
                                                include_ctm=False, n_app=self._n_app)

    def _step_impact(self, clean_out, attacked_out, clean_idx):
        """DQN impact = normalized erosion of the victim's Q-margin toward a flip.
        clean_out / attacked_out are RAW Q-vectors (PressLightAgent.get_action).

            clean_margin   = Q_clean[a]     - Q_clean[other]   (>=0; victim's preference for a)
            attacked_margin = Q_attacked[a] - Q_attacked[other]
            impact = clip((clean_margin - attacked_margin) / (|clean_margin| + eps), 0, 1)

        0 = attack changed nothing; 1 = drove the victim to/past its flip boundary. This
        replaces the fabricated softmax(Q) confidence drop and is less compressed, giving
        the attacker a stronger, DQN-native learning signal.
        """
        other = 1 - clean_idx
        clean_margin = float(clean_out[clean_idx]) - float(clean_out[other])
        attacked_margin = float(attacked_out[clean_idx]) - float(attacked_out[other])
        eroded = clean_margin - attacked_margin
        return float(np.clip(eroded / (abs(clean_margin) + 1e-6), 0.0, 1.0))
