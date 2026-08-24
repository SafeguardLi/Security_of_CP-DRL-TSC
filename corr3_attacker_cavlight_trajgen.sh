#!/bin/bash
# ============================================================================
# TEST-ONLY trajgen for the CAVLight blackbox attacker on plymouth_corr_3.
#   Reuses the shared attacker trained by corr3_attacker_cavlight_retrain_fixed.sh
#   (blackbox / surrogate-JSMA, FIXED injection code = no fake-vehicle pileup) and
#   deploys it to ALL 3 targets, 3 seeds each, WITH the vehicle-trajectory-generation
#   module ENABLED (-traj_gen).
#
# WHY test-only: -traj_gen only changes injection REALIZATION at test (fake vehicles must
#   follow a physically-feasible Gurobi-optimized kinematic trajectory instead of static
#   placement); it does NOT change the attacker's target policy. So we hold the attacker
#   FIXED and vary only -traj_gen to isolate the realizability effect (parity with the
#   PressLight trajgen shell). If the optimizer is INFEASIBLE for a cycle it returns {} and
#   NO fake vehicles are injected that cycle (no fallback, by design) -> expected WEAKER than
#   the static-injection result.
#
# NOTE: CAVLight has NO out-block / out-injection, so -out_leave_speed does not apply here
#   (only the incoming Gurobi optimizer is realized). This is the CAVLight counterpart of
#   corr3_attacker_presslight_trajgen.sh.
#
# VICTIM: unsat CAVLight set, T=10, staged at
#   experiments/cavlight/CAV_pen_rate_5.0/plymouth_corr_3_bin_real_real/saved_models/{tsc}_15000.h5
#   TSC victim = REAL model (no -tsc_surrogate); JSMA runs on the surrogate (-surrogate_dir) = blackbox.
#
# BENIGN / STATIC baselines (fixed code, benign 85.9s):
#   blackbox-static @ 62477148 +19% / 62500824 +33% / 62532012 +82%.
#   The trajgen numbers here are the ones to report for "under realistic spoofing".
# ============================================================================

sim=plymouth_corr_3
flow_type=bin_real
turn_type=real
tsc_program=0
pen_rate=5
detect_range=80
tsc_updates=15000                 # pre-trained CAVLight victim checkpoint to load
updates=50

force_flip=""                     # REAL attack (JSMA + injection). Set "-force_flip" only to
                                  # isolate the oracle upper bound (bypass JSMA/injection).
jsma_nf="-jsma_no_fallback"       # pure JSMA (train/test consistency with the retrain shell)

# --- budget (matches corr3_attacker_cavlight_retrain_fixed.sh) ---
num_segments="-num_segments 3"
simlen="-simlen 18000"
gmin="-gmin 100"
gmax="-gmax 400"
detect_r="-detect_r 200"
nreplay="-nreplay 512"
n_step="-nsteps 1"

# ============================================================
# TEST-ONLY guard: reuse the attacker trained by corr3_attacker_cavlight_retrain_fixed.sh.
# Do NOT back up or retrain — just deploy the existing blackbox attacker with -traj_gen.
# ============================================================
if [ ! -d experiments/attacker_cavlight/CAV_pen_rate_5.0/plymouth_corr_3_bin_real_real/saved_models/actor_app ]; then
    echo "ERROR: no trained attacker in experiments/attacker_cavlight — run corr3_attacker_cavlight_retrain_fixed.sh first."; exit 1
fi
echo "========== TEST-ONLY trajgen: deploy existing (blackbox-static) CAVLight attacker, T=10, ALL 3 targets =========="

# ============================================================
# TESTING  —  deploy the shared attacker to EACH target (others benign), 3 seeds; archive per target
# ============================================================
for att_target in 62477148 62500824 62532012
do
    for seed in 13 23 33
    do
        echo "--- TEST target=$att_target seed=$seed (T=10, traj_gen) ---"
        python3 run.py \
            -sim $sim -tsc cavlight -nogui -load -mode test \
            -tsc_updates $tsc_updates -updates $updates \
            -shared_att -att_target $att_target \
            -gamma 0.99 -pen_rate $pen_rate \
            -flow_type $flow_type -temperature 10 \
            -global_critic sep -tsc_program $tsc_program \
            -turn_type $turn_type \
            -n 1 $detect_r $gmax $num_segments \
            -marl sarl $simlen \
            -sumo_detect -detect_mode CAV \
            -detect_range $detect_range \
            -succ_detect_rate 100 \
            -no_random_flow -seed $seed \
            $gmin $nreplay $n_step \
            -att_model drl -act_ctm -marginal_delay $jsma_nf -surrogate_dir experiments/SURROGATES/cavlight -traj_gen $force_flip
    done
    # archive this target's 3 seeds (per-target label so targets don't collide)
    bash archive_round.sh cavlight_blackbox_trajgen_target${att_target} || true
    echo "--- archived target=$att_target ---"
done
echo "Done. All 3 targets tested with -traj_gen + archived under the FIXED code."
echo "Compare vs blackbox-static: 62477148 +19% / 62500824 +33% / 62532012 +82% (benign 85.9s)."
