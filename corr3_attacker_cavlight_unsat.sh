#!/bin/bash
# ============================================================================
# Generalized DRL attacker vs the UNSAT CAVLight victim on plymouth_corr_3.
#   1 TRAINING round : ONE shared attacker network trained on merged experience from all 3
#                      CAVLight intersections (canonical padded state, -shared_att).
#   3 TESTING rounds : deploy that shared policy to ONE target intersection (same target, 3 seeds);
#                      others run benign.  Compare attacked delay vs the benign baseline.
#
# VICTIM: unsat CAVLight set (all 3 vehicle-dependent), staged at
#   experiments/cavlight/CAV_pen_rate_5.0/plymouth_corr_3_bin_real_real/saved_models/{tsc}_15000.h5
#   Attackability (argmax-sweep): 62477148=51%, 62500824=88% (TARGET), 62532012 attackable.
#
# *** TEMPERATURE = 10 ***  This victim was TRAINED at T=10 and its action is SAMPLED from
#   softmax(logits/T). The old attacker shells used T=100, which FLATTENS a T=10 policy -> the
#   attack would be learned against the wrong (near-random) policy. Attack + benign MUST use T=10.
#
# BENIGN REFERENCE (already measured, same config T=10 / real demand / pen 5):
#   mean timeLoss 85.9s; per-approach: arterial 33-73s, Nixon SB (-1162299121) 268s.
#   The attack is judged by how far it degrades this (esp. at the target intersection).
#
# DEMAND (verified): attacker TRAINING (default -mode train) reads plymouth_corr_3_train.rou.xml =
#   0.75x real-world, UNIFORM scale across every origin edge (SAME pattern as real) -> acceptable.
#   TESTING (-mode test) reads plymouth_corr_3_test.rou.xml = real-world (WB 1087 / SWEB 950 / ...).
# ============================================================================

sim=plymouth_corr_3
flow_type=bin_real
turn_type=real
tsc_program=0
pen_rate=5
detect_range=80
tsc_updates=15000                 # pre-trained CAVLight victim checkpoint to load

# --- generalized attacker config ---
att_target=62500824               # strongest target (88% vehicle-dependent). Switch to 62477148
                                  # to verify the newly-fixed middle intersection (51%).
force_flip=""                     # REAL attack (JSMA + injection). Set "-force_flip" only to
                                  # isolate the oracle upper bound (bypass JSMA/injection).

# PURE JSMA (no heuristic fallback): when JSMA finds no salient feature, inject NOTHING and
# penalize the attacker (s_eff=0, jsma reward=-1) instead of rescuing with a blind injection.
# -> every realized flip is attributable to genuine white-box JSMA (clean paper claim).
# Applied to BOTH train and test so the policy is optimized for, and evaluated under, pure JSMA
# (train/test consistency; a fallback-trained policy under-performs when the net is removed).
jsma_nf="-jsma_no_fallback"

# --- budget (matches the base cavlight attacker; save intermediate for best-checkpoint test) ---
train_n="-n 8"
l="-l 1"
updates=50
save_u="-save_u 5"
batch="-batch 128"
nreplay="-nreplay 512"
n_step="-nsteps 1"
num_segments="-num_segments 3"
simlen="-simlen 18000"
gmin="-gmin 100"
gmax="-gmax 400"
detect_r="-detect_r 200"

# --- fresh start: don't resume an attacker trained on a different victim/reward ---
if [ -d experiments/attacker_cavlight ]; then
    backup="experiments/attacker_cavlight_PREVvictim_$(date +%m%d_%H%M%S)"
    mv experiments/attacker_cavlight "$backup"
    echo "Backed up prior attacker -> $backup  (fresh start vs unsat victim)"
fi

# ============================================================
# 1 ROUND OF TRAINING  —  shared attacker over ALL 3 cavlight intersections
# ============================================================
echo "========== SHARED attacker TRAINING vs UNSAT CAVLight on $sim (3 intersections -> 1 net), T=10 =========="
python3 run.py \
    -sim $sim -tsc cavlight -nogui \
    -load -tsc_updates $tsc_updates \
    -shared_att \
    -save $save_u -updates $updates \
    -gamma 0.99 -pen_rate $pen_rate \
    -flow_type $flow_type -temperature 10 \
    -global_critic sep -tsc_program $tsc_program \
    -turn_type $turn_type \
    $l $train_n $detect_r $gmax \
    $num_segments $batch \
    -marl sarl -all_veh_r $simlen \
    -sumo_detect -detect_mode CAV \
    -detect_range $detect_range \
    -succ_detect_rate 100 \
    $gmin $nreplay $n_step \
    -att_model drl -act_ctm -marginal_delay $jsma_nf $force_flip

echo ""
echo "Training done. Deploying shared attacker to $att_target for 3 test rounds."

# ============================================================
# 3 ROUNDS OF TESTING  —  deploy shared policy to the SAME target intersection
# ============================================================
for seed in 13 23 33
do
    echo "--- TEST seed $seed : shared attacker deployed at $att_target (others benign), T=10 ---"
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
        -att_model drl -act_ctm -marginal_delay $jsma_nf $force_flip
done
echo "Done. Compare attacked timeLoss (exp_log/sumo) + realize_${att_target}.txt vs benign 85.9s."
echo "PURE JSMA: check realize_${att_target}.txt fail_rate (JSMA-miss cycles now inject NOTHING) and"
echo "           realized_rate — every realized flip is genuine JSMA (no fallback)."

# auto-archive this round to experiments/ATTACK_RESULTS_tripinfos/
bash archive_round.sh cavlight_whitebox_static || true
