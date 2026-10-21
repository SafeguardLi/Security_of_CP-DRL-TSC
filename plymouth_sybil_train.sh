#!/bin/bash
# BENCHMARK: Sybil-DQN (Arabi et al., CSR 2021) vs CAVLight on isolated plymouth (TL 79), DISTRIBUTED.
# N actor procs collect (s,a,r,s') into a SHARED replay; 1 learner proc trains the DQN and broadcasts
# weights. LOCATION-based action (pick segment(s) to flood with 15 veh each, fair vs CAVLight). Victim 79_10000 FROZEN.
# -n = parallel actors; -updates (UPD) = total DQN grad steps (hard cap). off-policy: PERSISTENT rolling
# replay; MULT relaxes the 1:1 update/collection rate-limit -> updates allowed up to MULT*n_exp (buffer reuse),
# so the DQN reaches many grad steps without proportionally more slow sims.
sim=plymouth; tsc=cavlight; pen=5; victim=10000
N=${N:-4}; UPD=${UPD:-10000}; MULT=${MULT:-6}; D=${D:-0.2}; OPT=${OPT:-adam}; LR=${LR:-0.02}
[ "${RESUME:-0}" = "1" ] || rm -f experiments/attacker_${tsc}/CAV_pen_rate_${pen}.0/${sim}_bin_real_real/saved_models/sybil/79.pt experiments/attacker_${tsc}/CAV_pen_rate_${pen}.0/${sim}_bin_real_real/saved_models/sybil/sybil_train_log.csv 2>/dev/null
echo "########## Sybil-DQN TRAIN (distributed): N=$N actors, $UPD updates, mult=$MULT (d=$D opt=$OPT lr=$LR) ##########"
python3 run.py -sim $sim -tsc $tsc -nogui -load -mode train -n $N \
  -tsc_updates $victim -updates $UPD -gamma 0.99 -pen_rate $pen \
  -flow_type bin_real -temperature 100 -global_critic sep -tsc_program 0 -turn_type real \
  -detect_r 200 -gmax 400 -gmin 100 -num_segments 3 -marl sarl -simlen 18000 \
  -sumo_detect -detect_mode CAV -detect_range 80 -succ_detect_rate 100 \
  -no_random_flow -seed 13 -nreplay 512 -nsteps 1 -batch 32 -save_u 20 \
  -att_model sybil -sybil_cost $D -sybil_opt $OPT -sybil_lr $LR -sybil_replay_mult $MULT
echo "Done. Weights -> experiments/attacker_${tsc}/CAV_pen_rate_${pen}.0/${sim}_bin_real_real/saved_models/sybil/79.pt"
