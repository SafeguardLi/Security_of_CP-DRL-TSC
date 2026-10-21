#!/bin/bash
# TEST trained Sybil-DQN vs CAVLight (TL 79), 3 seeds, greedy. Compare avg vehicle delay vs
# benign (no attack) and the paper's RANDOM baseline.
sim=plymouth; tsc=cavlight; pen=5; victim=10000
common="-sim $sim -tsc $tsc -nogui -load -mode test -tsc_updates $victim -updates 1 \
  -gamma 0.99 -pen_rate $pen -flow_type bin_real -temperature 100 -global_critic sep -tsc_program 0 \
  -turn_type real -n 1 -detect_r 200 -gmax 400 -gmin 100 -num_segments 3 -marl sarl -simlen 18000 \
  -sumo_detect -detect_mode CAV -detect_range 80 -succ_detect_rate 100 -no_random_flow -nreplay 512 -nsteps 1"
echo "### BENIGN ###";        for s in 13 23 33; do python3 run.py $common -seed $s; done
echo "### RANDOM baseline ###"; for s in 13 23 33; do python3 run.py $common -seed $s -att_model random; done
echo "### Sybil-DQN (trained) ###"; for s in 13 23 33; do python3 run.py $common -seed $s -att_model sybil; done
