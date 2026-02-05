#!/bin/bash

# core param
sim=plymouth
#flow_type=uniform_mixbd
updates=35 #3000 # 5000
test_round=3
turn_type=real #0.3l_0.1r
tsc_program=0 #1

# additional param
detect_r="-detect_r 200"
nexp="-nreplay 512" # 2000
succ_detect_rate=100 #"-succ_detect_rate 100"
# detect_range= #"-detect_range 80"
r=""
gmin="-gmin 100" # 
gmax=""
train_n="-n 16" # 16
test_n="-n 1"
n_step="-nsteps 1" # nstep=2 can not solve the delayed reward issue due to the PPO design
l="-l 1"
batch="-batch 128"
num_segments="-num_segments 3"
simlen="-simlen 21000"
NOW="$(date +"%m-%d-%Y")"
 
for k in 1 #2 3 4 5 6 7 8 #
do

  # # CAVLight with CAV detection
  for flow_type in bin_real # uniform_bd300 uniform_bd1200
  do
    for detect_range in 80 #60 100 #60 80 100
    do 
      # for i in 5 #1 3 5 8 10 30 50 # 
      # do
      #   python3 run.py -sim $sim -tsc cavlight -nogui -gamma 0.999 -save  -updates $updates -pen_rate $i -flow_type $flow_type -temperature 100 -global_critic sep -tsc_program $tsc_program -turn_type $turn_type   $l $train_n $detect_r $gmax $num_segments $batch -marl sarl -all_veh_r $simlen -sumo_detect -detect_mode CAV -act_lp -detect_range $detect_range -succ_detect_rate $succ_detect_rate $gmin $nexp $n_step -act_ctm -save_u 5 # -load #-load_replay  #-decaying_eps -eps 0.5 #-load_replay # -no_random_flow 
      # done

      for i in 5 #1 3 5 8 10 30 50 # 
      do
        for seed in 63 #13 23 33 # ((j=1;j<=$test_round;j++));
        do
          python3 run.py -sim $sim -tsc cavlight -nogui -load -mode test -updates $updates -pen_rate $i -flow_type $flow_type -temperature 100 -global_critic sep -tsc_program $tsc_program -turn_type $turn_type   $test_n $detect_r $gmax $num_segments -marl sarl  $simlen -sumo_detect -detect_mode CAV -no_random_flow -seed $seed -detect_range $detect_range -succ_detect_rate $succ_detect_rate $gmin $nexp $n_step -act_ctm #-act_lp # -record_position 
        done
      done

      # NO LD
      # for i in 1 3 5 8 10 30 50 
      # do
      #   for seed in 13 23 33 #((j=1;j<=$test_round;j++));
      #   do
      #     python3 run.py -sim $sim -tsc cavlight -nogui -load -mode test -gamma 0.99 -updates $updates -pen_rate $i -flow_type $flow_type -temperature 100 -global_critic sep -tsc_program $tsc_program -turn_type $turn_type   $test_n $detect_r $gmax $num_segments -marl sarl  $simlen -sumo_detect -detect_mode CAV  -no_random_flow -seed $seed -detect_range $detect_range -succ_detect_rate $succ_detect_rate -act_ctm # -record_position
      #   done
      # done
    done
  done

#  mkdir experiments/${NOW}_round_${k}_CTM_70120demand/
#  mv experiments/cavlight/ experiments/${NOW}_round_${k}_CTM_70120demand/cavlight

  # maxpressure with CAV
  # for flow_type in bin_real #uniform_bd50 uniform_bd300 uniform_dynamic
  # do
  #   for i in 1 3 5 8 10 30 50 # 1 3 5 8 10 20 30 40 50 60 70 80 90 100 # 
  #   do
  #     for seed in 13 23 33  #for ((j=1;j<=$test_round;j++));
  #     do
  #       python3 run.py -sim $sim -tsc maxpressure -nogui -load -mode test -gamma 0.99 -updates $updates -pen_rate $i -flow_type $flow_type -temperature 100 -tsc_program $tsc_program -turn_type $turn_type   $test_n $detect_r $gmax -no_random_flow $simlen -sumo_detect -detect_mode CAV -act_ctm -no_random_flow -seed $seed # -record_position
  #     done
  #   done
  # done

  # mkdir experiments/round_${k}_CAV/
  # mv experiments/cavlight/ experiments/round_${k}/cavlight
  # mv experiments/maxpressure/ experiments/round_${k}_CAV/maxpressure


  # OPT TSC
#   for flow_type in bin_real #uniform_bd50 uniform_bd300 uniform_dynamic
#   do
#     for i in 1 3 5 8 10 30 50 # 1 3 5 8 10 20 30 40 50 60 70 80 90 100 # 
#     do
#       for ((j=1;j<=$test_round;j++));
#       do
#         python3 run.py -sim $sim -tsc opt -nogui -mode test -gamma 0.99 -updates $updates -pen_rate $i -flow_type $flow_type -temperature 100 -tsc_program $tsc_program -turn_type $turn_type   $test_n $detect_r $gmax -no_random_flow $simlen -sumo_detect -detect_mode CAV -act_ctm # -record_position
#       done
#     done
#   done

  # for flow_type in bin_real #uniform_bd50 uniform_bd300 uniform_dynamic
  # do
  #   for i in 1 3 5 8 10 20 30 40 50 60 70 80 90 100
  #   do
  #     for ((j=1;j<=$test_round;j++));
  #     do
  #       python3 run.py -sim $sim -tsc maxpressure -nogui -load -mode test -gamma 0.99 -updates $updates -pen_rate $i -flow_type $flow_type -temperature 0.3 -tsc_program $tsc_program -turn_type $turn_type   $test_n $detect_r $gmax -no_random_flow $simlen -sumo_detect -detect_mode CAV # -record_position
  #     done
  #   done
  # done

  # mkdir experiments/round_${k}_CAV/
  # #mv experiments/cavlight/ experiments/round_${k}/cavlight
  # mv experiments/maxpressure/ experiments/round_${k}_CAV/maxpressure

  # for flow_type in bin_real #uniform_bd50 uniform_bd300 uniform_dynamic
  # do
  #   for i in 1 3 5 8 10 20 30 40 50 60 70 80 90 100
  #   do
  #     for ((j=1;j<=$test_round;j++));
  #     do
  #       python3 run.py -sim $sim -tsc maxpressure -nogui -load -mode test -gamma 0.99 -updates $updates -pen_rate $i -flow_type $flow_type -temperature 0.3 -tsc_program $tsc_program -turn_type $turn_type   $test_n $detect_r $gmax -no_random_flow $simlen -sumo_detect -detect_mode CAV_w_intersection # -record_position
  #     done
  #   done
  # done

  # mkdir experiments/round_${k}_CAV_w_intersection/
  # #mv experiments/cavlight/ experiments/round_${k}/cavlight
  # mv experiments/maxpressure/ experiments/round_${k}_CAV_w_intersection/maxpressure


done

# zip experiments_2_2_2_dynamic_turn.zip experiments/ -r

echo "End"