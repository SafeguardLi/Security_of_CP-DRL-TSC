#!/bin/bash
# BENIGN baseline for the Sybil-DQN benchmark: CAVLight (TL 79) with NO attack, matching the
# victim's own config: temperature 100, act_ctm False, 5% CAV MPR (-pen_rate 5), bin_real flow,
# scale 1.4 (default), no random flow, 1800s (18000 ticks).
# NOTE: current benign ~75s (NOT the old USENIX ~52s): the USENIX 52s used the Feb-2026 test
# demand; plymouth_test.rou.xml gained 12 extra flows afterward (commits Apr-3 / May-1), ~+4%
# volume, which near saturation pushes benign 52s -> 75s. Same MPR, heavier demand.
# Run 3 seeds; the tally at the end prints mean timeLoss per sim.
sim=plymouth; tsc=cavlight; pen=5; victim=10000
SEEDS=${SEEDS:-"13 23 33"}
common="-sim $sim -tsc $tsc -nogui -load -mode test -tsc_updates $victim -updates 1 \
  -gamma 0.99 -pen_rate $pen -flow_type bin_real -temperature 100 -global_critic sep -tsc_program 0 \
  -turn_type real -n 1 -detect_r 200 -gmax 400 -gmin 100 -num_segments 3 -marl sarl -simlen 18000 \
  -sumo_detect -detect_mode CAV -detect_range 80 -succ_detect_rate 100 -no_random_flow -nreplay 512 -nsteps 1"

echo "########## BENIGN CAVLight (T=100, act_ctm=False): pen=$pen MPR, seeds=$SEEDS ##########"
START=$(date +%s)
for s in $SEEDS; do
  echo "### BENIGN seed $s ###"
  python3 run.py $common -seed $s
done

echo "########## DONE. Per-sim mean timeLoss (benign runs since this script started): ##########"
python3 - "$START" <<'PY'
import glob, os, sys, time, numpy as np, xml.etree.ElementTree as ET
cut = float(sys.argv[1])
fs = [f for f in glob.glob('exp_log/sumo/*trip_info.xml') if os.path.getmtime(f) >= cut]
for f in sorted(fs, key=os.path.getmtime):
    try:
        tl = [float(e.get('timeLoss')) for _, e in ET.iterparse(f) if e.tag == 'tripinfo']
        if tl:
            print(f"  {time.strftime('%H:%M', time.localtime(os.path.getmtime(f)))}  "
                  f"n={len(tl):>5}  meanTimeLoss={np.mean(tl):.1f}s")
    except Exception:
        pass
PY
