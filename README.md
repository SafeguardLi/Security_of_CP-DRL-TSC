# DRL-attacker

Code for *Security Analysis of Deep Reinforcement Learning Based Traffic Signal Control Systems Under Cooperative Perception Environment*.

The project studies how a DRL traffic signal controller (TSC) that relies on cooperative perception (CAV-shared detections) can be attacked by injecting fake vehicles, and how a consistency check can defend against it. All experiments run in SUMO.

- **Victims:** CAVLight (A2C, `-tsc cavlight`) and PressLight (DQN, `-tsc presslight`).
- **Proposed attack (DRL):** a PPO attacker picks the phase it wants the victim to choose. JSMA finds the smallest state perturbation that flips the victim's decision. A cell transmission model (CTM) and a Gurobi trajectory optimizer turn that perturbation into realistic fake-vehicle trajectories.
- **Baselines:** Random target phase, Sybil-DQN (Arabi et al., 2021) and multi-PPO (VehSec 2025).
- **Defense:** SDSM occupancy-map consistency check (`-sdsm_defense`).

---

## 1. Setup

```bash
conda env create -f environment.yml      # creates env "cosim" (Python 3.9, TF 2.13, PyTorch 2.4)
conda activate cosim
pip install -r requirements.txt          # extras: ART (JSMA), memoization
```

- **SUMO 1.20** must be installed separately (<https://sumo.dlr.de/docs/Installing/>), with `SUMO_HOME` set.
- **Gurobi** (`gurobipy` plus a valid licence) is needed only for physically realised fake trajectories (`-traj_gen`). Without it, attacks place fake vehicles at fixed target positions (the "static" mode).

---

## 2. Usage

Everything runs through one entry point:

```bash
python3 run.py -sim <network> -tsc <controller> -mode {train,test} [options]
```

- `-mode train` starts `-n` parallel simulation actors and `-l` learner processes.
- `-mode test` runs evaluation; use `-n 1` and loop over seeds.
- The `*.sh` scripts in the repo root are ready-made experiment recipes. Read them for the exact flag sets used in the paper.

### 2.1 Common flag set (isolated Plymouth intersection, TL 79)

```bash
COMMON="-sim plymouth -tsc cavlight -nogui -load -tsc_updates 10000 -gamma 0.99 \
  -pen_rate 5 -flow_type bin_real -turn_type real -temperature 100 -global_critic sep \
  -tsc_program 0 -detect_r 200 -gmax 400 -gmin 100 -num_segments 3 -marl sarl \
  -simlen 18000 -sumo_detect -detect_mode CAV -detect_range 80 -succ_detect_rate 100 \
  -no_random_flow -nreplay 512 -nsteps 1"
```

### 2.2 Typical runs

| Goal | Command |
|---|---|
| Benign victim | `python3 run.py $COMMON -mode test -n 1 -seed 13` |
| DRL attack (static fakes) | `... -mode test -n 1 -seed 13 -att_model drl -act_ctm -jsma_no_fallback -max_attack_scale 15 -updates 35` |
| DRL attack with realised trajectories | add `-traj_gen` (requires Gurobi) |
| Random baseline | `... -mode test -n 1 -seed 13 -att_model random` |
| Sybil-DQN baseline | see Section 2.3 |
| Attack under defense | add `-sdsm_defense` (see `plymouth_cavlight_*.sh`) |
| Train the DRL attacker | `... -mode train -n 16 -l 1 -save -save_u 5 -updates 35 -att_model drl -act_ctm` |

Notes:
- **Units:** `-simlen`, `-gmin`, `-gmax`, `-y` and `-r` are in 0.1 s simulation steps, so `-simlen 18000` is 1800 s.
- **Seeds:** the paper uses 3 seeds (`13 23 33`) per condition.
- **CAV market penetration rate (MPR):** set with `-pen_rate {1,3,5,10}`. Each MPR has its own victim model under `experiments/cavlight/CAV_pen_rate_<MPR>/`.
- **Demand level (100% / 120%):** this is set in `networks/plymouth/plymouth_test.rou.xml`. The extra 120% flows are commented out there; uncomment them for the 120% runs. Training uses `plymouth_train.rou.xml`.
- **multi-PPO (VehSec 2025)** runs only on the older code at commit `04f1ceb`. Check it out in a separate `git worktree` and symlink `experiments/attacker_cavlight` to `experiments/attacker_cavlight_VehSec2025`.
- **Run one simulation job at a time.** Output files are named by timestamp, so overlapping jobs make them hard to tell apart.

### 2.3 Sybil-DQN baseline

Sybil-DQN reproduces the Sybil attack of Arabi et al. (2021) as a black-box baseline. A PyTorch DQN (`src/rlagents/sybil_dqn.py`) chooses one of 6 injection actions at every victim decision: a street (side street NB+SB, or main-street through lanes) combined with a distance segment. It always injects, up to 15 fake vehicles per segment. Its reward is the victim's delay, the same delay term the DRL attacker uses. It runs without the CTM (no `-act_ctm`).

**Train** (distributed: `N` actors + 1 learner, about 50 min):

```bash
bash plymouth_sybil_train.sh                  # defaults: N=4 actors, UPD=10000 updates, MULT=6
N=8 UPD=20000 bash plymouth_sybil_train.sh    # override via environment variables
RESUME=1 bash plymouth_sybil_train.sh         # continue from the saved model instead of starting fresh
```

The script trains against the 5% MPR victim and writes:

- `experiments/attacker_cavlight/CAV_pen_rate_5.0/plymouth_bin_real_real/saved_models/sybil/79.pt` (weights)
- `.../saved_models/sybil/sybil_train_log.csv` (per-episode log; plot it with `python3 plot_sybil_training.py`)

Without `RESUME=1`, the script deletes both files first.

**Test** (3 seeds; uses `$COMMON` from Section 2.1):

```bash
for s in 13 23 33; do python3 run.py $COMMON -mode test -n 1 -updates 1 -seed $s -att_model sybil; done
```

`bash plymouth_sybil_test.sh` runs Benign, Random and Sybil-DQN back to back at 5% MPR.

- **Check that the trained model was loaded:** the log must print `[SybilDQN] loaded weights from .../sybil/79.pt`. If that line is missing, the run used a random policy.
- **Other MPRs:** the 5% attacker is reused. Copy `79.pt` to `experiments/attacker_cavlight/CAV_pen_rate_<MPR>/plymouth_bin_real_real/saved_models/sybil/79.pt`, then run with `-pen_rate <MPR>`.

Sybil-specific options:

| Flag | Default | Meaning |
|---|---|---|
| `-sybil_max_veh` | 15 | Fake vehicles injected per segment |
| `-sybil_replay_mult` | 6.0 | Updates allowed per collected transition (1.0 = strict 1:1) |
| `-sybil_lr` | 0.02 | DQN learning rate |
| `-sybil_opt` | `adam` | Optimizer (`adam` or `sgd`) |

### 2.4 Black-box (surrogate) attack

1. **Collect data:** run a benign test with `-collect_sa`, which logs (state, action) pairs to `_surrogate_data.p`.
2. **Train the surrogate:** `python3 train_surrogate.py` (CAVLight) or `train_surrogate_presslight.py`.
3. **Attack through the surrogate:** add `-surrogate_dir <dir>`. JSMA then reads gradients from the surrogate while the real victim still makes the decisions.

### 2.5 Key options

| Flag | Meaning |
|---|---|
| `-sim` | Network folder under `networks/` (`plymouth`, `plymouth_corr_3`, `plymouth_corr_4`, ...) |
| `-tsc` | Controller: `cavlight`, `presslight`, `maxpressure`, `websters`, `sotl`, `actuated`, `uniform`, `opt`, `mmitiss` |
| `-att_model` | Attacker: `drl`, `random`, `sybil`; omit for a benign run |
| `-tsc_updates` / `-updates` | Checkpoint (update count) of the victim / attacker to load or train |
| `-act_ctm` | Use the CTM for the attacker's state estimation |
| `-traj_gen` | Realise fake vehicles with the Gurobi trajectory optimizer (default: static placement) |
| `-max_attack_scale` | Max fake vehicles per feature (15 in the paper) |
| `-jsma_no_fallback` | Skip injection when JSMA finds no flipping features |
| `-sdsm_defense` | Enable the SDSM consistency defense |
| `-att_target` | Corridor runs: attack only this intersection id |
| `-surrogate_dir`, `-collect_sa`, `-tsc_surrogate` | Black-box surrogate pipeline |

Run `python3 run.py -h` for the full list.

---

## 3. Outputs

| What | Where |
|---|---|
| Victim models | `experiments/<tsc>/CAV_pen_rate_<MPR>/<sim>_<flow>_<turn>/saved_models/{actor,critic}/<tl>_<updates>.h5` |
| Attacker models | `experiments/attacker_<tsc>/CAV_pen_rate_<MPR>/<sim>_<flow>_<turn>/saved_models/` (`actor_app/`, `critic/`, `sybil/<tl>.pt`) |
| Per-decision records (test mode) | `experiments/attacker_<tsc>/.../state_action_record/<tl>/<timestamp>_*.p`: `state_action_record`, `JSMA`, `attack_phase_dist`, `fake_traj_input`, `CTM_cmp`, ... |
| SUMO trip info | `exp_log/sumo/<timestamp>trip_info.xml`. Each run writes a small header-only file at start and the full file at the end. |
| Defense logs | `exp_log/defense/` (trust-score CSVs, occupancy-map videos) |
| Run arguments | `experiments/attacker_<tsc>/.../args_config.txt` |

The headline metric is the mean `timeLoss` over all trips in a tripinfo file.

**Archived results:**

- `exp_log/sumo/usenix_arch/1401/`: USENIX runs (Benign / Random / DRL).
- `experiments/attacker_cavlight_1401_USENIX/`: the matching per-decision records.
- `exp_log/sumo/202610_TRC/<condition>/<baseline>/`: Sybil-DQN / multi-PPO baseline runs, with `manifest.csv` mapping each file to its seed and mean delay.

---

## 4. Repository structure

```
run.py                     entry point (parses args, launches DistProcs)
src/
  argparse.py              all command-line options
  distprocs.py             spawns simulation actors + learners (multiprocessing)
  simproc.py, learnerproc.py   actor / learner processes
  sumosim.py               SUMO/TraCI wrapper; writes per-run records
  networkdata.py           parses the .net.xml into lanes/phases
  tsc_factory.py, rl_factory.py, nn_factory.py, att_factory.py, rl_att_factory.py
  trafficsignalcontrollers/
    nextphaserltsc.py      CAVLight controller + the attack pipeline (JSMA, injection, CTM)
    nextphasepresslight*.py   PressLight benign / attack controllers
    maxpressuretsc.py, websterstsc.py, sotltsc.py, actuatedtsc.py, ...  classical baselines
    OptimizeBasedSignalControl_*.py, optimizationtsc.py, MMITISStsc.py  optimisation-based TSCs
  rlagents/                A2C (CAVLight), PPO (attacker), PressLight DQN, Sybil-DQN
  neuralnets/              actor / critic / DQN network definitions
  attackers/               attacker wrappers (next-phase attacker, Sybil action/injection)
  jsma.py, saliency_map.py JSMA perturbation search
  CTM.py                   cell transmission model for state estimation
  fake_veh_traj_gen.py     Gurobi fake-vehicle trajectory generation
  sdsm_defense.py, occupancy_map.py   SDSM consistency defense
networks/                  SUMO scenarios (net, routes, sumocfg, CTM cell tables)
  plymouth/                isolated intersection (TL 79), main scenario
  plymouth_corr_3/, plymouth_corr_4/   3- and 4-intersection corridors
experiments/               trained models + per-run records (see Section 3)
exp_log/                   tripinfo outputs, defense logs, result notebooks
figures/                   generated paper figures
surrogate_model_train/     surrogate training data / notebooks
*.sh                       experiment recipes
                           (plymouth_* = isolated intersection, corr3_* = corridor)
```

### Analysis and plotting

| Script | Purpose |
|---|---|
| `exp_log/result_prc.ipynb` | Delay tables and defense curves from tripinfo files |
| `VIZ_CAVLight.ipynb` | Per-phase durations, JSMA feature statistics |
| `plot_phase_duration_baselines.py` | Per-phase green-duration figure (Benign / Random / DRL / Sybil-DQN / multi-PPO) → `figures/phase_duration/` |
| `plot_defense_drl.py` | Defense figure (Benign / Attack / Defense vs MPR) → `figures/defense/` |
| `plot_sybil_training.py` | Sybil-DQN training curves |
| `analyze_*.py` | Diagnostics (attack table, CTM accuracy, flip clustering, trajectory-optimizer feasibility, ...) |

Result write-ups: `ATTACK_RESULTS_SUMMARY.md`, `CAVLight_attack_results.md`, `PressLight_*.md`.
