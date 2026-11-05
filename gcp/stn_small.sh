#!/usr/bin/env bash
# The full STN rules (TP_MCTS_WILAO_RULES=stn, blur 0) on SMALLER hand-picked patterns, against the windows PDB, on
# nasa_rover 1 object at D 25 -- so the STN's table builds in about the windows table's time, and TP-MCTS decides:
#   win3_pairs  : windows, blur 3, the automatic pairs patterns (the benchmark table's setting)
#   <tag>_<growth> for growth in llm_by_hand_nasa_one_rock | llm_by_hand_nasa_one_joint (hand_patterns.py):
#       stn  : the STN rules, blur 0      win0 : windows, blur 0      win3 : windows, blur 3
# Phase 1 builds every table (windows ones together, the two STN ones one after the other, 40 GB each);
# phase 2 runs Script 3 (experiments.ipynb cells 1 3 19, 100 runs) with 6 workers -- 12 workers x ~10 GB of nasa
# tables ran the 96 GB VM out of memory -- one scenario at a time, so every config runs under the same load.
#   bash gcp/stn_small.sh   -> results/stn_small/{prebuild,log,csv}/ and results/stn_small/summary.txt
# Other configs / folder: STN_SMALL_CONFIGS="name|rules|blur|growth ..." STN_SMALL_OUT=results/<dir> bash gcp/stn_small.sh
#   STN_SMALL_PHASES=tables  stop after the tables;  STN_SMALL_SEEDS="123 124"  one TP-MCTS run (100 runs) per seed;
#   STN_SMALL_STN_PARALLEL=1 build the STN tables together (STN_SMALL_STN_MEM_KB each, STN_SMALL_TIMEOUT seconds)
set -u
cd /opt/tp_mcts
PY=.venv/bin/python
OUT=${STN_SMALL_OUT:-results/stn_small}
export STN_SMALL_OUT=$OUT
mkdir -p $OUT/prebuild $OUT/log $OUT/csv
DOM=nasa_rover; OBJ=1; GARB=0; DL=${STN_SMALL_DEADLINE:-25}
WORKERS=${STN_SMALL_WORKERS:-6}
TIMECMD=""; [ -x /usr/bin/time ] && TIMECMD="/usr/bin/time -v"

wilao_env() {   # rules blur growth cache
  export PYTHONPATH=/opt/tp_mcts PYTHONIOENCODING=utf-8
  export TP_MCTS_WILAO_OFFLINE_SECONDS=0 TP_MCTS_WILAO_OFFLINE_EXTEND=1 TP_MCTS_WILAO_PATTERN_GOALS=1
  export TP_MCTS_WILAO_PATTERN_FACTS=10 TP_MCTS_WILAO_MISS=one TP_MCTS_WILAO_QUERY_SECONDS=0.05
  export TP_MCTS_WILAO_QUERY_EXPANSIONS=200 TP_MCTS_WILAO_AGG=same TP_MCTS_WILAO_GROUPING=1
  export TP_MCTS_WILAO_NO_STN_PASS=0 TP_MCTS_WILAO_EXACT_STATICS=1
  export TP_MCTS_WILAO_LOGIC_SOURCES=random TP_MCTS_WILAO_LOGIC_ROLLOUTS=20 TP_MCTS_WILAO_LOGIC_DEPTH=60
  export TP_MCTS_WILAO_LOGIC_RANK=frequency
  export TP_MCTS_WILAO_RULES=$1 TP_MCTS_WILAO_BLUR=$2 TP_MCTS_WILAO_PATTERN_GROWTH=$3 TP_MCTS_WILAO_TABLE_CACHE=$4
}

# name | rules | blur | growth
CONFIGS=("win3_pairs|windows|3|pairs"
         "stn_rock|stn|0|llm_by_hand_nasa_one_rock" "win0_rock|windows|0|llm_by_hand_nasa_one_rock"
         "win3_rock|windows|3|llm_by_hand_nasa_one_rock"
         "stn_joint|stn|0|llm_by_hand_nasa_one_joint" "win0_joint|windows|0|llm_by_hand_nasa_one_joint"
         "win3_joint|windows|3|llm_by_hand_nasa_one_joint")
[ -n "${STN_SMALL_CONFIGS:-}" ] && read -r -a CONFIGS <<< "$STN_SMALL_CONFIGS"

prebuild() {   # name rules blur growth mem_kb timeout
  ( wilao_env $2 $3 $4 /opt/tp_mcts/$OUT/cache_$1; ulimit -v $5
    $TIMECMD timeout $6 $PY -u scripts/prebuild_wilao_tables.py --domain $DOM --object_amount $OBJ \
      --garbage_amount $GARB --deadlines $DL ) > $OUT/prebuild/$1.log 2>&1 < /dev/null
  echo "[$(date +%H:%M:%S)] prebuild $1: $(grep -c '^\[prebuild\]' $OUT/prebuild/$1.log) line(s)," \
       "max RSS $(grep -o 'Maximum resident set size (kbytes): [0-9]*' $OUT/prebuild/$1.log | grep -o '[0-9]*$') kB"
}

script3() {   # name rules blur growth seed
  local run=$1; [ -n "${STN_SMALL_SEEDS:-}" ] && run=$1_s$5
  [ -f $OUT/csv/$run.csv ] && { echo "[skip] $run (csv exists)"; return; }
  grep -q '^\[prebuild\]' $OUT/prebuild/$1.log 2>/dev/null || { echo "[skip] $run (no table)"; return; }
  echo "[$(date +%H:%M:%S)] mcts $run"
  $PY -u scripts/run_notebook_cells.py experiments.ipynb --cells 1 3 19 \
    --set "DOMAINS=['$DOM']" --set "OBJECTS=[$OBJ]" --set "GARBAGE_AMOUNTS=[$GARB]" --set "DEADLINES=[$DL]" \
    --set "HEURISTICS=['windows_ilao_pdb']" --set "WORKERS=$WORKERS" --set "SEED=$5" \
    --set "SCRIPT3_CSV='/opt/tp_mcts/$OUT/csv/$run.tmp.csv'" \
    --env TP_MCTS_WILAO_RULES=$2 --env TP_MCTS_WILAO_BLUR=$3 --env TP_MCTS_WILAO_PATTERN_GROWTH=$4 \
    --env TP_MCTS_WILAO_TABLE_CACHE=/opt/tp_mcts/$OUT/cache_$1 \
    > $OUT/log/$run.log 2>&1 < /dev/null
  [ -f $OUT/csv/$run.tmp.csv ] && mv $OUT/csv/$run.tmp.csv $OUT/csv/$run.csv
  echo "[$(date +%H:%M:%S)] done $run $(grep -o 'success=[0-9]*/[0-9]*' $OUT/log/$run.log | tail -1)"
}

echo "[$(date +%H:%M:%S)] phase 1: tables (D $DL)"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name rules blur growth <<< "$cfg"
  [ "$rules" = windows ] && prebuild $name $rules $blur $growth 16777216 3600 &
done
STN_MEM=${STN_SMALL_STN_MEM_KB:-41943040}; STN_TIMEOUT=${STN_SMALL_TIMEOUT:-3600}
if [ "${STN_SMALL_STN_PARALLEL:-0}" = 1 ]; then
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name rules blur growth <<< "$cfg"
    [ "$rules" = stn ] && prebuild $name $rules $blur $growth $STN_MEM $STN_TIMEOUT &
  done
else
  ( for cfg in "${CONFIGS[@]}"; do
      IFS='|' read -r name rules blur growth <<< "$cfg"
      [ "$rules" = stn ] && prebuild $name $rules $blur $growth $STN_MEM $STN_TIMEOUT
    done ) &
fi
wait
if [ "${STN_SMALL_PHASES:-all}" != tables ]; then
  echo "[$(date +%H:%M:%S)] phase 2: TP-MCTS ($WORKERS workers)"
  for seed in ${STN_SMALL_SEEDS:-123}; do
    for cfg in "${CONFIGS[@]}"; do
      IFS='|' read -r name rules blur growth <<< "$cfg"
      script3 $name $rules $blur $growth $seed
    done
  done
fi

$PY - <<'PY' > $OUT/summary.txt
import csv, glob, json, os
OUT = os.environ.get("STN_SMALL_OUT", "results/stn_small")
print("config | build s | table states (per pattern) | h(s0) | score | wall s")
for log in sorted(glob.glob(f"{OUT}/prebuild/*.log")):
    name = os.path.basename(log)[:-4]
    pre = [l for l in open(log) if l.startswith("[prebuild]")]
    b = json.loads(pre[-1][len("[prebuild] "):]) if pre else None
    ts = b["report"].get("table_states") if b else None
    rows = [r for f in sorted(glob.glob(f"{OUT}/csv/{name}.csv") + glob.glob(f"{OUT}/csv/{name}_s*.csv"))
            for r in csv.DictReader(open(f))]
    score = (f'{sum(int(r["amount_success"]) for r in rows)}/{sum(int(r["runs_total"]) for r in rows)} '
             f'({" ".join(r["amount_success"] for r in rows)})') if rows else "-"
    wall = " ".join(r["wall_time_sec"] for r in rows) if rows else "-"
    print(f"{name} | {b['build_seconds'] if b else 'failed'} | {ts} | {b['h_s0'] if b else '-'} | {score} | {wall}")
PY
cat $OUT/summary.txt
echo "[$(date +%H:%M:%S)] all done"
