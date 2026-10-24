#!/usr/bin/env bash
# The table's LLM windows-PDB cells with the UB on (windows_lao.py "UB"): each table state keeps T(s), the
# least time to a goal (one Dijkstra from the goals), and a fitting state with r + x < T answers 0; on a
# table miss a forward Dijkstra from the leaf answers 0 when no goal is reachable in time (else the miss
# stays 1, MISS=one). Window rules and the shift fit are the default ones. Same settings as
# gcp/table_fill_blur.sh otherwise; its own cache (UB changes the table-cache key) and output folder.
#   TF_BLUR=3 bash gcp/table_fill_ub.sh prebuild   # builds every table, TF_PAR at a time (spare cores)
#   TF_BLUR=3 bash gcp/table_fill_ub.sh mcts       # Script 3 per cell once its table is built
# Outputs: results/table_fill_ub_blur$TF_BLUR/{csv,log,marks}/ (same names as table_fill.sh).
set -u
cd /opt/tp_mcts
PY=.venv/bin/python
BLUR=${TF_BLUR:-3}
OUT=results/table_fill_ub_blur$BLUR
CACHE=/opt/tp_mcts/results/wilao_cache_ub
PAR=${TF_PAR:-3}                                # parallel builds (12 MCTS workers + 3 = 15 of 16 cores)
PREBUILD_TIMEOUT=${PREBUILD_TIMEOUT:-1800}
PREBUILD_MEM_KB=${PREBUILD_MEM_KB:-31457280}    # 30 GB address-space cap per build
mkdir -p $OUT/csv $OUT/log $OUT/marks $CACHE

# tag | growth | domain | objects | garbage | deadlines -- in run order: light first, failure-prone last
CELLS=(
  "stuck1|llm_by_hand|stuck_car|1|0|10 15 30"
  "conc|llm_by_hand|conc|1|0|10 15 30"
  "nasa1|llm_by_hand|nasa_rover|1|0|25 35 50"
  "stuck2|llm_by_hand|stuck_car|2|0|10 20 30"
  "nasa2|llm_by_hand|nasa_rover|2|0|25 35 50"
  "ms2|llm_by_hand|machine_shop|2|0|17 27 40" "ms3|llm_by_hand|machine_shop|3|0|17 27 40"
  "nasa3|llm_by_hand|nasa_rover|3|0|25 35 50"
  "ms4|llm_by_hand|machine_shop|4|0|17 27 40"
)

wilao_env() {   # the notebook's windows settings (cell 3) with this blur and the UB on
  export PYTHONPATH=/opt/tp_mcts PYTHONIOENCODING=utf-8
  export TP_MCTS_WILAO_OFFLINE_SECONDS=0 TP_MCTS_WILAO_OFFLINE_EXTEND=1 TP_MCTS_WILAO_PATTERN_GOALS=1
  export TP_MCTS_WILAO_PATTERN_FACTS=10 TP_MCTS_WILAO_MISS=one TP_MCTS_WILAO_QUERY_SECONDS=0.05
  export TP_MCTS_WILAO_QUERY_EXPANSIONS=200 TP_MCTS_WILAO_AGG=same TP_MCTS_WILAO_GROUPING=1
  export TP_MCTS_WILAO_NO_STN_PASS=0 TP_MCTS_WILAO_BLUR=$BLUR TP_MCTS_WILAO_EXACT_STATICS=1
  export TP_MCTS_WILAO_LOGIC_SOURCES=random TP_MCTS_WILAO_LOGIC_ROLLOUTS=20 TP_MCTS_WILAO_LOGIC_DEPTH=60
  export TP_MCTS_WILAO_LOGIC_RANK=frequency TP_MCTS_WILAO_TABLE_CACHE=$CACHE
  export TP_MCTS_WILAO_UB=1 TP_MCTS_WILAO_MISS_UB=1
}

build_one() {   # tag growth domain objects garbage deadline
  local tag=$1 growth=$2 dom=$3 obj=$4 garb=$5 dl=$6
  local mark=$OUT/marks/${tag}_${growth}_d${dl}
  [ -f $mark.ok ] || [ -f $mark.fail ] && return
  echo "[$(date +%H:%M:%S)] prebuild $tag $growth d$dl"
  ( wilao_env; export TP_MCTS_WILAO_PATTERN_GROWTH=$growth; ulimit -v $PREBUILD_MEM_KB
    exec timeout -s KILL $PREBUILD_TIMEOUT $PY -u scripts/prebuild_wilao_tables.py --domain $dom \
      --object_amount $obj --garbage_amount $garb --deadlines $dl ) > $mark.log 2>&1 < /dev/null
  if grep -q '^\[prebuild\]' $mark.log; then touch $mark.ok; else touch $mark.fail; fi
  echo "[$(date +%H:%M:%S)]   -> $tag $growth d$dl $( [ -f $mark.ok ] && echo ok || echo FAIL )"
}

lane_prebuild() {
  local c tag growth dom obj garb dls dl
  export -f build_one wilao_env
  export OUT CACHE PY BLUR PREBUILD_TIMEOUT PREBUILD_MEM_KB
  for c in "${CELLS[@]}"; do
    IFS='|' read -r tag growth dom obj garb dls <<< "$c"
    for dl in $dls; do echo "$tag $growth $dom $obj $garb $dl"; done
  done | xargs -P $PAR -L 1 bash -c 'build_one "$@"' _
  echo "[$(date +%H:%M:%S)] prebuild lane finished"
}

lane_mcts() {
  local c tag growth dom obj garb dls dl mark name rc
  for c in "${CELLS[@]}"; do
    IFS='|' read -r tag growth dom obj garb dls <<< "$c"
    for dl in $dls; do
      mark=$OUT/marks/${tag}_${growth}_d${dl}
      while [ ! -f $mark.ok ] && [ ! -f $mark.fail ]; do sleep 20; done
      if [ -f $mark.fail ]; then echo "[skip] $tag $growth d$dl: no table in ${PREBUILD_TIMEOUT}s"; continue; fi
      name=${tag}_windows_ilao_pdb_${growth}_d${dl}
      [ -f $OUT/csv/$name.csv ] && continue
      echo "[$(date +%H:%M:%S)] start $name"
      $PY -u scripts/run_notebook_cells.py experiments.ipynb --cells 1 3 19 \
        --set "DOMAINS=['$dom']" --set "OBJECTS=[$obj]" --set "GARBAGE_AMOUNTS=[$garb]" --set "DEADLINES=[$dl]" \
        --set "HEURISTICS=['windows_ilao_pdb']" --set "SCRIPT3_CSV='/opt/tp_mcts/$OUT/csv/$name.tmp.csv'" \
        --env TP_MCTS_WILAO_PATTERN_GROWTH=$growth --env TP_MCTS_WILAO_TABLE_CACHE=$CACHE \
        --env TP_MCTS_WILAO_BLUR=$BLUR --env TP_MCTS_WILAO_UB=1 --env TP_MCTS_WILAO_MISS_UB=1 \
        > $OUT/log/$name.log 2>&1 < /dev/null
      rc=$?
      [ -f $OUT/csv/$name.tmp.csv ] && mv $OUT/csv/$name.tmp.csv $OUT/csv/$name.csv
      echo "[$(date +%H:%M:%S)] done  $name rc=$rc $(grep -o 'success=[0-9]*/[0-9]*' $OUT/log/$name.log | tail -1)"
    done
  done
  echo "[$(date +%H:%M:%S)] mcts lane finished"
}

case "${1:-}" in
  prebuild) lane_prebuild ;;
  mcts) lane_mcts ;;
  *) echo "usage: TF_BLUR=3 $0 prebuild|mcts"; exit 2 ;;
esac
