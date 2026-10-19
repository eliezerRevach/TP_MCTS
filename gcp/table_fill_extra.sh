#!/usr/bin/env bash
# Extra LLM (hand-picked) windows-PDB cells for the table, after gcp/table_fill.sh:
#   bash gcp/table_fill_extra.sh
# Builds each table now (one core, 30 min limit), then waits for the main Script 3 lane to finish
# ("mcts lane finished" in lane_mcts.log) so the 12 workers never overlap, then runs the cells.
# Same outputs / naming as table_fill.sh: results/table_fill/{csv,log,marks}/.
set -u
cd /opt/tp_mcts
PY=.venv/bin/python
OUT=results/table_fill
CACHE=/opt/tp_mcts/results/wilao_cache_table
mkdir -p $OUT/csv $OUT/log $OUT/marks $CACHE

# tag | domain | objects | garbage | deadlines   (growth llm_by_hand)
ROWS=(
  "stuck1|stuck_car|1|0|10 15 30"
  "conc|conc|1|0|10 15 30"
)

wilao_env() {
  export PYTHONPATH=/opt/tp_mcts PYTHONIOENCODING=utf-8
  export TP_MCTS_WILAO_OFFLINE_SECONDS=0 TP_MCTS_WILAO_OFFLINE_EXTEND=1 TP_MCTS_WILAO_PATTERN_GOALS=1
  export TP_MCTS_WILAO_PATTERN_FACTS=10 TP_MCTS_WILAO_MISS=one TP_MCTS_WILAO_QUERY_SECONDS=0.05
  export TP_MCTS_WILAO_QUERY_EXPANSIONS=200 TP_MCTS_WILAO_AGG=same TP_MCTS_WILAO_GROUPING=1
  export TP_MCTS_WILAO_NO_STN_PASS=0 TP_MCTS_WILAO_BLUR=3 TP_MCTS_WILAO_EXACT_STATICS=1
  export TP_MCTS_WILAO_LOGIC_SOURCES=random TP_MCTS_WILAO_LOGIC_ROLLOUTS=20 TP_MCTS_WILAO_LOGIC_DEPTH=60
  export TP_MCTS_WILAO_LOGIC_RANK=frequency TP_MCTS_WILAO_TABLE_CACHE=$CACHE
  export TP_MCTS_WILAO_PATTERN_GROWTH=llm_by_hand
}

for r in "${ROWS[@]}"; do
  IFS='|' read -r tag dom obj garb dls <<< "$r"
  for dl in $dls; do
    mark=$OUT/marks/${tag}_llm_by_hand_d${dl}
    [ -f $mark.ok ] || [ -f $mark.fail ] && continue
    echo "[$(date +%H:%M:%S)] prebuild $tag llm_by_hand d$dl"
    ( wilao_env; ulimit -v 62914560
      timeout 1800 $PY -u scripts/prebuild_wilao_tables.py --domain $dom --object_amount $obj \
        --garbage_amount $garb --deadlines $dl ) > $mark.log 2>&1 < /dev/null
    if grep -q '^\[prebuild\]' $mark.log; then touch $mark.ok; else touch $mark.fail; fi
  done
done

until grep -q "mcts lane finished" $OUT/lane_mcts.log; do sleep 60; done

for r in "${ROWS[@]}"; do
  IFS='|' read -r tag dom obj garb dls <<< "$r"
  for dl in $dls; do
    [ -f $OUT/marks/${tag}_llm_by_hand_d${dl}.ok ] || { echo "[skip] $tag d$dl: no table"; continue; }
    name=${tag}_windows_ilao_pdb_llm_by_hand_d${dl}
    [ -f $OUT/csv/$name.csv ] && { echo "[skip] $name (csv exists)"; continue; }
    echo "[$(date +%H:%M:%S)] start $name"
    $PY -u scripts/run_notebook_cells.py experiments.ipynb --cells 1 3 19 \
      --set "DOMAINS=['$dom']" --set "OBJECTS=[$obj]" --set "GARBAGE_AMOUNTS=[$garb]" --set "DEADLINES=[$dl]" \
      --set "HEURISTICS=['windows_ilao_pdb']" --set "SCRIPT3_CSV='/opt/tp_mcts/$OUT/csv/$name.tmp.csv'" \
      --env TP_MCTS_WILAO_PATTERN_GROWTH=llm_by_hand --env TP_MCTS_WILAO_TABLE_CACHE=$CACHE \
      > $OUT/log/$name.log 2>&1 < /dev/null
    rc=$?
    [ -f $OUT/csv/$name.tmp.csv ] && mv $OUT/csv/$name.tmp.csv $OUT/csv/$name.csv
    echo "[$(date +%H:%M:%S)] done  $name rc=$rc $(grep -o 'success=[0-9]*/[0-9]*' $OUT/log/$name.log | tail -1)"
  done
done
echo "[$(date +%H:%M:%S)] extra lane finished"
