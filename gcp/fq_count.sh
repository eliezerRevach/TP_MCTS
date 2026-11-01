#!/usr/bin/env bash
# Size of the (facts, end queue) MDP -- no windows, no c, no deadline -- against the windows table of the same
# patterns (scripts/count_fq_states.py): nasa_rover 2 objects with the LLM patterns at blur 3 and blur 1000, and
# nasa_rover 3 objects at blur 3. The table's windows settings, deadline FQ_DEADLINE (25), no table cache.
#   bash gcp/fq_count.sh      -> results/fq_count/<tag>.log, one [fq] line per table + a [fq-total] line
set -u
cd /opt/tp_mcts
PY=.venv/bin/python
OUT=results/fq_count
DL=${FQ_DEADLINE:-25}
MEM_KB=${FQ_MEM_KB:-41943040}                   # 40 GB address-space cap per run
mkdir -p $OUT

run() {   # tag objects blur
  ( export PYTHONPATH=/opt/tp_mcts PYTHONIOENCODING=utf-8
    export TP_MCTS_WILAO_OFFLINE_SECONDS=0 TP_MCTS_WILAO_OFFLINE_EXTEND=1 TP_MCTS_WILAO_PATTERN_GOALS=1
    export TP_MCTS_WILAO_PATTERN_FACTS=10 TP_MCTS_WILAO_MISS=one TP_MCTS_WILAO_QUERY_SECONDS=0.05
    export TP_MCTS_WILAO_QUERY_EXPANSIONS=200 TP_MCTS_WILAO_AGG=same TP_MCTS_WILAO_GROUPING=1
    export TP_MCTS_WILAO_NO_STN_PASS=0 TP_MCTS_WILAO_BLUR=$3 TP_MCTS_WILAO_EXACT_STATICS=1
    export TP_MCTS_WILAO_LOGIC_SOURCES=random TP_MCTS_WILAO_LOGIC_ROLLOUTS=20 TP_MCTS_WILAO_LOGIC_DEPTH=60
    export TP_MCTS_WILAO_LOGIC_RANK=frequency TP_MCTS_WILAO_PATTERN_GROWTH=llm_by_hand TP_MCTS_WILAO_REPORT=0
    unset TP_MCTS_WILAO_TABLE_CACHE
    ulimit -v $MEM_KB
    exec $PY -u scripts/count_fq_states.py --domain nasa_rover --object_amount $2 --deadline $DL --cap 20000000
  ) > $OUT/$1_d$DL.log 2>&1 < /dev/null
  echo "[$(date +%H:%M:%S)] $1 d$DL finished: $(grep -c '^\[fq-total\]' $OUT/$1_d$DL.log) total line(s)"
}

echo "[$(date +%H:%M:%S)] start: deadline $DL"
run nasa2_llm_blur3 2 3 &
run nasa2_llm_blur1000 2 1000 &
run nasa3_llm_blur3 3 3 &
wait
echo "[$(date +%H:%M:%S)] all done"
