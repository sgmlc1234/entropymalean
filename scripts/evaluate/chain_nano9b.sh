#!/bin/bash
# Control then treatment for nemotron_nano_9b, without a gap between them.
#
# The cell's API is deprecated today, so the two arms have to run back to back;
# waiting for a human to notice the control finished would cost an hour of the
# window. The order still holds: the treatment does not start until the control
# has all 300 episodes, because the budget parity gate compares the two and a
# treatment that starts first has nothing to be checked against.
cd "$(dirname "$0")/../.." || exit 1
set -a; source .env; set +a
CTRL=$(python3 -c "import json;print(json.load(open('config/exam_cells.json'))['controls']['nemotron_nano_9b'])")
count() { cat "$CTRL"/episodes_*.jsonl 2>/dev/null | grep -c . || echo 0; }
until [ "$(count)" -ge 300 ]; do sleep 60; done
until [ "$(pgrep -f 'model-label nemotron_nano_9b' | wc -l | tr -d ' ')" = "0" ]; do sleep 20; done
echo "control complete at $(count); starting treatment"
exec env LEAN_REPL_POOL_SIZE=2 python3 scripts/evaluate/run_panel.py --model nemotron_nano_9b --arm treatment
