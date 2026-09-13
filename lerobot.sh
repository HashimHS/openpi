#!/usr/bin/env bash
#
# Collect episodes of a subtask-labeled RoboTwin task and convert them into a LeRobot
# dataset, with the atomic actions (pick / place / ...) labeled per frame.
#
#   bash collect_to_lerobot.sh [episode_num] [task_name] [task_config] [repo_id] [gpu_id]
#
# Defaults collect 100 episodes of stack_blocks_three_atomic into the LeRobot repo
# "stack_sample":
#
#   bash collect_to_lerobot.sh
#   bash collect_to_lerobot.sh 100 stack_blocks_three_atomic demo_clean stack_sample 0
#
# Environment overrides:
#   CONDA_ENV=robotwin     conda environment holding the RoboTwin dependencies
#   MODE=image             LeRobot image storage: "image" (PNG frames) or "video" (mp4,
#                          roughly 10x smaller, needs a working video backend)
#   SPLIT=0                1 = one LeRobot episode per atomic action instead of one per
#                          demonstration
#   CLEAN_PROCESSED=0      1 = delete the intermediate processed_data/ after converting
#
# Collection is resumable: seeds already in data/<task>/<config>/seed.txt and episodes
# that already have an hdf5 file are kept, so re-running tops the dataset up to
# episode_num instead of starting over. Conversion, in contrast, always rebuilds the
# LeRobot dataset at repo_id from scratch (the existing one is deleted).

set -euo pipefail

TASK_NAME=${1:-stack_blocks_three_atomic}
REPO_ID=${2:-$TASK_NAME}
SUBTASKS=${3:-0} # 0 = no subtask labels, 1 = subtask labels
GPU_ID=${4:-0}

CONDA_ENV=${CONDA_ENV:-robotwin}
MODE=${MODE:-image}
SPLIT=${SPLIT:-0}
CLEAN_PROCESSED=${CLEAN_PROCESSED:-0}

cd ../../
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $REPO_ROOT/policy/pi05/

# Converted straight from processed_data/. The training_data/ copy in the RoboTwin doc
# only exists to merge several tasks into one dataset, and would duplicate every frame.
CONVERT_ARGS=(--raw_dir "processed_data/${TASK_NAME}"
              --repo_id "$REPO_ID" --mode "$MODE")
[ "$SUBTASKS" = "1" ] && CONVERT_ARGS+=(--subtasks)
[ "$SPLIT" = "1" ] && CONVERT_ARGS+=(--split-subtask-episodes)

export XDG_CACHE_HOME="${REPO_ROOT}/policy/pi05/.cache"
uv run examples/kuka/convert_kuka_data_to_lerobot_robotwin.py "${CONVERT_ARGS[@]}"

# if [ "$CLEAN_PROCESSED" = "1" ]; then
#     log "Removing intermediate ${PROCESSED_DIR}"
#     rm -rf "processed_data/${TASK_NAME}-${TASK_CONFIG}-${COLLECTED}"
# fi

# ------------------------------------------------------------------ summary
OUT_DIR="${XDG_CACHE_HOME}/huggingface/lerobot/${REPO_ID}"
python - "$OUT_DIR" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
info = json.loads((out / "meta/info.json").read_text())
tasks = [json.loads(l)["task"] for l in (out / "meta/tasks.jsonl").open()]
print(f"  dataset  : {out}")
print(f"  episodes : {info['total_episodes']}   frames: {info['total_frames']}   fps: {info['fps']}")
print(f"  tasks    : {info['total_tasks']} distinct language instructions")
for task in tasks[:8]:
    print(f"      - {task}")
if len(tasks) > 8:
    print(f"      ... and {len(tasks) - 8} more")
PY