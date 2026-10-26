#!/usr/bin/env bash
set -euo pipefail
cd /data/user02/Xiaoshaoxinxing
mkdir -p logs/evaluation/peer_lora_epoch2_run02/checkpoint_search_20_seed42
exec .venv/bin/python -B -u tools/search_peer_lora_checkpoints.py pipeline \
    >> logs/evaluation/peer_lora_epoch2_run02/checkpoint_search_20_seed42/supervisor.log 2>&1
