#!/usr/bin/env bash
set -euo pipefail
cd /data/user02/Xiaoshaoxinxing
mkdir -p logs/training/peer_lora_epoch2_run02
exec .venv/bin/python -B -u tools/peer_lora_epoch2.py supervise \
    >> logs/training/peer_lora_epoch2_run02/supervisor.log 2>&1
