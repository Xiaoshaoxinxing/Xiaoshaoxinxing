#!/usr/bin/env bash
set -euo pipefail
cd /data/user02/Xiaoshaoxinxing
mkdir -p logs/evaluation/affectgpt_official_face_screen20_run01
exec .venv-eval/bin/python -B -u tools/run_affectgpt_official_face_merunibench.py queue \
  >> logs/evaluation/affectgpt_official_face_screen20_run01/supervisor.log 2>&1
