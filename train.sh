#!/usr/bin/env bash
# Train the padding model with main CE for one complete pass through MER-Caption+.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$project_dir"

# A cloned checkout can use either its own venv or an explicitly chosen environment.
if [[ -n "${MER_OMNI_PYTHON:-}" ]]; then
    training_python="$MER_OMNI_PYTHON"
elif [[ -x .venv/bin/python ]]; then
    training_python="$project_dir/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    training_python=python
else
    training_python=python3
fi
if ! command -v "$training_python" >/dev/null 2>&1; then
    printf '%s\n' '找不到 Python；请激活环境或设置 MER_OMNI_PYTHON。' >&2
    exit 1
fi

# An interactive SSH launch gets a persistent terminal automatically.
if [[ -z "${TMUX:-}" && -t 0 && -t 1 ]] && command -v tmux >/dev/null 2>&1; then
    printf -v training_command 'bash %q' "$project_dir/train.sh"
    exec tmux new-session -A -s mer_omni_padding_epoch01 -c "$project_dir" "$training_command"
fi

if [[ -e runs/mer_omni_padding_epoch01 ]]; then
    printf '%s\n' '训练目录 runs/mer_omni_padding_epoch01 已存在；请使用已有检查点续训，避免覆盖。' >&2
    exit 1
fi
if [[ ! -f data/calibration/mercaptionplus-atsuko-padding-bands.json ]]; then
    printf '%s\n' '新补零方案尚未全量校准，请先执行：' \
        'python tools/mer_omni.py --config config/mer_omni_atsuko_padding.json calibrate --device cuda:0' >&2
    exit 1
fi

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
training_log_dir="logs/training/mer_omni_padding_epoch01"
mkdir -p "$training_log_dir"

printf '\n%s\n' '  Atsuko + Omni 3B · 一轮训练' \
    '  目标：完整遍历 31,327 条样本一次，共 10,443 步，最后一步 1 条。' \
    '  损失：仅主融合路径的语言 CE，不计算辅助损失。' \
    "  日志：$training_log_dir/console.log" ''
"$training_python" -u tools/mer_omni.py --config config/mer_omni_atsuko_padding.json train \
    --device cuda:0 \
    --output runs/mer_omni_padding_epoch01 \
    --log-every 1 \
    2>&1 | tee -a "$training_log_dir/console.log"
