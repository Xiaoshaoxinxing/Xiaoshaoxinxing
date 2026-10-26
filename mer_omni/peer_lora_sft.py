#!/usr/bin/env python3
"""Qwen2.5-Omni Thinker LoRA SFT for MER-Caption+ text targets.

Reads a JSONL manifest produced by prepare_mercaption.py. The CH-SIMS manifest
exists solely for a one-step pipeline check, not for MER-Caption+ training.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
          "capable of perceiving auditory and visual inputs, as well as generating text and speech.")


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    for row in rows:
        for key in ("id", "task", "question", "text", "audio", "target"):
            if key not in row:
                raise ValueError(f"Missing {key} in {path}")
    return rows


def video_frames(row: dict, cache: Path, max_frames: int) -> list[str]:
    if row.get("frames"):
        return [str(p) for p in row["frames"][:max_frames]]
    video = Path(row["video"])
    if not video.is_file():
        raise FileNotFoundError(video)
    import decord
    reader = decord.VideoReader(str(video), ctx=decord.cpu(0))
    if not len(reader):
        raise ValueError(f"Empty video: {video}")
    count = min(max_frames, len(reader))
    indices = np.linspace(0, len(reader)-1, count).round().astype(int).tolist()
    cache.mkdir(parents=True, exist_ok=True)
    result = []
    for slot, index in enumerate(indices):
        dest = cache / f"{row['id']}-{slot}.jpg"
        if not dest.is_file():
            temp = dest.with_suffix(".tmp.jpg")
            Image.fromarray(reader[index].asnumpy()).convert("RGB").save(temp, quality=85)
            os.replace(temp, dest)
        result.append(str(dest))
    return result


def conversation_for(row: dict, cache: Path, max_frames: int) -> list[dict]:
    content = [{"type": "image", "path": path} for path in video_frames(row, cache, max_frames)]
    content.extend([{"type": "audio", "path": str(row["audio"])},
                    {"type": "text", "text": row["text"]},
                    {"type": "text", "text": row["question"]}])
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content}]


def encode(processor, row: dict, cache: Path, max_frames: int, max_target_tokens: int, device):
    conversation = conversation_for(row, cache, max_frames)
    prompt = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt", padding=False)
    target_ids = processor.tokenizer(row["target"], add_special_tokens=False).input_ids
    target_ids = target_ids[:max_target_tokens]
    if not target_ids:
        raise ValueError(f"Empty target for {row['id']}")
    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    suffix = torch.tensor([target_ids + [end_id]], dtype=prompt["input_ids"].dtype)
    prefix_len = prompt["input_ids"].shape[1]
    prompt["input_ids"] = torch.cat((prompt["input_ids"], suffix), dim=1)
    if "attention_mask" in prompt:
        prompt["attention_mask"] = torch.cat((prompt["attention_mask"],
                                              torch.ones_like(suffix)), dim=1)
    labels = prompt["input_ids"].clone()
    labels[:, :prefix_len] = -100
    prompt["labels"] = labels
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in prompt.items()}


def save_checkpoint(model, optimizer, output: Path, epoch: int, next_index: int, global_step: int):
    ckpt = output / f"checkpoint-{global_step:07d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt / "adapter")
    torch.save({"optimizer": optimizer.state_dict(), "epoch": epoch,
                "next_index": next_index, "global_step": global_step}, ckpt / "state.pt")
    temp = output / "latest.tmp"
    temp.write_text(str(ckpt.resolve()), encoding="utf-8")
    os.replace(temp, output / "latest")
    print(f"checkpoint={ckpt}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--valid-manifest", type=Path,
                    help="Validation subset split by sample ID; never a MER-UniBench test set")
    ap.add_argument("--model-path", default="/home/zhaobohan/Qwen/models/Qwen2.5-Omni-3B")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--frame-cache", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-frames", type=int, default=2)
    ap.add_argument("--max-pixels", type=int, default=100352)
    ap.add_argument("--max-target-tokens", type=int, default=384)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Slurm GPU allocation is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.frame_cache.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.manifest)
    valid_rows = load_rows(args.valid_manifest) if args.valid_manifest else []
    if valid_rows and {row["id"] for row in rows} & {row["id"] for row in valid_rows}:
        raise ValueError("Training and validation sample IDs overlap")
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False
    from peft import LoraConfig, PeftModel, get_peft_model
    ckpt = None
    if args.resume:
        latest = args.output_dir / "latest"
        if not latest.is_file():
            raise FileNotFoundError(latest)
        ckpt = Path(latest.read_text(encoding="utf-8").strip())
        model = PeftModel.from_pretrained(model, ckpt / "adapter", is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules=r".*model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
            bias="none", task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    stop_requested = [False]
    signal.signal(signal.SIGUSR1, lambda _signum, _frame: stop_requested.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_requested.__setitem__(0, True))
    epoch_start = 1; index_start = 0; global_step = 0
    if ckpt is not None:
        state = torch.load(ckpt / "state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        epoch_start, index_start, global_step = state["epoch"], state["next_index"], state["global_step"]
    print(json.dumps({"rows": len(rows), "epoch_start": epoch_start, "index_start": index_start,
                      "global_step": global_step, "job_id": os.getenv("SLURM_JOB_ID")}), flush=True)
    model.train()
    for epoch in range(epoch_start, args.epochs + 1):
        order = list(range(len(rows)))
        random.Random(args.seed + epoch).shuffle(order)
        for position in range(index_start if epoch == epoch_start else 0, len(order)):
            row = rows[order[position]]
            inputs = encode(processor, row, args.frame_cache, args.max_frames,
                            args.max_target_tokens, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs, use_cache=False)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss for {row['id']}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            global_step += 1
            print(json.dumps({"epoch": epoch, "index": position, "step": global_step,
                              "id": row["id"], "task": row["task"], "loss": float(loss.item()),
                              "peak_gb": round(torch.cuda.max_memory_allocated()/1024**3, 3)}), flush=True)
            if global_step % args.save_every == 0 or (args.max_steps and global_step >= args.max_steps):
                save_checkpoint(model, optimizer, args.output_dir, epoch, position + 1, global_step)
            if stop_requested[0]:
                save_checkpoint(model, optimizer, args.output_dir, epoch, position + 1, global_step)
                print("Stop signal received; resumable checkpoint saved", flush=True)
                return
            if args.max_steps and global_step >= args.max_steps:
                return
        index_start = 0
        # Keep a checkpoint pointing at the completed epoch until validation
        # finishes, so a Slurm interruption can repeat validation on resume.
        save_checkpoint(model, optimizer, args.output_dir, epoch, len(order), global_step)
        if valid_rows:
            model.eval()
            losses = []
            with torch.inference_mode():
                for row in valid_rows:
                    if stop_requested[0]:
                        print("Stop signal during validation; validation will resume next job", flush=True)
                        return
                    inputs = encode(processor, row, args.frame_cache, args.max_frames,
                                    args.max_target_tokens, device)
                    losses.append(float(model(**inputs, use_cache=False).loss.item()))
            valid_loss = float(np.mean(losses))
            history_path = args.output_dir / "validation.jsonl"
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"epoch": epoch, "step": global_step,
                                         "validation_loss": valid_loss,
                                         "validation_rows": len(valid_rows)}) + "\n")
            previous = args.output_dir / "best_validation.json"
            prior_loss = json.loads(previous.read_text())["validation_loss"] if previous.is_file() else float("inf")
            if valid_loss < prior_loss:
                previous.write_text(json.dumps({"epoch": epoch, "step": global_step,
                                                "validation_loss": valid_loss,
                                                "adapter": str((args.output_dir / f"checkpoint-{global_step:07d}" / "adapter").resolve())},
                                               indent=2), encoding="utf-8")
            print(json.dumps({"epoch": epoch, "validation_loss": valid_loss}), flush=True)
            model.train()
        save_checkpoint(model, optimizer, args.output_dir, epoch + 1, 0, global_step)

    (args.output_dir / "completed.json").write_text(
        json.dumps({"epochs": args.epochs, "global_step": global_step,
                    "training_rows_per_epoch": len(rows), "validation_rows": len(valid_rows)},
                   indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
