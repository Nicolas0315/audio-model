# -*- coding: utf-8 -*-
"""DDSP オートエンコーダを学習完走させる。

やること:
  1. モノフォニック調波楽器コーパスを生成（学習/検証を音高で分割）
  2. [f0, loudness] を条件に harmonic+noise シンセを再構成するよう学習
  3. multi-scale STFT 損失で最適化（RTX4090 / CUDA 自動）
  4. 検証（学習外の音高）で汎化を測定、target/model を wav 出力
  5. 学習曲線と最終指標を docs/ddsp_training_report.md に記録、checkpoint 保存

使い方:
  python examples/train_ddsp.py            # 既定 4000 step
  python examples/train_ddsp.py --steps 6000 --batch 16
"""
import os
import sys
import time
import argparse
import json

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from neural.ddsp import (  # noqa: E402
    DDSPAutoencoder, multiscale_stft_loss, compute_loudness, scale_f0,
)
from neural.data import build_corpus, SR, N  # noqa: E402
from physmod.core import write_wav  # noqa: E402

HOP = 64
OUT = os.path.join(ROOT, "outputs")
DOCS = os.path.join(ROOT, "docs")


def notes_to_tensors(notes, device):
    audio = torch.tensor(np.stack([n.audio for n in notes]), device=device)  # (B, N)
    f0 = torch.tensor(np.stack([n.f0 for n in notes]), device=device)        # (B, N)
    # f0 をフレームレートへ間引き（音声レート -> フレーム）
    n_frames = N // HOP
    idx = torch.linspace(0, N - 1, n_frames, device=device).long()
    f0_frame = f0[:, idx].unsqueeze(-1)                                       # (B, T, 1)
    loud = compute_loudness(audio, SR, HOP).unsqueeze(-1)                     # (B, T, 1)
    # 正規化
    loud = (loud - loud.mean()) / (loud.std() + 1e-5)
    f0_scaled = scale_f0(f0_frame)
    # フレーム数の整合（compute_loudness の T に合わせる）
    T = min(f0_frame.shape[1], loud.shape[1])
    return audio, f0_frame[:, :T], f0_scaled[:, :T], loud[:, :T]


def evaluate(model, notes, device):
    model.eval()
    with torch.no_grad():
        audio, f0_hz, f0_sc, loud = notes_to_tensors(notes, device)
        out = model(f0_hz, f0_sc, loud, N)
        loss = multiscale_stft_loss(audio, out["audio"])
    model.train()
    return loss.item(), audio, out["audio"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DOCS, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"device={device}  torch={torch.__version__}")

    print("コーパス生成中 ...")
    train, val = build_corpus()
    print(f"  train={len(train)} notes  val={len(val)} notes (音高汎化用に分離)")

    # 学習データを事前にテンソル化（GPU 常駐）
    tr_audio, tr_f0, tr_f0s, tr_loud = notes_to_tensors(train, device)
    n_train = tr_audio.shape[0]

    model = DDSPAutoencoder(sample_rate=SR, hop=HOP).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"モデル: DDSP autoencoder  params={n_params/1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    history = []
    best_val = float("inf")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n_train, (args.batch,), device=device)
        audio = tr_audio[idx]
        out = model(tr_f0[idx], tr_f0s[idx], tr_loud[idx], N)
        loss = multiscale_stft_loss(audio, out["audio"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == 1:
            val_loss, v_tgt, v_out = evaluate(model, val, device)
            dt = time.time() - t0
            print(f"step {step:5d}/{args.steps}  train={loss.item():.4f}  "
                  f"val={val_loss:.4f}  lr={sched.get_last_lr()[0]:.1e}  {dt:.0f}s")
            history.append({"step": step, "train": round(loss.item(), 5),
                            "val": round(val_loss, 5), "sec": round(dt, 1)})
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "step": step,
                            "val_loss": val_loss, "config": {"sr": SR, "hop": HOP}},
                           os.path.join(OUT, "ddsp_best.pt"))

    total = time.time() - t0
    print(f"\n学習完了: {args.steps} steps / {total:.0f}s  best_val={best_val:.4f}")

    # 最良モデルで検証音を再合成し、聴き比べ用に出力
    ckpt = torch.load(os.path.join(OUT, "ddsp_best.pt"), map_location=device,
                      weights_only=True)
    model.load_state_dict(ckpt["model"])
    val_loss, v_tgt, v_out = evaluate(model, val, device)
    # 先頭の検証ノート（学習外の音高）を書き出し
    tgt = v_tgt[0].detach().cpu().numpy()
    mdl = v_out[0].detach().cpu().numpy()
    write_wav(os.path.join(OUT, "ddsp_val_target.wav"), tgt, fs=SR)
    write_wav(os.path.join(OUT, "ddsp_val_model.wav"), mdl, fs=SR)
    # target と model を連結（前半=本物 / 後半=DDSP再現）
    gap = np.zeros(int(0.3 * SR), dtype=np.float32)
    write_wav(os.path.join(OUT, "ddsp_val_AB.wav"),
              np.concatenate([tgt, gap, mdl]), fs=SR)

    # レポート出力
    report = {
        "device": device,
        "params_million": round(n_params / 1e6, 3),
        "steps": args.steps,
        "batch": args.batch,
        "train_notes": len(train),
        "val_notes": len(val),
        "held_out_pitches_midi": [50, 57, 64, 71, 78],
        "best_val_mss_loss": round(best_val, 4),
        "final_val_mss_loss": round(val_loss, 4),
        "wall_sec": round(total, 1),
        "history": history,
    }
    with open(os.path.join(OUT, "ddsp_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = [
        "# DDSP 学習レポート",
        "",
        f"- device: **{device}**（torch {torch.__version__}）",
        f"- モデル: DDSP autoencoder, params **{n_params/1e6:.2f}M**",
        f"- 学習: **{args.steps} steps**, batch {args.batch}, {total:.0f}s",
        f"- データ: 合成モノフォニック調波楽器 train {len(train)} / val {len(val)} notes",
        f"- 検証は**学習に含めない音高**（MIDI 50/57/64/71/78）で音高汎化を測定",
        "",
        f"## 結果",
        f"- best val multi-scale STFT loss: **{best_val:.4f}**",
        f"- final val loss: **{val_loss:.4f}**",
        "- 出力: `outputs/ddsp_val_AB.wav`（前半=本物 / 後半=DDSP再現）",
        "",
        "## 学習曲線",
        "",
        "| step | train | val | sec |",
        "|---:|---:|---:|---:|",
    ]
    for h in history:
        md.append(f"| {h['step']} | {h['train']:.4f} | {h['val']:.4f} | {h['sec']:.0f} |")
    md += [
        "",
        "## 意味",
        "この学習器は f0 と loudness だけを条件に楽器音を再構成する。",
        "学習外の音高でも損失が下がることは、音色を**制御可能な形で**獲得したことを示す。",
        "f0/loudness を MIDI から与えれば任意ピッチ・強弱で演奏でき、",
        "実運用では入力を Demucs 分離ステム＋CREPE 抽出の f0 に差し替えるだけで、",
        "実楽器の音色学習に移行できる。",
    ]
    with open(os.path.join(DOCS, "ddsp_training_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"レポート -> docs/ddsp_training_report.md  metrics -> outputs/ddsp_metrics.json")


if __name__ == "__main__":
    main()
