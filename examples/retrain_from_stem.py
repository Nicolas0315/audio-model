# -*- coding: utf-8 -*-
"""実録音（またはDemucs分離ステム）から DDSP を再学習する。

ステップ1: 実楽曲 -> Demucs 分離 -> ソロ楽器ステム -> f0/loudness 抽出 ->
           固定長ウィンドウに切り出し -> DDSP を実音色で学習 -> 再合成で検証。

使い方:
  # 既にモノフォニックのソロ録音がある場合（分離不要）
  python examples/retrain_from_stem.py samples/solo.wav

  # ミックスから分離してから学習
  python examples/retrain_from_stem.py samples/song.wav --separate --instrument guitar

出力:
  outputs/real_ddsp_best.pt          学習済みモデル
  outputs/real_val_AB.wav            検証ウィンドウ（前半=実音 / 後半=DDSP再現）
  docs/real_ddsp_training_report.md  結果レポート
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
from physmod.core import write_wav  # noqa: E402
from pipeline.f0_track import track_f0, slice_windows  # noqa: E402
from pipeline.demucs_separate import separate, load_stem_mono  # noqa: E402

SR = 16000
HOP = 64
OUT = os.path.join(ROOT, "outputs")
DOCS = os.path.join(ROOT, "docs")


def load_audio_16k(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    x, fs = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    if fs != SR:
        g = gcd(int(fs), SR)
        x = resample_poly(x, SR // g, fs // g)
    x = x / (np.max(np.abs(x)) + 1e-9)
    return x.astype(np.float32)


def windows_to_tensors(windows, device):
    audio = torch.tensor(np.stack([w["audio"] for w in windows]), device=device)
    f0 = torch.tensor(np.stack([w["f0"] for w in windows]), device=device).unsqueeze(-1)
    n_samples = audio.shape[1]
    loud = compute_loudness(audio, SR, HOP).unsqueeze(-1)
    loud = (loud - loud.mean()) / (loud.std() + 1e-5)
    T = min(f0.shape[1], loud.shape[1])
    f0 = f0[:, :T]
    return audio, f0, scale_f0(f0), loud[:, :T], n_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="ソロ録音 or ミックス音源")
    ap.add_argument("--separate", action="store_true", help="先にDemucs分離する")
    ap.add_argument("--instrument", default="guitar")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True); os.makedirs(DOCS, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # 1. 入力（必要なら分離）
    src = args.input
    if args.separate:
        print(f"[分離] Demucs -> {args.instrument}")
        stems = separate(args.input, os.path.join(OUT, "stems"))
        if args.instrument not in stems:
            args.instrument = "other" if "other" in stems else list(stems)[0]
        x, _ = load_stem_mono(stems[args.instrument], target_fs=SR)
        x = x.astype(np.float32)
    else:
        x = load_audio_16k(src)
    print(f"[音声] {len(x)/SR:.1f}s @ {SR}Hz")

    # 2. f0/有声度 抽出
    print("[f0] 自己相関トラッキング ...")
    f0, voiced, loud = track_f0(x, SR, hop=HOP)
    print(f"  有声フレーム率 {np.mean(voiced)*100:.0f}%  "
          f"f0中央値 {np.median(f0[voiced]) if voiced.any() else 0:.1f}Hz")

    # 3. ウィンドウ化
    windows = slice_windows(x, SR, f0, voiced, hop=HOP)
    if len(windows) < 4:
        print(f"エラー: 学習に足るウィンドウが得られない（{len(windows)}）。"
              f"より長い/有声なモノフォニック録音が必要。")
        sys.exit(1)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(windows))
    n_val = max(1, int(len(windows) * args.val_frac))
    val_w = [windows[i] for i in idx[:n_val]]
    train_w = [windows[i] for i in idx[n_val:]]
    print(f"[データ] windows train={len(train_w)} val={len(val_w)} "
          f"(win 1.5s, stride 0.75s)")

    tr_audio, tr_f0, tr_f0s, tr_loud, n_samples = windows_to_tensors(train_w, device)
    va_audio, va_f0, va_f0s, va_loud, _ = windows_to_tensors(val_w, device)

    # 4. 学習
    model = DDSPAutoencoder(sample_rate=SR, hop=HOP).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    n_tr = tr_audio.shape[0]
    best = float("inf"); hist = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        b = torch.randint(0, n_tr, (min(args.batch, n_tr),), device=device)
        out = model(tr_f0[b], tr_f0s[b], tr_loud[b], n_samples)
        loss = multiscale_stft_loss(tr_audio[b], out["audio"])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step(); sched.step()
        if step % 250 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vo = model(va_f0, va_f0s, va_loud, n_samples)
                vloss = multiscale_stft_loss(va_audio, vo["audio"]).item()
            model.train()
            dt = time.time() - t0
            print(f"step {step:5d}/{args.steps} train={loss.item():.4f} val={vloss:.4f} {dt:.0f}s")
            hist.append({"step": step, "train": round(loss.item(), 5),
                         "val": round(vloss, 5), "sec": round(dt, 1)})
            if vloss < best:
                best = vloss
                torch.save({"model": model.state_dict(), "step": step,
                            "val_loss": vloss, "config": {"sr": SR, "hop": HOP}},
                           os.path.join(OUT, "real_ddsp_best.pt"))
    total = time.time() - t0
    print(f"学習完了 {args.steps} steps / {total:.0f}s best_val={best:.4f}")

    # 5. 検証再合成
    ckpt = torch.load(os.path.join(OUT, "real_ddsp_best.pt"),
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"]); model.eval()
    with torch.no_grad():
        vo = model(va_f0, va_f0s, va_loud, n_samples)
    tgt = va_audio[0].cpu().numpy(); mdl = vo["audio"][0].cpu().numpy()
    gap = np.zeros(int(0.3 * SR), dtype=np.float32)
    write_wav(os.path.join(OUT, "real_val_AB.wav"),
              np.concatenate([tgt, gap, mdl]), fs=SR)
    write_wav(os.path.join(OUT, "real_val_target.wav"), tgt, fs=SR)
    write_wav(os.path.join(OUT, "real_val_model.wav"), mdl, fs=SR)

    report = {"device": device, "source": os.path.basename(args.input),
              "separated": args.separate, "instrument": args.instrument if args.separate else "(direct)",
              "audio_sec": round(len(x) / SR, 1), "voiced_ratio": round(float(np.mean(voiced)), 3),
              "windows_train": len(train_w), "windows_val": len(val_w),
              "steps": args.steps, "best_val_mss_loss": round(best, 4),
              "wall_sec": round(total, 1), "history": hist}
    with open(os.path.join(OUT, "real_ddsp_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = [f"# 実録音からの DDSP 学習レポート", "",
          f"- 入力: `{os.path.basename(args.input)}`"
          + (f"（Demucs分離 → {args.instrument}）" if args.separate else "（直接）"),
          f"- 音声長: {len(x)/SR:.1f}s, 有声率 {np.mean(voiced)*100:.0f}%",
          f"- ウィンドウ: train {len(train_w)} / val {len(val_w)}（1.5s, stride 0.75s）",
          f"- device: {device}, {args.steps} steps, {total:.0f}s, params 0.70M",
          "", "## 結果",
          f"- best val multi-scale STFT loss: **{best:.4f}**",
          "- 出力: `outputs/real_val_AB.wav`（前半=実音 / 後半=DDSP再現）",
          "", "## 学習曲線", "", "| step | train | val | sec |", "|---:|---:|---:|---:|"]
    for h in hist:
        md.append(f"| {h['step']} | {h['train']:.4f} | {h['val']:.4f} | {h['sec']:.0f} |")
    with open(os.path.join(DOCS, "real_ddsp_training_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("レポート -> docs/real_ddsp_training_report.md")


if __name__ == "__main__":
    main()
