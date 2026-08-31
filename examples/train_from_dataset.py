# -*- coding: utf-8 -*-
"""公開の単音データセット（TinySOL / NSynth / 汎用単音フォルダ）で DDSP を学習する。

単音はピッチが既知なので f0 は定数（音名/MIDIから）で条件付けし、CREPE 不要。
loudness は音声から A-weighting で抽出。音高で train/val を分割して汎化を測る。

使い方:
  # TinySOL の1楽器（例: Violin）で学習
  python examples/train_from_dataset.py --tinysol PATH/TinySOL --metadata PATH/TinySOL_metadata.csv \
         --instrument Violin --steps 4000

  # 汎用: ファイル名に音名を含む単音フォルダ
  python examples/train_from_dataset.py --folder PATH/notes --steps 4000

  # NSynth
  python examples/train_from_dataset.py --nsynth-json examples.json --nsynth-audio audio/ \
         --family keyboard --source acoustic --steps 4000
"""
import os
import sys
import time
import json
import argparse

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from neural.ddsp import (  # noqa: E402
    DDSPAutoencoder, multiscale_stft_loss, compute_loudness, scale_f0,
)
from neural.data import midi_to_hz  # noqa: E402
from physmod.core import write_wav  # noqa: E402
from pipeline.note_dataset import (  # noqa: E402
    load_tinysol, load_note_folder, load_nsynth,
)

SR = 16000
HOP = 64
OUT = os.path.join(ROOT, "outputs")
DOCS = os.path.join(ROOT, "docs")


def notes_to_tensors(notes, device):
    audio = torch.tensor(np.stack([nte.audio for nte in notes]), device=device)  # (B,N)
    n_samples = audio.shape[1]
    n_frames = n_samples // HOP
    # 既知ピッチ → 定数 f0 トラック
    f0_hz = torch.tensor([[midi_to_hz(nte.midi)] for nte in notes],
                         device=device).unsqueeze(1).repeat(1, n_frames, 1)  # (B,T,1)
    loud = compute_loudness(audio, SR, HOP).unsqueeze(-1)
    loud = (loud - loud.mean()) / (loud.std() + 1e-5)
    T = min(f0_hz.shape[1], loud.shape[1])
    f0_hz = f0_hz[:, :T]
    return audio, f0_hz, scale_f0(f0_hz), loud[:, :T], n_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tinysol", help="TinySOL 展開先ルート")
    ap.add_argument("--metadata", help="TinySOL_metadata.csv")
    ap.add_argument("--instrument", default=None, help="楽器名で絞る（推奨）")
    ap.add_argument("--folder", help="音名入りファイル名の単音フォルダ")
    ap.add_argument("--nsynth-json"); ap.add_argument("--nsynth-audio")
    ap.add_argument("--family"); ap.add_argument("--source")
    ap.add_argument("--max-notes", type=int, default=None)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--tag", default="dataset")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True); os.makedirs(DOCS, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 読み込み
    if args.tinysol:
        notes = load_tinysol(args.tinysol, args.metadata, instrument=args.instrument,
                             target_fs=SR, max_notes=args.max_notes)
        src = f"TinySOL/{args.instrument or 'all'}"
    elif args.nsynth_json:
        notes = load_nsynth(args.nsynth_json, args.nsynth_audio, family=args.family,
                            source=args.source, target_fs=SR, max_notes=args.max_notes)
        src = f"NSynth/{args.family or 'all'}"
    elif args.folder:
        notes = load_note_folder(args.folder, target_fs=SR, max_notes=args.max_notes)
        src = f"folder:{os.path.basename(args.folder)}"
    else:
        print("エラー: --tinysol / --folder / --nsynth-json のいずれかを指定"); sys.exit(1)

    if len(notes) < 8:
        print(f"エラー: 音数が少なすぎる（{len(notes)}）"); sys.exit(1)
    pitches = sorted(set(n.midi for n in notes))
    print(f"[データ] {src}  notes={len(notes)}  音域MIDI {min(pitches)}..{max(pitches)}")

    # 音高で train/val 分割（汎化を測る: 一部の音高を学習から除外）
    rng = np.random.default_rng(0)
    val_pitches = set(rng.choice(pitches, size=max(1, int(len(pitches) * args.val_frac)),
                                 replace=False).tolist())
    train = [n for n in notes if n.midi not in val_pitches]
    val = [n for n in notes if n.midi in val_pitches]
    if not val:
        val = train[:2]
    print(f"  train={len(train)}  val={len(val)}（学習外音高 {sorted(val_pitches)}）")

    tr = notes_to_tensors(train, device)
    va = notes_to_tensors(val, device)
    tr_audio, tr_f0, tr_f0s, tr_loud, n_samples = tr
    va_audio, va_f0, va_f0s, va_loud, _ = va

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
                torch.save({"model": model.state_dict(), "step": step, "val_loss": vloss,
                            "config": {"sr": SR, "hop": HOP}, "instrument": args.instrument or src},
                           os.path.join(OUT, f"ddsp_{args.tag}_best.pt"))
    total = time.time() - t0
    print(f"学習完了 {args.steps} steps / {total:.0f}s best_val={best:.4f}")

    ckpt = torch.load(os.path.join(OUT, f"ddsp_{args.tag}_best.pt"),
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"]); model.eval()
    with torch.no_grad():
        vo = model(va_f0, va_f0s, va_loud, n_samples)
    tgt = va_audio[0].cpu().numpy(); mdl = vo["audio"][0].cpu().numpy()
    gap = np.zeros(int(0.3 * SR), dtype=np.float32)
    write_wav(os.path.join(OUT, f"ddsp_{args.tag}_AB.wav"),
              np.concatenate([tgt, gap, mdl]), fs=SR)

    report = {"source": src, "instrument": args.instrument, "notes": len(notes),
              "pitch_range": [min(pitches), max(pitches)],
              "val_pitches": sorted(val_pitches), "train": len(train), "val": len(val),
              "steps": args.steps, "best_val_mss_loss": round(best, 4),
              "wall_sec": round(total, 1), "device": device, "history": hist}
    with open(os.path.join(OUT, f"ddsp_{args.tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"AB -> outputs/ddsp_{args.tag}_AB.wav  metrics -> outputs/ddsp_{args.tag}_metrics.json")


if __name__ == "__main__":
    main()
