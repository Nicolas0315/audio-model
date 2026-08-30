# -*- coding: utf-8 -*-
"""ミックス音源 -> Demucs 分離 -> ステムから物理パラメータ推定 -> 再合成。

Demucs の物理モデリングへの応用を通しで実証するパイプライン。
  1. ミックス音源を htdemucs_6s で分離
  2. 指定楽器ステム（guitar/piano/bass 等）を取り出し
  3. 単発ノートを検出して analysis-by-synthesis でモード推定
  4. 推定パラメータから再合成し、MSS loss でターゲットとの近さを測る

使い方:
  python examples/mix_to_model.py samples/track.wav --instrument guitar
"""
import os
import sys
import argparse

import numpy as np
import scipy.signal as sig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from physmod.core import (  # noqa: E402
    estimate_modes, resynth_modes, mss_loss, measure_f0, write_wav, FS,
)
from pipeline.demucs_separate import separate, load_stem_mono  # noqa: E402

OUT = os.path.join(ROOT, "outputs")


def first_note_segment(x, fs, min_sec=0.4, max_sec=2.5):
    """最初の明瞭なオンセットを検出し、そこから1音分を切り出す。"""
    env = np.abs(sig.hilbert(x))
    env = sig.savgol_filter(env, min(1001, len(env) // 2 * 2 + 1), 3)
    thr = np.max(env) * 0.15
    onset = int(np.argmax(env > thr))
    seg = x[onset:onset + int(max_sec * fs)]
    if len(seg) < int(min_sec * fs):
        seg = x[onset:]
    return seg, onset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="ミックス音源 (wav/mp3/flac)")
    ap.add_argument("--instrument", default="guitar",
                    help="対象ステム名 (guitar/piano/bass/other/drums/vocals)")
    ap.add_argument("--model", default="htdemucs_6s")
    ap.add_argument("--n-modes", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    stem_dir = os.path.join(OUT, "stems")

    print(f"[1/4] Demucs 分離: {args.input}")
    stems = separate(args.input, stem_dir, model=args.model)
    if args.instrument not in stems:
        print(f"  警告: '{args.instrument}' は分離結果にない。利用可能: {list(stems)}")
        args.instrument = "other" if "other" in stems else list(stems)[0]
        print(f"  -> '{args.instrument}' を使用")

    print(f"[2/4] ステム読み込み: {args.instrument}")
    x, fs = load_stem_mono(stems[args.instrument], target_fs=FS)

    print("[3/4] オンセット検出 + モード推定")
    seg, onset = first_note_segment(x, fs)
    f0 = measure_f0(seg[:fs]) if len(seg) >= fs else measure_f0(seg)
    print(f"  onset={onset/fs:.3f}s  推定f0={f0:.2f} Hz")
    est = estimate_modes(seg, n_modes=args.n_modes)
    for f_e, tau_e, a_e in est:
        print(f"    mode {f_e:8.2f} Hz  tau {tau_e:.3f}s  amp {a_e:.4f}")

    print("[4/4] 再合成 + 評価")
    dur = len(seg) / fs
    resyn = resynth_modes(est, dur)
    loss = mss_loss(seg, resyn)
    print(f"  MSS loss (ステム音 vs モデル再合成) = {loss:.4f}")

    write_wav(os.path.join(OUT, f"stem_{args.instrument}_target.wav"), seg)
    write_wav(os.path.join(OUT, f"stem_{args.instrument}_model.wav"), resyn)
    print(f"  -> outputs/stem_{args.instrument}_target.wav / _model.wav")
    print("完了。target と model を聴き比べて、モデルの説得力を確認する。")


if __name__ == "__main__":
    main()
