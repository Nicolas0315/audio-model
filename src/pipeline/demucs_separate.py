# -*- coding: utf-8 -*-
"""Demucs による音源分離ラッパ。

ミックス音源を楽器ステムへ分離し、物理モデリングの「クリーンなターゲット音」を得る。
htdemucs_6s モデルは drums / bass / other / vocals に加え guitar / piano を分離できるため、
撥弦（ギター）・打弦（ピアノ）の物理モデル用ターゲット抽出に向く。

依存: demucs, torch, soundfile
GPU があれば自動で利用する（RTX4090 等）。
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _to_mono(x: np.ndarray) -> np.ndarray:
    """(channels, samples) or (samples, channels) を mono float64 にする。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x
    # demucs のテンソルは (channels, samples)
    if x.shape[0] <= 8 and x.shape[0] < x.shape[1]:
        return x.mean(axis=0)
    return x.mean(axis=1)


def separate(
    input_path: str,
    out_dir: str,
    model: str = "htdemucs_6s",
    device: Optional[str] = None,
) -> Dict[str, str]:
    """input_path を分離し、各ステムを out_dir に wav 保存する。

    Returns: {stem_name: wav_path}
    """
    import torch
    import soundfile as sf
    from demucs.api import Separator, save_audio

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(out_dir, exist_ok=True)
    sep = Separator(model=model, device=device)
    print(f"[demucs] model={model} device={device} -> {input_path}")

    origin, stems = sep.separate_audio_file(input_path)
    sr = sep.samplerate

    stem = Path(input_path).stem
    out: Dict[str, str] = {}
    for name, tensor in stems.items():
        p = os.path.join(out_dir, f"{stem}__{name}.wav")
        save_audio(tensor, p, samplerate=sr)
        out[name] = p
        print(f"[demucs]   {name:8s} -> {p}")
    return out


def load_stem_mono(wav_path: str, target_fs: int = 48000):
    """分離済みステムを mono / target_fs で読み込み、物理モデル分析に渡せる形にする。

    Returns: (mono_float64, fs)
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    x, fs = sf.read(wav_path, always_2d=False)
    x = _to_mono(x)
    if fs != target_fs:
        # 近似有理数リサンプル
        from math import gcd
        g = gcd(int(fs), int(target_fs))
        x = resample_poly(x, target_fs // g, fs // g)
        fs = target_fs
    peak = np.max(np.abs(x)) + 1e-12
    return x / peak, fs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Demucs で音源をステム分離する")
    ap.add_argument("input", help="入力音源 (wav/mp3/flac)")
    ap.add_argument("--out", default="outputs/stems", help="出力ディレクトリ")
    ap.add_argument("--model", default="htdemucs_6s",
                    help="demucs モデル (htdemucs / htdemucs_6s / mdx_extra 等)")
    ap.add_argument("--device", default=None, help="cuda / cpu (既定: 自動)")
    args = ap.parse_args()

    res = separate(args.input, args.out, model=args.model, device=args.device)
    print("done:", res)
