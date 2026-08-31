# -*- coding: utf-8 -*-
"""物理ピアノからモノフォニック独奏（DDSP学習用のクリーン素材）を生成する。

UVI 等の商用サンプルを使わず、自作の物理ピアノ（src/physmod/piano.py）で
ライセンス問題ゼロの学習素材を作る。単音を音階順に並べ（重ねない）、
f0 トラッカが効くようにする。出力は 16kHz mono。

使い方:
  python examples/make_piano_corpus.py            # samples/piano_solo.wav
  python examples/make_piano_corpus.py --lo 36 --hi 90 --step 1
"""
import os
import sys
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from physmod.piano import piano_note  # noqa: E402
from physmod.core import FS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=40, help="最低ノート(MIDI)")
    ap.add_argument("--hi", type=int, default=88, help="最高ノート(MIDI)")
    ap.add_argument("--step", type=int, default=2, help="半音ステップ")
    ap.add_argument("--note-sec", type=float, default=1.15)
    ap.add_argument("--out", default="samples/piano_solo.wav")
    args = ap.parse_args()

    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    rng = np.random.default_rng(3)
    midis = list(range(args.lo, args.hi + 1, args.step))
    gap = int(0.05 * FS)
    parts = []
    for m in midis:
        v = float(rng.uniform(0.45, 0.95))
        y = piano_note(m, args.note_sec, velocity=v, sustain=False, release=0.12)
        parts.append(y)
        parts.append(np.zeros(gap))
    solo = np.concatenate(parts)
    solo = solo / (np.max(np.abs(solo)) + 1e-9) * 0.9

    g = gcd(FS, 16000)
    solo16 = resample_poly(solo, 16000 // g, FS // g)

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sf.write(out, solo16.astype(np.float32), 16000)
    print(f"wrote {args.out}  dur={len(solo16)/16000:.1f}s  notes={len(midis)}")


if __name__ == "__main__":
    main()
