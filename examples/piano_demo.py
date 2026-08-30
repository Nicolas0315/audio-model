# -*- coding: utf-8 -*-
"""物理ピアノモデルのデモと検証。

  1. 単音を数音レンダリング
  2. 設計インハーモニシティ B と、合成音から回収した B を比較（物理の正しさ検証）
  3. ベロシティ違い（弱/中/強）で輝度が変わることを確認
  4. MIDI（和音進行）をピアノで演奏
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from physmod.core import write_wav, write_midi, parse_midi, midi_to_freq, FS  # noqa: E402
from physmod.piano import (  # noqa: E402
    piano_note, render_piano_midi, inharmonicity, measure_inharmonicity,
)

OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=== 1. インハーモニシティの設計 vs 回収（物理の検証） ===")
    for midi in [40, 52, 64, 76, 88]:
        f0 = midi_to_freq(midi)
        y = piano_note(midi, 3.0, velocity=0.8, sustain=True)
        B_design = inharmonicity(midi)
        B_meas, ns, fs_ = measure_inharmonicity(y, f0, n_partials=12)
        # 第8部分音のズレ（セント）で体感
        n = 8
        ideal = n * f0
        inharm = n * f0 * np.sqrt(1 + B_design * n * n)
        cents = 1200 * np.log2(inharm / ideal)
        print(f"  MIDI{midi:3d} f0={f0:6.1f}Hz  B設計={B_design:.2e} 回収={B_meas:.2e}"
              f"  8倍音の上ずり {cents:+.1f}cent")

    print("\n=== 2. ベロシティ→輝度（スペクトル重心） ===")
    for v in [0.25, 0.6, 1.0]:
        y = piano_note(64, 2.0, velocity=v, sustain=True)
        spec = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1 / FS)
        centroid = np.sum(freqs * spec) / (np.sum(spec) + 1e-9)
        print(f"  velocity {v:.2f} -> スペクトル重心 {centroid:6.0f} Hz")

    print("\n=== 3. 単音デモ音源 ===")
    demo = np.concatenate([
        piano_note(40, 2.2, velocity=0.7, sustain=True),
        piano_note(52, 2.0, velocity=0.6, sustain=True),
        piano_note(64, 1.8, velocity=0.9, sustain=True),
        piano_note(76, 1.6, velocity=0.8, sustain=True),
        piano_note(88, 1.4, velocity=0.7, sustain=True),
    ])
    write_wav(os.path.join(OUT, "piano_notes.wav"), demo)
    print("  -> outputs/piano_notes.wav (A1, E3, E4, E5, E6)")

    print("\n=== 4. MIDI 和音進行をピアノで演奏 ===")
    tpq = 480

    def chord(beat, dur_b, notes, vel):
        ev = []
        for m in notes:
            ev += [(int(beat * tpq), "on", m, vel, 0),
                   (int((beat + dur_b) * tpq), "off", m, 0, 0)]
        return ev

    events = []
    # I-V-vi-IV (C-G-Am-F) 風
    events += chord(0, 1.9, [48, 60, 64, 67], 90)   # C
    events += chord(2, 1.9, [43, 59, 62, 67], 85)   # G
    events += chord(4, 1.9, [45, 60, 64, 69], 88)   # Am
    events += chord(6, 1.9, [41, 60, 65, 69], 82)   # F
    # 右手メロディ
    for b, m, v in [(0, 72, 100), (0.5, 76, 80), (1, 79, 95), (2, 74, 90),
                    (3, 71, 75), (4, 76, 100), (5, 72, 80), (6, 69, 90), (7, 72, 85)]:
        events += [(int(b * tpq), "on", m, v, 0), (int((b + 0.45) * tpq), "off", m, 0, 0)]

    write_midi(os.path.join(OUT, "piano_demo.mid"), events, tpq=tpq, bpm=90)
    parsed = parse_midi(os.path.join(OUT, "piano_demo.mid"))
    mix = render_piano_midi(parsed)
    write_wav(os.path.join(OUT, "piano_performance.wav"), mix)
    print(f"  note-on {sum(1 for e in parsed if e[1]=='on')} 件 -> outputs/piano_performance.wav")
    print("完了。")


if __name__ == "__main__":
    main()
