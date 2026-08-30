# -*- coding: utf-8 -*-
"""物理モデル PoC 一式を再実行し、outputs/ に音源を書き出す。

依存: numpy, scipy のみ（機械学習なし）。
検証内容:
  1. 波形導波管 撥弦のピッチ精度
  2. モーダル合成のモード周波数一致
  3. analysis-by-synthesis の往復（録音 -> 推定 -> 再合成 -> 誤差）
  4. 標準MIDIファイルの生成 -> 独立パーサ -> レンダリング
"""
import os
import sys

import numpy as np
import scipy.signal as sig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from physmod.core import (  # noqa: E402
    ks_pluck, modal_strike, estimate_modes, resynth_modes, mss_loss,
    measure_f0, write_wav, write_midi, parse_midi, render_midi,
    MARIMBA_RATIOS, MARIMBA_TAUS, FS,
)

OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=== 1. 波形導波管 撥弦: ピッチ精度 ===")
    for name, f in [("A2", 110.0), ("A3", 220.0), ("E4", 329.63)]:
        y = ks_pluck(f, 2.5, velocity=0.9)
        fm = measure_f0(y[:FS])
        print(f"  {name}: 目標 {f:.2f} -> 実測 {fm:.2f} Hz ({1200*np.log2(fm/f):+.1f} cent)")
    demo1 = np.concatenate([
        ks_pluck(110.0, 2.0, velocity=0.4), ks_pluck(110.0, 2.0, velocity=0.9),
        ks_pluck(220.0, 2.0, velocity=0.7), ks_pluck(329.63, 2.5, velocity=0.8)])
    write_wav(os.path.join(OUT, "01_waveguide_pluck.wav"), demo1)

    print("=== 2. モーダル合成 マリンバ: モード一致 ===")
    y2 = modal_strike(261.63, 3.0, velocity=0.9)
    spec = np.abs(np.fft.rfft(y2 * np.hanning(len(y2))))
    freqs = np.fft.rfftfreq(len(y2), 1 / FS)
    pk, _ = sig.find_peaks(spec, height=np.max(spec) * 2e-4, distance=int(30 * len(y2) / FS))
    print(f"  設計: {[round(261.63*r) for r in MARIMBA_RATIOS[:3]]} / 実測: {[round(f) for f in freqs[pk][:3]]}")
    demo2 = np.concatenate([modal_strike(261.63, 1.2, velocity=v) for v in (0.3, 0.6, 1.0)])
    write_wav(os.path.join(OUT, "02_modal_marimba.wav"), demo2)

    print("=== 3. analysis-by-synthesis 往復 ===")
    true_f0 = 196.0
    target = modal_strike(true_f0, 2.5, velocity=0.85)
    est = estimate_modes(target, n_modes=5)
    for f_e, tau_e, a_e in est[:3]:
        print(f"    est {f_e:8.2f} Hz, tau {tau_e:.3f}s")
    resyn = resynth_modes(est, 2.5)
    print(f"  MSS loss 再合成={mss_loss(target, resyn):.4f}")
    write_wav(os.path.join(OUT, "03_target_vs_resynth.wav"),
              np.concatenate([target, np.zeros(int(0.4 * FS)), resyn]))

    print("=== 4. MIDI経路 ===")
    tpq = 480

    def nev(beat, dur_b, note, vel, ch):
        return [(int(beat * tpq), "on", note, vel, ch),
                (int((beat + dur_b) * tpq), "off", note, 0, ch)]

    events = []
    for b, n, v in [(0, 45, 96), (1, 45, 70), (2, 48, 96), (3, 43, 80), (4, 45, 100), (6, 40, 90)]:
        events += nev(b, 0.9, n, v, 0)
    for b, n, v in [(0, 69, 80), (0.5, 72, 60), (1, 76, 90), (2, 79, 100),
                    (3, 76, 70), (4, 81, 110), (5, 76, 60), (6, 72, 80), (7, 69, 90)]:
        events += nev(b, 0.45, n, v, 1)
    midi_path = os.path.join(OUT, "04_demo.mid")
    write_midi(midi_path, events, tpq=tpq, bpm=100)
    parsed = parse_midi(midi_path)
    n_on = sum(1 for e in parsed if e[1] == "on")
    print(f"  SMF {len(events)} events -> note-on {n_on} 読戻し")
    write_wav(os.path.join(OUT, "04_midi_render.wav"), render_midi(parsed))
    print("完了 -> outputs/")


if __name__ == "__main__":
    main()
