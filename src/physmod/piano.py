# -*- coding: utf-8 -*-
"""物理モデリング・ピアノ（Pianoteq 系のアプローチを参照した自作実装）。

サンプルを一切使わず、ピアノの音を物理から組み立てる。モデル化する要素:

  1. 弦の剛性 → インハーモニシティ  f_n = n·f0·√(1+B·n²)（B は音域で変化）
  2. ハンマー打弦                  打弦位置のコーム（sin(nπp)）+ ベロシティ依存の輝度
  3. 複弦（ユニゾン）のうなり       わずかに離調した 2〜3 本の弦の和
  4. 二段階減衰                    prompt（速い）+ aftersound（遅い）の混合
  5. 高次ほど速い減衰              tau_n ∝ 1/(1+c·n²)
  6. 響板（soundboard）の色付け     低域の共鳴を軽く付与
  7. ダンパー                      ノートオフで減衰を速める（サステイン制御）

すべて numpy。ベロシティ・サステインを与えれば MIDI で演奏できる。
これ自体が物理モデル音源であり、DDSP 学習用のクリーンな素材にもなる。
"""
from __future__ import annotations
import numpy as np
import scipy.signal as sig

from .core import FS, midi_to_freq, write_wav, write_midi, parse_midi


def inharmonicity(midi: float) -> float:
    """音域による B の近似。中音を基準に高音で顕著に増える（Conklin/Fletcher 的傾向）。"""
    B0 = 8e-5
    B = B0 * 2 ** (0.55 * (midi - 60) / 12.0)
    # 低音側は巻線でやや増える方向に軽く持ち上げ
    if midi < 48:
        B *= 1.0 + (48 - midi) * 0.03
    return float(np.clip(B, 1e-5, 2e-3))


def piano_note(midi: float, dur: float, fs: int = FS, velocity: float = 0.8,
               strike_pos: float = 0.125, n_strings: int = 2,
               detune_cents: float = 1.4, sustain: bool = True,
               release: float = 0.18, seed: int = 0) -> np.ndarray:
    """1音を合成して返す（[-1,1] 付近）。"""
    rng = np.random.default_rng(seed + int(midi))
    n = int(dur * fs)
    t = np.arange(n) / fs
    f0 = midi_to_freq(midi)
    B = inharmonicity(midi)

    # ベロシティ → 輝度（高次倍音の届く範囲）。実ピアノは重心1-3kHz程度なので抑えめ
    n_cut = 4.0 + 20.0 * velocity
    # 全体の減衰スケール（低音ほど長く鳴る）
    dec_scale = 0.55 * 2 ** ((60 - midi) / 22.0)
    dec_scale = float(np.clip(dec_scale, 0.15, 6.0))

    y = np.zeros(n)
    for s in range(n_strings):
        cents = (s - (n_strings - 1) / 2.0) * detune_cents
        fs0 = f0 * 2 ** (cents / 1200.0)
        phase0 = rng.uniform(0, 2 * np.pi)
        k = 1
        while True:
            fn = k * fs0 * np.sqrt(1.0 + B * k * k)
            if fn >= 0.45 * fs:
                break
            comb = abs(np.sin(k * np.pi * strike_pos))          # 打弦位置（節を抑圧）
            roll = 1.0 / (1.0 + (k / n_cut) ** 2.2)             # ハンマー輝度
            amp = comb * roll / (k ** 1.6)                       # 打弦の点励振 ~sin(nπβ)/n^~2
            if amp > 1e-4:
                tau_slow = dec_scale / (1.0 + 0.006 * k * k)
                tau_fast = tau_slow * 0.28
                g = 0.45                                        # prompt 比率
                env = g * np.exp(-t / tau_fast) + (1 - g) * np.exp(-t / tau_slow)
                y += amp * env * np.sin(2 * np.pi * fn * t + phase0)
            k += 1
    y /= n_strings

    # ハンマーの打撃トランジェント（フェルトの当たり）
    na = int(0.006 * fs)
    att = rng.standard_normal(na) * np.hanning(na) * 0.12 * (0.5 + velocity)
    y[:na] += att

    # 響板の色付け（低域の共鳴を軽く）
    for f_res, q, gain in [(105.0, 6.0, 0.12), (215.0, 8.0, 0.08)]:
        b, a = sig.iirpeak(f_res / (fs / 2), q)
        y = y + gain * sig.lfilter(b, a, y)

    # ダンパー（ノートオフ相当）: sustain=False なら末尾を速く減衰
    if not sustain:
        r = int(release * fs)
        if r < n:
            damp = np.ones(n)
            damp[n - r:] = np.exp(-np.arange(r) / (0.05 * fs))
            y = y * damp

    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak * 0.9 * (0.4 + 0.6 * velocity)).astype(np.float64)


def render_piano_midi(events, fs: int = FS, sustain_tail: float = 2.5) -> np.ndarray:
    """parse_midi のイベント列をピアノでレンダリング（ポリフォニック）。"""
    ons, notes = {}, []
    for sec, kind, note, vel, ch in events:
        if kind == "on":
            ons[note] = (sec, vel)
        elif note in ons:
            s, v = ons.pop(note)
            notes.append((s, max(sec - s, 0.08), note, v))
    if not notes:
        return np.zeros(int(fs))
    total = max(s + d for s, d, *_ in notes) + sustain_tail
    mix = np.zeros(int(total * fs))
    for s, d, note, vel in notes:
        y = piano_note(note, d + sustain_tail, fs=fs, velocity=vel / 127.0,
                       sustain=True)
        i0 = int(s * fs)
        m = min(len(y), len(mix) - i0)
        mix[i0:i0 + m] += y[:m]
    peak = np.max(np.abs(mix)) + 1e-9
    if peak > 0.98:
        mix = mix / peak * 0.98
    return mix


def measure_inharmonicity(y: np.ndarray, f0: float, fs: int = FS,
                          n_partials: int = 12):
    """レンダリング音から部分音周波数を拾い、B をフィットして回収精度を測る。"""
    win = np.hanning(len(y))
    spec = np.abs(np.fft.rfft(y * win))
    freqs = np.fft.rfftfreq(len(y), 1 / fs)
    part_f, part_n = [], []
    for k in range(1, n_partials + 1):
        target = k * f0 * np.sqrt(1 + 1e-4 * k * k)
        lo, hi = target * 0.96, target * 1.04
        band = (freqs > lo) & (freqs < hi)
        if not band.any():
            continue
        idx = np.where(band)[0]
        j = idx[np.argmax(spec[idx])]
        if 1 <= j < len(spec) - 1:
            a, b, c = np.log(spec[j-1]+1e-12), np.log(spec[j]+1e-12), np.log(spec[j+1]+1e-12)
            d = 0.5 * (a - c) / (a - 2*b + c + 1e-12)
            fpk = (j + d) * fs / len(y)
        else:
            fpk = freqs[j]
        part_f.append(fpk); part_n.append(k)
    part_f = np.array(part_f); part_n = np.array(part_n)
    # (f_n/(n f0))^2 = 1 + B n^2  → 線形回帰で B
    yv = (part_f / (part_n * f0)) ** 2 - 1.0
    xv = part_n ** 2
    B_fit = float(np.sum(xv * yv) / (np.sum(xv * xv) + 1e-12))
    return B_fit, part_n, part_f
