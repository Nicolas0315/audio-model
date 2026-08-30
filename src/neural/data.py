# -*- coding: utf-8 -*-
"""DDSP 学習用のモノフォニック楽器コーパス生成。

DDSP が本領を発揮する「持続する調波楽器」を合成で用意する。
入力次元（音声 + f0 トラック + ラウドネス）は、Demucs で分離した実楽器ステムと
同じ形なので、実データへの差し替えはローダを変えるだけで済む。

楽器モデル（合成ターゲット。学習器はこれを "知らない" 状態から音だけで再現する）:
  - ビブラート付き f0
  - ラウドネスで明るさ（高次倍音の量）が変わる調波スペクトル
  - フォルマント的な山
  - ADSR 振幅包絡
  - 微量のブレス雑音
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


SR = 16000
DUR = 1.5
N = int(SR * DUR)


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def _adsr(n, sr, a=0.02, d=0.1, s=0.7, r=0.25):
    env = np.ones(n)
    ai, di, ri = int(a * sr), int(d * sr), int(r * sr)
    ai = max(ai, 1); di = max(di, 1); ri = max(ri, 1)
    env[:ai] = np.linspace(0, 1, ai)
    env[ai:ai + di] = np.linspace(1, s, di)
    env[ai + di:n - ri] = s
    env[n - ri:] = np.linspace(env[n - ri - 1] if n - ri - 1 >= 0 else s, 0, ri)
    return env


def render_note(midi: float, loudness: float, vib_rate: float = 5.0,
                vib_depth: float = 0.004, rng: np.random.Generator = None):
    """1音を合成し、(audio, f0_track) を返す。audio は [-1,1] 付近。"""
    if rng is None:
        rng = np.random.default_rng(0)
    t = np.arange(N) / SR
    f0 = midi_to_hz(midi)
    # ビブラート付き f0 トラック
    f0_track = f0 * (1.0 + vib_depth * np.sin(2 * np.pi * vib_rate * t + rng.uniform(0, 6.28)))
    # 位相
    phase = 2 * np.pi * np.cumsum(f0_track) / SR

    # 調波スペクトル: 基本スロープ + ラウドネスで明るさ増、フォルマント山
    n_harm = 60
    slope = 2.4 - 1.1 * loudness          # 強いほど倍音が減衰しにくい（明るい）
    ks = np.arange(1, n_harm + 1)
    base = ks ** (-slope)
    # フォルマント（固定周波数帯を強調）
    formant_hz = 1400.0
    form = np.exp(-0.5 * ((ks * f0 - formant_hz) / 600.0) ** 2) * 0.6
    amps = base + form
    amps /= amps.sum()

    sig = np.zeros(N)
    for k, a in zip(ks, amps):
        if k * f0 >= SR / 2:
            break
        sig += a * np.sin(k * phase)

    # ブレス雑音（微量、高域）
    noise = rng.standard_normal(N)
    from scipy.signal import butter, lfilter
    b, a = butter(2, 2000 / (SR / 2), btype="high")
    noise = lfilter(b, a, noise) * 0.02 * (0.5 + loudness)

    env = _adsr(N, SR)
    audio = (sig + noise) * env * (0.3 + 0.7 * loudness)
    peak = np.max(np.abs(audio)) + 1e-9
    audio = audio / peak * 0.9
    return audio.astype(np.float32), f0_track.astype(np.float32)


@dataclass
class Note:
    audio: np.ndarray
    f0: np.ndarray
    midi: float
    loudness: float


def build_corpus(val_pitches=(50, 57, 64, 71, 78),
                 pitch_lo=48, pitch_hi=84,
                 loudnesses=(0.4, 0.7, 1.0),
                 seed=0) -> Tuple[List[Note], List[Note]]:
    """学習/検証コーパスを生成。val_pitches は音高汎化を測るため学習から除外。"""
    rng = np.random.default_rng(seed)
    train, val = [], []
    for midi in range(pitch_lo, pitch_hi + 1):
        for ld in loudnesses:
            vib_rate = rng.uniform(4.5, 6.0)
            audio, f0 = render_note(midi, ld, vib_rate=vib_rate, rng=rng)
            note = Note(audio, f0, float(midi), float(ld))
            (val if midi in val_pitches else train).append(note)
    return train, val
