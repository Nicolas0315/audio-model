# -*- coding: utf-8 -*-
"""モノフォニック録音からフレーム単位の f0 と有声度を抽出する。

CREPE を導入しなくても動くよう、自己相関 + 放物線補間 + メディアン平滑 +
有声ゲートで実装する。ソロ楽器（単音の旋律）向け。
将来 torchcrepe に差し替える場合も、出力（frame f0[Hz], voiced flag）は同形。
"""
from __future__ import annotations
import numpy as np
import scipy.signal as sig


def track_f0(x: np.ndarray, fs: int, hop: int = 64, win: int = 1024,
             fmin: float = 55.0, fmax: float = 1500.0,
             voiced_thresh: float = 0.5):
    """
    x:   mono float [-1,1]
    return: (f0_hz[T], voiced[T], loudness_db[T])  すべてフレーム系列
    """
    n = len(x)
    n_frames = 1 + n // hop
    f0 = np.zeros(n_frames)
    voiced = np.zeros(n_frames, dtype=bool)
    loud = np.full(n_frames, -80.0)

    lag_min = int(fs / fmax)
    lag_max = int(fs / fmin)
    half = win // 2
    window = np.hanning(win)

    for i in range(n_frames):
        c = i * hop
        a = max(0, c - half)
        b = min(n, c + half)
        frame = x[a:b]
        if len(frame) < win:
            frame = np.pad(frame, (0, win - len(frame)))
        frame = frame * window
        # ラウドネス（RMS dB）
        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        loud[i] = 20 * np.log10(rms + 1e-9)
        # 自己相関
        ac = sig.correlate(frame, frame, mode="full")[win - 1:]
        ac0 = ac[0] + 1e-9
        seg = ac[lag_min:lag_max]
        if len(seg) == 0:
            continue
        k = int(np.argmax(seg)) + lag_min
        peak = ac[k] / ac0  # 正規化ピーク（有声度）
        # 放物線補間
        if 1 <= k < len(ac) - 1:
            al, be, ga = ac[k - 1], ac[k], ac[k + 1]
            d = 0.5 * (al - ga) / (al - 2 * be + ga + 1e-12)
            k = k + d
        f0[i] = fs / k
        voiced[i] = peak > voiced_thresh

    # メディアン平滑（外れ値除去）
    f0_s = sig.medfilt(f0, kernel_size=5)
    # 無声フレームは直近の有声 f0 を保持（合成の破綻回避）
    last = 110.0
    for i in range(n_frames):
        if voiced[i] and f0_s[i] > 0:
            last = f0_s[i]
        else:
            f0_s[i] = last
    return f0_s, voiced, loud


def slice_windows(x: np.ndarray, fs: int, f0: np.ndarray, voiced: np.ndarray,
                  hop: int = 64, win_sec: float = 1.5, stride_sec: float = 0.75,
                  min_voiced_ratio: float = 0.4):
    """録音を学習用の固定長ウィンドウに切り出す。

    return: list of dict{audio, f0(frame), voiced_ratio}
    有声比率が低い（無音/雑音のみ）ウィンドウは捨てる。
    """
    win_n = int(win_sec * fs)
    stride_n = int(stride_sec * fs)
    fpw = win_n // hop  # frames per window
    out = []
    pos = 0
    while pos + win_n <= len(x):
        fa = pos // hop
        fb = fa + fpw
        if fb > len(f0):
            break
        vr = float(np.mean(voiced[fa:fb]))
        if vr >= min_voiced_ratio:
            out.append({
                "audio": x[pos:pos + win_n].astype(np.float32),
                "f0": f0[fa:fb].astype(np.float32),
                "voiced_ratio": vr,
            })
        pos += stride_n
    return out
