# -*- coding: utf-8 -*-
"""DDSP: 微分可能な harmonic-plus-noise シンセと損失・特徴量。

参照: Engel et al., "DDSP: Differentiable Digital Signal Processing", ICLR 2020.

構成:
  - HarmonicOscillator : f0(t) と倍音分布 a_k(t) から加算合成（ナイキスト以上をマスク）
  - FilteredNoise      : 周波数サンプリング法による時変 FIR で白色雑音を整形
  - DDSPDecoder        : [f0, loudness] 特徴 → GRU → 大域振幅 / 倍音分布 / 雑音応答
  - DDSPAutoencoder    : デコーダ + シンセ（f0 と loudness を条件に音を再構成）
  - multiscale_stft_loss : 複数解像度スペクトログラム L1（DDSP 標準の目的関数）
  - compute_loudness   : A-weighting 近似の対数パワー（フレーム毎）

すべて PyTorch。制御条件（f0, loudness）を与えれば任意ピッチで再合成できるため、
MIDI からの駆動に直結する。
"""
from __future__ import annotations
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
def modified_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """DDSP の正値・有界活性化（振幅を安定に出す）。"""
    return 2.0 * torch.sigmoid(x) ** math.log(10.0) + 1e-7


def upsample(x: torch.Tensor, n_samples: int) -> torch.Tensor:
    """フレーム系列 (B, T, C) を音声レート (B, n_samples, C) へ線形補間。"""
    x = x.transpose(1, 2)  # (B, C, T)
    x = F.interpolate(x, size=n_samples, mode="linear", align_corners=True)
    return x.transpose(1, 2)  # (B, n_samples, C)


# ---------------------------------------------------------------------------
# シンセ・モジュール
# ---------------------------------------------------------------------------
class HarmonicOscillator(nn.Module):
    def __init__(self, sample_rate: int, n_harmonics: int = 80):
        super().__init__()
        self.sr = sample_rate
        self.n_harmonics = n_harmonics
        self.register_buffer("harm_ids", torch.arange(1, n_harmonics + 1).float())

    def forward(self, f0: torch.Tensor, harm_dist: torch.Tensor,
                amp: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        f0:        (B, T, 1)   Hz
        harm_dist: (B, T, K)   倍音分布（正値、内部で正規化）
        amp:       (B, T, 1)   大域振幅（正値）
        return:    (B, n_samples)
        """
        # 音声レートへ
        f0_up = upsample(f0, n_samples)                      # (B, N, 1)
        amp_up = upsample(amp, n_samples)                    # (B, N, 1)
        harm_up = upsample(harm_dist, n_samples)             # (B, N, K)

        # 倍音周波数とナイキスト・マスク
        harm_freqs = f0_up * self.harm_ids.view(1, 1, -1)    # (B, N, K)
        mask = (harm_freqs < self.sr / 2.0).float()
        harm_up = harm_up * mask
        # 分布を正規化（合計1）→ 大域振幅を乗算
        harm_up = harm_up / (harm_up.sum(dim=-1, keepdim=True) + 1e-8)
        harm_amp = harm_up * amp_up                          # (B, N, K)

        # 位相 = 基本波の瞬時位相の累積 × 倍数
        omega = 2.0 * math.pi * f0_up / self.sr              # (B, N, 1)
        base_phase = torch.cumsum(omega.squeeze(-1), dim=1)  # (B, N)
        phases = base_phase.unsqueeze(-1) * self.harm_ids.view(1, 1, -1)  # (B, N, K)
        signal = (torch.sin(phases) * harm_amp).sum(dim=-1)  # (B, N)
        return signal


class FilteredNoise(nn.Module):
    """周波数サンプリング法によるフレーム毎 FIR フィルタ雑音。"""
    def __init__(self, sample_rate: int, n_bands: int = 65, hop: int = 64):
        super().__init__()
        self.sr = sample_rate
        self.n_bands = n_bands
        self.hop = hop
        self.fir_size = 2 * (n_bands - 1)  # irfft 出力長

    def forward(self, noise_mag: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        noise_mag: (B, T, n_bands)  正値の振幅応答（フレーム毎）
        return:    (B, n_samples)
        """
        B, T, _ = noise_mag.shape
        # 線形位相 FIR を magnitude から生成
        mag = noise_mag  # (B, T, n_bands)
        # 複素スペクトル（位相ゼロ）→ 時間領域インパルス応答
        ir = torch.fft.irfft(mag.to(torch.complex64) if not torch.is_complex(mag) else mag,
                             n=self.fir_size, dim=-1)          # (B, T, fir_size)
        # 因果化のため fftshift + 窓
        ir = torch.roll(ir, shifts=self.fir_size // 2, dims=-1)
        win = torch.hann_window(self.fir_size, device=ir.device)
        ir = ir * win.view(1, 1, -1)

        # フレーム毎に白色雑音を生成し、FIR で畳み込み → overlap-add
        frame_len = self.hop
        pad = self.fir_size
        out = torch.zeros(B, n_samples + pad, device=noise_mag.device)
        white = torch.rand(B, T, frame_len, device=noise_mag.device) * 2.0 - 1.0
        # 各フレーム: conv(white_frame, ir_frame) を配置
        # FFT ベースで一括畳み込み
        conv_len = frame_len + self.fir_size - 1
        n_fft = 1
        while n_fft < conv_len:
            n_fft *= 2
        W = torch.fft.rfft(white, n=n_fft, dim=-1)            # (B, T, F)
        H = torch.fft.rfft(ir, n=n_fft, dim=-1)               # (B, T, F)
        y = torch.fft.irfft(W * H, n=n_fft, dim=-1)           # (B, T, conv_len<=n_fft)
        y = y[..., :conv_len]
        # overlap-add
        for t in range(T):
            start = t * frame_len
            end = start + conv_len
            if start >= n_samples:
                break
            seg = y[:, t, :]
            e = min(end, n_samples + pad)
            out[:, start:e] += seg[:, : e - start]
        return out[:, :n_samples]


# ---------------------------------------------------------------------------
# デコーダ（制御特徴 → シンセ・パラメータ）
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, layers=2):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.1)]
            d = hidden
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class DDSPDecoder(nn.Module):
    def __init__(self, n_harmonics=80, n_noise_bands=65, hidden=256):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.mlp_f0 = MLP(1, hidden, 1)
        self.mlp_ld = MLP(1, hidden, 1)
        self.gru = nn.GRU(hidden * 2, hidden, batch_first=True)
        self.mlp_out = MLP(hidden, hidden, 1)
        self.head_amp = nn.Linear(hidden, 1)
        self.head_harm = nn.Linear(hidden, n_harmonics)
        self.head_noise = nn.Linear(hidden, n_noise_bands)

    def forward(self, f0_scaled: torch.Tensor, loudness: torch.Tensor):
        """
        f0_scaled: (B, T, 1)  スケール済み f0（例 log Hz を正規化）
        loudness:  (B, T, 1)  正規化ラウドネス
        """
        h = torch.cat([self.mlp_f0(f0_scaled), self.mlp_ld(loudness)], dim=-1)
        h, _ = self.gru(h)
        h = self.mlp_out(h)
        amp = modified_sigmoid(self.head_amp(h))
        harm = modified_sigmoid(self.head_harm(h))
        noise = modified_sigmoid(self.head_noise(h))
        return amp, harm, noise


class DDSPAutoencoder(nn.Module):
    def __init__(self, sample_rate=16000, hop=64,
                 n_harmonics=80, n_noise_bands=65, hidden=256):
        super().__init__()
        self.sr = sample_rate
        self.hop = hop
        self.decoder = DDSPDecoder(n_harmonics, n_noise_bands, hidden)
        self.harmonic = HarmonicOscillator(sample_rate, n_harmonics)
        self.noise = FilteredNoise(sample_rate, n_noise_bands, hop)

    def forward(self, f0_hz: torch.Tensor, f0_scaled: torch.Tensor,
                loudness: torch.Tensor, n_samples: int) -> Dict[str, torch.Tensor]:
        amp, harm, noise_mag = self.decoder(f0_scaled, loudness)
        harm_sig = self.harmonic(f0_hz, harm, amp, n_samples)
        noise_sig = self.noise(noise_mag, n_samples)
        audio = harm_sig + noise_sig
        return {"audio": audio, "harmonic": harm_sig, "noise": noise_sig,
                "amp": amp, "harm_dist": harm, "noise_mag": noise_mag}


# ---------------------------------------------------------------------------
# 損失・特徴量
# ---------------------------------------------------------------------------
def multiscale_stft_loss(x: torch.Tensor, y: torch.Tensor,
                         fft_sizes=(2048, 1024, 512, 256, 128, 64),
                         eps: float = 1e-7) -> torch.Tensor:
    """複数解像度スペクトログラムの L1（線形 + 対数）。DDSP の標準損失。"""
    loss = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=x.device)
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win,
                       return_complex=True, center=True)
        Y = torch.stft(y, n_fft=n_fft, hop_length=hop, window=win,
                       return_complex=True, center=True)
        Xm, Ym = X.abs(), Y.abs()
        loss = loss + (Xm - Ym).abs().mean()
        loss = loss + (torch.log(Xm + eps) - torch.log(Ym + eps)).abs().mean()
    return loss / len(fft_sizes)


def compute_loudness(audio: torch.Tensor, sr: int, hop: int,
                     n_fft: int = 1024) -> torch.Tensor:
    """A-weighting 近似の対数パワーをフレーム毎に（(B, T) を返す）。"""
    win = torch.hann_window(n_fft, device=audio.device)
    S = torch.stft(audio, n_fft=n_fft, hop_length=hop, window=win,
                   return_complex=True, center=True)             # (B, F, T)
    power = S.abs() ** 2
    freqs = torch.linspace(0, sr / 2, S.shape[1], device=audio.device)
    # A-weighting（dB）近似
    f2 = freqs ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / (
        (f2 + 20.6 ** 2)
        * torch.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    )
    a_weight = 2.0 + 20.0 * torch.log10(ra + 1e-12)
    weighting = (10.0 ** (a_weight / 10.0)).view(1, -1, 1)
    loud = torch.log(torch.sum(power * weighting, dim=1) + 1e-7)  # (B, T)
    return loud


def scale_f0(f0_hz: torch.Tensor) -> torch.Tensor:
    """f0[Hz] を log スケールで概ね [0,1] に（MIDI 24..108 相当）。"""
    midi = 69.0 + 12.0 * torch.log2((f0_hz + 1e-5) / 440.0)
    return (midi.clamp(24, 108) - 24.0) / 84.0
