# -*- coding: utf-8 -*-
"""単音（isolated note）データセットのローダ。

TinySOL / NSynth のような「1ファイル=1音・ピッチ既知」のデータを DDSP 学習用の
コーパスに変換する。単音はピッチが既知なので f0 は定数として与えられ、CREPE 不要。
（表現力のある実演奏に移る段では CREPE/torchcrepe に差し替える。）

対応:
  - TinySOL: メタCSV（Pitch 列が "A4" 等の科学的音名）
  - 汎用: ファイル名に含まれる音名（例 "Vn-ord-A4-mf.wav"）からピッチ推定
  - NSynth: examples.json の "pitch"(MIDI 整数)
"""
from __future__ import annotations
import os
import re
import csv
import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


_NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NAME_RE = re.compile(r"([A-Ga-g])([#b]?)(-?\d)")


def note_name_to_midi(name: str) -> Optional[int]:
    """'A4' 'C#3' 'Db5' → MIDI 番号。A4=69。見つからなければ None。"""
    m = _NAME_RE.search(name)
    if not m:
        return None
    letter, acc, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    semis = _NOTE[letter] + (1 if acc == "#" else -1 if acc == "b" else 0)
    return 12 * (octave + 1) + semis


@dataclass
class NoteSample:
    audio: np.ndarray   # mono float32 @ target_fs
    midi: int
    label: str          # 楽器名など（分割・分類用）


def _load_mono(path: str, target_fs: int):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    x, fs = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    if fs != target_fs:
        g = gcd(int(fs), int(target_fs))
        x = resample_poly(x, target_fs // g, fs // g)
    return x.astype(np.float32)


def _fix_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def load_tinysol(root: str, metadata_csv: str, instrument: Optional[str] = None,
                 target_fs: int = 16000, clip_sec: float = 4.0,
                 max_notes: Optional[int] = None) -> List[NoteSample]:
    """TinySOL を読み込む。instrument 指定で1楽器に絞る（推奨）。"""
    n = int(clip_sec * target_fs)
    out: List[NoteSample] = []
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in reader.fieldnames}
        path_col = cols.get("path") or cols.get("filepath") or reader.fieldnames[0]
        inst_col = cols.get("instrument (abbr.)") or cols.get("instrument") or None
        pitch_col = cols.get("pitch")
        for row in reader:
            rel = row[path_col]
            inst = row[inst_col] if inst_col else "unknown"
            if instrument and instrument.lower() not in inst.lower():
                continue
            midi = None
            if pitch_col and row.get(pitch_col):
                midi = note_name_to_midi(row[pitch_col])
            if midi is None:
                midi = note_name_to_midi(os.path.basename(rel))
            if midi is None:
                continue
            wav = os.path.join(root, rel)
            if not os.path.exists(wav):
                # メタのパスが root 直下でない場合の保険
                cand = os.path.join(root, os.path.basename(rel))
                wav = cand if os.path.exists(cand) else wav
            if not os.path.exists(wav):
                continue
            x = _fix_length(_load_mono(wav, target_fs), n)
            x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
            out.append(NoteSample(x.astype(np.float32), int(midi), inst))
            if max_notes and len(out) >= max_notes:
                break
    return out


def load_note_folder(folder: str, target_fs: int = 16000, clip_sec: float = 4.0,
                     max_notes: Optional[int] = None) -> List[NoteSample]:
    """ファイル名に音名を含む単音 wav 群を読み込む汎用ローダ。"""
    n = int(clip_sec * target_fs)
    out: List[NoteSample] = []
    for dp, _, files in os.walk(folder):
        for fn in sorted(files):
            if not fn.lower().endswith((".wav", ".aiff", ".flac")):
                continue
            midi = note_name_to_midi(fn)
            if midi is None:
                continue
            x = _fix_length(_load_mono(os.path.join(dp, fn), target_fs), n)
            x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
            out.append(NoteSample(x.astype(np.float32), int(midi), fn))
            if max_notes and len(out) >= max_notes:
                return out
    return out


def load_nsynth(json_path: str, audio_dir: str, family: Optional[str] = None,
                source: Optional[str] = None, target_fs: int = 16000,
                clip_sec: float = 4.0, max_notes: Optional[int] = None) -> List[NoteSample]:
    """NSynth の examples.json + audio/ を読み込む。family/source でフィルタ。"""
    n = int(clip_sec * target_fs)
    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)
    out: List[NoteSample] = []
    for key, ex in meta.items():
        if family and ex.get("instrument_family_str") != family:
            continue
        if source and ex.get("instrument_source_str") != source:
            continue
        wav = os.path.join(audio_dir, key + ".wav")
        if not os.path.exists(wav):
            continue
        x = _fix_length(_load_mono(wav, target_fs), n)
        x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
        out.append(NoteSample(x.astype(np.float32), int(ex["pitch"]),
                              ex.get("instrument_str", "nsynth")))
        if max_notes and len(out) >= max_notes:
            break
    return out
