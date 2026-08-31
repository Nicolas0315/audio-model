# -*- coding: utf-8 -*-
"""学習済み DDSP モデルを MIDI 楽器として演奏する。

DDSP は f0 と loudness を条件に音を合成するため、MIDI ノート列から
制御トラックを作れば任意の旋律を「学習した音色」で演奏できる。
これが「音から学んだ楽器を MIDI で弾く」最終形態の最小実装。

使い方:
  python examples/play_ddsp_midi.py               # 内蔵デモ旋律
  python examples/play_ddsp_midi.py song.mid       # 任意のSMF(モノフォニック)
"""
import os
import sys
import argparse

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from neural.ddsp import DDSPAutoencoder, scale_f0  # noqa: E402
from neural.data import SR, midi_to_hz  # noqa: E402
from physmod.core import write_wav, write_midi, parse_midi  # noqa: E402

HOP = 64
OUT = os.path.join(ROOT, "outputs")
CKPT = os.path.join(OUT, "ddsp_best.pt")


def demo_notes():
    """内蔵デモ旋律（モノフォニック, (start_sec, dur_sec, midi, vel)）。"""
    seq = [(60, 100), (62, 80), (64, 110), (65, 70), (67, 120),
           (65, 80), (64, 100), (62, 70), (60, 110)]
    notes, t = [], 0.0
    for m, v in seq:
        notes.append((t, 0.45, m, v))
        t += 0.5
    return notes


def midi_file_to_notes(path):
    """SMFをモノフォニック(start,dur,midi,vel)へ（同時発音は最初のchを優先）。"""
    events = parse_midi(path)
    ons, notes = {}, []
    for sec, kind, note, vel, ch in events:
        if kind == "on":
            ons[note] = (sec, vel)
        elif note in ons:
            s, v = ons.pop(note)
            notes.append((s, max(sec - s, 0.05), note, v))
    notes.sort()
    return notes


def build_control(notes, fs=SR, hop=HOP):
    """ノート列 → フレームレートの f0[Hz] と loudness トラック（モノフォニック）。"""
    total = max(s + d for s, d, *_ in notes) + 0.6
    n_frames = int(total * fs / hop)
    tf = np.arange(n_frames) * hop / fs
    f0 = np.zeros(n_frames)
    loud = np.full(n_frames, -4.0)  # 無音時の下限
    for s, d, m, v in notes:
        i0 = int(s * fs / hop)
        i1 = int((s + d) * fs / hop)
        i1 = min(i1, n_frames)
        if i1 <= i0:
            continue
        f0[i0:i1] = midi_to_hz(m)
        # velocity → 正規化ラウドネス（学習時の分布に概ね合わせる）
        lv = (v / 127.0) * 3.0 - 1.5
        seg = np.linspace(lv + 0.3, lv, i1 - i0)  # 軽い減衰
        loud[i0:i1] = seg
    # 発音が無いフレームの f0 は直近の有効値で補完（合成の破綻回避）
    last = 110.0
    for i in range(n_frames):
        if f0[i] <= 0:
            f0[i] = last
        else:
            last = f0[i]
    return f0, loud, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midi", nargs="?", default=None, help="SMFファイル(省略で内蔵デモ)")
    ap.add_argument("--ckpt", default=CKPT, help="学習済みモデル(.pt)。既定はdDDSP汎用")
    ap.add_argument("--tag", default=None, help="出力ファイル名のタグ(例: violin)")
    args = ap.parse_args()

    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        print(f"エラー: 学習済みモデルが無い: {ckpt_path}。先に学習を実行。")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DDSPAutoencoder(sample_rate=SR, hop=HOP).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"モデル読込: {os.path.basename(ckpt_path)} step={ckpt.get('step')} "
          f"val_loss={ckpt.get('val_loss'):.4f} device={device}")

    if args.midi:
        notes = midi_file_to_notes(args.midi)
        tag = args.tag or os.path.splitext(os.path.basename(args.midi))[0]
    else:
        notes = demo_notes()
        tag = args.tag or "demo"
        # デモのSMFも書き出しておく（DAWで開ける）
        tpq = 480
        ev = []
        for s, d, m, v in notes:
            ev += [(int(s / 0.5 * tpq), "on", m, v, 0),
                   (int((s + d) / 0.5 * tpq), "off", m, 0, 0)]
        write_midi(os.path.join(OUT, "ddsp_demo_melody.mid"), ev, tpq=tpq, bpm=120)

    print(f"ノート数: {len(notes)}")
    f0, loud, total = build_control(notes)
    n_samples = int(total * SR)

    f0_t = torch.tensor(f0[None, :, None], dtype=torch.float32, device=device)
    loud_t = torch.tensor(loud[None, :, None], dtype=torch.float32, device=device)
    f0_sc = scale_f0(f0_t)

    with torch.no_grad():
        out = model(f0_t, f0_sc, loud_t, n_samples)
    audio = out["audio"][0].cpu().numpy()

    path = os.path.join(OUT, f"ddsp_performance_{tag}.wav")
    write_wav(path, audio, fs=SR)
    print(f"演奏を書き出し -> outputs/ddsp_performance_{tag}.wav ({total:.1f}s)")
    print("学習した音色で MIDI 旋律を演奏した結果。f0/velocity を変えれば任意に弾ける。")


if __name__ == "__main__":
    main()
