# audio-model

物理モデリング音源の内製に向けた研究・実装リポジトリ。
実楽器の音を「サンプルとして貼る」のではなく、物理モデル（波形導波管・モーダル合成）と
そのパラメータ推定（analysis-by-synthesis）で再構築し、最終的に MIDI / MIDI 2.0・MPE で
演奏可能な音源（VST3 / CLAP）へ育てることを目標とする。

## なぜ Demucs か

物理モデルを実楽器に似せるには「その楽器だけのクリーンなターゲット音」が要る。
しかし現実の音源はミックス済みで、単一楽器の素材は手に入りにくい。
[Demucs](https://github.com/adefossez/demucs)（Meta 発の音源分離、MIT）でミックスを
ステム（ドラム / ベース / ギター / ピアノ / ボーカル / その他）に分離し、
分離したステムをパラメータ推定パイプラインの入力にする。

```
ミックス音源 ──Demucs──▶ 楽器ステム ──分析──▶ 物理パラメータ ──合成──▶ 物理モデル音源
                          (guitar/piano/bass)   (モード/減衰/f0/B)      (MIDIで演奏)
```

htdemucs_6s モデルは guitar / piano を独立ステムとして出せるため、
撥弦（ギター＝waveguide）と打弦（ピアノ＝モーダル/waveguide）の
モデリング・ターゲット抽出に直接使える。

## 構成

| パス | 役割 |
|---|---|
| `src/physmod/core.py` | 物理モデル本体。拡張Karplus-Strong（波形導波管）撥弦、モーダル合成打楽器、モード推定（analysis-by-synthesis）、SMF読み書き＋レンダリング |
| `src/physmod/piano.py` | 物理モデルピアノ（Pianoteq参照・サンプル不使用）。剛性インハーモニシティ＋ハンマー打弦＋複弦うなり＋二段階減衰＋響板 |
| `src/neural/ddsp.py` | DDSP（微分可能DSP）。harmonic+noise 微分可能シンセ、GRUデコーダ、multi-scale STFT損失、f0/ラウドネス特徴 |
| `src/neural/data.py` | DDSP学習用のモノフォニック楽器コーパス生成（実ステムと同形） |
| `src/pipeline/demucs_separate.py` | Demucsラッパ。ミックス→ステム分離、分離stemを物理モデル分析に橋渡し |
| `examples/run_poc.py` | 物理モデルPoC一式（ピッチ精度・モード一致・推定往復・MIDI経路）を再実行 |
| `examples/piano_demo.py` | 物理ピアノの検証（設計vs回収インハーモニシティ）＋デモ音源＋MIDI演奏 |
| `examples/make_piano_corpus.py` | 物理ピアノからDDSP学習用のモノフォニック独奏を生成（ライセンス問題ゼロの素材） |
| `examples/retrain_from_stem.py` | 実録音/Demucs分離ステムからDDSPを実音色で再学習 |
| `examples/train_from_dataset.py` | 公開単音データセット(TinySOL/NSynth/汎用)でDDSP学習（ピッチ既知→CREPE不要） |
| `src/pipeline/note_dataset.py` | 単音データセットのローダ（TinySOL/NSynth/音名入りフォルダ） |
| `examples/mix_to_model.py` | ミックス音源→Demucs分離→ステムのモード推定→再合成までを通す実パイプライン |
| `examples/train_ddsp.py` | DDSPオートエンコーダを学習完走（GPU）。f0/ラウドネスから音色を制御可能に再構成 |
| `docs/architecture.md` | 音声生成の2系譜（コーデック+LM / 微分可能DSP）と本プロジェクトの立ち位置、Suno公開アーキ参照 |
| `docs/piano_model.md` | 物理ピアノの設計（Pianoteq公開アーキとの対応）と検証結果 |
| `docs/datasets.md` | 公開データセットのライセンス整理（商用可否）とDDSP学習の実務要件 |
| `docs/ddsp_training_report.md` | DDSP学習の結果（学習曲線・汎化指標）※学習実行で生成 |
| `samples/` | 入力音源置き場（gitignore） |
| `outputs/` | 生成物置き場（gitignore） |

## セットアップ

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
# GPU (RTX4090等) を使う場合は torch を CUDA 版で:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 使い方

```bash
# 物理モデルPoC（依存 numpy/scipy のみ、機械学習なし）
python examples/run_poc.py

# ミックス音源を分離して、ステムから物理パラメータを推定
python examples/mix_to_model.py samples/your_track.wav --instrument guitar
```

## ライセンス / 出所の方針

本リポジトリは公開研究・OSS（Demucs, STK, Faust, DDSP 等）のみを土台とする独自実装。
第三者の漏洩ソースコード・非公開の営業秘密は一切取り込まない。
