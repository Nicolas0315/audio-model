# 公開データ（TinySOL Violin）での DDSP 学習レポート

初めて**公開の実楽器データ**で DDSP を学習した記録。

## データ

- **TinySOL**（Zenodo record 3685367, **CC BY 4.0＝商用可**・帰属のみ）から Violin を使用。
- Violin 284音（4秒/16kHz mono、MIDI 55–100=G3–E7、≈19分＝DDSP論文の13分閾値超）。
- 単音なのでピッチは既知 → f0 は定数で条件付け（CREPE 不要）、loudness は A-weighting で抽出。
- 音高で train/val 分割（学習外の音高 MIDI [56,66,68,76,81,89] で汎化を測定）。

## 結果

- best val multi-scale STFT loss **1.46**（step 2250 で収束、以降 val 改善なし）。
- 出力: `outputs/ddsp_violin_AB.wav`（学習外の音高 E5: 前半=本物 / 後半=DDSP再現）、
  `outputs/ddsp_performance_violin.wav`（学習音色で MIDI 旋律を演奏）。

### 損失の比較（何が難しいか）

| 学習データ | best val MSS | 難しさの要因 |
|---|---:|---|
| 合成調波楽器（持続・整数倍音） | 0.79 | 最も易しい（DDSPの理想ドメイン） |
| 物理ピアノ（自作・非整数倍音） | 1.16 | インハーモニシティを整数倍音モデルが平滑化 |
| **TinySOL Violin（実録音）** | **1.46** | 実録音のノイズ・部屋・**ビブラート**（下記） |

Violin が最も高いのは、実録音由来の要因に加え、**ヴァイオリンの強いビブラート（f0 の揺れ）を
「音名からの定数 f0」では追えていない**ため。DDSP のデコーダは与えられた f0 に倍音を乗せるだけなので、
実際のピッチ変調を条件に含めないと再現が甘くなる。→ **次段: torchcrepe で f0 曲線を抽出して条件付け**すれば改善見込み。

## 学習を途中で停止した経緯（正直な記録）

- 4秒クリップでの学習が想定より遅く、GPU を長時間 100% 占有してマシンが重くなった。
- **根本原因**: `FilteredNoise` の overlap-add に `for t in range(T)` の Python ループがあり、
  4秒クリップだと T=1000 回/forward の GPU カーネル起動でオーバーヘッド律速していた。
- モデルは step 2250（val 1.46）で既に収束しており、以降の step は best を更新せず無駄だったため、
  収束済み best.pt を残して学習を停止した。
- **修正**: overlap-add を `F.fold`（col2im）で**ベクトル化**。単体で T=1000 が 3.9ms/forward、
  全学習ループで **0.077s/step（従来比 約8倍高速）**。4000step が 40分→約5分、GPU 占有も激減。

## 再現手順

```bash
# TinySOL 取得（CC BY 4.0）: TinySOL.tar.gz を Zenodo から、Violin を展開
python examples/train_from_dataset.py --tinysol data \
       --metadata data/TinySOL_metadata.csv --instrument Violin --steps 4000 --tag violin
# 学習済みモデルで MIDI 演奏
python examples/play_ddsp_midi.py --ckpt outputs/ddsp_violin_best.pt --tag violin
```

## 次段

1. **torchcrepe で f0 曲線**を抽出して条件付け（ビブラート対応）→ val 改善を狙う。
2. 他楽器（Flute/Trumpet/Cello 等 TinySOL 14楽器）へ横展開。
3. 物理モデル（インハーモニック倍音）× DDSP（音色・制御）のハイブリッド。
