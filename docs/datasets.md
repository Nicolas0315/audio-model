# 公開データセット（学習素材）とライセンス

DDSP/物理モデル音源の学習に使える公開データセットを、**ライセンス（商用可否）**と
**モノフォニック適性**で整理する。将来の収益化（音源プラグイン販売）を視野に入れるため、
**CC-NC（非商用）は製品に組み込めない**ことを最重要の判断軸とする。

## 商用可（CC0 / CC BY）× モノフォニック — 使える

| 優先 | データセット | ライセンス | 内容 | サイズ | 備考 |
|---:|---|---|---|---|---|
| ★1 | **TinySOL** | CC BY 4.0 | 14楽器×2,913の**単音**（Ircam品質） | ~1.0GB | 小さく商用可。**最初の実データはこれ**。単音=ピッチ既知でCREPE不要 |
| 2 | **NSynth** | CC BY 4.0 | 305k単音、16kHz/4秒、DDSP論文の実績 | 数十GB | canonical だが巨大。サブセット抽出前提 |
| 3 | **Medley-solos-DB** | CC BY 4.0 | 8楽器のソロクリップ 21,571件 | 7.9GB | F0なし→抽出必要 |
| 4 | **GuitarSet** | CC BY 4.0 | ギター、弦別ピッチ輪郭付き（和音含む） | 8.2GB | 撥弦モデル用。単音区間抽出 |

## 商用不可（CC-NC）— 製品に使わない（研究・評価のみ）

| データセット | ライセンス | 警告 |
|---|---|---|
| **MAESTRO**（ピアノ） | CC BY-NC-SA 4.0 | ポリフォニック＋NC。商用ピアノ音源に転用不可 |
| **MDB-stem-synth** | CC BY-NC 4.0 | 完璧なF0付きで技術的に理想だが商用不可 |
| **Good-sounds** | CC BY-NC 4.0 | 単音・音階で理想だが商用不可 |
| IRMAS / MusicNet | CC BY-NC(-SA) | 商用不可、かつ単音でない |

## 要確認（一次ソースでライセンス未確定）

- **URMP**（個別ステム+F0、技術的に◎）: 申請制・ライセンス明記なし → 商用前に要問い合わせ
- **University of Iowa MIS**（単音・多楽器）: 「無制限に使用可」の二次情報のみ、公式条文未取得 → 要確認
- **Philharmonia**: 商用可だが「サンプルをそのまま再配布・サンプラー化して販売」は禁止 → DDSP学習(波形からモデル学習)はグレーだが可能性高、法務確認推奨

## DDSP 学習の実務要件（研究で確認済み）

- **16kHz mono / 4秒クリップ / frame_rate 250Hz**（本リポは hop=64@16k=250Hz で既に一致）。
- f0: 単音はピッチ既知→定数で条件付け（CREPE不要）。実演奏に移る段で **torchcrepe** に差し替え。
- loudness: A-weighting のパワー平均（本リポ `compute_loudness` が実装済み）。
- **モノフォニック単一楽器・ドライ・単一環境が必須**（harmonic osc は単一f0前提。ポリフォニックはCREPEが破綻）。
- データ量: DDSP論文は「**13分未満**」の表現力あるソロで学習実績（条件が揃った場合の下限）。
- 音域: 学習した音域＋やや外側まで実用。極端な外挿は倍音が破綻。

## 本リポでの受け入れ経路（実装・検証済み）

```bash
# TinySOL の1楽器で学習（例: Violin）
python examples/train_from_dataset.py --tinysol PATH/TinySOL \
       --metadata PATH/TinySOL_metadata.csv --instrument Violin --steps 4000
# 学習済みモデルで MIDI 演奏
python examples/play_ddsp_midi.py            # ckpt を指すよう最小改修で対応
```

`src/pipeline/note_dataset.py` が TinySOL / NSynth / 汎用単音フォルダを読み、
`examples/train_from_dataset.py` が音高分割で学習・汎化測定する。単音フォルダでの
通し動作は検証済み（合成単音25個で loader→分割→学習→AB出力）。

## 出所の方針

商用可(CC0/CC BY)を優先。CC-NCは製品に組み込まない。要確認ライセンスは商用化前に一次確認。
第三者の漏洩物・暗号化商用サンプル（UVI等）は使わない。
