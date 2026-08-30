"""ニューラル音源合成（DDSP）。

Suno 系譜（neural audio codec + 自己回帰 Transformer）のうち、
「音から楽器音色を学習し、ピッチ/ラウドネスで制御して再合成する」目的に最適な
DDSP（微分可能 DSP, Engel et al. 2020）を実装する。
"""
