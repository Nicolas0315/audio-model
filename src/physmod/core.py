# -*- coding: utf-8 -*-
"""
物理モデリング音源 最小実証 (PoC)
1. 拡張Karplus-Strong (digital waveguide) 撥弦モデル
2. モーダル合成 打楽器モデル (マリンバ風)
3. analysis-by-synthesis 往復検証: ターゲット音 -> モード推定 -> 再合成 -> 誤差測定
4. 標準MIDIファイル(SMF)を生成し、独立パーサで読み戻して両モデルでレンダリング
依存: numpy, scipy (stdlib: wave, struct)
"""
import numpy as np
import scipy.signal as sig
import wave as wavemod
import struct, os, sys, io

FS = 48000
OUT = os.path.dirname(os.path.abspath(__file__))

def write_wav(path, x, fs=FS):
    x = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x)) + 1e-12
    if peak > 0.98:
        x = x / peak * 0.98
    pcm = (x * 32767).astype(np.int16)
    with wavemod.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs)
        w.writeframes(pcm.tobytes())

# ---------------------------------------------------------------
# 1. 拡張 Karplus-Strong (digital waveguide 撥弦)
#    - ピック位置コームフィルタ
#    - 周波数依存減衰 (2点平均ローパス + 減衰係数 rho)
#    - 分数遅延 (1次オールパス) でピッチ精度を確保
# ---------------------------------------------------------------
def ks_pluck(f0, dur, fs=FS, velocity=1.0, pick_pos=0.13, brightness=0.55, rho=0.9985):
    n_total = int(dur * fs)
    # 弦ループ長: ループ内フィルタ遅延(約0.5)を差し引く
    P = fs / f0 - 0.5
    N = int(np.floor(P))
    frac = P - N
    # 分数遅延オールパス係数
    C = (1 - frac) / (1 + frac)
    # 励振: ノイズバースト + ベロシティで帯域を変化(強いほど明るい)
    exc = np.random.default_rng(42).uniform(-1, 1, N)
    cutoff = 0.15 + 0.8 * velocity           # 正規化カットオフ相当
    b, a = sig.butter(2, min(cutoff, 0.99))
    exc = sig.lfilter(b, a, exc)
    # ピック位置コーム: 弦上の位置 pick_pos で撥弦した励振形状
    D = max(1, int(pick_pos * N))
    comb = np.zeros(N); comb[:] = exc
    comb[D:] -= exc[:-D]
    dl = comb / (np.max(np.abs(comb)) + 1e-12) * velocity
    y = np.zeros(n_total)
    ap_x1 = 0.0; ap_y1 = 0.0
    prev = 0.0
    idx = 0
    dl = dl.copy()
    for n in range(n_total):
        s = dl[idx]
        # 損失フィルタ: 2点平均 (brightnessで混合) + 全体減衰 rho
        lp = brightness * s + (1 - brightness) * prev
        prev = s
        v = rho * lp
        # 分数遅延オールパス
        ap_y = C * v + ap_x1 - C * ap_y1
        ap_x1 = v; ap_y1 = ap_y
        y[n] = ap_y
        dl[idx] = ap_y
        idx = (idx + 1) % N
    return y

# ---------------------------------------------------------------
# 2. モーダル合成 (マリンバ風バー + 共鳴管の気積共鳴を1本追加)
#    y(t) = sum_k a_k * exp(-t/tau_k) * sin(2*pi*f_k*t)
# ---------------------------------------------------------------
MARIMBA_RATIOS = np.array([1.0, 3.984, 9.90, 17.3, 24.5])   # 調律済みバーの典型比 1:4:10 + 高次
MARIMBA_TAUS   = np.array([0.9, 0.35, 0.16, 0.07, 0.04])    # 秒 (高次ほど速く減衰)
MARIMBA_AMPS   = np.array([1.0, 0.42, 0.18, 0.08, 0.04])

def modal_strike(f0, dur, fs=FS, velocity=1.0,
                 ratios=MARIMBA_RATIOS, taus=MARIMBA_TAUS, amps=MARIMBA_AMPS):
    t = np.arange(int(dur * fs)) / fs
    y = np.zeros_like(t)
    # ベロシティ -> 高次モードの相対増強 (強打で明るくなる挙動モデル)
    tilt = velocity ** np.linspace(1.0, 2.2, len(ratios))
    for r, tau, a, w in zip(ratios, taus, amps, tilt):
        f = f0 * r
        if f > fs * 0.45:
            continue
        y += a * w * np.exp(-t / tau) * np.sin(2 * np.pi * f * t)
    # 打撃アタック: 短いノイズトランジェント
    n_att = int(0.004 * fs)
    att = np.random.default_rng(7).uniform(-1, 1, n_att) * np.hanning(n_att) * 0.25 * velocity
    y[:n_att] += att
    return y * velocity

# ---------------------------------------------------------------
# 3. analysis-by-synthesis: ターゲット音からモード推定 -> 再合成 -> 誤差
# ---------------------------------------------------------------
def estimate_modes(x, fs=FS, n_modes=5, fmin=60.0):
    """FFTピークピッキングでモード周波数、ヘテロダイン包絡の対数線形回帰で減衰と振幅を推定"""
    n = len(x)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    # ピーク検出 (卓越ピーク上位 n_modes)
    pk, props = sig.find_peaks(spec, height=np.max(spec) * 1e-3, distance=int(30 * n / fs))
    pk = pk[freqs[pk] > fmin]
    order = np.argsort(spec[pk])[::-1][:n_modes]
    cand = np.sort(freqs[pk[order]])
    est = []
    t = np.arange(n) / fs
    for f in cand:
        # 放物線補間で周波数を精密化
        k = int(round(f * n / fs))
        if 1 <= k < len(spec) - 1:
            a, b, c = np.log(spec[k-1] + 1e-12), np.log(spec[k] + 1e-12), np.log(spec[k+1] + 1e-12)
            delta = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
            f = (k + delta) * fs / n
        # ヘテロダイン + ローパスで複素包絡
        z = x * np.exp(-2j * np.pi * f * t)
        blp, alp = sig.butter(4, 40 / (fs / 2))
        env = np.abs(sig.filtfilt(blp, alp, z)) * 2
        # 減衰フィット: 包絡ピーク後、ピークの1/100に落ちるまでの区間で log-linear 回帰
        i0 = int(np.argmax(env))
        floor = env[i0] * 0.01
        i1 = i0 + np.argmax(env[i0:] < floor) if np.any(env[i0:] < floor) else n
        i1 = max(i1, i0 + int(0.05 * fs))
        seg = env[i0:i1]
        tt = t[i0:i1]
        A = np.vstack([tt, np.ones_like(tt)]).T
        slope, intercept = np.linalg.lstsq(A, np.log(seg + 1e-12), rcond=None)[0]
        tau = -1.0 / slope if slope < 0 else 10.0
        amp = np.exp(intercept)
        est.append((f, tau, amp))
    return est

def resynth_modes(est, dur, fs=FS):
    t = np.arange(int(dur * fs)) / fs
    y = np.zeros_like(t)
    for f, tau, a in est:
        y += a * np.exp(-t / tau) * np.sin(2 * np.pi * f * t)
    return y

def mss_loss(x, y, fs=FS, sizes=(256, 512, 1024, 2048, 4096)):
    """multi-scale spectral loss (簡易版, log-magnitude L1)"""
    L = 0.0
    n = min(len(x), len(y))
    for s in sizes:
        _, _, X = sig.stft(x[:n], fs, nperseg=s)
        _, _, Y = sig.stft(y[:n], fs, nperseg=s)
        L += np.mean(np.abs(np.log(np.abs(X) + 1e-6) - np.log(np.abs(Y) + 1e-6)))
    return L / len(sizes)

def measure_f0(x, fs=FS, fmin=50, fmax=2000):
    """自己相関によるf0実測"""
    x = x - np.mean(x)
    ac = sig.correlate(x, x, mode="full")[len(x)-1:]
    lo, hi = int(fs / fmax), int(fs / fmin)
    lag = lo + int(np.argmax(ac[lo:hi]))
    # 放物線補間
    if 1 <= lag < len(ac) - 1:
        a, b, c = ac[lag-1], ac[lag], ac[lag+1]
        d = 0.5 * (a - c) / (a - 2*b + c + 1e-12)
        lag = lag + d
    return fs / lag

# ---------------------------------------------------------------
# 4. 標準MIDIファイル (SMF type-0): 独立した writer / parser
# ---------------------------------------------------------------
def varlen(v):
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))

def write_midi(path, events, tpq=480, bpm=120):
    """events: (tick, kind, note, vel, ch)"""
    trk = io.BytesIO()
    tempo = int(60_000_000 / bpm)
    trk.write(b"\x00\xff\x51\x03" + struct.pack(">I", tempo)[1:])
    last = 0
    for tick, kind, note, vel, ch in sorted(events, key=lambda e: e[0]):
        trk.write(varlen(tick - last)); last = tick
        status = (0x90 if kind == "on" else 0x80) | ch
        trk.write(bytes([status, note, vel]))
    trk.write(b"\x00\xff\x2f\x00")
    data = trk.getvalue()
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq))
        f.write(b"MTrk" + struct.pack(">I", len(data)) + data)

def parse_midi(path):
    """最小SMFパーサ (writerと独立実装): [(sec, kind, note, vel, ch)] を返す"""
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:4] == b"MThd"
    _, fmt, ntrk, tpq = struct.unpack(">IHHH", raw[4:14])
    p = 14
    assert raw[p:p+4] == b"MTrk"
    tlen = struct.unpack(">I", raw[p+4:p+8])[0]
    p += 8
    end = p + tlen
    tick = 0; tempo = 500000
    out = []
    running = 0
    while p < end:
        dv = 0
        while True:
            b_ = raw[p]; p += 1
            dv = (dv << 7) | (b_ & 0x7F)
            if not (b_ & 0x80):
                break
        tick += dv
        b_ = raw[p]
        if b_ == 0xFF:
            meta = raw[p+1]; ln = raw[p+2]
            if meta == 0x51:
                tempo = int.from_bytes(raw[p+3:p+3+ln], "big")
            p += 3 + ln
            continue
        if b_ & 0x80:
            running = b_; p += 1
        status = running
        kind = status & 0xF0; ch = status & 0x0F
        if kind in (0x90, 0x80):
            note, vel = raw[p], raw[p+1]; p += 2
            sec = tick * tempo / (tpq * 1_000_000)
            k = "on" if (kind == 0x90 and vel > 0) else "off"
            out.append((sec, k, note, vel, ch))
        else:
            p += 2
    return out

def midi_to_freq(note):
    return 440.0 * 2 ** ((note - 69) / 12)

def render_midi(events, fs=FS):
    """ch0 -> 撥弦(waveguide), ch1 -> マリンバ(modal)"""
    ons = {}
    notes = []   # (start_sec, dur_sec, note, vel, ch)
    for sec, kind, note, vel, ch in events:
        if kind == "on":
            ons[(ch, note)] = (sec, vel)
        else:
            if (ch, note) in ons:
                s, v = ons.pop((ch, note))
                notes.append((s, max(sec - s, 0.05), note, v, ch))
    total = max(s + d for s, d, *_ in notes) + 2.0
    mix = np.zeros(int(total * fs))
    for s, d, note, vel, ch in notes:
        f0 = midi_to_freq(note)
        v = vel / 127
        tail = 1.8
        if ch == 0:
            y = ks_pluck(f0, d + tail, velocity=v)
        else:
            y = modal_strike(f0, d + tail, velocity=v)
        i0 = int(s * fs)
        mix[i0:i0+len(y)] += y * 0.5
    return mix

# ================================================================
if __name__ == "__main__":
    rep = []
    def log(s):
        print(s); rep.append(s)

    log("=== 1. 拡張Karplus-Strong 撥弦: ピッチ精度検証 ===")
    for name, f_target in [("A2", 110.0), ("A3", 220.0), ("E4", 329.63)]:
        y = ks_pluck(f_target, 2.5, velocity=0.9)
        f_meas = measure_f0(y[:FS])
        cents = 1200 * np.log2(f_meas / f_target)
        log(f"  {name}: 目標 {f_target:.2f} Hz -> 実測 {f_meas:.2f} Hz (誤差 {cents:+.1f} cent)")
    demo1 = np.concatenate([
        ks_pluck(110.0, 2.0, velocity=0.4), ks_pluck(110.0, 2.0, velocity=0.9),
        ks_pluck(220.0, 2.0, velocity=0.7), ks_pluck(329.63, 2.5, velocity=0.8)])
    write_wav(os.path.join(OUT, "01_waveguide_pluck.wav"), demo1)
    log("  -> 01_waveguide_pluck.wav (弱/強ベロシティのA2, A3, E4)")

    log("=== 2. モーダル合成 マリンバ風: モード周波数検証 ===")
    f0 = 261.63  # C4
    y2 = modal_strike(f0, 3.0, velocity=0.9)
    spec = np.abs(np.fft.rfft(y2 * np.hanning(len(y2))))
    freqs = np.fft.rfftfreq(len(y2), 1 / FS)
    pk, _ = sig.find_peaks(spec, height=np.max(spec) * 0.01, distance=int(50 * len(y2) / FS))
    found = freqs[pk][:5]
    log(f"  設計モード: {[f'{f0*r:.0f}' for r in MARIMBA_RATIOS[:4]]} Hz")
    log(f"  実測ピーク: {[f'{f:.0f}' for f in found[:4]]} Hz")
    demo2 = np.concatenate([modal_strike(261.63, 1.2, velocity=v) for v in (0.3, 0.6, 1.0)]
                           + [modal_strike(392.0, 1.2, velocity=0.8), modal_strike(523.25, 2.0, velocity=0.9)])
    write_wav(os.path.join(OUT, "02_modal_marimba.wav"), demo2)
    log("  -> 02_modal_marimba.wav (C4 弱中強 + G4 + C5)")

    log("=== 3. analysis-by-synthesis 往復検証 (録音->推定->再合成) ===")
    # ターゲット: 「未知の楽器録音」と見立てたモーダル音 (真値は既知なので回収精度を測れる)
    true_f0 = 196.0  # G3
    target = modal_strike(true_f0, 2.5, velocity=0.85)
    est = estimate_modes(target, n_modes=5)
    log("  推定結果 (周波数Hz / 減衰tau秒):  真値と比較")
    true_modes = [(true_f0 * r, tau) for r, tau in zip(MARIMBA_RATIOS, MARIMBA_TAUS) if true_f0 * r < FS * 0.45]
    for f_e, tau_e, a_e in est:
        # 最近傍の真値モード
        tf, ttau = min(true_modes, key=lambda m: abs(m[0] - f_e))
        log(f"    est {f_e:8.2f} Hz, tau {tau_e:.3f}s | true {tf:8.2f} Hz, tau {ttau:.3f}s"
            f" | Δf {f_e-tf:+.2f} Hz, Δtau {(tau_e-ttau)/ttau*100:+.1f}%")
    resyn = resynth_modes(est, 2.5)
    loss = mss_loss(target, resyn)
    loss_ref = mss_loss(target, np.random.default_rng(1).uniform(-0.5, 0.5, len(target)))  # 参照: ノイズとの距離
    log(f"  multi-scale spectral loss: 再合成 vs 目標 = {loss:.4f} (参照: ノイズ vs 目標 = {loss_ref:.4f})")
    ab = np.concatenate([target, np.zeros(int(0.4 * FS)), resyn])
    write_wav(os.path.join(OUT, "03_target_vs_resynth.wav"), ab)
    log("  -> 03_target_vs_resynth.wav (前半=目標『録音』, 後半=推定パラメータからの再合成)")

    log("=== 4. MIDI経路検証 (SMF生成 -> 独立パーサ -> レンダリング) ===")
    tpq = 480
    def note_ev(beat, dur_b, note, vel, ch):
        t0 = int(beat * tpq); t1 = int((beat + dur_b) * tpq)
        return [(t0, "on", note, vel, ch), (t1, "off", note, 0, ch)]
    events = []
    # ch0 撥弦: ベースライン / ch1 マリンバ: メロディ
    bass = [(0, 45, 96), (1, 45, 70), (2, 48, 96), (3, 43, 80), (4, 45, 100), (6, 40, 90)]
    for b, n, v in bass:
        events += note_ev(b, 0.9, n, v, 0)
    mel = [(0, 69, 80), (0.5, 72, 60), (1, 76, 90), (1.5, 72, 50), (2, 79, 100),
           (3, 76, 70), (4, 81, 110), (5, 76, 60), (6, 72, 80), (7, 69, 90)]
    for b, n, v in mel:
        events += note_ev(b, 0.45, n, v, 1)
    midi_path = os.path.join(OUT, "04_demo.mid")
    write_midi(midi_path, events, tpq=tpq, bpm=100)
    parsed = parse_midi(midi_path)
    n_on = sum(1 for e in parsed if e[1] == "on")
    log(f"  SMF書出し {len(events)} イベント -> 読戻し note-on {n_on} 件 (期待 {len(bass)+len(mel)})")
    mix = render_midi(parsed)
    write_wav(os.path.join(OUT, "04_midi_render.wav"), mix)
    log(f"  -> 04_demo.mid (DAWで開けます) + 04_midi_render.wav (ch0=撥弦, ch1=マリンバ)")

    with open(os.path.join(OUT, "poc_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    log("完了")
