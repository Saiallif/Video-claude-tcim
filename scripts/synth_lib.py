"""Pustaka sintesis bunyi (oud/piano-pluck, pad, loceng, dram, reverb) untuk skor muzik prosedural.

Diekstrak daripada make_music.py. Panggil set_length(saat) sebelum mencipta Stem.
"""
import numpy as np

SR = 44100
N = 0
rng = np.random.default_rng(7)


def set_length(seconds):
    global N
    N = int(SR * seconds)
    return N


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


class Stem:
    def __init__(self):
        self.buf = np.zeros((2, N), dtype=np.float32)

    def add(self, sig, t0, pan=0.0, gain=1.0):
        i = int(t0 * SR)
        if i >= N:
            return
        n = min(len(sig), N - i)
        l = np.cos((pan + 1) * np.pi / 4)
        r = np.sin((pan + 1) * np.pi / 4)
        self.buf[0, i:i + n] += sig[:n] * gain * l
        self.buf[1, i:i + n] += sig[:n] * gain * r


def fft_eq(x, shape_fn):
    """Terapkan EQ (fungsi gandaan ikut frekuensi) pada isyarat stereo (2,N)."""
    n = x.shape[1]
    X = np.fft.rfft(x, axis=1)
    f = np.fft.rfftfreq(n, 1 / SR)
    return np.fft.irfft(X * shape_fn(f)[None, :], n=n, axis=1).astype(np.float32)


def lowpass(fc, slope=1.0):
    return lambda f: 1.0 / (1.0 + (f / fc) ** (4 * slope))


def highpass(fc):
    return lambda f: 1.0 / (1.0 + (fc / np.maximum(f, 1)) ** 4)


def reverb_ir(seconds=2.8, tau=0.75, damp=4500):
    n = int(SR * seconds)
    ir = np.zeros((2, n), dtype=np.float32)
    t = np.arange(n) / SR
    for c in range(2):
        noise = rng.standard_normal(n)
        F = np.fft.rfft(noise)
        f = np.fft.rfftfreq(n, 1 / SR)
        noise = np.fft.irfft(F / (1 + (f / damp) ** 4), n=n)
        ir[c] = noise * np.exp(-t / tau) * (1 - np.exp(-t / 0.02))
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True)) + 1e-9
    return ir


def convolve(x, ir):
    n = x.shape[1] + ir.shape[1]
    nf = 1 << (n - 1).bit_length()
    out = np.zeros_like(x)
    for c in range(2):
        X = np.fft.rfft(x[c], nf)
        # silang: bunyi kiri guna IR kiri, kanan guna IR kanan (lebar stereo)
        Y = np.fft.irfft(X * np.fft.rfft(ir[c], nf), nf)[:x.shape[1]]
        out[c] = Y
    return out


# ---------------------------------------------------------------- instrumen
def pluck(freq, dur, bright=1.0, courses=True):
    """Oud/qanun: sintesis aditif, harmonik reput lebih laju di frekuensi tinggi."""
    n = int(SR * (dur + 0.25))
    t = np.arange(n) / SR
    out = np.zeros(n)
    for detune in ((1.0, 1.0035) if courses else (1.0,)):
        for k in range(1, 15):
            fk = freq * k * detune
            if fk > 9000:
                break
            amp = (1.0 / k ** 1.15) * (bright if k > 3 else 1.0)
            decay = 2.2 + 1.35 * k
            out += amp * np.sin(2 * np.pi * fk * t + k) * np.exp(-t * decay)
    # bunyi petikan (pick)
    click = rng.standard_normal(int(SR * 0.012))
    click = np.diff(click, prepend=0) * np.exp(-np.arange(len(click)) / (SR * 0.003))
    out[:len(click)] += click * 0.35
    env = np.minimum(1, t / 0.002) * np.minimum(1, (n / SR - t) / 0.12)
    return out * env


def bell(freq, dur=2.5):
    n = int(SR * dur)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for r, a, d in ((1, 1, 1.6), (2.76, 0.45, 2.6), (5.4, 0.25, 4.0), (8.93, 0.12, 6.0)):
        out += a * np.sin(2 * np.pi * freq * r * t) * np.exp(-t * d)
    return out * np.minimum(1, t / 0.003)


def pad_note(freq, dur, harm=9, attack=1.1, release=1.6, vib=0.0):
    n = int(SR * (dur + release))
    t = np.arange(n) / SR
    out = np.zeros(n)
    for det in (-0.0045, 0.0, 0.0045):
        ph = 2 * np.pi * freq * (1 + det) * t
        if vib:
            ph = ph + vib * np.sin(2 * np.pi * 5.3 * t)
        for k in range(1, harm + 1):
            out += np.sin(k * ph + k * 0.7) / k
    env = np.minimum(1, t / attack) * np.minimum(1, np.maximum(0, (n / SR - t)) / release)
    return out * env / 3.0


def dum(vel=1.0):
    n = int(SR * 0.45)
    t = np.arange(n) / SR
    f = 62 + 95 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 9)
    thump = rng.standard_normal(n) * np.exp(-t * 90) * 0.15
    return (body + thump) * vel


def tek(vel=1.0):
    n = int(SR * 0.09)
    t = np.arange(n) / SR
    noise = np.diff(rng.standard_normal(n), prepend=0)
    ring = np.sin(2 * np.pi * 1900 * t) * 0.5 + np.sin(2 * np.pi * 2730 * t) * 0.3
    return (noise * 0.6 + ring) * np.exp(-t * 55) * vel * 0.55


def ka(vel=1.0):
    n = int(SR * 0.07)
    t = np.arange(n) / SR
    noise = np.diff(rng.standard_normal(n), prepend=0)
    ring = np.sin(2 * np.pi * 1350 * t) * 0.4
    return (noise * 0.5 + ring) * np.exp(-t * 70) * vel * 0.4


def frame_drum(vel=1.0, deep=True):
    n = int(SR * (1.1 if deep else 0.6))
    t = np.arange(n) / SR
    f = 52 + 70 * np.exp(-t * 14)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * (3.4 if deep else 6))
    skin = rng.standard_normal(n) * np.exp(-t * 55) * 0.12
    return (body + skin) * vel


def clap(vel=1.0):
    n = int(SR * 0.16)
    out = np.zeros(n)
    t = np.arange(n) / SR
    for i, off in enumerate((0, 0.009, 0.019, 0.03)):
        s = int(off * SR)
        m = n - s
        burst = np.diff(rng.standard_normal(m), prepend=0)
        out[s:] += burst * np.exp(-np.arange(m) / SR * (160 if i < 3 else 38)) * (0.6 if i < 3 else 1.0)
    return out * vel * 0.5


def shaker(vel=1.0):
    n = int(SR * 0.05)
    t = np.arange(n) / SR
    noise = np.diff(np.diff(rng.standard_normal(n), prepend=0), prepend=0)
    return noise * np.exp(-t * 80) * np.minimum(1, t / 0.004) * vel * 0.25


def riser(dur, f0=300, f1=6000):
    n = int(SR * dur)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    # sapuan: bahagi kepada segmen pendek, tapis jalur lalu naikkan frekuensi
    seg = 2048
    out = np.zeros(n)
    for s in range(0, n - seg, seg // 2):
        frac = s / n
        fc = f0 * (f1 / f0) ** frac
        chunk = noise[s:s + seg] * np.hanning(seg)
        C = np.fft.rfft(chunk)
        ff = np.fft.rfftfreq(seg, 1 / SR)
        C *= np.exp(-((np.log(np.maximum(ff, 1)) - np.log(fc)) ** 2) / 0.5)
        out[s:s + seg] += np.fft.irfft(C, n=seg)
    out /= np.abs(out).max() + 1e-9
    return out * (t / dur) ** 2.2


def boom(dur=2.2):
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = 40 + 90 * np.exp(-t * 7)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 1.6) + rng.standard_normal(n) * np.exp(-t * 18) * 0.15


