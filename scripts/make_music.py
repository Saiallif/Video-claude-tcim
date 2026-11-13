#!/usr/bin/env python3
"""Jana skor muzik asli (prosedural) bertema Islamik / Umrah untuk video PMDIUC PJJ Batch 2.

- Maqam Hijaz pada D (D Eb F# G A Bb C), 96 BPM, 4/4, 27 bar = 67.5 saat
- Bunyi: oud, qanun, pad tali, koir "ooh", rebana/darbuka (irama maqsum), tepukan, loceng
- Struktur sepadan dengan keratan video (setiap bar = 2.5 saat):
    bar  0- 3  Pembuka sinematik (intro logo)
    bar  4- 9  Syahdu: oud + rebana perlahan (kad tajuk + 2 gambar)
    bar 10-21  Meriah: irama maqsum penuh (montaj gambar)
    bar 22-26  Penutup: kumpulan penuh + terima kasih

Guna: python3 scripts/make_music.py assets/audio/muzik_pmdiuc.wav
"""
import sys
import numpy as np

SR = 44100
BPM = 96
BEAT = 60.0 / BPM
BAR = 4 * BEAT
BARS = 27
LEN_S = BARS * BAR + 4.0  # +4s ekor
N = int(SR * LEN_S)
rng = np.random.default_rng(7)


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


# ---------------------------------------------------------------- komposisi
D4 = 62
CH = {  # akord (MIDI) ikut bar dalam kitaran 4 bar: D | Eb | C | D
    0: dict(root=38, notes=[50, 54, 57, 62]),
    1: dict(root=39, notes=[51, 55, 58, 63]),
    2: dict(root=36, notes=[48, 52, 55, 60]),
    3: dict(root=38, notes=[50, 54, 57, 62]),
}

PHRASE_A = [  # (mula_eighth, semiton_dari_D4, panjang_eighth)
    (0, 7, 3), (3, 5, 1), (4, 4, 2), (6, 5, 1), (7, 7, 1),
    (8, 8, 3), (11, 7, 1), (12, 5, 2), (14, 4, 1), (15, 5, 1),
    (16, 7, 2), (18, 5, 2), (20, 4, 2), (22, 1, 2),
    (24, 0, 4), (28, 4, 2), (30, 5, 1), (31, 7, 1),
]
PHRASE_B = [
    (0, 12, 3), (3, 10, 1), (4, 8, 2), (6, 7, 2),
    (8, 8, 3), (11, 10, 1), (12, 12, 2), (14, 13, 2),
    (16, 12, 2), (18, 10, 2), (20, 8, 2), (22, 7, 2),
    (24, 5, 2), (26, 4, 2), (28, 0, 4),
]

oud, qanun, pad, choir, bass, perc, sparkle, fx = (Stem() for _ in range(8))
E = BEAT / 2  # satu eighth


def bar_t(b):
    return b * BAR


def chord_of(bar):
    return CH[bar % 4]


def melody(start_bar, phrase, transpose=0, vel=1.0, stem=oud, bright=1.0, doubled=False):
    for (e0, semi, ln) in phrase:
        t0 = bar_t(start_bar) + e0 * E
        m = D4 + semi + transpose
        dur = ln * E * 1.05
        stem.add(pluck(mtof(m), dur, bright=bright), t0, pan=rng.uniform(-0.15, 0.15), gain=0.55 * vel)
        # tremolo (rap) pada nota panjang gaya oud
        if ln >= 3:
            for k in range(1, int(ln * 2)):
                tt = t0 + k * E / 2
                stem.add(pluck(mtof(m), E / 2, bright=bright), tt, gain=0.22 * vel)


# --- Seksyen I: pembuka (bar 0-3)
for b in range(0, 4):
    c = chord_of(b)
    ns = [c['root']] + c['notes']
    for n_ in ns:
        pad.add(pad_note(mtof(n_), BAR, attack=1.6 if b == 0 else 0.9), bar_t(b), pan=rng.uniform(-0.6, 0.6),
                gain=(0.07 + 0.025 * b))
    bass.add(pad_note(mtof(c['root']), BAR, harm=2, attack=0.5), bar_t(b), gain=0.3)
    perc.add(frame_drum(0.55 + 0.12 * b), bar_t(b), gain=0.8)
    if b >= 2:
        perc.add(frame_drum(0.35, deep=False), bar_t(b) + 2 * BEAT, gain=0.6)
# loceng berkilau mengikut logo (naik skala hijaz)
scale_up = [74, 75, 78, 79, 81, 82, 84, 86, 87, 90, 91, 93]
for i, m in enumerate(scale_up):
    sparkle.add(bell(mtof(m), 2.2), 0.9 + i * 0.62, pan=rng.uniform(-0.8, 0.8), gain=0.10)
sparkle.add(bell(mtof(86), 4), 7.4, gain=0.14)
fx.add(riser(2 * BAR, 300, 7000), bar_t(2), gain=0.10)
fx.add(boom(2.6), bar_t(4), gain=0.45)  # tepat pada kad tajuk
choir_notes = lambda c: [c['notes'][1] + 12, c['notes'][2] + 12, c['notes'][0] + 24]
for b in range(2, 4):
    c = chord_of(b)
    for n_ in choir_notes(c):
        choir.add(pad_note(mtof(n_), BAR, harm=6, attack=1.4, vib=0.004), bar_t(b), pan=rng.uniform(-0.5, 0.5), gain=0.05)

# --- Seksyen II: syahdu (bar 4-9)
for b in range(4, 10):
    c = chord_of(b)
    for n_ in [c['root']] + c['notes']:
        pad.add(pad_note(mtof(n_), BAR, attack=0.8), bar_t(b), pan=rng.uniform(-0.6, 0.6), gain=0.085)
    for n_ in choir_notes(c):
        choir.add(pad_note(mtof(n_), BAR, harm=6, attack=1.0, vib=0.004), bar_t(b), pan=rng.uniform(-0.5, 0.5), gain=0.045)
    bass.add(pad_note(mtof(c['root']), BAR, harm=2, attack=0.4), bar_t(b), gain=0.32)
    perc.add(frame_drum(0.8), bar_t(b), gain=0.8)
    perc.add(frame_drum(0.5, deep=False), bar_t(b) + 2 * BEAT, gain=0.6)
    perc.add(frame_drum(0.35, deep=False), bar_t(b) + 3.5 * BEAT, gain=0.4)
melody(4, PHRASE_A, vel=0.95)
melody(8, PHRASE_B[:8] + [(16, 12, 2), (18, 10, 2)], vel=0.95)  # bar 8-9 sahaja
# bina tenaga bar 9: sapuan + gulungan darbuka 16th
fx.add(riser(BAR, 500, 8000), bar_t(9), gain=0.12)
for i in range(8):
    perc.add(tek(0.25 + 0.6 * i / 8), bar_t(9) + BAR / 2 + i * (BAR / 16), gain=0.9)
    perc.add(ka(0.2 + 0.5 * i / 8), bar_t(9) + BAR / 2 + i * (BAR / 16) + BAR / 32, gain=0.7)

# --- Seksyen III: meriah (bar 10-21)  maqsum: D T _ T D _ T _
MAQSUM = [(0, 'D'), (1, 'T'), (3, 'T'), (4, 'D'), (6, 'T')]  # indeks eighth
for b in range(10, 22):
    c = chord_of(b)
    t0 = bar_t(b)
    first = (b == 10)
    if first:
        fx.add(boom(2.4), t0, gain=0.5)
    for n_ in [c['root']] + c['notes']:
        pad.add(pad_note(mtof(n_), BAR, attack=0.5), t0, pan=rng.uniform(-0.7, 0.7), gain=0.075)
    for n_ in choir_notes(c):
        choir.add(pad_note(mtof(n_), BAR, harm=6, attack=0.7, vib=0.004), t0, pan=rng.uniform(-0.6, 0.6), gain=0.05)
    # bass: akar pada ketukan 1, 2.5, 3 dan 4.5
    for off, ln in ((0, 1.4), (2, 1.0), (2.5, 0.8), (3.5, 0.5)):
        bass.add(pluck(mtof(c['root']), ln * BEAT, bright=0.3, courses=False), t0 + off * BEAT, gain=0.38 if off == 0 else 0.26)
    # irama
    for e, kind in MAQSUM:
        tt = t0 + e * E
        if kind == 'D':
            perc.add(dum(0.95 if e == 0 else 0.8), tt, gain=0.85)
        else:
            perc.add(tek(0.9), tt, pan=0.25, gain=0.8)
    for e in (2, 5, 7):  # ka lembut mengisi
        perc.add(ka(0.6), t0 + e * E, pan=-0.25, gain=0.7)
    # rebana (frame drum) pada ketukan 2 & 4 + tepukan tangan
    perc.add(frame_drum(0.4, deep=False), t0 + BEAT, gain=0.5)
    perc.add(clap(0.8), t0 + BEAT, pan=-0.2, gain=0.55)
    perc.add(clap(0.9), t0 + 3 * BEAT, pan=0.2, gain=0.6)
    perc.add(frame_drum(0.4, deep=False), t0 + 3 * BEAT, gain=0.5)
    for s in range(8):  # shaker 16th
        perc.add(shaker(1.0 if s % 2 == 0 else 0.6), t0 + s * E + (0 if s % 2 == 0 else 0) , pan=0.5, gain=0.7)
        perc.add(shaker(0.5), t0 + s * E + E / 2, pan=0.5, gain=0.6)
# melodi (oud + qanun satu oktaf atas)
for i, (sb, ph) in enumerate(((10, PHRASE_A), (14, PHRASE_B), (18, PHRASE_A))):
    melody(sb, ph, vel=1.0, stem=oud)
    melody(sb, ph, transpose=12, vel=0.55, stem=qanun, bright=1.6)
    if i == 2:
        pass
# bar 21 -> peralihan: sapuan kecil
fx.add(riser(BAR * 0.8, 800, 9000), bar_t(21) + BAR * 0.2, gain=0.09)

# --- Seksyen IV: penutup (bar 22-26)
for b in range(22, 24):  # kumpulan penuh: masih bertenaga tetapi separuh ketumpatan
    c = chord_of(b)
    t0 = bar_t(b)
    for n_ in [c['root']] + c['notes']:
        pad.add(pad_note(mtof(n_), BAR, attack=0.5), t0, pan=rng.uniform(-0.7, 0.7), gain=0.085)
    for n_ in choir_notes(c):
        choir.add(pad_note(mtof(n_), BAR, harm=6, attack=0.6, vib=0.004), t0, pan=rng.uniform(-0.6, 0.6), gain=0.06)
    bass.add(pluck(mtof(c['root']), 1.8 * BEAT, bright=0.3, courses=False), t0, gain=0.38)
    perc.add(dum(0.9), t0, gain=0.85)
    perc.add(dum(0.7), t0 + 2 * BEAT, gain=0.8)
    perc.add(clap(0.8), t0 + BEAT, gain=0.5)
    perc.add(clap(0.8), t0 + 3 * BEAT, gain=0.5)
    perc.add(tek(0.7), t0 + 1.5 * BEAT, gain=0.7)
    perc.add(tek(0.7), t0 + 3.5 * BEAT, gain=0.7)
melody(22, [(0, 12, 3), (3, 10, 1), (4, 8, 2), (6, 7, 2), (8, 8, 3), (11, 10, 1), (12, 12, 2), (14, 13, 2)], vel=1.0)
melody(22, [(0, 12, 3), (3, 10, 1), (4, 8, 2), (6, 7, 2), (8, 8, 3), (11, 10, 1), (12, 12, 2), (14, 13, 2)],
       transpose=12, vel=0.5, stem=qanun, bright=1.6)
# akhir: Eb -> C -> D tenang, tamat dengan akord D besar
end_chords = [(24, CH[1]), (25, CH[2]), (26, CH[0])]
for b, c in end_chords:
    last = (b == 26)
    dur = BAR * (2.4 if last else 1.0)
    for n_ in [c['root']] + c['notes']:
        pad.add(pad_note(mtof(n_), dur, attack=0.5, release=2.5), bar_t(b), pan=rng.uniform(-0.7, 0.7), gain=0.09)
    for n_ in choir_notes(c):
        choir.add(pad_note(mtof(n_), dur, harm=6, attack=0.7, release=2.5, vib=0.004), bar_t(b), pan=rng.uniform(-0.6, 0.6), gain=0.06)
    bass.add(pluck(mtof(c['root']), 2.5 * BEAT, bright=0.3, courses=False), bar_t(b), gain=0.4)
    perc.add(frame_drum(0.9 if last else 0.6), bar_t(b), gain=0.8)
melody(24, [(0, 8, 4), (4, 7, 2), (6, 5, 2), (8, 7, 3), (11, 5, 1), (12, 4, 2), (14, 1, 2)], vel=0.85)
melody(26, [(0, 0, 8)], vel=0.9)
for i, m in enumerate([86, 90, 93, 98]):
    sparkle.add(bell(mtof(m), 3.5), bar_t(26) + 0.25 + i * 0.35, pan=rng.uniform(-0.8, 0.8), gain=0.11)

# ---------------------------------------------------------------- mixing
def process(stem, eq, send, rev_ir):
    dry = eq(stem.buf)
    wet = convolve(dry, rev_ir) * send
    return dry + wet


ir_hall = reverb_ir(2.8, 0.8, 4500)
ir_room = reverb_ir(1.2, 0.3, 6000)

stems = {
    'oud': process(oud, lambda x: fft_eq(x, lambda f: lowpass(4800)(f) * highpass(110)(f) * (1 + 0.7 * np.exp(-((f - 420) / 300) ** 2))), 0.55, ir_hall) * 1.0,
    'qanun': process(qanun, lambda x: fft_eq(x, lambda f: lowpass(6500)(f) * highpass(300)(f)), 0.6, ir_hall) * 0.9,
    'pad': process(pad, lambda x: fft_eq(x, lambda f: lowpass(1900)(f) * highpass(80)(f)), 0.5, ir_hall) * 1.0,
    'choir': process(choir, lambda x: fft_eq(x, lambda f: (np.exp(-((f - 330) / 120) ** 2) * 1.0 + np.exp(-((f - 850) / 220) ** 2) * 0.6 + np.exp(-((f - 2300) / 400) ** 2) * 0.15)), 0.8, ir_hall) * 1.7,
    'bass': process(bass, lambda x: fft_eq(x, lambda f: lowpass(260)(f) * highpass(32)(f)), 0.0, ir_room) * 1.0,
    'perc': process(perc, lambda x: fft_eq(x, lambda f: highpass(35)(f)), 0.28, ir_room) * 1.0,
    'sparkle': process(sparkle, lambda x: fft_eq(x, lambda f: highpass(500)(f)), 0.6, ir_hall) * 1.0,
    'fx': process(fx, lambda x: fft_eq(x, lambda f: highpass(30)(f)), 0.35, ir_hall) * 1.0,
}
mix = sum(stems.values())
# fade-in pendek & fade-out ekor
t = np.arange(N) / SR
mix *= np.minimum(1, t / 0.5)[None, :]
end_fade_start = BARS * BAR + 0.2
mix *= np.clip((LEN_S - t) / (LEN_S - end_fade_start), 0, 1)[None, :] ** 1.5
# automasi dinamik: seksyen syahdu lebih lembut, naik perlahan ke seksyen meriah
auto = np.interp(t, [0, bar_t(4) - 0.01, bar_t(4), bar_t(9), bar_t(10), bar_t(22), bar_t(24), bar_t(27)],
                 [1.0, 1.0, 0.72, 0.9, 1.05, 1.05, 0.9, 0.85])
mix *= auto[None, :]
# pengehad lembut
peak = np.abs(mix).max()
mix = np.tanh(mix / (peak * 0.55)) * 0.9
mix = mix.astype(np.float32)

out = sys.argv[1] if len(sys.argv) > 1 else 'assets/audio/muzik_pmdiuc.wav'
import wave
pcm = (np.clip(mix, -1, 1).T * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print(f"OK {out}  {LEN_S:.1f}s  peak={np.abs(mix).max():.2f}")
