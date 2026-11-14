#!/usr/bin/env python3
"""Skor muzik asli #3: pop ceria "happening" (G major, 120 BPM, 48 bar = 96 saat).

Kick empat-pada-lantai, tepukan, hi-hat, bass funky, stab kord off-beat, piano, melodi catchy, loceng,
dan kesan "sidechain pump". Setiap bar = 2.0 saat supaya mudah disegerakkan dengan visual.

    bar  0- 1  Pembuka berkilau (logo)
    bar  2- 5  Membina (piano, hi-hat, gulungan tepukan)
    bar  6- 9  DROP pertama (orbit laju)
    bar 10-25  Groove (spotlight 8 kenangan)
    bar 26-29  Korus (orbit pecah, konfeti)
    bar 30-41  Groove variasi (spotlight 6 kenangan)
    bar 42-45  Korus akhir (terima kasih)
    bar 46-47  Penutup

Guna: python3 scripts/make_music3.py assets/audio/muzik_happy.wav
"""
import sys
import wave

import numpy as np

import synth_lib as S
from synth_lib import SR, Stem, bell, boom, clap, convolve, dum, fft_eq, highpass, lowpass, mtof, pad_note, pluck, reverb_ir, riser, shaker

BPM = 120
BEAT = 60.0 / BPM
BAR = 4 * BEAT
BARS = 48
LEN_S = BARS * BAR + 3.0
S.set_length(LEN_S)
N = S.N
rng = S.rng


def piano(freq, dur, vel=1.0):
    n = int(SR * (dur + 0.4))
    t = np.arange(n) / SR
    out = np.zeros(n)
    for det in (1.0, 1.003):
        for k in range(1, 10):
            fk = freq * k * (1 + 0.0004 * k * k)
            if fk > 9000:
                break
            out += (1.0 / k ** 1.05) * np.sin(2 * np.pi * fk * det * t + k * 0.6) * np.exp(-t * (1.1 + 1.0 * k))
    click = np.diff(rng.standard_normal(int(SR * 0.008)), prepend=0) * np.exp(-np.arange(int(SR * 0.008)) / (SR * 0.002))
    out[:len(click)] += click * 0.12
    env = np.minimum(1, t / 0.003) * np.minimum(1, np.maximum(0, (n / SR - t)) / 0.15)
    return out * env * vel * 0.5


def stab(freq, dur, vel=1.0):
    """Stab synth cerah (zigzag harmonik, reput pantas)."""
    n = int(SR * (dur + 0.12))
    t = np.arange(n) / SR
    out = np.zeros(n)
    for det in (0.996, 1.0, 1.004):
        for k in range(1, 9):
            out += np.sin(2 * np.pi * freq * k * det * t + k) / k * (1.0 if k % 2 else 0.6)
    env = np.minimum(1, t / 0.004) * np.exp(-t * 11) * np.minimum(1, np.maximum(0, (n / SR - t)) / 0.05)
    return out * env * vel * 0.22


CH = [  # G | D | Em | C
    dict(root=43, tri=[67, 71, 74], arp=[67, 71, 74, 79]),
    dict(root=38, tri=[66, 69, 74], arp=[66, 69, 74, 78]),
    dict(root=40, tri=[67, 71, 76], arp=[67, 71, 76, 79]),
    dict(root=36, tri=[67, 72, 76], arp=[67, 72, 76, 79]),
]
HOOK = [  # 4 bar: (beat, midi, length-beats)
    [(0, 79, .5), (.5, 83, .5), (1, 86, 1), (2.5, 83, .5), (3, 81, 1)],
    [(0, 81, .5), (.5, 86, .5), (1, 90, 1), (2.5, 86, .5), (3, 81, 1)],
    [(0, 83, .5), (.5, 79, .5), (1, 88, 1), (2.5, 83, .5), (3, 79, 1)],
    [(0, 84, .5), (.5, 88, .5), (1, 91, 1.5), (3, 88, .5), (3.5, 84, .5)],
]

pno, pad, stabs, bass, perc, lead, spark, fx = (Stem() for _ in range(8))


def T(bar, beat=0.0):
    return bar * BAR + beat * BEAT


def c(bar):
    return CH[bar % 4]


def drums(bar, vel=1.0, hat16=False, kick=True):
    if kick:
        for b in range(4):
            perc.add(dum(0.95 * vel), T(bar, b), gain=0.9)
    perc.add(clap(0.9 * vel), T(bar, 1), pan=-0.1, gain=0.7)
    perc.add(clap(1.0 * vel), T(bar, 3), pan=0.1, gain=0.75)
    for s in range(4):
        perc.add(shaker(1.1 * vel), T(bar, s + 0.5), pan=0.4, gain=0.75)
    for s in range(8):
        perc.add(shaker(0.45 * vel), T(bar, s * 0.5), pan=-0.4, gain=0.5)
    if hat16:
        for s in range(16):
            perc.add(shaker(0.35 * vel), T(bar, s * 0.25 + 0.125), pan=0.5, gain=0.5)


def add_bass(bar, vel=1.0):
    r = c(bar)['root']
    for beat, m, ln in ((0, r, 0.6), (0.75, r + 12, 0.3), (1.5, r, 0.4), (2, r, 0.6), (2.75, r + 12, 0.3), (3.5, r + 7, 0.4)):
        bass.add(pluck(mtof(m), BEAT * ln, bright=0.3, courses=False), T(bar, beat), gain=0.42 * vel)


def add_stabs(bar, vel=1.0):
    for beat in (0.5, 1.5, 2.5, 3.5):
        for m in c(bar)['tri']:
            stabs.add(stab(mtof(m), BEAT * 0.3, vel), T(bar, beat), pan=rng.uniform(-0.4, 0.4), gain=0.42)


def add_pad(bar, gain=0.07, length=1):
    for m in c(bar)['tri'] + [c(bar)['root'] + 24]:
        pad.add(pad_note(mtof(m - 12), BAR * length, attack=0.35, release=0.9), T(bar), pan=rng.uniform(-0.7, 0.7), gain=gain)


def add_arp(bar, vel=0.6):
    a = c(bar)['arp']
    for i, ix in enumerate([0, 1, 2, 3, 2, 1, 2, 3]):
        pno.add(piano(mtof(a[ix]), BEAT * 0.5, vel * (1 if i % 2 == 0 else 0.75)), T(bar, i * 0.5), pan=-0.2 + 0.05 * i, gain=0.5)


def add_chords(bar, vel=0.7):
    for beat in (0, 2):
        for m in c(bar)['tri']:
            pno.add(piano(mtof(m - 12), BEAT * 1.8, vel), T(bar, beat), pan=rng.uniform(-0.2, 0.2), gain=0.45)


def add_hook(bar0, vel=1.0, octave_bell=True):
    for i, ph in enumerate(HOOK):
        for beat, m, ln in ph:
            lead.add(piano(mtof(m), BEAT * ln * 1.05, vel), T(bar0 + i, beat), pan=rng.uniform(-0.1, 0.1), gain=0.55)
            if octave_bell:
                spark.add(bell(mtof(m + 12), 1.2), T(bar0 + i, beat), pan=rng.uniform(-0.6, 0.6), gain=0.06 * vel)


def roll(bar, beat0, n, v0=0.25, v1=1.0):
    span = 4 - beat0
    for i in range(n):
        perc.add(clap(v0 + (v1 - v0) * i / n), T(bar, beat0 + span * i / n), gain=0.6)


# ----- bar 0-1
for i, m in enumerate([79, 83, 86, 91, 95, 98, 91, 86]):
    spark.add(bell(mtof(m), 3.0), 0.4 + i * 0.38, pan=rng.uniform(-0.8, 0.8), gain=0.13)
for b in (0, 1):
    add_pad(b, gain=0.06 + 0.03 * b)
fx.add(riser(BAR * 0.9, 400, 8000), T(1, 0.5), gain=0.09)
# ----- bar 2-5 membina
for b in range(2, 6):
    add_pad(b, gain=0.09)
    add_chords(b, 0.7)
    if b >= 3:
        add_arp(b, 0.5)
    if b >= 4:
        for s in range(8):
            perc.add(shaker(0.6), T(b, s * 0.5), pan=0.4, gain=0.6)
        perc.add(dum(0.7), T(b), gain=0.7)
        perc.add(dum(0.7), T(b, 2), gain=0.7)
roll(5, 1.5, 12)
fx.add(riser(BAR, 500, 9500), T(5), gain=0.14)
# ----- bar 6-9 DROP
fx.add(boom(2.4), T(6), gain=0.5)
perc.add(clap(1.2), T(6), gain=0.6)
for b in range(6, 10):
    drums(b, 1.0)
    add_bass(b)
    add_stabs(b)
    add_pad(b, gain=0.09)
add_hook(6, 1.05)
# ----- bar 10-25 groove
for b in range(10, 26):
    drums(b, 0.9, kick=True)
    add_bass(b, 0.9)
    add_stabs(b, 0.7)
    add_pad(b, gain=0.07)
    add_arp(b, 0.55 if b < 18 else 0.65)
for k in (14, 22):
    add_hook(k, 0.95)
roll(25, 1.5, 12)
fx.add(riser(BAR, 500, 9500), T(25), gain=0.14)
# ----- bar 26-29 korus
fx.add(boom(2.4), T(26), gain=0.5)
perc.add(clap(1.2), T(26), gain=0.6)
for b in range(26, 30):
    drums(b, 1.05, hat16=True)
    add_bass(b, 1.05)
    add_stabs(b, 0.9)
    add_pad(b, gain=0.1)
    add_arp(b, 0.65)
add_hook(26, 1.1)
# ----- bar 30-41 groove variasi
for b in range(30, 42):
    drums(b, 0.95)
    add_bass(b, 0.95)
    add_stabs(b, 0.75)
    add_pad(b, gain=0.075)
    add_arp(b, 0.6)
for k in (30, 34, 38):
    add_hook(k, 1.0)
roll(41, 1.5, 12)
fx.add(riser(BAR, 500, 9500), T(41), gain=0.14)
# ----- bar 42-45 korus akhir
fx.add(boom(2.4), T(42), gain=0.55)
perc.add(clap(1.3), T(42), gain=0.6)
for b in range(42, 46):
    drums(b, 1.1, hat16=True)
    add_bass(b, 1.1)
    add_stabs(b, 1.0)
    add_pad(b, gain=0.11)
    add_arp(b, 0.7)
add_hook(42, 1.15)
# ----- bar 46-47 penutup
for beat in (0, 1, 2):
    perc.add(dum(0.9), T(46, beat), gain=0.8)
perc.add(clap(1.0), T(46, 1), gain=0.6)
perc.add(clap(1.0), T(46, 3), gain=0.6)
add_stabs(46, 0.8)
add_pad(46, gain=0.1)
bass.add(pluck(mtof(43), BEAT * 2, bright=0.3, courses=False), T(46), gain=0.45)
perc.add(dum(1.0), T(47), gain=0.9)
perc.add(clap(1.2), T(47), gain=0.8)
for m in (43, 55, 59, 62, 67, 71, 74):
    pad.add(pad_note(mtof(m), BAR * 1.5, attack=0.05, release=2.2), T(47), pan=rng.uniform(-0.7, 0.7), gain=0.1)
for m, d_ in ((67, 0), (71, 0.05), (74, 0.1), (79, 0.15)):
    pno.add(piano(mtof(m), BAR * 1.5, 1.0), T(47) + d_, pan=rng.uniform(-0.3, 0.3), gain=0.55)
bass.add(pluck(mtof(31), BAR, bright=0.2, courses=False), T(47), gain=0.5)
for i, m in enumerate([91, 95, 98, 103]):
    spark.add(bell(mtof(m), 3.5), T(47, 0.1) + i * 0.15, pan=rng.uniform(-0.8, 0.8), gain=0.13)


# ---------------------------------------------------------------- mixing
def process(stem, eq, send, ir):
    dry = eq(stem.buf)
    return dry + convolve(dry, ir) * send


ir_hall = reverb_ir(2.0, 0.55, 5500)
ir_room = reverb_ir(0.9, 0.25, 7000)
t = np.arange(N) / SR
duck = 1 - 0.55 * np.exp(-(t % BEAT) / 0.10)
duck = np.where((t >= T(6)) & (t < T(46)), duck, 1.0).astype(np.float32)[None, :]

stems = {
    'pno': process(pno, lambda x: fft_eq(x, lambda f: lowpass(6500)(f) * highpass(90)(f)), 0.35, ir_hall) * duck,
    'pad': process(pad, lambda x: fft_eq(x, lambda f: lowpass(2600)(f) * highpass(80)(f)), 0.45, ir_hall) * duck,
    'stab': process(stabs, lambda x: fft_eq(x, lambda f: lowpass(5500)(f) * highpass(200)(f)), 0.25, ir_room) * duck,
    'lead': process(lead, lambda x: fft_eq(x, lambda f: lowpass(7500)(f) * highpass(250)(f)), 0.4, ir_hall) * 1.15,
    'bass': process(bass, lambda x: fft_eq(x, lambda f: lowpass(300)(f) * highpass(32)(f)), 0.0, ir_room),
    'perc': process(perc, lambda x: fft_eq(x, lambda f: highpass(30)(f)), 0.18, ir_room),
    'spark': process(spark, lambda x: fft_eq(x, lambda f: highpass(500)(f)), 0.55, ir_hall) * duck,
    'fx': process(fx, lambda x: fft_eq(x, lambda f: highpass(30)(f)), 0.3, ir_hall),
}
mix = sum(stems.values())
mix *= np.minimum(1, t / 0.3)[None, :]
end = BARS * BAR + 0.5
mix *= np.clip((LEN_S - t) / (LEN_S - end), 0, 1)[None, :] ** 1.5
peak = np.abs(mix).max()
mix = np.tanh(mix / (peak * 0.5)) * 0.92

out = sys.argv[1] if len(sys.argv) > 1 else 'assets/audio/muzik_happy.wav'
pcm = (np.clip(mix, -1, 1).T * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print(f"OK {out}  {LEN_S:.1f}s bar={BAR:.2f}s")
