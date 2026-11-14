#!/usr/bin/env python3
"""Skor muzik asli #2: sinematik hangat dan megah (D major, 90 BPM, 28 bar = 74.7 saat).

Piano arpeggio + tali + loceng + dram lembut. Tema: penghargaan yang hangat dan memberi harapan.
Struktur (setiap bar = 2.67s):
    bar  0- 1  Pembuka berkilau (logo)
    bar  2- 5  Piano + tali, melodi masuk bar 4
    bar  6- 9  Melodi A + bass + dram lembut
    bar 10-13  Membina tenaga (tepukan, hi-hat, sapuan)
    bar 14-21  Korus penuh (orbit)
    bar 22-25  Surut perlahan (ucapan)
    bar 26-27  Penyelesaian + loceng

Guna: python3 scripts/make_music2.py assets/audio/muzik_orbit.wav
"""
import sys
import wave
import numpy as np

import synth_lib as S
from synth_lib import SR, Stem, bell, clap, dum, fft_eq, highpass, lowpass, mtof, pad_note, pluck, shaker, riser, boom, reverb_ir, convolve

BPM = 90
BEAT = 60.0 / BPM
BAR = 4 * BEAT
BARS = 28
LEN_S = BARS * BAR + 3.5
S.set_length(LEN_S)
N = S.N
rng = S.rng


def piano(freq, dur, vel=1.0):
    """Piano elektrik/lembut: harmonik lebih panjang di asas, reput pantas di tinggi."""
    n = int(SR * (dur + 0.5))
    t = np.arange(n) / SR
    out = np.zeros(n)
    for det in (1.0, 1.0025):
        for k in range(1, 11):
            fk = freq * k * (1 + 0.0004 * k * k)
            if fk > 9000:
                break
            out += (1.0 / k ** 1.05) * np.sin(2 * np.pi * fk * det * t + k * 0.6) * np.exp(-t * (0.9 + 0.95 * k))
    click = np.diff(rng.standard_normal(int(SR * 0.01)), prepend=0) * np.exp(-np.arange(int(SR * 0.01)) / (SR * 0.002))
    out[:len(click)] += click * 0.12
    env = np.minimum(1, t / 0.003) * np.minimum(1, np.maximum(0, (n / SR - t)) / 0.25)
    return out * env * vel * 0.5


CH = [
    dict(arp=[62, 66, 69, 74], pad=[50, 57, 62, 66], root=38),   # D
    dict(arp=[57, 61, 64, 69], pad=[45, 52, 57, 61], root=33),   # A
    dict(arp=[59, 62, 66, 71], pad=[47, 54, 59, 62], root=35),   # Bm
    dict(arp=[55, 59, 62, 67], pad=[43, 50, 55, 59], root=31),   # G
]
PAT = [0, 1, 2, 3, 2, 1, 2, 1]
MEL_A = [
    [(0, 78, 1.5), (1.5, 81, 0.5), (2, 83, 1), (3, 81, 1)],
    [(0, 85, 1.5), (1.5, 83, 0.5), (2, 81, 2)],
    [(0, 83, 1), (1, 86, 1.5), (2.5, 83, 0.5), (3, 81, 1)],
    [(0, 79, 1), (1, 78, 1), (2, 76, 1), (3, 74, 1)],
]
MEL_B = [
    [(0, 86, 1.5), (1.5, 85, 0.5), (2, 83, 1), (3, 81, 1)],
    [(0, 85, 1.5), (1.5, 88, 0.5), (2, 86, 2)],
    [(0, 86, 1), (1, 83, 1), (2, 81, 1), (3, 83, 1)],
    [(0, 79, 1.5), (1.5, 81, 0.5), (2, 78, 2)],
]

pno, strings, bass, perc, lead, spark, fx = (Stem() for _ in range(7))


def T(bar, beat=0.0):
    return bar * BAR + beat * BEAT


def add_arp(bar, vel=0.6, octave=0):
    c = CH[bar % 4]
    for i, p in enumerate(PAT):
        pno.add(piano(mtof(c['arp'][p] + octave), BEAT * 1.0, vel * (1.0 if i % 2 == 0 else 0.75)), T(bar, i * 0.5),
                pan=-0.25 + 0.1 * (i % 4), gain=0.5)


def add_pad(bar, length_bars=1, gain=0.08, attack=0.7):
    c = CH[bar % 4]
    for n_ in c['pad']:
        strings.add(pad_note(mtof(n_), BAR * length_bars, attack=attack, release=1.4), T(bar), pan=rng.uniform(-0.7, 0.7), gain=gain)


def add_bass(bar, gain=0.4, busy=False):
    c = CH[bar % 4]
    bass.add(pluck(mtof(c['root']), BEAT * 1.8, bright=0.25, courses=False), T(bar), gain=gain)
    bass.add(pluck(mtof(c['root'] + 12), BEAT * 0.9, bright=0.25, courses=False), T(bar, 2.5), gain=gain * 0.6)
    if busy:
        bass.add(pluck(mtof(c['root']), BEAT * 0.9, bright=0.25, courses=False), T(bar, 2), gain=gain * 0.8)
        bass.add(pluck(mtof(c['root'] + 7), BEAT * 0.9, bright=0.25, courses=False), T(bar, 3.5), gain=gain * 0.55)


def add_mel(bar0, mel, vel=1.0, octave=0, gain=0.5, stem=None):
    st = stem or lead
    for i, phrase in enumerate(mel):
        for (b, m, ln) in phrase:
            st.add(piano(mtof(m + octave), ln * BEAT * 1.05, vel), T(bar0 + i, b), pan=rng.uniform(-0.1, 0.1), gain=gain)


def add_drums(bar, kick=True, clap_=True, hat=True, vel=1.0):
    if kick:
        perc.add(dum(0.85 * vel), T(bar), gain=0.7)
        perc.add(dum(0.55 * vel), T(bar, 2.5), gain=0.6)
        perc.add(dum(0.7 * vel), T(bar, 2), gain=0.5)
    if clap_:
        perc.add(clap(0.7 * vel), T(bar, 1), pan=-0.1, gain=0.5)
        perc.add(clap(0.8 * vel), T(bar, 3), pan=0.1, gain=0.55)
    if hat:
        for s in range(8):
            perc.add(shaker((1.0 if s % 2 == 0 else 0.55) * vel), T(bar, s * 0.5), pan=0.4, gain=0.6)


# --- bar 0-1: pembuka
for b in range(0, 2):
    add_pad(b, gain=0.07 + 0.02 * b, attack=1.4)
for i, m in enumerate([74, 78, 81, 86, 90, 93, 86, 81]):
    spark.add(bell(mtof(m), 3.0), 0.6 + i * 0.55, pan=rng.uniform(-0.8, 0.8), gain=0.12)
fx.add(riser(BAR * 0.9, 400, 7000), T(1, 0.5), gain=0.08)
# --- bar 2-5: piano + tali
for b in range(2, 6):
    add_arp(b, vel=0.55)
    add_pad(b, gain=0.09)
fx.add(boom(2.0), T(2), gain=0.3)
add_mel(4, MEL_A, vel=0.8, gain=0.42)
# --- bar 6-9: melodi A + bass + dram lembut
for b in range(6, 10):
    add_arp(b, vel=0.6)
    add_pad(b, gain=0.1)
    add_bass(b, gain=0.36)
    add_drums(b, kick=True, clap_=False, hat=False, vel=0.55)
add_mel(6, MEL_A, vel=0.95, gain=0.5)
# --- bar 10-13: membina
for b in range(10, 14):
    add_arp(b, vel=0.7)
    add_pad(b, gain=0.12)
    add_bass(b, gain=0.4, busy=True)
    add_drums(b, vel=0.75)
add_mel(10, MEL_A, vel=1.0, gain=0.5)
add_mel(10, MEL_A, vel=0.5, octave=12, gain=0.4, stem=spark)
fx.add(riser(BAR, 500, 9000), T(13), gain=0.12)
for i in range(8):
    perc.add(clap(0.3 + 0.07 * i), T(13, 2 + i * 0.25), gain=0.5)
# --- bar 14-21: korus penuh
for b in range(14, 22):
    if b == 14:
        fx.add(boom(2.4), T(14), gain=0.45)
    add_arp(b, vel=0.78)
    add_pad(b, gain=0.14)
    add_bass(b, gain=0.45, busy=True)
    add_drums(b, vel=1.0)
for k in (14, 18):
    add_mel(k, MEL_B, vel=1.05, gain=0.52)
    add_mel(k, MEL_B, vel=0.55, octave=12, gain=0.42, stem=spark)
# harmoni tali pada korus
for b in range(14, 22):
    c = CH[b % 4]
    for n_ in (c['arp'][1] + 12, c['arp'][2] + 12):
        strings.add(pad_note(mtof(n_), BAR, attack=0.4, release=1.0, vib=0.003), T(b), pan=rng.uniform(-0.6, 0.6), gain=0.06)
# --- bar 22-25: surut
for b in range(22, 26):
    add_arp(b, vel=0.5)
    add_pad(b, gain=0.1)
    add_bass(b, gain=0.28)
    perc.add(dum(0.4), T(b), gain=0.45)
add_mel(22, MEL_A, vel=0.85, gain=0.46)
# --- bar 26-27: penyelesaian
add_arp(26, vel=0.45)
add_pad(26, gain=0.1)
bass.add(pluck(mtof(31), BEAT * 3, bright=0.2, courses=False), T(26), gain=0.3)
for n_ in (50, 57, 62, 66, 69, 74):
    strings.add(pad_note(mtof(n_), BAR * 1.6, attack=0.6, release=3.2), T(27), pan=rng.uniform(-0.7, 0.7), gain=0.1)
for n_, d_ in ((62, 0.0), (66, 0.12), (69, 0.24), (74, 0.36), (78, 0.5), (81, 0.62)):
    pno.add(piano(mtof(n_), BEAT * 6, 0.7), T(27) + d_, pan=rng.uniform(-0.3, 0.3), gain=0.5)
bass.add(pluck(mtof(38), BEAT * 6, bright=0.2, courses=False), T(27), gain=0.4)
for i, m in enumerate([86, 90, 93, 98]):
    spark.add(bell(mtof(m), 4.0), T(27, 0.5) + i * 0.4, pan=rng.uniform(-0.8, 0.8), gain=0.12)
perc.add(dum(0.9), T(27), gain=0.6)

# ---------------------------------------------------------------- mixing
def process(stem, eq, send, ir):
    dry = eq(stem.buf)
    return dry + convolve(dry, ir) * send


ir_hall = reverb_ir(2.6, 0.75, 5000)
ir_room = reverb_ir(1.1, 0.3, 6500)
stems = {
    'pno': process(pno, lambda x: fft_eq(x, lambda f: lowpass(6500)(f) * highpass(90)(f)), 0.45, ir_hall),
    'lead': process(lead, lambda x: fft_eq(x, lambda f: lowpass(7000)(f) * highpass(250)(f)), 0.6, ir_hall) * 1.1,
    'strings': process(strings, lambda x: fft_eq(x, lambda f: lowpass(2400)(f) * highpass(80)(f)), 0.5, ir_hall),
    'bass': process(bass, lambda x: fft_eq(x, lambda f: lowpass(280)(f) * highpass(32)(f)), 0.0, ir_room),
    'perc': process(perc, lambda x: fft_eq(x, lambda f: highpass(35)(f)), 0.25, ir_room),
    'spark': process(spark, lambda x: fft_eq(x, lambda f: highpass(500)(f)), 0.65, ir_hall),
    'fx': process(fx, lambda x: fft_eq(x, lambda f: highpass(30)(f)), 0.35, ir_hall),
}
mix = sum(stems.values())
t = np.arange(N) / SR
mix *= np.minimum(1, t / 0.4)[None, :]
end_start = BARS * BAR + 0.5
mix *= np.clip((LEN_S - t) / (LEN_S - end_start), 0, 1)[None, :] ** 1.5
auto = np.interp(t, [0, T(2), T(6), T(10), T(14), T(22), T(26), T(28)],
                 [0.9, 0.85, 0.92, 1.0, 1.1, 0.88, 0.85, 0.9])
mix *= auto[None, :]
mix = np.tanh(mix / (np.abs(mix).max() * 0.55)) * 0.9

out = sys.argv[1] if len(sys.argv) > 1 else 'assets/audio/muzik_orbit.wav'
pcm = (np.clip(mix, -1, 1).T * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print(f"OK {out}  {LEN_S:.1f}s  bar={BAR:.3f}s")
