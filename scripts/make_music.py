#!/usr/bin/env python3
"""Synthesise an energetic 128 BPM EDM / pop-anthem backing track + SFX.

Everything is generated procedurally (no samples, no copyright issues).
Timeline is expressed in beats so the video can be locked to it.

Usage: make_music.py OUT.wav
"""
import sys
import numpy as np
from scipy import signal

SR = 44100
BPM = 128.0
BEAT = 60.0 / BPM
TOTAL_BEATS = 184
N = int(SR * BEAT * TOTAL_BEATS) + SR  # + 1s tail
rng = np.random.default_rng(7)


def b2s(b):
    return int(round(b * BEAT * SR))


def note(m):  # midi -> Hz
    return 440.0 * 2 ** ((m - 69) / 12)


NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def n(name):  # "E5" -> midi
    return 12 * (int(name[-1]) + 1) + NOTE[name[0]]


def lp(x, fc, order=2):
    sos = signal.butter(order, fc, "low", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def hp(x, fc, order=2):
    sos = signal.butter(order, fc, "high", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def bp(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, hi], "band", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def add(buf, start, sig, gain=1.0, pan=0.0):
    """Mix mono `sig` into stereo `buf` at sample `start`."""
    if start < 0:
        sig = sig[-start:]
        start = 0
    end = min(buf.shape[1], start + len(sig))
    if end <= start:
        return
    s = sig[: end - start] * gain
    l = np.cos((pan + 1) * np.pi / 4)
    r = np.sin((pan + 1) * np.pi / 4)
    buf[0, start:end] += s * l * 1.414
    buf[1, start:end] += s * r * 1.414


def saw(f, t, detune=0.0):
    ph = (f * (1 + detune) * t) % 1.0
    return 2 * ph - 1


def square(f, t, pw=0.5):
    ph = (f * t) % 1.0
    return np.where(ph < pw, 1.0, -1.0)


# ---------------------------------------------------------------- drums
def kick(length=0.38):
    t = np.arange(int(SR * length)) / SR
    f = 48 + 120 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 7.5)
    click = hp(rng.standard_normal(len(t)), 3000) * np.exp(-t * 180) * 0.25
    return np.tanh((body + click) * 1.6)


def clap():
    t = np.arange(int(SR * 0.28)) / SR
    nz = bp(rng.standard_normal(len(t)), 900, 7500)
    env = np.zeros_like(t)
    for off in (0.0, 0.011, 0.023):  # layered flams
        env += np.where(t >= off, np.exp(-(t - off) * 55), 0)
    env += np.where(t >= 0.03, np.exp(-(t - 0.03) * 18) * 0.7, 0)
    return nz * env * 0.9


def snare():
    t = np.arange(int(SR * 0.3)) / SR
    tone = np.sin(2 * np.pi * (190 + 60 * np.exp(-t * 40)) * t) * np.exp(-t * 22)
    nz = hp(rng.standard_normal(len(t)), 1500) * np.exp(-t * 17)
    return np.tanh(tone * 0.8 + nz * 0.9)


def hat(open_=False):
    L = 0.25 if open_ else 0.06
    t = np.arange(int(SR * L)) / SR
    nz = hp(rng.standard_normal(len(t)), 7000)
    return nz * np.exp(-t * (14 if open_ else 70))


def crash(length=2.6):
    t = np.arange(int(SR * length)) / SR
    nz = hp(rng.standard_normal(len(t)), 4500)
    return nz * np.exp(-t * 1.9)


def riser(length, f0=300, f1=9000):
    L = int(SR * length)
    t = np.arange(L) / SR
    nz = rng.standard_normal(L)
    # sweep by blending a few band-passed versions
    out = np.zeros(L)
    edges = np.geomspace(f0, f1, 9)
    for i in range(8):
        band = bp(nz, edges[i], min(edges[i + 1], SR / 2 - 100))
        centre = (i + 0.5) / 8
        prog = t / length
        w = np.exp(-((prog - centre) ** 2) / 0.02)
        out += band * w
    env = (t / length) ** 1.6
    return out * env


def subdrop(length=1.2):
    t = np.arange(int(SR * length)) / SR
    f = 28 + 90 * np.exp(-t * 4)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)


def whoosh(length=0.5):
    L = int(SR * length)
    nz = rng.standard_normal(L)
    out = np.zeros(L)
    edges = np.geomspace(400, 9000, 7)
    prog = np.arange(L) / L
    for i in range(6):
        band = bp(nz, edges[i], edges[i + 1])
        centre = (i + 0.5) / 6
        out += band * np.exp(-((prog - centre) ** 2) / 0.03)
    return out * np.sin(np.pi * prog) ** 1.5


def pin_crash():
    L = int(SR * 1.3)
    out = np.zeros(L)
    for k in range(46):
        st = int(rng.random() ** 1.5 * 0.8 * SR)
        f = rng.uniform(500, 2400)
        d = int(0.05 * SR) + int(rng.random() * 0.08 * SR)
        t = np.arange(d) / SR
        clack = np.sin(2 * np.pi * f * t) * np.exp(-t * 55) + hp(rng.standard_normal(d), 2500) * np.exp(-t * 90) * 0.6
        out[st : st + d] += clack * rng.uniform(0.25, 1.0)
    return np.tanh(out * 0.9)


def bell(f, dur=0.6):
    t = np.arange(int(SR * dur)) / SR
    s = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 2.01 * t) * np.exp(-t * 9) + 0.3 * np.sin(2 * np.pi * f * 3.0 * t) * np.exp(-t * 14)
    return s * np.exp(-t * 6) * np.minimum(t / 0.002, 1)


# ---------------------------------------------------------------- tonal
def bass_note(f, dur):
    t = np.arange(int(SR * dur)) / SR
    s = saw(f, t) * 0.6 + square(f, t) * 0.4 + np.sin(2 * np.pi * f * t) * 0.8
    s = lp(s, 700)
    env = np.minimum(t / 0.004, 1) * np.exp(-t * 2.5)
    return np.tanh(s * env * 1.5)


def pluck(f, dur=0.25):
    t = np.arange(int(SR * dur)) / SR
    s = saw(f, t) * 0.5 + square(f * 2, t, 0.3) * 0.25 + saw(f, t, 0.006) * 0.4
    cutoff = 1200 + 5000 * np.exp(-t * 22)
    # time-varying cutoff via 3 fixed filters crossfaded
    a = lp(s, 1300)
    b = lp(s, 3000)
    c = lp(s, 7000)
    m = np.exp(-t * 22)
    out = a * (1 - m) + (b * 0.5 + c * 0.5) * m
    return out * np.exp(-t * 11) * np.minimum(t / 0.002, 1)


def supersaw(f, dur, gain=1.0):
    t = np.arange(int(SR * dur)) / SR
    s = np.zeros_like(t)
    for d in (-0.012, -0.006, 0, 0.007, 0.013):
        s += saw(f, t, d)
    s = s / 5
    env = np.minimum(t / 0.01, 1) * np.minimum((dur - t) / 0.03, 1).clip(0)
    return s * env * gain


def lead_note(f, dur, vib=True):
    t = np.arange(int(SR * dur)) / SR
    v = 1 + (0.004 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(t / 0.15, 1) if vib else 0)
    ph = 2 * np.pi * np.cumsum(f * v) / SR
    s = (
        2 * ((ph / (2 * np.pi)) % 1) - 1
    ) * 0.55 + np.where(np.sin(ph * 1.003) > 0, 1, -1) * 0.25 + saw(f * 2, t, 0.004) * 0.2
    s = lp(s, 5200)
    env = np.minimum(t / 0.006, 1) * np.exp(-t * 1.2) * np.minimum((dur - t) / 0.04, 1).clip(0)
    return s * env


def pad_chord(freqs, dur):
    t = np.arange(int(SR * dur)) / SR
    s = np.zeros_like(t)
    for f in freqs:
        for d in (-0.004, 0.0, 0.005):
            s += np.sin(2 * np.pi * f * (1 + d) * t) + 0.3 * saw(f, t, d)
    s = lp(s / (len(freqs) * 3), 2400)
    env = np.minimum(t / 0.2, 1) * np.minimum((dur - t) / 0.25, 1).clip(0)
    return s * env


# ---------------------------------------------------------------- arrangement
CHORDS = [  # I V vi IV in C major
    {"root": 36, "tri": ["C4", "E4", "G4"], "arp": ["C4", "E4", "G4", "C5"]},
    {"root": 31, "tri": ["G3", "B3", "D4"], "arp": ["G3", "B3", "D4", "G4"]},
    {"root": 33, "tri": ["A3", "C4", "E4"], "arp": ["A3", "C4", "E4", "A4"]},
    {"root": 29, "tri": ["F3", "A3", "C4"], "arp": ["F3", "A3", "C4", "F4"]},
]
# anthem hook, 8 eighth-note slots per bar (None = rest)
HOOK = [
    ["E5", "E5", "G5", None, "G5", "A5", "G5", "E5"],
    ["D5", "D5", "G5", None, "G5", "B5", "A5", "G5"],
    ["C5", "C5", "E5", None, "E5", "A5", "G5", "E5"],
    ["F5", "F5", "A5", None, "A5", "G5", "F5", "E5"],
]
HOOK_HI = [
    ["E6", "E6", "G6", None, "G6", "A6", "G6", "E6"],
    ["D6", "D6", "G6", None, "G6", "B6", "A6", "G6"],
    ["C6", "C6", "E6", None, "E6", "A6", "G6", "E6"],
    ["F6", "F6", "A6", None, "A6", "G6", "F6", "E6"],
]

mix = np.zeros((2, N))
drums = np.zeros((2, N))
tonal = np.zeros((2, N))  # sidechained group
fx = np.zeros((2, N))

K, CL, SN = kick(), clap(), snare()
HH, OH = hat(), hat(True)


def section(beat):
    """Return flags describing what plays at a given absolute beat."""
    return dict(
        intro=beat < 8,
        brk=88 <= beat < 92,
        lead=(20 <= beat < 88) or (92 <= beat < 180),
        lead_hi=(148 <= beat < 180),
        full=(8 <= beat < 88) or (92 <= beat < 180),
    )


# --- sections of the timeline
for beat in range(TOTAL_BEATS):
    s = section(beat)
    t0 = b2s(beat)
    bar = beat // 4
    chord = CHORDS[bar % 4]
    # kick
    if (s["intro"] and beat >= 2) or s["full"]:
        add(drums, t0, K, 1.0)
    # claps / snare on 2 and 4
    if (s["full"] or s["intro"]) and beat % 2 == 1 and beat >= 3:
        add(drums, t0, CL, 0.55, 0.0)
        add(drums, t0, SN, 0.35, 0.0)
    # off-beat open hat + 16th closed hats
    if s["full"] or (s["intro"] and beat >= 4):
        add(drums, t0 + b2s(0.5), OH, 0.16, 0.2)
        for sub in range(4):
            if sub % 2 == 1 or s["full"]:
                add(drums, t0 + b2s(sub * 0.25), HH, 0.07 if sub % 2 == 0 else 0.11, -0.2)
    # bass: offbeat 8ths
    if s["full"]:
        for sub in (0.5,):
            f = note(chord["root"])
            add(tonal, t0 + b2s(sub), bass_note(f, BEAT * 0.48), 0.9)
        # root on the beat too, softer (rolling feel)
        add(tonal, t0, bass_note(note(chord["root"]) , BEAT * 0.45), 0.55)
    # arp plucks (16ths)
    if s["full"] or s["intro"]:
        for sub in range(4):
            idx = (beat * 4 + sub) % 4
            fn = note(n(chord["arp"][idx]) + 12)
            g = 0.30 if s["full"] else 0.34
            add(tonal, t0 + b2s(sub * 0.25), pluck(fn), g, pan=(-0.4 if sub % 2 == 0 else 0.4))

# chords (one per bar) - supersaw stabs on every beat (pumping) + pad
for bar in range(TOTAL_BEATS // 4):
    beat = bar * 4
    s = section(beat)
    ch = CHORDS[bar % 4]
    freqs = [note(n(x)) for x in ch["tri"]]
    t0 = b2s(beat)
    if s["intro"] or s["brk"]:
        add(tonal, t0, pad_chord(freqs, BEAT * 4), 0.5)
    if s["full"]:
        for i in range(4):
            stab = sum(supersaw(f, BEAT * 0.9, 1.0) for f in freqs) / 3
            add(tonal, b2s(beat + i), lp(stab, 4500), 0.33 if bar < 14 else 0.38)
        add(tonal, t0, pad_chord(freqs, BEAT * 4), 0.22)

# lead hook
for bar in range(TOTAL_BEATS // 4):
    beat = bar * 4
    s = section(beat)
    if not s["lead"]:
        continue
    pattern = (HOOK_HI if s["lead_hi"] else HOOK)[bar % 4]
    for i, nm in enumerate(pattern):
        if nm is None:
            continue
        dur = BEAT * 0.5 * (1.8 if pattern[(i + 1) % 8] is None else 0.95)
        sig = lead_note(note(n(nm)), dur)
        add(tonal, b2s(beat + i * 0.5), sig, 0.36, pan=0.1)
        if s["lead_hi"]:
            add(tonal, b2s(beat + i * 0.5), lead_note(note(n(nm)) / 2, dur), 0.18, pan=-0.1)

# --- FX: intro hit, risers, snare rolls, impacts
def snare_roll(start_beat, beats, gmax=0.8):
    cnt = int(beats * 8)
    for i in range(cnt):
        p = i / cnt
        add(drums, b2s(start_beat + i * beats / cnt), SN, 0.12 + gmax * p * p)


# intro: crash + sub drop at 0, riser to beat 8 with snare roll
add(fx, 0, crash(2.8), 0.45)
add(fx, 0, subdrop(), 0.8)
add(fx, 0, K, 1.0)
add(fx, 0, CL, 0.7)
add(fx, b2s(4), riser(BEAT * 4), 0.55)
snare_roll(6, 2)
# drop at 8 (title slam)
add(fx, b2s(8), crash(2.6), 0.5)
add(fx, b2s(8), subdrop(), 0.85)
# strike impact at 12 (bowling strike moment)
add(fx, b2s(12), crash(2.4), 0.6)
add(fx, b2s(12), subdrop(1.0), 0.9)
add(fx, b2s(11), riser(BEAT * 1.0, 800, 12000), 0.4)
# montage start 20
add(fx, b2s(20), crash(2.0), 0.4)
add(fx, b2s(18), riser(BEAT * 2), 0.4)
# short breakdown before the results section
add(fx, b2s(88), crash(1.5), 0.35)
add(fx, b2s(88), riser(BEAT * 4, 200, 11000), 0.8)
snare_roll(88.0, 4, 0.9)
# results start (92): team standings
add(fx, b2s(92), crash(3.0), 0.6)
add(fx, b2s(92), subdrop(), 1.0)
# champions reveal 106, men's section 120, women's section 134
for hb in (106, 120, 134):
    add(fx, b2s(hb - 1), riser(BEAT * 1.0, 800, 12000), 0.35)
    add(fx, b2s(hb), crash(2.2), 0.45)
    add(fx, b2s(hb), subdrop(0.9), 0.8)
# group-photo finale 148
add(fx, b2s(148), crash(2.4), 0.5)
add(fx, b2s(148), subdrop(), 0.8)
# end card 168
add(fx, b2s(166), riser(BEAT * 2), 0.5)
add(fx, b2s(168), crash(3.0), 0.55)
add(fx, b2s(168), subdrop(), 0.9)
# final build + hit at 180
add(fx, b2s(176), riser(BEAT * 4, 300, 12000), 0.7)
snare_roll(178, 2, 0.9)
add(fx, b2s(180), crash(3.8), 0.75)
add(fx, b2s(180), subdrop(1.6), 1.1)
add(fx, b2s(180), K, 1.0)
add(fx, b2s(180), CL, 0.8)
final = sum(supersaw(note(n(x)), 2.0, 1.0) for x in ["C3", "G3", "C4", "E4", "G4", "C5"]) / 6
add(fx, b2s(180), lp(final, 5000), 0.55)

# whooshes on every scene wipe, pin crash on the strike, bells on each #1 reveal
for wc in (7.55, 19.55, 91.55, 119.55, 133.55, 147.55, 167.55):
    add(fx, b2s(wc - 0.5), whoosh(BEAT * 1.0), 0.55)
add(fx, b2s(12.0), pin_crash(), 0.9)
for hb in (106.0, 127.0, 141.0):
    for nm in ("C6", "E6", "G6"):
        add(fx, b2s(hb), bell(note(n(nm)), 1.4), 0.30)

# --- cut everything after the final hit (except the hit's own tail) - handled by fades below
# ---------------------------------------------------------------- sidechain + mix
sc = np.ones(N)
t = np.arange(N) / SR
phase = (t / BEAT) % 1.0
pump = 1 - 0.8 * np.exp(-phase * 9)
for beat in range(TOTAL_BEATS):
    sidechain_on = section(beat)["full"] or (section(beat)["intro"] and beat >= 2)
    a, bnd = b2s(beat), b2s(beat + 1)
    if sidechain_on:
        sc[a:bnd] = pump[a:bnd]
tonal *= sc

# silence the pump/tonal in the breakdown's low end: highpass bass out
mix = drums * 0.95 + tonal * 0.95 + fx * 0.9

# gentle stereo width: tiny haas delay on the tonal bus
# master: soft clip, normalise
mix = np.tanh(mix * 0.95)
# end fade
total_len = b2s(TOTAL_BEATS)
fade_start = b2s(182)
fade = np.ones(N)
fade[fade_start : total_len] = np.linspace(1, 0, total_len - fade_start) ** 1.5
fade[total_len:] = 0
mix *= fade
# slight fade in
mix[:, :200] *= np.linspace(0, 1, 200)
peak = np.abs(mix).max()
mix = mix / peak * 0.89
out = (mix[:, :total_len + SR // 2].T * 32767).astype(np.int16)

import wave

with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(out.tobytes())
print("wrote", sys.argv[1], len(out) / SR, "s")
