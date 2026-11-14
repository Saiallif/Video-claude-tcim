#!/usr/bin/env python3
"""TCIM Bowling Night - energetic teamwork video (1920x1080 @ 30fps).

All animation is locked to the beat of the soundtrack (128 BPM).

  make_video.py preview OUT.png beat1,beat2,...     contact sheet of frames
  make_video.py render  OUT.mp4 AUDIO.wav           final render (4 procs)
"""
import math
import os
import random
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_data import AWARDS, TEAMS  # noqa: E402

W, H, FPS = 1920, 1080, 30
BPM = 128.0
BEAT = 60.0 / BPM
TOTAL_BEATS = 184
NFRAMES = int(round(TOTAL_BEATS * BEAT * FPS))

SRC = os.environ.get("TCIM_SRC", "src")

NAVY = (14, 32, 120)
DEEP = (7, 14, 58)
RED = (226, 30, 42)
WHITE = (255, 255, 255)
GOLD = (255, 205, 60)
SKY = (60, 120, 240)

FD = "/usr/share/fonts/opentype/inter/"
F_BLACK_IT = FD + "Inter-BlackItalic.otf"
F_XB_IT = FD + "Inter-ExtraBoldItalic.otf"
F_B_IT = FD + "Inter-BoldItalic.otf"
F_BLACK = FD + "Inter-Black.otf"
F_BOLD = FD + "Inter-Bold.otf"
F_SEMI = FD + "Inter-SemiBold.otf"


# ------------------------------------------------------------------ easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lerp(a, b, u):
    return a + (b - a) * u


def eo3(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def eo5(x):
    x = clamp(x)
    return 1 - (1 - x) ** 5


def eio(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def back(x, s=1.9):
    x = clamp(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def seg(bt, a, b):
    """0..1 progress of beat-time bt through [a, b]."""
    return clamp((bt - a) / (b - a))


# ------------------------------------------------------------------ text
@lru_cache(maxsize=1024)
def text_layer(txt, font, size, fill, stroke=0, stroke_fill=(0, 0, 0), shadow=0,
               shadow_fill=(0, 0, 0), tracking=0):
    f = ImageFont.truetype(font, size)
    if tracking:
        adv = [f.getlength(c) + tracking for c in txt]
        tw = int(sum(adv))
    else:
        tw = int(f.getlength(txt))
    asc, desc = f.getmetrics()
    pad = stroke + shadow + 10
    w, h = tw + 2 * pad, asc + desc + 2 * pad
    img = Image.new("RGBA", (w, h), fill + (0,))
    d = ImageDraw.Draw(img)
    if tracking:
        x = pad
        for c, a in zip(txt, adv):
            if shadow:
                d.text((x + shadow, pad + shadow), c, font=f, fill=shadow_fill + (255,),
                       stroke_width=stroke, stroke_fill=shadow_fill + (255,))
            d.text((x, pad), c, font=f, fill=fill + (255,), stroke_width=stroke,
                   stroke_fill=stroke_fill + (255,))
            x += a
    else:
        if shadow:
            d.text((pad + shadow, pad + shadow), txt, font=f, fill=shadow_fill + (255,),
                   stroke_width=stroke, stroke_fill=shadow_fill + (255,))
        d.text((pad, pad), txt, font=f, fill=fill + (255,), stroke_width=stroke,
               stroke_fill=stroke_fill + (255,))
    return img


def put(canvas, layer, cx, cy, scale=1.0, rot=0.0, alpha=1.0, anchor="c"):
    """Composite RGBA `layer` onto RGB canvas centred at (cx, cy)."""
    if alpha <= 0.003 or scale <= 0.01:
        return
    if scale != 1.0:
        layer = layer.resize((max(1, int(layer.width * scale)), max(1, int(layer.height * scale))),
                             Image.BICUBIC)
    if rot:
        layer = layer.rotate(rot, resample=Image.BICUBIC, expand=True)
    if alpha < 0.999:
        a = layer.getchannel("A").point(lambda v: int(v * alpha))
        layer = layer.copy()
        layer.putalpha(a)
    if anchor == "c":
        x, y = int(cx - layer.width / 2), int(cy - layer.height / 2)
    elif anchor == "l":
        x, y = int(cx), int(cy - layer.height / 2)
    else:  # right
        x, y = int(cx - layer.width), int(cy - layer.height / 2)
    canvas.paste(layer, (x, y), layer)


@lru_cache(maxsize=256)
def chip_layer(w, h, color, slant=0.28, edge=None):
    S = 3
    dx = int(h * slant * S)
    img = Image.new("RGBA", (w * S + dx, h * S), color + (0,))
    d = ImageDraw.Draw(img)
    if edge:
        e = 7 * S
        d.polygon([(dx, 0), (w * S + dx, 0), (w * S, h * S), (0, h * S)], fill=edge + (255,))
        d.polygon([(dx + e, e), (w * S + dx - e, e), (w * S - e, h * S - e), (e, h * S - e)],
                  fill=color + (255,))
    else:
        d.polygon([(dx, 0), (w * S + dx, 0), (w * S, h * S), (0, h * S)], fill=color + (255,))
    return img.resize((w + int(h * slant), h), Image.LANCZOS)


def chip(canvas, txt, cx, cy, size, color=RED, tcolor=WHITE, font=F_BLACK_IT, padx=46,
         scale=1.0, alpha=1.0, rot=0.0, edge=None, tracking=0, anchor="c", shadow=0):
    tl = text_layer(txt, font, size, tcolor, tracking=tracking, shadow=shadow,
                    shadow_fill=(0, 0, 0))
    w = tl.width + padx * 2
    h = int(size * 1.45)
    cl = chip_layer(w, h, color, 0.28, edge)
    comp = Image.new("RGBA", cl.size, (0, 0, 0, 0))
    comp.paste(cl, (0, 0))
    comp.alpha_composite(tl, (int((cl.width - tl.width) / 2), int((cl.height - tl.height) / 2)))
    put(canvas, comp, cx, cy, scale, rot, alpha, anchor)
    return comp.width


def slam(canvas, layer, cx, cy, bt, start, dur=0.55, s0=2.4, rot0=0.0, rot=0.0, hold_alpha=1.0,
         exit_at=None, exit_dur=0.5):
    """Beat-synced 'slam' entrance (and optional exit)."""
    u = (bt - start) / dur
    if u < 0:
        return
    sc = lerp(s0, 1.0, back(u, 1.4))
    al = clamp(u * 5) * hold_alpha
    rt = rot + lerp(rot0, 0, eo3(u))
    if exit_at is not None and bt > exit_at:
        v = clamp((bt - exit_at) / exit_dur)
        al *= 1 - v
        sc *= 1 + 0.25 * v
    put(canvas, layer, cx, cy, sc, rt, al)


# ------------------------------------------------------------------ sprites
@lru_cache(maxsize=8)
def pin_sprite(h=150):
    S = 4
    hh = h * S
    prof = [(0, 0.0), (0.02, 0.07), (0.06, 0.10), (0.12, 0.112), (0.18, 0.095), (0.24, 0.068),
            (0.30, 0.060), (0.40, 0.085), (0.52, 0.135), (0.64, 0.172), (0.78, 0.178),
            (0.90, 0.150), (0.97, 0.118), (1.0, 0.105)]
    ys = np.array([p[0] for p in prof])
    ws = np.array([p[1] for p in prof])
    f = PchipInterpolator(ys, ws)
    wd = int(0.40 * hh)
    yy, xx = np.mgrid[0:hh, 0:wd]
    cx = wd / 2
    hw = f(np.clip(yy / hh, 0, 1)) * hh
    nx = (xx - cx) / np.maximum(hw, 1)
    mask = (np.abs(nx) <= 1.0).astype(np.float32)
    nz = np.sqrt(np.clip(1 - nx ** 2, 0, 1))
    light = 0.50 + 0.52 * (0.65 * nz + 0.45 * (-nx))
    light = np.clip(light, 0.25, 1.08)
    col = np.stack([light * 252, light * 252, light * 255], -1)
    yn = yy / hh
    stripe = ((yn > 0.205) & (yn < 0.238)) | ((yn > 0.262) & (yn < 0.295))
    red = np.stack([light * 222, light * 30, light * 38], -1)
    col = np.where(stripe[..., None], red, col)
    rgba = np.dstack([np.clip(col, 0, 255), mask * 255]).astype(np.uint8)
    img = Image.fromarray(rgba, "RGBA")
    return img.resize((wd // S, hh // S), Image.LANCZOS)


@lru_cache(maxsize=8)
def ball_sprite(d=150):
    S = 3
    D = d * S
    yy, xx = np.mgrid[0:D, 0:D]
    x = (xx - D / 2) / (D / 2)
    y = (yy - D / 2) / (D / 2)
    r2 = x ** 2 + y ** 2
    mask = (r2 <= 1.0)
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    L = np.array([-0.45, -0.55, 0.70])
    L = L / np.linalg.norm(L)
    diff = np.clip(x * L[0] + y * L[1] + z * L[2], 0, 1)
    spec = np.clip(diff, 0, 1) ** 40
    base = np.array([214, 24, 38], dtype=np.float32)
    col = base[None, None, :] * (0.30 + 0.85 * diff[..., None]) + 255 * spec[..., None] * 0.9
    rim = (1 - z) ** 2.5
    col = col * (1 - 0.45 * rim[..., None])
    img = Image.fromarray(np.dstack([np.clip(col, 0, 255), mask * 255]).astype(np.uint8), "RGBA")
    dr = ImageDraw.Draw(img)
    for (hx, hy, hr) in [(0.30, -0.42, 0.085), (0.48, -0.22, 0.085), (0.12, -0.20, 0.105)]:
        px, py, pr = D / 2 + hx * D / 2, D / 2 + hy * D / 2, hr * D / 2
        dr.ellipse([px - pr, py - pr * 0.85, px + pr, py + pr * 0.85], fill=(30, 4, 8, 255))
    return img.resize((d, d), Image.LANCZOS)


def shadow_ellipse(canvas, cx, cy, rx, ry, alpha=0.35):
    lay = Image.new("L", (int(rx * 2 + 20), int(ry * 2 + 20)), 0)
    ImageDraw.Draw(lay).ellipse([10, 10, 10 + rx * 2, 10 + ry * 2], fill=int(255 * alpha))
    lay = lay.filter(ImageFilter.GaussianBlur(6))
    canvas.paste((0, 0, 0), (int(cx - lay.width / 2), int(cy - lay.height / 2)), lay)


def star_burst(canvas, cx, cy, r, spikes=14, rot=0, outer=GOLD, inner=WHITE):
    if r < 3:
        return
    d = ImageDraw.Draw(canvas)
    for scale, col in ((1.0, outer), (0.62, inner)):
        pts = []
        for i in range(spikes * 2):
            a = rot + i * math.pi / spikes
            rr = r * scale * (1.0 if i % 2 == 0 else 0.55)
            pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
        d.polygon(pts, fill=col)


def tcim_wordmark(size, fill=WHITE, dot=RED):
    f = ImageFont.truetype(F_BLACK_IT, size)
    letters = "TCIM"
    adv = [f.getlength(c) - size * 0.01 for c in letters]
    pad = int(size * 0.2)
    w = int(sum(adv)) + 2 * pad
    asc, desc = f.getmetrics()
    h = asc + desc + 2 * pad
    img = Image.new("RGBA", (w, h), fill + (0,))
    d = ImageDraw.Draw(img)
    x = pad
    ibox = None
    for c, a in zip(letters, adv):
        d.text((x, pad), c, font=f, fill=fill + (255,))
        if c == "I":
            bb = f.getbbox("I")
            ibox = (x + bb[0], pad + bb[1], x + bb[2], pad + bb[3])
        x += a
    if ibox:
        dw = (ibox[2] - ibox[0]) * 1.15
        dh = size * 0.12
        cx = (ibox[0] + ibox[2]) / 2
        top = ibox[1] - dh - size * 0.04
        d.polygon([(cx - dw / 2 + size * 0.05, top), (cx + dw / 2 + size * 0.05, top),
                   (cx + dw / 2, top + dh), (cx - dw / 2, top + dh)], fill=dot + (255,))
    return img


# ------------------------------------------------------------------ photos
_PH = {}


def photo(name):
    if name in _PH:
        return _PH[name]
    p = lambda f: os.path.join(SRC, f)
    if name == "building":
        im = Image.open(p("building.png")).convert("RGB")
        im = im.filter(ImageFilter.UnsharpMask(1.6, 90, 2))
        im = ImageEnhance.Color(im).enhance(1.15)
    elif name == "wait":
        im = Image.open(p("wait.jpg")).convert("RGB").resize((2400, 1800), Image.LANCZOS)
        im = ImageEnhance.Color(im).enhance(1.25)
        im = ImageEnhance.Contrast(im).enhance(1.08)
    elif name == "score":
        im = Image.open(p("score.jpg")).convert("RGB").crop((600, 950, 2750, 2560))
        im = im.resize((1800, 1348), Image.LANCZOS)
        im = ImageEnhance.Color(im).enhance(1.2)
    elif name == "food":
        im = Image.open(p("food.jpg")).convert("RGB")
        im = ImageEnhance.Color(im).enhance(1.2)
    else:  # group shots
        im = Image.open(p(name + ".jpg")).convert("RGB")
        im = ImageEnhance.Brightness(im).enhance(1.20)
        im = ImageEnhance.Contrast(im).enhance(1.06)
        im = ImageEnhance.Color(im).enhance(1.22)
    _PH[name] = im
    return im


def view(img, b0, b1, u, pulse=0.0, shift=(0, 0)):
    """Ken-Burns view. Boxes are (fx, fy, fw) fractions of the image; 16:9 window."""
    fx = lerp(b0[0], b1[0], u)
    fy = lerp(b0[1], b1[1], u)
    fw = lerp(b0[2], b1[2], u)
    iw, ih = img.size
    bw = fw * iw
    bh = bw * 9 / 16
    cx = fx * iw + bw / 2
    cy = fy * ih + bh / 2
    z = 1 - 0.045 * pulse
    bw *= z
    bh *= z
    x0 = clamp(cx - bw / 2 + shift[0], 0, iw - bw)
    y0 = clamp(cy - bh / 2 + shift[1], 0, ih - bh)
    return img.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + bw, y0 + bh))


# ------------------------------------------------------------------ backgrounds / overlays
@lru_cache(maxsize=1)
def radial_bg():
    yy, xx = np.mgrid[0:H // 4, 0:W // 4]
    x = (xx - W / 8) / (W / 8)
    y = (yy - H / 8) / (H / 8)
    r = np.sqrt(x ** 2 * 0.6 + y ** 2)
    t = np.clip(r / 1.3, 0, 1)[..., None]
    c0 = np.array([28, 60, 190], dtype=np.float32)
    c1 = np.array([6, 12, 52], dtype=np.float32)
    img = c0 * (1 - t) + c1 * t
    return Image.fromarray(img.astype(np.uint8), "RGB").resize((W, H), Image.BICUBIC)


@lru_cache(maxsize=1)
def _polar():
    yy, xx = np.mgrid[0:H // 4, 0:W // 4]
    x = (xx - W / 8) / (W / 8) * 1.78
    y = (yy - H / 8) / (H / 8)
    return np.arctan2(y, x), np.sqrt(x ** 2 + y ** 2)


def sunburst(canvas, t, color=(60, 110, 255), strength=0.22, n=18, cx_off=0.0):
    th, r = _polar()
    ph = (th / (2 * math.pi) * n + t * 0.35) % 1.0
    m = (ph < 0.5).astype(np.float32) * np.clip(r / 0.35, 0, 1) * np.clip(1.6 - r * 0.55, 0, 1)
    mask = Image.fromarray((m * 255 * strength).astype(np.uint8), "L").resize((W, H), Image.BILINEAR)
    canvas.paste(color, (0, 0), mask)


@lru_cache(maxsize=1)
def vignette():
    yy, xx = np.mgrid[0:H // 4, 0:W // 4]
    x = (xx - W / 8) / (W / 8)
    y = (yy - H / 8) / (H / 8)
    r = np.sqrt(x ** 2 * 0.8 + y ** 2)
    a = np.clip((r - 0.65) / 0.95, 0, 1) ** 1.6 * 165
    return Image.fromarray(a.astype(np.uint8), "L").resize((W, H), Image.BICUBIC)


@lru_cache(maxsize=1)
def bottom_grad():
    a = np.linspace(0, 1, H)[:, None] ** 2.2 * 210
    a = np.repeat(a, W, axis=1)
    return Image.fromarray(a.astype(np.uint8), "L")


@lru_cache(maxsize=1)
def top_grad():
    return bottom_grad().transpose(Image.FLIP_TOP_BOTTOM)


@lru_cache(maxsize=1)
def left_grad():
    a = np.linspace(1, 0, W)[None, :] ** 1.6 * 190
    a = np.repeat(a, H, axis=0)
    return Image.fromarray(a.astype(np.uint8), "L")


@lru_cache(maxsize=1)
def lane_img():
    h = 300
    yy = np.linspace(0, 1, h)[:, None]
    top = np.array([236, 196, 140], dtype=np.float32)
    bot = np.array([150, 100, 56], dtype=np.float32)
    col = top * (1 - yy[..., None]) + bot * yy[..., None]
    img = np.repeat(col, W, axis=1).astype(np.uint8)
    im = Image.fromarray(img, "RGB")
    d = ImageDraw.Draw(im)
    for i in range(0, h, 22):
        d.line([(0, i), (W, i)], fill=(120, 76, 40), width=1)
    d.rectangle([0, 0, W, 8], fill=(40, 24, 18))
    d.rectangle([0, h - 34, W, h], fill=(36, 22, 18))
    gl = Image.new("L", (W, h), 0)
    gd = ImageDraw.Draw(gl)
    for i in range(14):
        x = i * 170 - 60
        gd.polygon([(x, 0), (x + 90, 0), (x + 20, h), (x - 70, h)], fill=26)
    gl = gl.filter(ImageFilter.GaussianBlur(14))
    im.paste((255, 245, 225), (0, 0), gl)
    return im


def flash(canvas, a, color=WHITE):
    if a > 0.01:
        canvas.paste(color, (0, 0), Image.new("L", (W, H), int(255 * clamp(a))))


def confetti(canvas, tsec, origin, seed, n=110, spread=1.0, speed=(500, 1500), gravity=1500,
             up=0.7, life=3.0, kind="burst"):
    if tsec < 0 or tsec > life:
        return
    rnd = random.Random(seed)
    d = ImageDraw.Draw(canvas)
    cols = [RED, WHITE, GOLD, SKY, (255, 120, 140), (120, 180, 255)]
    for i in range(n):
        ang = rnd.uniform(-math.pi, 0) * spread + (math.pi / 2 if kind == "fall" else 0)
        if kind == "burst":
            ang = rnd.uniform(0, 2 * math.pi)
            sp = rnd.uniform(*speed)
            vx, vy = math.cos(ang) * sp, math.sin(ang) * sp - speed[1] * up
            x0, y0 = origin
        else:  # rain from the top
            vx, vy = rnd.uniform(-120, 120), rnd.uniform(60, 300)
            x0, y0 = rnd.uniform(0, W), -rnd.uniform(0, 400)
        drag = math.exp(-1.4 * tsec) if kind == "burst" else 1.0
        x = x0 + vx * (1 - drag) / 1.4 if kind == "burst" else x0 + vx * tsec
        y = y0 + vy * (1 - drag) / 1.4 + 0.5 * gravity * tsec ** 2 * (0.6 if kind == "burst" else 0.25) \
            if kind == "burst" else y0 + vy * tsec + 0.5 * 120 * tsec ** 2
        if not (-30 < x < W + 30 and -30 < y < H + 30):
            continue
        w, h = rnd.uniform(10, 20), rnd.uniform(6, 11)
        rot = rnd.uniform(0, 6.28) + tsec * rnd.uniform(-9, 9)
        flip = abs(math.cos(tsec * rnd.uniform(3, 8) + i))
        c, s = math.cos(rot), math.sin(rot)
        pts = [(-w / 2, -h / 2 * flip), (w / 2, -h / 2 * flip), (w / 2, h / 2 * flip), (-w / 2, h / 2 * flip)]
        d.polygon([(x + px * c - py * s, y + px * s + py * c) for px, py in pts], fill=rnd.choice(cols))


def equalizer(canvas, tsec, bt, y0=H, maxh=80, color=RED, n=64):
    d = ImageDraw.Draw(canvas)
    bw = W / n
    pulse = (1 - (bt % 1.0)) ** 2
    for i in range(n):
        v = 0.5 + 0.5 * math.sin(tsec * 6 + i * 0.7) * math.sin(tsec * 3.3 + i * 0.31)
        h = maxh * (0.2 + 0.8 * v * (0.55 + 0.45 * pulse))
        c = color if i % 3 else WHITE
        d.rectangle([i * bw + 3, y0 - h, (i + 1) * bw - 3, y0], fill=c)


def wipe(canvas, bt, center):
    half = 0.5
    u = (bt - (center - half)) / (2 * half)
    if not 0 < u < 1:
        return
    dx = 380
    w = 4200
    p = lerp(-dx, W + w, u)  # right edge of main band

    def band(pp, ww, col):
        l, r = pp - ww, pp
        ImageDraw.Draw(canvas).polygon([(l + dx, 0), (r + dx, 0), (r, H), (l, H)], fill=col)

    band(p + 330, w, RED)
    band(p + 250, w, WHITE)
    band(p, w, NAVY)
    band(p - 70, w, DEEP)


def draw_pin(canvas, x, base_y, rot=0.0, alpha=1.0, h=150, shadow=True):
    sp = pin_sprite(h)
    if shadow:
        shadow_ellipse(canvas, x, base_y, sp.width * 0.5, 9, 0.35 * alpha)
    cy = base_y - sp.height / 2
    if rot == 0:
        put(canvas, sp, x, cy, alpha=alpha)
    else:
        put(canvas, sp, x, cy, rot=rot, alpha=alpha)


def shake(bt, start, amp=14, dur=0.7):
    u = (bt - start) / dur
    if not 0 <= u <= 1:
        return 0, 0
    a = amp * (1 - u) ** 2
    t = bt * 60
    return math.sin(t * 1.7) * a, math.cos(t * 2.3) * a


# ------------------------------------------------------------------ montage table
# (photo, box0, box1, caption, beats)  boxes = (fx, fy, fw) as fractions of the image
def D2K(x, y, w):  # displayed 2000x1125 coords -> fractions
    return (x / 2000, y / 1125, w / 2000)


def D4(x, y, w):  # displayed 2000x1500 coords (4:3 photo)
    return (x / 2000, y / 1500, w / 2000)


MONT = [
    # (photo | "clip:<file>", box0 | clip start sec, box1, caption, beats)   - 4 beats per shot (~1.9 s)
    ("wait", D4(0, 375, 2000), D4(280, 470, 1500), "LET'S BOWL!", 4),
    ("score", (0.04, 0.30, 0.86), (0.20, 0.34, 0.56), "GAME ON!", 4),
    ("g0224", D2K(0, 470, 820), D2K(60, 500, 700), "TEAMWORK", 4),
    ("food", (0.0, 0.0, 1.0), (0.12, 0.10, 0.74), "GOOD FOOD", 4),
    ("g0214", D2K(1180, 600, 820), D2K(1220, 640, 740), "GREAT PEOPLE", 4),
    ("g0252", D2K(560, 660, 760), D2K(500, 600, 880), "ONE TEAM", 4),
    ("wait", D4(900, 700, 1100), D4(1000, 820, 900), "FEEL THE ENERGY", 4),
    ("g0217", D2K(1000, 520, 700), D2K(1060, 560, 620), "FUN TOGETHER", 4),
    ("g0222", D2K(0, 470, 820), D2K(80, 520, 700), "STRONGER TEAM", 4),
    ("g0226", D2K(620, 480, 760), D2K(660, 520, 660), "PLAY · LAUGH · BOND", 4),
    ("score", (0.20, 0.55, 0.62), (0.28, 0.60, 0.42), "SCORE BIG!", 4),
    ("g0228", D2K(1180, 600, 820), D2K(1240, 640, 700), "SPIRIT", 4),
    ("g0214", D2K(1000, 520, 700), D2K(1060, 560, 640), "CONNECT", 4),
    ("g0224", D2K(620, 480, 760), D2K(660, 520, 680), "UNWIND", 4),
    ("g0222", D2K(1180, 600, 820), D2K(1230, 640, 720), "ENERGY", 4),
    ("wait", D4(0, 375, 1400), D4(120, 430, 1250), "FUN", 4),
    ("g0252", D2K(0, 470, 820), D2K(60, 510, 700), "TOGETHER", 4),
    ("g0230", D2K(0, 250, 2000), D2K(100, 300, 1800), "ALL IN!", 4),
]
MONT_START = 20.0
WIPES = [7.55, 19.55, 91.55, 119.55, 133.55, 147.55, 167.55]


def prep_clips():
    """Extract frames for every clip shot in MONT (ffmpeg, 30 fps)."""
    for m in MONT:
        if not m[0].startswith("clip:"):
            continue
        f, ss, beats = m[0][5:], m[1], m[4]
        d = os.path.join(SRC, "clipframes", "%s_%s" % (f, ss))
        if os.path.isdir(d) and os.listdir(d):
            continue
        os.makedirs(d, exist_ok=True)
        subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-ss", str(ss), "-t", "%.2f" % (beats * BEAT + 0.2),
                               "-i", os.path.join(SRC, "clips", f), "-vf",
                               "fps=30,scale='if(gt(iw,ih),1920,-2)':'if(gt(iw,ih),-2,1080)'", "-q:v", "3",
                               os.path.join(d, "%04d.jpg")])


def clip_view(f, ss, tin, pulse=0.0):
    d = os.path.join(SRC, "clipframes", "%s_%s" % (f, ss))
    n = len(os.listdir(d))
    idx = max(1, min(int(tin * FPS) + 1, n))
    im = Image.open(os.path.join(d, "%04d.jpg" % idx)).convert("RGB")
    iw, ih = im.size
    z = 1 - 0.03 * pulse
    if iw / ih >= 1.6:
        bw, bh = iw * z, iw * z * 9 / 16
        x0, y0 = (iw - bw) / 2, (ih - bh) / 2
        return im.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + bw, y0 + bh))
    # portrait / square clip: blurred cover background + centred sharp frame
    bg = im.resize((W, int(W * ih / iw)), Image.BILINEAR)
    top = (bg.height - H) // 2
    bg = bg.crop((0, top, W, top + H)).filter(ImageFilter.GaussianBlur(28))
    bg = ImageEnhance.Brightness(bg).enhance(0.55)
    fh = int(H * z)
    fw = int(iw * fh / ih)
    fg = im.resize((fw, fh), Image.BICUBIC)
    bg.paste(fg, ((W - fw) // 2, (H - fh) // 2))
    return bg


@lru_cache(maxsize=1)
def _logo_img():
    p = os.path.join(SRC, "logo.png")
    return Image.open(p).convert("RGBA") if os.path.exists(p) else None


def logo_layer(size):
    """Real TCIM logo if src/logo.png exists, otherwise the typeset wordmark."""
    lg = _logo_img()
    if lg is None:
        return tcim_wordmark(size)
    bb = lg.getbbox()
    lg = lg.crop(bb) if bb else lg
    h = int(size * 0.95)
    lg = lg.resize((int(lg.width * h / lg.height), h), Image.LANCZOS)
    pad = 30
    out = Image.new("RGBA", (lg.width + 2 * pad, lg.height + 2 * pad), (255, 255, 255, 0))
    out.alpha_composite(lg, (pad, pad))
    return out


def mont_index(bt):
    t = MONT_START
    for i, m in enumerate(MONT):
        if bt < t + m[4]:
            return i, t
        t += m[4]
    return len(MONT) - 1, t - MONT[-1][4]


# ------------------------------------------------------------------ scenes
def scene_intro(bt, tsec):
    cv = Image.new("RGB", (W, H))
    u = seg(bt, 0, 8)
    shx, shy = shake(bt, 0, 12, 0.5)
    pulse = (1 - (bt % 1.0)) ** 3
    img = photo("building")
    v = view(img, (0.30, 0.0, 0.60), (0.31, 0.12, 0.40), eio(u) if False else u, pulse * 0.6,
             (shx * 1.5, shy * 1.5))
    cv.paste(v)
    cv.paste((8, 18, 80), (0, 0), left_grad())
    cv.paste((6, 12, 60), (0, 0), bottom_grad())

    # header chip
    cu = back(seg(bt, 0.4, 1.2))
    chip(cv, "TCIM  ·  YEC 2026", 130 + 290 * cu - 290, 110, 44, RED, WHITE, F_XB_IT, 38,
         alpha=clamp(cu * 3) * (1 - seg(bt, 3.6, 4.0)), anchor="l", tracking=3)
    # staggered lines
    L1 = text_layer("STAFF", F_BLACK_IT, 210, WHITE, stroke=0, shadow=8, shadow_fill=(0, 0, 40))
    L2 = text_layer("ENGAGEMENT", F_BLACK_IT, 150, WHITE, shadow=7, shadow_fill=(0, 0, 40))
    slam(cv, L1, 130 + L1.width / 2 + shx, 480 + shy, bt, 1.0, 0.5, 2.0)
    slam(cv, L2, 130 + L2.width / 2 + shx, 650 + shy, bt, 1.6, 0.5, 2.0)
    if bt > 2.4:
        cu2 = back(seg(bt, 2.4, 3.3))
        chip(cv, "ACTIVITY", 130 + 700 * cu2 - 700, 820 + shy, 120, RED, WHITE, F_BLACK_IT, 60,
             alpha=clamp(cu2 * 3), anchor="l", shadow=4)
    # get ready + countdown
    if 4.0 <= bt < 8:
        gr = back(seg(bt, 4.0, 4.6))
        fade = 1 - seg(bt, 7.0, 7.6)
        chip(cv, "GET READY TO ROLL!", 960, 160, 70, NAVY, WHITE, F_BLACK_IT, 50, scale=gr,
             alpha=clamp(gr * 3) * fade, edge=WHITE)
    for k, num in enumerate(("3", "2", "1")):
        b0 = 5.0 + k
        if b0 <= bt < b0 + 1:
            p = bt - b0
            sc = lerp(1.9, 1.0, back(p / 0.5, 1.2))
            al = 1 - clamp((p - 0.7) / 0.3)
            nl = text_layer(num, F_BLACK_IT, 520, WHITE, stroke=0, shadow=14, shadow_fill=RED)
            put(cv, nl, 1560 + shx, 520 + shy, sc, 0, al)
            d = ImageDraw.Draw(cv)
            r = lerp(380, 280, eo3(p))
            if al > 0.1:
                d.ellipse([1560 - r, 520 - r, 1560 + r, 520 + r], outline=RED, width=12)
    flash(cv, 1 - seg(bt, 0, 0.5), WHITE)
    return cv


def scene_title(bt, tsec):
    cv = radial_bg().copy()
    sunburst(cv, tsec, (70, 120, 255), 0.28, 20)
    shx, shy = shake(bt, 12.0, 18, 0.9)
    shx2, shy2 = shake(bt, 8.1, 8, 0.5)
    shx += shx2
    shy += shy2
    # top chip
    cu = back(seg(bt, 8.1, 8.9))
    chip(cv, "STAFF ENGAGEMENT ACTIVITY", 960, 80, 40, RED, WHITE, F_XB_IT, 40, scale=cu,
         alpha=clamp(cu * 3), tracking=3)
    # lane + pins
    cv.paste(lane_img(), (0, 790))
    # pins
    pins = []
    idx = 0
    for ci in range(1, 5):
        for k in range(ci):
            y = 935 + (k - (ci - 1) / 2) * 82
            pins.append((1380 + (ci - 1) * 78, y, ci, idx))
            idx += 1
    pins.sort(key=lambda p: p[1])
    hit = 12.0
    for x, y, ci, i in pins:
        appear = 8.4 + (idx - i) * 0.05
        drop = clamp((bt - appear) / 0.5)
        if drop <= 0:
            continue
        yy = y - (1 - eo3(drop)) * 700 + (math.sin(drop * 9) * (1 - drop) * 30 if drop < 1 else 0)
        al = 1.0
        rot = 0
        if bt >= hit:
            rnd = random.Random(i * 7 + 3)
            tau = (bt - hit) * BEAT - ci * 0.025
            if tau > 0:
                vx = (450 + rnd.random() * 900) * (0.7 + ci * 0.12)
                vy = -(350 + rnd.random() * 950)
                yy = y + vy * tau + 0.5 * 2300 * tau ** 2 - 20
                x = x + vx * tau
                rot = (1 if rnd.random() > 0.5 else -1) * (300 + rnd.random() * 900) * tau
                al = 1 - clamp((tau - 0.9) / 0.5)
            if x > W + 200 or yy > H + 250 or al <= 0:
                continue
        draw_pin(cv, x, yy, rot, al, 140, shadow=(bt < hit))
    # ball rolling in
    if 10.6 <= bt < 15.5:
        bx = None
        d = 150
        if bt < hit:
            p = (bt - 10.6) / (hit - 10.6)
            bx = lerp(-160, 1320, p ** 1.15)
        else:
            tau = (bt - hit) * BEAT
            bx = 1320 + 1900 * tau
        by = 950 - (d / 2 - 12)
        bs = ball_sprite(d)
        if bx is not None and bx < W + 200:
            # motion trail
            for k in range(6, 0, -1):
                put(cv, bs, bx - k * 55, by, rot=-(bx - k * 55) * 1.2, alpha=0.10 * (7 - k) / 6 * (1 if bt < hit + 0.3 else 0.3))
            shadow_ellipse(cv, bx, 960, 62, 10, 0.4)
            put(cv, bs, bx, by, rot=-bx * 1.2)
    # strike impact: burst, flash, ring, confetti
    if bt >= hit:
        tau = (bt - hit) * BEAT
        r = 300 * eo3(tau / 0.30)
        if tau < 0.5:
            star_burst(cv, 1420, 880, r * (1 - clamp((tau - 0.3) / 0.2)), 16, tau * 2)
            d = ImageDraw.Draw(cv)
            rr = 40 + 900 * eo3(tau / 0.5)
            d.ellipse([1450 - rr, 880 - rr, 1450 + rr, 880 + rr], outline=WHITE, width=max(1, int(8 * (1 - tau / 0.5))))
        confetti(cv, tau, (1450, 880), 5, 140, speed=(500, 1700), up=0.5)
        confetti(cv, tau - 0.2, (960, 0), 9, 80, kind="rain", life=3.5)
        flash(cv, 0.85 * (1 - clamp(tau / 0.2)))
    # title
    t1 = text_layer("BOWLING", F_BLACK_IT, 250, WHITE, stroke=16, stroke_fill=NAVY, shadow=12,
                    shadow_fill=RED)
    slam(cv, t1, 960 + shx, 250 + shy, bt, 8.1, 0.6, 2.6, -6, rot=-3)
    t2 = text_layer("NIGHT", F_BLACK_IT, 300, RED, stroke=16, stroke_fill=WHITE, shadow=14,
                    shadow_fill=NAVY)
    slam(cv, t2, 1010 + shx, 470 + shy, bt, 9.8, 0.6, 2.6, 6, rot=-3)
    # ribbon
    if bt > 12.8:
        ru = eo5(seg(bt, 12.8, 13.8))
        cw = chip(cv, "BOWL  •  CONNECT  •  UNWIND  •  REPEAT", 960 - (1 - ru) * 1900, 680, 54,
                  NAVY, WHITE, F_XB_IT, 60, edge=RED, tracking=2)
    if bt > 15.0:
        du = back(seg(bt, 15.0, 15.8))
        chip(cv, "13 NOVEMBER 2026  ·  WANGSA BOWL @ SETIA CITY MALL", 960, 960, 44, RED, WHITE,
             F_XB_IT, 44, scale=du, alpha=clamp(du * 3), tracking=2)
    return cv


def caption_for(cv, txt, bt, start, end, y=930, size=104):
    u = back(seg(bt, start, start + 0.45), 1.5)
    out = eo3(seg(bt, end - 0.35, end))
    x = 90 - (1 - u) * 900 - out * 900
    chip(cv, txt, x, y, size, RED, WHITE, F_BLACK_IT, 50, anchor="l", shadow=5,
         alpha=clamp(u * 4) * (1 - out))
    # navy underline bar echo
    if u > 0.05:
        ImageDraw.Draw(cv).polygon([(x + 20, y + size * 0.78), (x + 520 * u + 20, y + size * 0.78),
                                    (x + 500 * u, y + size * 0.78 + 12), (x, y + size * 0.78 + 12)],
                                   fill=WHITE) if False else None


def scene_montage(bt, tsec):
    i, t0 = mont_index(bt)
    name, b0, b1, cap, dur = MONT[i]
    u = clamp((bt - t0) / dur)
    pulse = (1 - ((bt - t0) % 1.0)) ** 3
    # slight direction variety via shift
    sx = (i % 3 - 1) * 14 * (1 - u)
    if name.startswith("clip:"):
        cv = clip_view(name[5:], b0, (bt - t0) * BEAT, pulse * 0.6).copy()
    else:
        cv = view(photo(name), b0, b1, u, pulse * 0.55, (sx, 0)).copy()
    cv.paste((6, 12, 60), (0, 0), bottom_grad())
    # diagonal colour corners
    d = ImageDraw.Draw(cv)
    d.polygon([(0, 0), (430, 0), (0, 150)], fill=RED)
    d.polygon([(0, 0), (300, 0), (0, 105)], fill=NAVY)
    d.polygon([(W, H), (W - 520, H), (W, H - 180)], fill=NAVY)
    d.polygon([(W, H), (W - 320, H), (W, H - 110)], fill=RED)
    # tag
    tag = text_layer("TCIM BOWLING NIGHT 2026", F_XB_IT, 36, WHITE, shadow=3, tracking=3)
    put(cv, tag, W - 40, 60, anchor="r")
    # counter  "01/20" style
    cnt = text_layer("%02d" % (i + 1), F_BLACK_IT, 62, WHITE, shadow=3)
    put(cv, cnt, 38, 36, anchor="l") if False else None
    caption_for(cv, cap, bt, t0, t0 + dur, 920, 104)
    equalizer(cv, tsec, bt, H, 64, RED)
    cv.paste((0, 0, 0), (0, 0), vignette())
    flash(cv, 0.22 * (1 - clamp((bt - t0) / 0.28)))
    return cv


MEDAL = {1: GOLD, 2: (205, 212, 230), 3: (214, 134, 64)}
INK = (12, 26, 100)


def fmt(n):
    return "{:,}".format(n)


@lru_cache(maxsize=64)
def row_layer(rank, team, g1, g2, total):
    w, h = 1320, 52
    if rank <= 3:
        col, tcol = MEDAL[rank], INK
    else:
        col, tcol = ((22, 52, 170) if rank % 2 == 0 else (16, 40, 138)), WHITE
    base = chip_layer(w, h, col, 0.28)
    img = Image.new("RGBA", base.size, (0, 0, 0, 0))
    img.paste(base, (0, 0))

    def t(txt, cx, size=36, left=False):
        tl = text_layer(txt, F_BLACK_IT, size, tcol)
        x = cx if left else cx - tl.width // 2
        img.alpha_composite(tl, (int(x), (h - tl.height) // 2))

    t(str(rank), 60, 38)
    t(team, 140, 38, left=True)
    t(str(g1), 800)
    t(str(g2), 1000)
    t(fmt(total), 1190, 40)
    return img


def scene_teams(bt, tsec):
    L0 = 92.0
    cv = radial_bg().copy()
    sunburst(cv, tsec, (70, 120, 255), 0.18, 16)
    shx, shy = shake(bt, 106.0, 14, 0.8)
    tl = text_layer("TEAM STANDINGS", F_BLACK_IT, 104, WHITE, shadow=8, shadow_fill=RED)
    slam(cv, tl, 960, 92 + shy, bt, L0 + 0.1, 0.6, 2.2)
    sub = text_layer("TCIM BOWLING NIGHT  ·  13 NOVEMBER 2026  ·  TOTAL PINS (2 GAMES)", F_XB_IT, 30,
                     (200, 215, 255), tracking=3)
    put(cv, sub, 960, 168, alpha=seg(bt, L0 + 1, L0 + 2))
    ha = seg(bt, L0 + 1.5, L0 + 2.5)
    for txt, cx in (("RANK", 330), ("TEAM", 475), ("GAME 1", 1095), ("GAME 2", 1295), ("TOTAL", 1485)):
        lb = text_layer(txt, F_XB_IT, 26, (170, 190, 255), tracking=3)
        put(cv, lb, cx, 236, alpha=ha)
    for rank, team, g1, g2, total in TEAMS:
        ta = L0 + 2 + (12 - rank) if rank > 1 else L0 + 14.0
        u = (bt - ta) / 0.55
        if u <= 0:
            continue
        y = 292 + (rank - 1) * 58 + (shy if rank == 1 else 0)
        lay = row_layer(rank, team, g1, g2, total)
        put(cv, lay, 960 + (1 - eo5(u)) * 1100, y, alpha=clamp(u * 4))
    if bt >= 106.0:
        tau = (bt - 106.0) * BEAT
        y = 292
        if tau < 0.6:
            star_burst(cv, 960, y, 520 * eo3(tau / 0.3) * (1 - clamp((tau - 0.3) / 0.3)), 18, tau * 2)
            put(cv, row_layer(*TEAMS[0]), 960, y + shy)
        confetti(cv, tau, (960, y), 31, 150, speed=(600, 1900), up=0.55)
        confetti(cv, tau - 0.1, (960, 0), 32, 80, kind="rain", life=4.0)
        flash(cv, 0.7 * (1 - clamp(tau / 0.22)))
    # bottom message
    if 106.0 <= bt < 113.0:
        a = back(seg(bt, 106.2, 106.9), 1.4)
        chip(cv, "CHAMPIONS: TEAM 7  ·  1,114 PINS", 960, 1010, 54, GOLD, INK, F_BLACK_IT, 56, scale=a,
             alpha=clamp(a * 4) * (1 - seg(bt, 112.6, 113.0)), edge=WHITE)
    elif bt >= 113.0:
        a = back(seg(bt, 113.0, 113.7), 1.4)
        chip(cv, "HIGHEST TEAM GAME: TEAM 7  ·  576", 960, 1010, 54, RED, WHITE, F_BLACK_IT, 56, scale=a,
             alpha=clamp(a * 4), edge=WHITE)
    cv.paste((0, 0, 0), (0, 0), vignette())
    return cv


@lru_cache(maxsize=64)
def card_layer(rank, name, score, w=820, h=150):
    S = 2
    col = MEDAL[rank]
    dx = int(h * 0.22)
    big = Image.new("RGBA", ((w + dx) * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    d.polygon([(dx * S, 0), ((w + dx) * S, 0), (w * S, h * S), (0, h * S)], fill=col + (255,))
    e = 6 * S
    d.polygon([(dx * S + e * 1.2, e), ((w + dx) * S - e, e), (w * S - e * 1.2, h * S - e), (e, h * S - e)],
              fill=INK + (255,))
    img = big.resize((w + dx, h), Image.LANCZOS)
    bd = ImageDraw.Draw(img)
    r, cx, cy = 46, 100, h // 2
    bd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    nl = text_layer(str(rank), F_BLACK_IT, 64, INK)
    img.alpha_composite(nl, (cx - nl.width // 2, cy - nl.height // 2))
    tl = text_layer(name, F_BLACK_IT, 70, WHITE, shadow=3, shadow_fill=(0, 0, 40))
    img.alpha_composite(tl, (170, (h - tl.height) // 2))
    sl = text_layer(str(score), F_BLACK_IT, 88, col, shadow=3, shadow_fill=(0, 0, 40))
    img.alpha_composite(sl, (w + dx - 40 - sl.width, (h - sl.height) // 2))
    return img


def scene_people(bt, tsec, L0, title, series, game, extras):
    cv = radial_bg().copy()
    sunburst(cv, tsec, (70, 120, 255), 0.18, 18)
    shx, shy = shake(bt, L0 + 7.0, 16, 0.8)
    tl = text_layer(title, F_BLACK_IT, 104, WHITE, shadow=8, shadow_fill=RED)
    slam(cv, tl, 960, 92 + shy, bt, L0 + 0.1, 0.6, 2.2)
    hu = back(seg(bt, L0 + 1.0, L0 + 1.8), 1.4)
    chip(cv, "HIGHEST SERIES  (2 GAMES)", 500, 262, 38, NAVY, WHITE, F_XB_IT, 40, scale=hu,
         alpha=clamp(hu * 4), edge=RED, tracking=2)
    chip(cv, "HIGHEST SINGLE GAME", 1420, 262, 38, NAVY, WHITE, F_XB_IT, 40, scale=hu,
         alpha=clamp(hu * 4), edge=RED, tracking=2)
    reveal = {3: L0 + 2.0, 2: L0 + 4.5, 1: L0 + 7.0}
    for rank in (3, 2, 1):
        u = (bt - reveal[rank]) / 0.55
        if u <= 0:
            continue
        y = 450 + (rank - 1) * 200 + shy
        for lst, cx, side in ((series, 500, -1), (game, 1420, 1)):
            nm, sc = lst[rank - 1]
            put(cv, card_layer(rank, nm, sc), cx + side * (1 - eo5(u)) * 1100, y, alpha=clamp(u * 4))
    if bt >= L0 + 7.0:
        tau = (bt - (L0 + 7.0)) * BEAT
        if tau < 0.6:
            for cx in (500, 1420):
                star_burst(cv, cx, 450, 190 * eo3(tau / 0.3) * (1 - clamp((tau - 0.3) / 0.3)), 14, tau * 2)
                put(cv, card_layer(1, *(series if cx == 500 else game)[0]), cx, 450 + shy)
        confetti(cv, tau, (960, 450), int(L0) + 7, 140, speed=(600, 1800), up=0.55)
        flash(cv, 0.6 * (1 - clamp(tau / 0.2)))
    for k, txt in enumerate(extras):
        a0 = L0 + 9.5 + k * 2.2
        a1 = a0 + 2.2 if k < len(extras) - 1 else L0 + 15
        if a0 <= bt < a1 + 0.3:
            a = back(seg(bt, a0, a0 + 0.7), 1.4)
            out = seg(bt, a1 - 0.3, a1) if k < len(extras) - 1 else 0
            chip(cv, txt, 960, 1010, 46, GOLD if k == 0 else RED, INK if k == 0 else WHITE, F_BLACK_IT, 50,
                 scale=a, alpha=clamp(a * 4) * (1 - out), edge=WHITE)
    cv.paste((0, 0, 0), (0, 0), vignette())
    return cv


def scene_men(bt, tsec):
    return scene_people(bt, tsec, 120.0, "MEN'S TOP 3", AWARDS["men_series"], AWARDS["men_game"],
                        ["RUDY: HIGH GAME 157  ·  HIGH SERIES 286"])


def scene_women(bt, tsec):
    return scene_people(bt, tsec, 134.0, "WOMEN'S TOP 3", AWARDS["women_series"], AWARDS["women_game"],
                        ["AISHA: HIGH GAME 125  ·  HIGH SERIES 225",
                         "STAR BOWLER: AIN  ·  +16 PINS OVER AVERAGE"])


def scene_hero(bt, tsec):
    b0 = 148.0
    lb = bt - b0
    second = lb >= 10.0
    pulse = (1 - (bt % 1.0)) ** 3
    if not second:
        u = seg(lb, -0.5, 10.0)
        cv = view(photo("g0214"), (0.0, 0.0, 1.0), D2K(160, 200, 1680), u, pulse * 0.5)
    else:
        u = seg(lb, 10.0, 20.0)
        cv = view(photo("g0224"), (0.0, 0.0, 1.0), D2K(100, 150, 1800), u, pulse * 0.5)
    cv = cv.copy()
    cv.paste((6, 12, 60), (0, 0), top_grad())
    shx, shy = shake(bt, b0 + 0.5, 8, 0.4)

    def line(txt, size, start, exit_at, y):
        l = text_layer(txt, F_BLACK_IT, size, WHITE, shadow=9, shadow_fill=(0, 0, 40))
        slam(cv, l, 120 + l.width / 2 + shx, y + shy, bt, b0 + start, 0.5, 2.2, exit_at=b0 + exit_at)

    def bar(txt, size, start, end, y):
        if b0 + start <= bt < b0 + end + 0.5:
            cu = back(seg(bt, b0 + start, b0 + start + 0.7), 1.5)
            chip(cv, txt, 120 + (1 - cu) * -1400, y, size, RED, WHITE, F_BLACK_IT, 56,
                 anchor="l", shadow=5, alpha=clamp(cu * 4) * (1 - eo3(seg(bt, b0 + end, b0 + end + 0.4))))

    if not second:
        line("SAME TEAM.", 150, 0.6, 5.6, 190)
        bar("BIGGER MEMORIES!", 118, 2.4, 5.4, 350)
        line("IT'S NOT JUST A GAME.", 112, 6.2, 9.7, 190)
        bar("IT'S A STRONGER US.", 118, 8.0, 9.6, 350)
    else:
        line("THANK YOU, TEAM!", 140, 10.6, 15.6, 190)
        bar("YOU MADE IT UNFORGETTABLE!", 100, 12.4, 15.4, 350)
        l3 = text_layer("SEE YOU AT THE NEXT ONE!", F_BLACK_IT, 100, WHITE, shadow=8, shadow_fill=(0, 0, 40))
        slam(cv, l3, 120 + l3.width / 2, 210, bt, b0 + 16.0, 0.5, 2.0)
    d = ImageDraw.Draw(cv)
    d.polygon([(W, H), (W - 520, H), (W, H - 180)], fill=NAVY)
    d.polygon([(W, H), (W - 320, H), (W, H - 110)], fill=RED)
    d.polygon([(0, H), (430, H), (0, H - 150)], fill=RED)
    d.polygon([(0, H), (300, H), (0, H - 105)], fill=NAVY)
    tag = text_layer("TCIM BOWLING NIGHT 2026", F_XB_IT, 36, WHITE, shadow=3, tracking=3)
    put(cv, tag, W - 40, 60, anchor="r")
    for t0, s_ in ((0.0, 1), (6.0, 2), (10.0, 3)):
        if lb >= t0:
            confetti(cv, (lb - t0) * BEAT, (960, 420), 40 + s_, 90, speed=(500, 1500), up=0.6, life=2.6)
    equalizer(cv, tsec, bt, H, 54, RED)
    cv.paste((0, 0, 0), (0, 0), vignette())
    flash(cv, 0.6 * (1 - clamp(lb / 0.4)) if lb < 2 else (0.5 * (1 - clamp((lb - 10) / 0.35)) if 10 <= lb < 12 else 0))
    return cv


def scene_end(bt, tsec):
    b0 = 168.0
    lb = bt - b0
    cv = radial_bg().copy()
    sunburst(cv, tsec, (80, 130, 255), 0.30, 22)
    d = ImageDraw.Draw(cv)
    for k in range(-2, 8):
        x = k * 380 + (tsec * 40) % 380
        d.polygon([(x, 0), (x + 60, 0), (x - 220, H), (x - 280, H)], fill=(18, 40, 140))
    cv.paste((0, 0, 0), (0, 0), vignette())
    shx, shy = shake(bt, b0, 14, 0.7)
    shx3, shy3 = shake(bt, b0 + 12, 20, 1.0)
    shx += shx3
    shy += shy3
    wm = logo_layer(340)
    slam(cv, wm, 960 + shx, 300 + shy, bt, b0 + 0.1, 0.6, 2.6, -5)
    if lb >= 2.0:
        cu = back(seg(lb, 2.0, 2.8), 1.4)
        chip(cv, "BOWLING NIGHT 2026", 960 + shx, 560 + shy, 120, RED, WHITE, F_BLACK_IT, 70, scale=cu,
             alpha=clamp(cu * 4), edge=WHITE, shadow=5)
    if lb >= 4.0:
        a = clamp(seg(lb, 4.0, 4.8))
        t1 = text_layer("STAFF ENGAGEMENT ACTIVITY  ·  YEC 2026", F_XB_IT, 50, WHITE, tracking=4, shadow=3)
        put(cv, t1, 960, 700 + (1 - eo3(a)) * 30, alpha=a)
    if lb >= 5.0:
        a = clamp(seg(lb, 5.0, 5.8))
        t2 = text_layer("13 NOVEMBER 2026  ·  WANGSA BOWL @ SETIA CITY MALL", F_SEMI, 40, (210, 225, 255), tracking=3)
        put(cv, t2, 960, 770 + (1 - eo3(a)) * 30, alpha=a)
    if lb >= 6.5:
        cu = back(seg(lb, 6.5, 7.3), 1.4)
        t3 = text_layer("PLAY  ·  LAUGH  ·  BOND", F_BLACK_IT, 92, GOLD, shadow=6, shadow_fill=(0, 0, 40), tracking=4)
        put(cv, t3, 960, 880, cu, alpha=clamp(cu * 4) * (1 - seg(lb, 11.2, 11.7)))
    for k in range(9):
        px = 130 + k * 207
        ph = (bt + k * 0.18) % 1.0
        bounce = abs(math.sin(ph * math.pi)) * 34
        if lb >= 1.0:
            appear = back(seg(lb, 1.0 + k * 0.12, 1.6 + k * 0.12))
            draw_pin(cv, px, 1052 - bounce - (1 - clamp(appear)) * 200, 7 * math.sin((bt + k) * math.pi),
                     clamp(appear * 3), 120)
    if lb >= 12.0:
        tau = (lb - 12.0) * BEAT
        flash(cv, 0.9 * (1 - clamp(tau / 0.3)))
        confetti(cv, tau, (960, 520), 77, 170, speed=(700, 2100), up=0.5, life=3.0)
        t4 = text_layer("IT'S A STRONGER US!", F_BLACK_IT, 112, WHITE, stroke=11, stroke_fill=NAVY, shadow=8,
                        shadow_fill=RED)
        slam(cv, t4, 960, 880, bt, b0 + 12.0, 0.5, 2.2, 4, rot=-2)
    if 0 <= lb < 1:
        flash(cv, 0.6 * (1 - clamp(lb / 0.3)))
    return cv


# ------------------------------------------------------------------ frame
def render_frame(i):
    tsec = i / FPS
    bt = tsec / BEAT
    scenes = [scene_intro, scene_title, scene_montage, scene_teams, scene_men, scene_women, scene_hero, scene_end]
    k = sum(1 for c in WIPES if bt >= c)
    cv = scenes[k](bt, tsec)
    for c in WIPES:
        wipe(cv, bt, c)
    fo = seg(bt, 182.2, 184.0)
    if fo > 0:
        flash(cv, fo, DEEP)
    return cv


def render_range(args):
    a, b, path = args
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
           "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(a, b):
        p.stdin.write(render_frame(i).tobytes())
    p.stdin.close()
    p.wait()
    return path


def main():
    mode = sys.argv[1]
    prep_clips()
    if mode == "preview":
        out = sys.argv[2]
        beats = [float(x) for x in sys.argv[3].split(",")]
        tiles = []
        for b in beats:
            f = int(round(b * BEAT * FPS))
            tiles.append(render_frame(f).resize((960, 540), Image.LANCZOS))
        cols = 2
        rows = (len(tiles) + 1) // 2
        sheet = Image.new("RGB", (960 * cols, 540 * rows), (0, 0, 0))
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % cols) * 960, (k // cols) * 540))
        sheet.save(out)
    elif mode == "render":
        out, audio = sys.argv[2], sys.argv[3]
        tmp = os.path.dirname(os.path.abspath(out))
        nproc = 4
        step = math.ceil(NFRAMES / nproc)
        jobs = [(k * step, min(NFRAMES, (k + 1) * step), os.path.join(tmp, "seg%d.mp4" % k)) for k in range(nproc)]
        with Pool(nproc) as pool:
            segs = pool.map(render_range, jobs)
        lst = os.path.join(tmp, "segs.txt")
        with open(lst, "w") as f:
            for s in segs:
                f.write("file '%s'\n" % s)
        dur = NFRAMES / FPS
        subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                               "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                               "-b:a", "256k", "-t", "%.3f" % dur, "-movflags", "+faststart", out])
        print("done", out)


if __name__ == "__main__":
    main()
