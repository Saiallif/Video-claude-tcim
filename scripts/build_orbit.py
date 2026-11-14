#!/usr/bin/env python3
"""Video penghargaan versi 2 "Orbit & Spotlight" (ceria): gambar/video kenangan mengorbit potret
Ustaz Dr. Mohd Nazri Bin Mat Yasin, dan setiap kenangan terbang dari orbit lalu di-zoom DEKAT satu-satu.

1920x1080 / 30fps / 96s, 120 BPM (1 bar = 2s). Muzik: assets/audio/muzik_happy.wav (make_music3.py).
Konfeti, denyutan ketukan dan perubahan laju orbit mengikut seksyen muzik.

Guna:
    python3 scripts/build_orbit.py --media <folder media> [--work <folder kerja>] [--preview 30]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tribute import (F_SANS_M, F_SANS_SB, F_SERIF, LOGO_FULL, LOGO_PLAIN, PORTRAIT, W, H, FPS,  # noqa: E402
                           logo_plate, night_bg, place, text_layer, run)

ROOT = Path(__file__).resolve().parent.parent
TOTAL = 96.0
BEAT = 0.5
CX, CY = W // 2, 585          # pusat gelang orbit
MX, MY = W // 2, 455          # medalion potret
BX, BY = 150, 125             # kedudukan lencana medalion semasa spotlight
MED_D = 440
PANEL_W, PANEL_H = 1200, 675
PCX, PCY = W // 2, 545        # pusat panel spotlight

OUTER = ["IMG-20261114-WA0057.jpg", "IMG-20261114-WA0073.jpg", "IMG-20261114-WA0014.jpg", "IMG-20261114-WA0033.jpg",
         "IMG-20261114-WA0086.jpg", "REQ2_women.jpg", "IMG-20261114-WA0036.jpg", "IMG-20261114-WA0090.jpg",
         "IMG-20261114-WA0078.jpg", "IMG-20261114-WA0066.jpg", "REQ5_women.jpg", "IMG-20261114-WA0068.jpg",
         "IMG-20261114-WA0082.jpg", "IMG-20261114-WA0044.jpg"]
INNER = ["IMG-20261114-WA0095.jpg", "IMG-20261114-WA0031.jpg", "IMG-20261114-WA0077.jpg", "IMG-20261114-WA0088.jpg",
         "IMG-20261114-WA0062.jpg", "IMG-20261114-WA0069.jpg", "REQ6_women.jpg", "IMG-20261114-WA0076.jpg",
         "VID-20261114-WA0071.mp4@3", "VID-20261114-WA0085.mp4@4"]
RINGS = [
    dict(name="outer", files=OUTER, R=740, y=0, w=300, dir=+1, speed=1.0, t_in=6.0),
    dict(name="inner", files=INNER, R=450, y=20, w=225, dir=-1, speed=1.35, t_in=8.0),
]

# (jenis, fail, mula dalam video) -- setiap spotlight 4 saat (2 bar)
SPOT_FILES = [
    ("img", "REQ1_men.jpg", 0), ("vid", "VID-20261114-WA0083.mp4", 1.5), ("vid", "VID-20261114-WA0085.mp4", 2.0),
    ("img", "IMG-20261114-WA0075.jpg", 0), ("vid", "VID-20261114-WA0070.mp4", 0.0), ("img", "REQ7_speaker.jpg", 0),
    ("vid", "VID-20261114-WA0072.mp4", 2.0), ("img", "REQ4_lunch.jpg", 0),
    ("vid", "VID-20261114-WA0084.mp4", 1.0), ("img", "REQ3_mixed.jpg", 0), ("vid", "VID-20261114-WA0074.mp4", 1.0),
    ("vid", "VID-20261114-WA0071.mp4", 2.0), ("vid", "VID-20261114-WA0080.mp4", 3.0), ("vid", "VID-20261114-WA0089.mp4", 12.0),
]
SPOT_T0 = [20 + 4 * i for i in range(8)] + [60 + 4 * i for i in range(6)]
SPOT_D = 4.0
SPOT_CAPS = ["Bermula dengan satu perjalanan…", "Bersama Ustaz, setiap hari ada ilmu baharu", "Suasana kelas yang kami rindui",
             "Kenangan yang tak ternilai", "Nasihat dan bimbingan Ustaz kami simpan", "Terima kasih atas kesabaran Ustaz",
             "Ketawa dan belajar bersama-sama", "Meja makan, cerita dan gelak tawa",
             "Ukhuwah yang terbina kerana Ustaz", "Dari pendaftaran hingga ke dewan peperiksaan", "Ustaz sentiasa ada untuk kami",
             "Setiap detik bersama Ustaz berharga", "Semoga sentiasa dalam lindungan Allah", "Kami akan sentiasa merindui Ustaz"]
SPOT_WINDOWS = [(20.0, 52.0), (60.0, 84.0)]


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease_out_back(x, c1=1.3):
    x = float(np.clip(x, 0, 1))
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def pulse(t):
    return float(np.exp(-(t % BEAT) / 0.11)) if 12.0 <= t < 92 else 0.0


def presence(t):
    p = 0.0
    for a, b in SPOT_WINDOWS:
        p = max(p, float(smooth((t - a + 1.0) / 1.4) * (1 - smooth((t - b + 0.4) / 1.4))))
    return p


def omega(t):
    pts = [(0, 0.22), (6, 0.3), (11.5, 0.3), (12.5, 1.05), (19.5, 1.05), (21, 0.38), (51, 0.38), (52.5, 1.2), (59, 1.2),
           (61, 0.42), (83, 0.42), (85, 0.9), (92, 0.7), (96, 0.5)]
    xs, ys = zip(*pts)
    return float(np.interp(t, xs, ys))


# ---------------------------------------------------------------- bahan visual
def load_image(media, spec, work):
    if "@" in spec:
        name, ss = spec.split("@")
        out = work / f"fr_{name}.jpg"
        if not out.exists():
            run(["ffmpeg", "-v", "error", "-y", "-ss", ss, "-i", str(media / name), "-frames:v", "1", str(out)])
        return Image.open(out).convert("RGB")
    return Image.open(media / spec).convert("RGB")


def make_card(im, base_w, box_ratio=0.76):
    bw, bh = base_w, int(base_w * box_ratio)
    sc = min(bw / im.width, bh / im.height)
    iw, ih = int(im.width * sc), int(im.height * sc)
    im = im.resize((iw, ih), Image.LANCZOS)
    b = 7
    card = Image.new("RGBA", (iw + 2 * b, ih + 2 * b), (0, 0, 0, 0))
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, card.width - 1, card.height - 1), 14, fill=255)
    card.paste(Image.new("RGBA", card.size, (232, 190, 98, 255)), (0, 0), mask)
    inner = Image.new("L", card.size, 0)
    ImageDraw.Draw(inner).rounded_rectangle((b, b, card.width - 1 - b, card.height - 1 - b), 9, fill=255)
    card.paste(im, (b, b), inner.crop((b, b, b + iw, b + ih)))
    a = np.asarray(card).astype(np.float32)
    rgb, al = a[..., :3] * (a[..., 3:4] / 255.0), a[..., 3:4]
    return np.clip(np.concatenate([rgb[..., ::-1], al], -1), 0, 255).astype(np.uint8)


def make_medallion(media, d):
    im = Image.open(media / PORTRAIT).convert("RGB").crop((140, 95, 860, 815)).resize((d, d), Image.LANCZOS)
    ring = 9
    sz = d + 2 * ring
    out = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    m1 = Image.new("L", (sz * 3, sz * 3), 0)
    ImageDraw.Draw(m1).ellipse((0, 0, sz * 3 - 1, sz * 3 - 1), fill=255)
    out.paste(Image.new("RGBA", (sz, sz), (236, 196, 108, 255)), (0, 0), m1.resize((sz, sz), Image.LANCZOS))
    m2 = Image.new("L", (sz * 3, sz * 3), 0)
    ImageDraw.Draw(m2).ellipse((ring * 3, ring * 3, (sz - ring) * 3 - 1, (sz - ring) * 3 - 1), fill=255)
    full = Image.new("RGB", (sz, sz), (255, 255, 255))
    full.paste(im, (ring, ring))
    out.paste(full, (0, 0), m2.resize((sz, sz), Image.LANCZOS))
    a = np.asarray(out).astype(np.float32)
    rgb, al = a[..., :3] * (a[..., 3:4] / 255.0), a[..., 3:4]
    return np.clip(np.concatenate([rgb[..., ::-1], al], -1), 0, 255).astype(np.uint8)


def glow_patch(size=1500, color=(255, 205, 120)):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 2)
    return (np.exp(-(r ** 2) * 5.0)[..., None] * np.array(color[::-1], dtype=np.float32)[None, None, :]).astype(np.float32)


def make_panel(src_bgr, zoom=1.0):
    """Panel 16:9 berbingkai emas: latar kabur + gambar/bingkai video di tengah (zoom perlahan). BGRA 'premultiplied'."""
    h, w = src_bgr.shape[:2]
    pw, ph = PANEL_W, PANEL_H
    sc = max(160 / w, 90 / h)
    small = cv2.resize(src_bgr, (max(160, int(w * sc) + 1), max(90, int(h * sc) + 1)), interpolation=cv2.INTER_AREA)
    y0, x0 = (small.shape[0] - 90) // 2, (small.shape[1] - 160) // 2
    small = cv2.GaussianBlur(small[y0:y0 + 90, x0:x0 + 160], (0, 0), 2.2)
    bg = (cv2.resize(small, (pw, ph), interpolation=cv2.INTER_CUBIC).astype(np.float32) * 0.5).astype(np.uint8)
    fit = min((pw - 30) / w, (ph - 24) / h)
    fw, fh = int(w * fit), int(h * fit)
    if zoom > 1.0:
        cw, ch = int(w / zoom), int(h / zoom)
        cx0, cy0 = (w - cw) // 2, (h - ch) // 2
        src = src_bgr[cy0:cy0 + ch, cx0:cx0 + cw]
    else:
        src = src_bgr
    fg = cv2.resize(src, (fw, fh), interpolation=cv2.INTER_AREA if fit < 1 else cv2.INTER_CUBIC)
    fg = cv2.copyMakeBorder(fg, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(98, 190, 232))
    oy, ox = (ph - fg.shape[0]) // 2, (pw - fg.shape[1]) // 2
    bg[oy:oy + fg.shape[0], ox:ox + fg.shape[1]] = fg
    panel = cv2.copyMakeBorder(bg, 9, 9, 9, 9, cv2.BORDER_CONSTANT, value=(98, 190, 232))
    return np.concatenate([panel, np.full(panel.shape[:2] + (1,), 255, np.uint8)], -1)


# ---------------------------------------------------------------- geometri 3D
class Cam:
    def __init__(self, t):
        self.phi = 0.30 + 0.05 * np.sin(0.35 * t)
        self.D = (2300 - 240 * smooth(t / TOTAL)) * (1 - 0.014 * pulse(t))
        self.cp, self.sp = np.cos(self.phi), np.sin(self.phi)

    def proj(self, p):
        x, y, z = p
        yp = y * self.cp + z * self.sp
        zp = -y * self.sp + z * self.cp
        s = self.D / (self.D + zp)
        return CX + x * s, CY - yp * s, zp, s


def draw_premult(frame, card, quad, shade, alpha_mul):
    q = np.asarray(quad, np.float32)
    x0, y0 = int(np.floor(q[:, 0].min())), int(np.floor(q[:, 1].min()))
    x1, y1 = int(np.ceil(q[:, 0].max())), int(np.ceil(q[:, 1].max()))
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4 or bw > 2400 or bh > 1500 or x1 < 0 or y1 < 0 or x0 >= W or y0 >= H or alpha_mul <= 0.003:
        return
    ch, cw = card.shape[:2]
    tw = np.hypot(*(q[1] - q[0]))
    if tw < 0.7 * cw:
        f = max(0.15, tw / cw * 1.15)
        card = cv2.resize(card, (max(4, int(cw * f)), max(4, int(ch * f))), interpolation=cv2.INTER_AREA)
        ch, cw = card.shape[:2]
    src = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32)
    M = cv2.getPerspectiveTransform(src, q - np.array([x0, y0], np.float32))
    warped = cv2.warpPerspective(card, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    ix0, iy0, ix1, iy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    wr = warped[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0].astype(np.float32)
    roi = frame[iy0:iy1, ix0:ix1].astype(np.float32)
    a = wr[..., 3:4] / 255.0 * alpha_mul
    frame[iy0:iy1, ix0:ix1] = np.clip(wr[..., :3] * shade * alpha_mul + roi * (1 - a), 0, 255).astype(np.uint8)


def add_glow(frame, glow, cx, cy, k):
    gh = glow.shape[0]
    s = int(gh * (0.55 + 0.45 * min(k, 1.2)))
    if s < 8 or k <= 0:
        return
    g = cv2.resize(glow, (s, s), interpolation=cv2.INTER_LINEAR) * k
    x0, y0 = int(cx - s / 2), int(cy - s / 2)
    ix0, iy0, ix1, iy1 = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    roi = frame[iy0:iy1, ix0:ix1].astype(np.float32) + g[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    frame[iy0:iy1, ix0:ix1] = np.clip(roi, 0, 255).astype(np.uint8)


class Confetti:
    COLORS = [(98, 190, 232), (255, 255, 255), (235, 200, 90), (150, 110, 245), (60, 160, 250), (210, 235, 120)]

    def __init__(self):
        rg = np.random.default_rng(21)
        self.p = []
        for tb, n in [(12.0, 200), (20.0, 90), (52.0, 220), (60.0, 90), (84.0, 220)]:
            for _ in range(n):
                self.p.append(dict(tb=tb + rg.uniform(0, 0.5), x=rg.uniform(0, W), y=rg.uniform(-300, -20),
                                   vx=rg.normal(0, 90), vy=rg.uniform(60, 220), g=rg.uniform(120, 220), rot=rg.uniform(-9, 9),
                                   ph=rg.uniform(0, 6.28), w=rg.uniform(10, 22), h=rg.uniform(5, 11), col=self.COLORS[rg.integers(0, 6)],
                                   sw=rg.uniform(20, 70), life=rg.uniform(4.0, 6.2)))

    def draw(self, f, t):
        for c in self.p:
            a = t - c["tb"]
            if a < 0 or a > c["life"]:
                continue
            x = c["x"] + c["vx"] * a + c["sw"] * np.sin(2 * a + c["ph"])
            y = c["y"] + c["vy"] * a + 0.25 * c["g"] * a * a
            if y > H + 30:
                continue
            ang = c["ph"] + c["rot"] * a
            sq = abs(np.cos(ang * 0.7)) * 0.8 + 0.2
            dx, dy = c["w"] * 0.5, c["h"] * 0.5 * sq
            ca, sa = np.cos(ang), np.sin(ang)
            pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]]) @ np.array([[ca, sa], [-sa, ca]]) + [x, y]
            cv2.fillConvexPoly(f, pts.astype(np.int32), c["col"], cv2.LINE_AA)


# ---------------------------------------------------------------- adegan
class Scene:
    def __init__(self, media, work):
        self.media, self.work = media, work
        self.bg = np.asarray(night_bg((MX, MY), 0.16).convert("RGB"))[..., ::-1].copy()
        self.glow = glow_patch()
        self.med = make_medallion(media, MED_D)
        self.rings = [dict(rd, cards=[make_card(load_image(media, f, work), 470) for f in rd["files"]]) for rd in RINGS]
        ts = np.arange(0, TOTAL + 1, 1 / FPS)
        self.Theta = np.cumsum([omega(x) for x in ts]) / FPS
        self.confetti = Confetti()
        self.spots = []
        for (kind, fn, ss), t0 in zip(SPOT_FILES, SPOT_T0):
            if kind == "img":
                img = cv2.imread(str(media / fn))
                if max(img.shape[:2]) > 1800:
                    s = 1800 / max(img.shape[:2])
                    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                self.spots.append(dict(kind="img", img=img, t0=t0))
            else:
                clip = work / f"spot_{fn}_{ss}.mp4"
                if not clip.exists():
                    run(["ffmpeg", "-v", "error", "-y", "-ss", str(ss), "-t", f"{SPOT_D + 1.0:.1f}", "-i", str(media / fn), "-an",
                         "-vf", "fps=30,scale=1280:720:force_original_aspect_ratio=decrease", "-c:v", "libx264", "-crf", "17", str(clip)])
                self.spots.append(dict(kind="vid", clip=clip, t0=t0, cap=None, k=-1, last=None))

    def theta(self, t):
        return float(np.interp(t, np.arange(len(self.Theta)) / FPS, self.Theta))

    def ring_cards(self, t, cam):
        out = []
        dim = 1 - 0.5 * presence(t)
        pl = pulse(t)
        for ri, rg in enumerate(self.rings):
            n = len(rg["cards"])
            Th = self.theta(t) * rg["speed"] * rg["dir"]
            for i, card in enumerate(rg["cards"]):
                e_in = smooth((t - rg["t_in"] - i * 0.22) / 3.4)
                e_out = smooth((t - 90.0 - i * 0.1) / 1.6)
                life = e_in * (1 - e_out)
                if life <= 0.003:
                    out.append(None)
                    continue
                rad = rg["R"] * (0.12 + 0.88 * e_in) * (1 - 0.8 * e_out) * (1 + 0.02 * pl)
                th = 2 * np.pi * i / n + Th + 0.4 * ri
                P = np.array([rad * np.cos(th), rg["y"], rad * np.sin(th)])
                tx, tz = -np.sin(th), np.cos(th)
                if tx < 0:
                    tx, tz = -tx, -tz
                d = np.array([0.6 * tx + 0.4, 0, 0.6 * tz])
                d /= np.linalg.norm(d)
                k = 0.5 + 0.5 * e_in
                hw = card.shape[1] / 470 * rg["w"] / 2 * k
                hh = card.shape[0] / 470 * rg["w"] / 2 * k
                up = np.array([0, 1, 0])
                corners = [P - d * hw + up * hh, P + d * hw + up * hh, P + d * hw - up * hh, P - d * hw - up * hh]
                quad = [(p[0], p[1]) for p in (cam.proj(c) for c in corners)]
                zc = cam.proj(P)[2]
                df = np.clip(1.0 - 0.52 * (zc + rg["R"]) / (2 * rg["R"] + 1e-6), 0.38, 1.0) * dim
                out.append(dict(zc=zc, card=card, quad=quad, df=df, life=life, ring=ri, i=i))
        return out

    def spot_frame(self, sp, u):
        if sp["kind"] == "img":
            return sp["img"], 1.0 + 0.07 * float(np.clip(u / SPOT_D, 0, 1))
        k = int(max(u, 0) * 30)
        if sp["cap"] is None or k < sp["k"]:
            sp["cap"] = cv2.VideoCapture(str(sp["clip"]))
            sp["k"] = -1
            sp["last"] = None
        while sp["k"] < k:
            ok, fr = sp["cap"].read()
            if not ok:
                break
            sp["last"] = fr
            sp["k"] += 1
        if sp["last"] is None:
            sp["last"] = np.zeros((720, 1280, 3), np.uint8)
        return sp["last"], 1.0 + 0.03 * float(np.clip(u / SPOT_D, 0, 1))

    def frame(self, t):
        f = self.bg.copy()
        pl = pulse(t)
        if 12 <= t < 92 and (t % 2.0) < 0.12:        # kilat halus pada setiap bar
            f = np.clip(f.astype(np.int16) + int(14 * (1 - (t % 2.0) / 0.12)), 0, 255).astype(np.uint8)
        cam = Cam(t)
        p = presence(t)
        if t > 3.4:
            add_glow(f, self.glow, MX, MY, (0.9 + 0.5 * smooth((t - 3.6) / 3)) * (0.9 + 0.35 * pl) * 0.55 * (1 - 0.6 * p))
        ring = self.ring_cards(t, cam)
        items = [(c["zc"], ("card", c)) for c in ring if c]
        if t >= 3.6:
            ms = ease_out_back((t - 3.6) / 2.4) * (1 + 0.10 * smooth((t - 84) / 6) + 0.03 * pl) * (1 - 0.66 * p)
            items.append((0.0, ("med", ms, MX + (BX - MX) * p, MY + (BY - MY) * p)))
        items.sort(key=lambda x: x[0], reverse=True)
        for rg in self.rings:
            self._orbit_line(f, cam, rg, t)
        for _, it in items:
            if it[0] == "card":
                c = it[1]
                draw_premult(f, c["card"], c["quad"], c["df"], c["life"])
            else:
                _, ms, mx, my = it
                sz = int(self.med.shape[0] * ms)
                if sz > 8:
                    m = cv2.resize(self.med, (sz, sz), interpolation=cv2.INTER_AREA if ms < 1 else cv2.INTER_CUBIC)
                    x0, y0 = int(mx - sz // 2), int(my - sz // 2)
                    draw_premult(f, m, [(x0, y0), (x0 + sz, y0), (x0 + sz, y0 + sz), (x0, y0 + sz)], 1.0, 1.0)
        # spotlight: terbang dari orbit, zoom DEKAT
        rc = [c for c in ring if c and c["ring"] == 0]
        for si, sp in enumerate(self.spots):
            u = t - sp["t0"]
            if u < -0.05 or u > SPOT_D + 0.1:
                continue
            e = ease_out_back(u / 0.85)
            x_out = float(smooth((u - 3.5) / 0.5))
            alpha = float(smooth(u / 0.22)) * (1 - x_out)
            src, zoom = self.spot_frame(sp, u)
            panel = make_panel(src, zoom)
            hw, hh = panel.shape[1] / 2, panel.shape[0] / 2
            k = (1 + 0.012 * pl) * (1 - 0.1 * x_out) * (1 + 0.04 * float(np.clip(u / SPOT_D, 0, 1)))
            tgt = np.array([[PCX - hw * k, PCY - hh * k], [PCX + hw * k, PCY - hh * k], [PCX + hw * k, PCY + hh * k], [PCX - hw * k, PCY + hh * k]], np.float32)
            if rc and e < 1.0:
                s0 = np.array(rc[(si * 3) % len(rc)]["quad"], np.float32)
                quad = s0 * (1 - e) + tgt * e
            else:
                quad = tgt * (1 if e <= 1 else 1) + 0
                if rc and e > 1.0:
                    s0 = np.array(rc[(si * 3) % len(rc)]["quad"], np.float32)
                    quad = s0 * (1 - e) + tgt * e
            draw_premult(f, panel, quad, 1.0, alpha)
        self.confetti.draw(f, t)
        return f

    def _orbit_line(self, f, cam, rg, t):
        rin = smooth((t - rg["t_in"]) / 5.0) * (1 - smooth((t - 90.0) / 1.6)) * (1 - 0.55 * presence(t))
        if rin <= 0.01:
            return
        R = rg["R"] * (0.4 + 0.6 * rin)
        pts = []
        for k in range(0, 121):
            a = 2 * np.pi * k / 120
            x, y, z, s = cam.proj((R * np.cos(a), rg["y"], R * np.sin(a)))
            pts.append((x, y, z))
        ov = np.zeros_like(f)
        for k in range(120):
            x1, y1, z1 = pts[k]
            x2, y2, z2 = pts[k + 1]
            zf = np.clip(1.0 - 0.75 * (z1 + R) / (2 * R), 0.2, 1.0)
            col = tuple(int(c * zf * rin) for c in (130, 190, 235))
            cv2.line(ov, (int(x1 * 16), int(y1 * 16)), (int(x2 * 16), int(y2 * 16)), col, 2, cv2.LINE_AA, 4)
        for j in range(3):
            a = self.theta(t) * rg["speed"] * rg["dir"] * 1.6 + j * 2.094 + (0.5 if rg["dir"] < 0 else 0)
            x, y, z, s = cam.proj((R * np.cos(a), rg["y"], R * np.sin(a)))
            zf = np.clip(1.0 - 0.6 * (z + R) / (2 * R), 0.35, 1.0)
            cv2.circle(ov, (int(x * 16), int(y * 16)), int(6 * 16 * s ** 0.5), (int(250 * zf), int(235 * zf), int(190 * zf)), -1, cv2.LINE_AA, 4)
            cv2.circle(ov, (int(x * 16), int(y * 16)), int(15 * 16 * s ** 0.5), (int(90 * zf * rin), int(130 * zf * rin), int(170 * zf * rin)), -1, cv2.LINE_AA, 4)
        ov = cv2.GaussianBlur(ov, (0, 0), 1.2)
        f[:] = np.clip(f.astype(np.int16) + ov.astype(np.int16), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- teks dan komposisi akhir
def gradient_png(path):
    y = np.arange(H, dtype=np.float32)
    a = np.maximum(np.clip((260 - y) / 260, 0, 1) ** 1.3 * 0.75, np.clip((y - 880) / 200, 0, 1) ** 1.3 * 0.72)
    arr = np.zeros((H, W, 4), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 4, 9, 24
    arr[..., 3] = (a[:, None] * 255).astype(np.uint8)
    Image.fromarray(arr, "RGBA").save(path)


def layers(work):
    gold = (245, 215, 140)
    L = []

    def add(img, name, t0, t1, fi=0.6, fo=0.6):
        p = work / f"olay_{name}.png"
        img.save(p)
        L.append((p, t0, t1, fi, fo))

    add(place(logo_plate(LOGO_PLAIN, 380), W // 2, 470), "logo1", 0.2, 3.8, 0.7, 0.7)
    add(text_layer("DENGAN PENUH PENGHARGAAN", F_SANS_SB, 34, gold, tracking=14, glow=False, cy=810), "t0", 0.9, 3.8, 0.8, 0.6)
    add(text_layer("Ustaz Dr. Mohd Nazri Bin Mat Yasin", F_SERIF, 62, "gold", cy=75), "name", 5.2, 91.0, 0.9, 0.8)
    add(text_layer("PMDIUC UKM BERANANG   •   SEJAK JANUARI 2026", F_SANS_SB, 26, "white", tracking=7, glow=False, cy=140), "sub", 6.0, 91.0, 0.9, 0.8)
    add(text_layer("TERIMA KASIH, USTAZ!", F_SERIF, 96, "gold", cy=975), "drop1", 12.3, 19.6, 0.4, 0.5)
    add(text_layer("KENANGAN TAK TERLUPA", F_SERIF, 96, "gold", cy=975), "drop2", 52.3, 59.6, 0.4, 0.5)
    for i, (t0, s) in enumerate(zip(SPOT_T0, SPOT_CAPS)):
        add(text_layer(s, F_SERIF, 54, "white", glow=False, cy=972), f"scap{i}", t0 + 0.7, t0 + 3.75, 0.5, 0.4)
    add(text_layer("Terima kasih atas ilmu, bimbingan dan kesabaran Ustaz", F_SERIF, 52, "white", glow=False, cy=965), "th1", 84.4, 88.6, 0.6, 0.5)
    add(text_layer("Semoga Allah membalas jasa Ustaz dan", F_SERIF, 50, "white", glow=False, cy=930), "th2a", 88.8, 92.4, 0.6, 0.5)
    add(text_layer("memudahkan urusan di tempat yang baharu", F_SERIF, 50, "gold", glow=True, cy=997), "th2b", 89.2, 92.4, 0.6, 0.5)
    end_bg = work / "olay_endbg.png"
    night_bg((W // 2, 430), 0.28).save(end_bg)
    L.append((end_bg, 92.3, TOTAL, 1.0, 0.01))
    add(place(logo_plate(LOGO_FULL, 520, pad=30), W // 2, 395), "end_logo", 93.0, TOTAL, 0.8, 0.01)
    add(text_layer("DENGAN PENUH PENGHARGAAN DARIPADA", F_SANS_SB, 28, gold, tracking=10, glow=False, cy=770), "end1", 93.6, TOTAL, 0.8, 0.01)
    add(text_layer("PELAJAR PJJ PMDIUC UKM BERANANG", F_SANS_SB, 40, "white", tracking=6, glow=False, cy=830), "end2", 94.0, TOTAL, 0.8, 0.01)
    add(text_layer("Ustaz Dr. Mohd Nazri Bin Mat Yasin", F_SERIF, 56, "gold", cy=935), "end3", 94.6, TOTAL, 0.8, 0.01)
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--work", default="/tmp/orbit_work")
    ap.add_argument("--out", default=str(ROOT / "output/Penghargaan_Ustaz_Orbit_v2b.mp4"))
    ap.add_argument("--particles", default="")
    ap.add_argument("--preview", type=float, default=None)
    ap.add_argument("--skip-base", action="store_true")
    a = ap.parse_args()
    media, work = Path(a.media), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    sc = Scene(media, work)
    if a.preview is not None:
        t = a.preview
        for sp in sc.spots:
            if sp["t0"] <= t <= sp["t0"] + SPOT_D:
                for tt in np.arange(sp["t0"], t, 1 / 30):
                    sc.spot_frame(sp, tt - sp["t0"])
        f = sc.frame(t)
        cv2.imwrite(str(work / f"preview_{t:.1f}.png"), f)
        print("preview", work / f"preview_{t:.1f}.png")
        return
    base = work / "base.mp4"
    if not a.skip_base:
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                              "-c:v", "libx264", "-crf", "14", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(base)], stdin=subprocess.PIPE)
        n = int(TOTAL * FPS)
        for i in range(n):
            p.stdin.write(sc.frame(i / FPS).tobytes())
            if i % 150 == 0:
                print(f"bingkai {i}/{n}", flush=True)
        p.stdin.close()
        p.wait()

    part = work / "particles.mp4"
    if a.particles and not part.exists():
        part.write_bytes(Path(a.particles).read_bytes())
    grad = work / "grad.png"
    gradient_png(grad)
    L = layers(work)
    ins = ["-i", str(base), "-stream_loop", "-1", "-i", str(part), "-i", str(ROOT / "assets/audio/muzik_happy.wav"),
           "-loop", "1", "-framerate", str(FPS), "-t", f"{TOTAL:.1f}", "-i", str(grad)]
    g = ["[3:v]format=rgba,fade=t=in:st=5.0:d=1.0:alpha=1,fade=t=out:st=91.5:d=0.8:alpha=1[gr]",
         "[0:v][gr]overlay=format=auto:shortest=1[v0]"]
    last = "v0"
    for j, (png, t0, t1, fi, fo) in enumerate(L):
        idx = 4 + j
        dur = t1 - t0
        ins += ["-loop", "1", "-framerate", str(FPS), "-t", f"{dur + 0.1:.2f}", "-i", str(png)]
        g.append(f"[{idx}:v]format=rgba,fade=t=in:st=0:d={fi}:alpha=1,fade=t=out:st={max(0, dur - fo):.2f}:d={fo}:alpha=1,setpts=PTS+{t0}/TB[L{j}]")
        g.append(f"[{last}][L{j}]overlay=format=auto:eof_action=pass[o{j}]")
        last = f"o{j}"
    g.append(f"[{last}]format=gbrp[vv]")
    g.append(f"[1:v]trim=duration={TOTAL:.1f},setpts=PTS-STARTPTS,fade=t=in:st=0:d=1.5,eq=brightness=-0.05,format=gbrp[pp]")
    g.append("[vv][pp]blend=all_mode=screen:all_opacity=0.45[vb]")
    g.append(f"[vb]format=yuv420p,eq=saturation=1.1:contrast=1.05,vignette=angle=0.35:eval=init,fade=t=in:st=0:d=0.6,fade=t=out:st={TOTAL - 1.2:.1f}:d=1.2[v]")
    g.append(f"[2:a]atrim=0:{TOTAL:.1f},asetpts=PTS-STARTPTS,aformat=sample_rates=48000:channel_layouts=stereo,afade=t=in:st=0:d=0.5,"
             f"afade=t=out:st={TOTAL - 2.0:.1f}:d=2.0,loudnorm=I=-14:TP=-1.5:LRA=11[a]")
    script = work / "final_graph.txt"
    script.write_text(";\n".join(g))
    print("Gabung akhir ...", flush=True)
    run(["ffmpeg", "-v", "error", "-y"] + ins + ["-filter_complex_script", str(script), "-map", "[v]", "-map", "[a]", "-t", f"{TOTAL:.1f}",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", a.out])
    print("SIAP:", a.out)


if __name__ == "__main__":
    main()
