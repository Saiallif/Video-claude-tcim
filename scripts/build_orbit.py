#!/usr/bin/env python3
"""Video penghargaan versi 2 "Orbit": gambar kenangan mengorbit potret Ustaz Dr. Mohd Nazri Bin Mat Yasin.

1920x1080 / 30fps / 75s. Dua gelang orbit 3D (perspektif sebenar, kedalaman, bayang) berputar berlawanan arah
mengelilingi medalion potret ustaz. Muzik: assets/audio/muzik_orbit.wav (make_music2.py).

Guna:
    python3 scripts/build_orbit.py --media <folder media> [--work <folder kerja>] [--preview 30]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tribute import (F_SANS_M, F_SANS_SB, F_SERIF, LOGO_FULL, LOGO_PLAIN, PORTRAIT, W, H, FPS,  # noqa: E402
                           logo_plate, night_bg, ornament, place, text_layer, run)

ROOT = Path(__file__).resolve().parent.parent
TOTAL = 75.0
BPM = 90
BEAT = 60 / BPM
CX, CY = W // 2, 585          # pusat orbit (gelang)
MX, MY = W // 2, 455          # medalion potret (lebih tinggi supaya tidak terlindung)
MED_D = 440                   # diameter medalion

OUTER = ["IMG-20261114-WA0057.jpg", "IMG-20261114-WA0073.jpg", "IMG-20261114-WA0014.jpg", "IMG-20261114-WA0033.jpg",
         "IMG-20261114-WA0086.jpg", "IMG-20261114-WA0075.jpg", "IMG-20261114-WA0036.jpg", "IMG-20261114-WA0090.jpg",
         "IMG-20261114-WA0078.jpg", "IMG-20261114-WA0066.jpg", "IMG-20261114-WA0092.jpg", "IMG-20261114-WA0068.jpg",
         "IMG-20261114-WA0082.jpg", "IMG-20261114-WA0044.jpg"]
INNER = ["IMG-20261114-WA0095.jpg", "IMG-20261114-WA0031.jpg", "IMG-20261114-WA0077.jpg", "IMG-20261114-WA0088.jpg",
         "IMG-20261114-WA0062.jpg", "IMG-20261114-WA0069.jpg", "IMG-20261114-WA0076.jpg", "IMG-20261114-WA0065.jpg",
         "VID-20261114-WA0071.mp4@3", "VID-20261114-WA0085.mp4@4"]
RINGS = [
    dict(name="outer", files=OUTER, R=740, y=0, w=300, dir=+1, speed=1.0, t_in=7.5),
    dict(name="inner", files=INNER, R=450, y=20, w=225, dir=-1, speed=1.35, t_in=10.0),
]


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease_out_back(x):
    x = float(np.clip(x, 0, 1))
    c1 = 1.5
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


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
    """Kad gambar: bingkai emas, sudut bulat; pulang BGRA tolak-alfa (premultiplied)."""
    bw, bh = base_w, int(base_w * box_ratio)
    sc = min(bw / im.width, bh / im.height)
    iw, ih = int(im.width * sc), int(im.height * sc)
    im = im.resize((iw, ih), Image.LANCZOS)
    b = 7
    card = Image.new("RGBA", (iw + 2 * b, ih + 2 * b), (0, 0, 0, 0))
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, card.width - 1, card.height - 1), 14, fill=255)
    gold = Image.new("RGBA", card.size, (232, 190, 98, 255))
    card.paste(gold, (0, 0), mask)
    inner = Image.new("L", card.size, 0)
    ImageDraw.Draw(inner).rounded_rectangle((b, b, card.width - 1 - b, card.height - 1 - b), 9, fill=255)
    card.paste(im, (b, b), inner.crop((b, b, b + iw, b + ih)))
    a = np.asarray(card).astype(np.float32)
    rgb, al = a[..., :3] * (a[..., 3:4] / 255.0), a[..., 3:4]
    bgra = np.concatenate([rgb[..., ::-1], al], -1)
    return np.clip(bgra, 0, 255).astype(np.uint8)


def make_medallion(media, d):
    im = Image.open(media / PORTRAIT).convert("RGB").crop((140, 95, 860, 815)).resize((d, d), Image.LANCZOS)
    ring = 9
    sz = d + 2 * ring
    out = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    m1 = Image.new("L", (sz * 3, sz * 3), 0)
    ImageDraw.Draw(m1).ellipse((0, 0, sz * 3 - 1, sz * 3 - 1), fill=255)
    m1 = m1.resize((sz, sz), Image.LANCZOS)
    out.paste(Image.new("RGBA", (sz, sz), (236, 196, 108, 255)), (0, 0), m1)
    m2 = Image.new("L", (sz * 3, sz * 3), 0)
    ImageDraw.Draw(m2).ellipse((ring * 3, ring * 3, (sz - ring) * 3 - 1, (sz - ring) * 3 - 1), fill=255)
    m2 = m2.resize((sz, sz), Image.LANCZOS)
    full = Image.new("RGB", (sz, sz), (255, 255, 255))
    full.paste(im, (ring, ring))
    out.paste(full, (0, 0), m2)
    a = np.asarray(out).astype(np.float32)
    rgb, al = a[..., :3] * (a[..., 3:4] / 255.0), a[..., 3:4]
    return np.clip(np.concatenate([rgb[..., ::-1], al], -1), 0, 255).astype(np.uint8)


def glow_patch(size=1500, color=(255, 205, 120)):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 2)
    g = np.exp(-(r ** 2) * 5.0)
    return (g[..., None] * np.array(color[::-1], dtype=np.float32)[None, None, :]).astype(np.float32)


def background():
    bg = night_bg((MX, MY), 0.10).convert("RGB")
    arr = np.asarray(bg)[..., ::-1].copy()
    return arr


# ---------------------------------------------------------------- geometri 3D
class Cam:
    def __init__(self, t):
        self.phi = 0.30 + 0.05 * np.sin(0.35 * t)
        self.D = 2300 - 240 * smooth(t / TOTAL)
        self.cp, self.sp = np.cos(self.phi), np.sin(self.phi)

    def proj(self, p):
        x, y, z = p
        yp = y * self.cp + z * self.sp
        zp = -y * self.sp + z * self.cp
        s = self.D / (self.D + zp)
        return CX + x * s, CY - yp * s, zp, s


def draw_premult(frame, card, quad, shade, alpha_mul):
    """Warp kad (BGRA tolak-alfa) ke quad [TL,TR,BR,BL] dan campur ke frame."""
    q = np.asarray(quad, np.float32)
    x0, y0 = int(np.floor(q[:, 0].min())), int(np.floor(q[:, 1].min()))
    x1, y1 = int(np.ceil(q[:, 0].max())), int(np.ceil(q[:, 1].max()))
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4 or bw > 1700 or bh > 1300 or x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
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
    out = wr[..., :3] * shade * alpha_mul + roi * (1 - a)
    frame[iy0:iy1, ix0:ix1] = np.clip(out, 0, 255).astype(np.uint8)


def add_glow(frame, glow, cx, cy, k):
    gh = glow.shape[0]
    s = int(gh * (0.55 + 0.45 * min(k, 1.2)))
    if s < 8:
        return
    g = cv2.resize(glow, (s, s), interpolation=cv2.INTER_LINEAR) * k
    x0, y0 = int(cx - s / 2), int(cy - s / 2)
    ix0, iy0, ix1, iy1 = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    roi = frame[iy0:iy1, ix0:ix1].astype(np.float32)
    roi += g[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    frame[iy0:iy1, ix0:ix1] = np.clip(roi, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- adegan
class Scene:
    def __init__(self, media, work):
        self.bg = background()
        self.glow = glow_patch()
        self.med = make_medallion(media, MED_D)
        self.rings = []
        for rd in RINGS:
            cards = [make_card(load_image(media, f, work), 470) for f in rd["files"]]
            self.rings.append(dict(rd, cards=cards))
        # sudut orbit kumulatif (laju berubah mengikut muzik)
        ts = np.arange(0, TOTAL + 1, 1 / FPS)
        om = 0.24 + 0.50 * np.sin(np.pi * np.clip((ts - 6) / 56, 0, 1)) ** 2
        self.Theta = np.cumsum(om) / FPS

    def theta(self, t):
        return float(np.interp(t, np.arange(len(self.Theta)) / FPS, self.Theta))

    def frame(self, t):
        f = self.bg.copy()
        cam = Cam(t)
        pulse = 0.88 + 0.12 * np.sin(2 * np.pi * t / BEAT)
        med_k = ease_out_back((t - 3.6) / 2.4)
        add_glow(f, self.glow, MX, MY, (0.9 + 0.5 * smooth((t - 3.6) / 3)) * pulse * 0.55 if t > 3.4 else 0.0)
        items = []  # (kedalaman, fungsi lukis)
        ext = smooth((t - 58) / 7.5)          # fasa keluar
        for ri, rg in enumerate(self.rings):
            n = len(rg["cards"])
            Th = self.theta(t) * rg["speed"] * rg["dir"]
            rin = smooth((t - rg["t_in"]) / 5.5)
            for i, card in enumerate(rg["cards"]):
                e_in = smooth((t - rg["t_in"] - i * 0.22) / 3.4)
                e_out = smooth((t - 58 - i * 0.18) / 5.5)
                life = e_in * (1 - e_out)
                if life <= 0.003:
                    continue
                rad = rg["R"] * (0.12 + 0.88 * e_in) * (1 - 0.8 * e_out)
                th = 2 * np.pi * i / n + Th + 0.4 * ri
                px, pz = rad * np.cos(th), rad * np.sin(th)
                py = rg["y"]
                tx = -np.sin(th)
                tz = np.cos(th)
                if tx < 0:
                    tx, tz = -tx, -tz
                d = np.array([0.6 * tx + 0.4, 0, 0.6 * tz])
                d /= np.linalg.norm(d)
                hw = card.shape[1] / 470 * rg["w"] / 2 * (0.5 + 0.5 * e_in)
                hh = card.shape[0] / 470 * rg["w"] / 2 * (0.5 + 0.5 * e_in)
                P = np.array([px, py, pz])
                up = np.array([0, 1, 0])
                corners = [P - d * hw + up * hh, P + d * hw + up * hh, P + d * hw - up * hh, P - d * hw - up * hh]
                pr = [cam.proj(c) for c in corners]
                quad = [(p[0], p[1]) for p in pr]
                zc = cam.proj(P)[2]
                depth_f = np.clip(1.0 - 0.52 * (zc + rg["R"]) / (2 * rg["R"] + 1e-6), 0.38, 1.0)
                items.append((zc, ("card", card, quad, depth_f, life)))
        mpos = 3.6
        # medalion (z=0) -- bersaiz mengikut ease
        if t >= mpos:
            ms = med_k * (1 + 0.10 * smooth((t - 56) / 9) + 0.015 * np.sin(2 * np.pi * t / BEAT))
            items.append((0.0, ("med", ms)))
        items.sort(key=lambda x: x[0], reverse=True)   # jauh -> dekat
        # garis orbit
        for rg in self.rings:
            self._orbit_line(f, cam, rg, t, ext)
        for _, it in items:
            if it[0] == "card":
                _, card, quad, df, life = it
                draw_premult(f, card, quad, df, life)
            else:
                ms = it[1]
                sz = int(self.med.shape[0] * ms)
                if sz > 8:
                    m = cv2.resize(self.med, (sz, sz), interpolation=cv2.INTER_AREA if ms < 1 else cv2.INTER_CUBIC)
                    x0, y0 = MX - sz // 2, MY - sz // 2
                    draw_premult(f, m, [(x0, y0), (x0 + sz, y0), (x0 + sz, y0 + sz), (x0, y0 + sz)], 1.0, 1.0)
        return f

    def _orbit_line(self, f, cam, rg, t, ext):
        rin = smooth((t - rg["t_in"]) / 5.0) * (1 - ext)
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
            col = tuple(int(c * zf * rin) for c in (130, 190, 235))   # BGR emas hangat
            cv2.line(ov, (int(x1 * 16), int(y1 * 16)), (int(x2 * 16), int(y2 * 16)), col, 2, cv2.LINE_AA, 4)
        # bintang satelit
        for j in range(3):
            a = self.theta(t) * rg["speed"] * rg["dir"] * 1.6 + j * 2.094 + (0.5 if rg["dir"] < 0 else 0)
            x, y, z, s = cam.proj((R * np.cos(a), rg["y"], R * np.sin(a)))
            zf = np.clip(1.0 - 0.6 * (z + R) / (2 * R), 0.35, 1.0)
            cv2.circle(ov, (int(x * 16), int(y * 16)), int(6 * 16 * s ** 0.5), (int(250 * zf), int(235 * zf), int(190 * zf)), -1, cv2.LINE_AA, 4)
            cv2.circle(ov, (int(x * 16), int(y * 16)), int(15 * 16 * s ** 0.5), (int(90 * zf * rin), int(130 * zf * rin), int(170 * zf * rin)), -1, cv2.LINE_AA, 4)
        ov = cv2.GaussianBlur(ov, (0, 0), 1.2)
        f[:] = np.clip(f.astype(np.int16) + ov.astype(np.int16), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- lapisan teks dan komposisi akhir
def gradient_png(path):
    y = np.arange(H, dtype=np.float32)
    a = np.clip((300 - y) / 300, 0, 1) ** 1.3 * 0.80
    arr = np.zeros((H, W, 4), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 4, 9, 24
    arr[..., 3] = (a[:, None] * 255).astype(np.uint8)
    Image.fromarray(arr, "RGBA").save(path)


def layers(work):
    gold = (245, 215, 140)
    L = []  # (png, mula, tamat, fade_in, fade_out)

    def add(img, name, t0, t1, fi=0.8, fo=0.8):
        p = work / f"olay_{name}.png"
        img.save(p)
        L.append((p, t0, t1, fi, fo))

    add(place(logo_plate(LOGO_PLAIN, 380), W // 2, 470), "logo1", 0.2, 3.8, 0.7, 0.7)
    add(text_layer("DENGAN PENUH PENGHARGAAN", F_SANS_SB, 34, gold, tracking=14, glow=False, cy=810), "t0", 0.9, 3.8, 0.8, 0.6)
    add(text_layer("Ustaz Dr. Mohd Nazri Bin Mat Yasin", F_SERIF, 66, "gold", cy=110), "name", 5.8, 14.4, 1.0, 0.9)
    add(text_layer("PMDIUC UKM BERANANG   •   SEJAK JANUARI 2026", F_SANS_SB, 28, "white", tracking=7, glow=False, cy=185), "sub", 6.8, 14.4, 1.0, 0.9)
    caps = [(15.0, 22.2, "Dari pendaftaran yang pertama…"),
            (22.6, 30.2, "…hingga ilmu yang dicurahkan dengan penuh sabar"),
            (30.6, 38.4, "Kesabaran dan keikhlasan Ustaz membentuk kami"),
            (38.8, 47.0, "Setiap kenangan bersama Ustaz, kami simpan selamanya"),
            (47.4, 57.2, "Terima kasih atas ilmu, bimbingan dan kesabaran")]
    for i, (a, b, s) in enumerate(caps):
        add(text_layer(s, F_SERIF, 56, "white", glow=False, cy=140), f"cap{i}", a, b, 0.9, 0.8)
    add(text_layer("Semoga Allah membalas jasa Ustaz dan", F_SERIF, 52, "white", glow=False, cy=110), "cap5a", 57.8, 66.2, 1.0, 0.8)
    add(text_layer("memudahkan urusan di tempat yang baharu", F_SERIF, 52, "gold", glow=True, cy=185), "cap5b", 58.6, 66.2, 1.0, 0.8)
    # kad penutup
    end_bg = work / "olay_endbg.png"
    night_bg((W // 2, 430), 0.28).save(end_bg)
    L.append((end_bg, 66.0, TOTAL, 1.4, 0.01))
    add(place(logo_plate(LOGO_FULL, 560, pad=30), W // 2, 400), "end_logo", 67.2, TOTAL, 1.0, 0.01)
    add(text_layer("DENGAN PENUH PENGHARGAAN DARIPADA", F_SANS_SB, 28, gold, tracking=10, glow=False, cy=790), "end1", 68.2, TOTAL, 1.0, 0.01)
    add(text_layer("PELAJAR PJJ PMDIUC UKM BERANANG", F_SANS_SB, 40, "white", tracking=6, glow=False, cy=850), "end2", 68.8, TOTAL, 1.0, 0.01)
    add(text_layer("Ustaz Dr. Mohd Nazri Bin Mat Yasin", F_SERIF, 56, "gold", cy=950), "end3", 69.8, TOTAL, 1.0, 0.01)
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--work", default="/tmp/orbit_work")
    ap.add_argument("--out", default=str(ROOT / "output/Penghargaan_Ustaz_Orbit_v2.mp4"))
    ap.add_argument("--particles", default="")
    ap.add_argument("--preview", type=float, default=None)
    ap.add_argument("--skip-base", action="store_true")
    a = ap.parse_args()
    media, work = Path(a.media), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    sc = Scene(media, work)
    if a.preview is not None:
        f = sc.frame(a.preview)
        cv2.imwrite(str(work / f"preview_{a.preview:.1f}.png"), f)
        print("preview", work / f"preview_{a.preview:.1f}.png")
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
    ins = ["-i", str(base), "-stream_loop", "-1", "-i", str(part), "-i", str(ROOT / "assets/audio/muzik_orbit.wav"),
           "-loop", "1", "-framerate", str(FPS), "-t", f"{TOTAL:.1f}", "-i", str(grad)]
    g = []
    g.append("[3:v]format=rgba,fade=t=in:st=5.0:d=1.0:alpha=1,fade=t=out:st=65.4:d=0.8:alpha=1[gr]")
    g.append("[0:v][gr]overlay=format=auto:shortest=1[v0]")
    last = "v0"
    for j, (png, t0, t1, fi, fo) in enumerate(L):
        idx = 4 + j
        dur = t1 - t0
        ins += ["-loop", "1", "-framerate", str(FPS), "-t", f"{dur + 0.1:.2f}", "-i", str(png)]
        g.append(f"[{idx}:v]format=rgba,fade=t=in:st=0:d={fi}:alpha=1,fade=t=out:st={max(0, dur - fo):.2f}:d={fo}:alpha=1,setpts=PTS+{t0}/TB[L{j}]")
        g.append(f"[{last}][L{j}]overlay=format=auto:eof_action=pass[o{j}]")
        last = f"o{j}"
    g.append(f"[{last}]format=gbrp[vv]")
    g.append(f"[1:v]trim=duration={TOTAL:.1f},setpts=PTS-STARTPTS,fade=t=in:st=0:d=1.5,format=gbrp[pp]")
    g.append("[vv][pp]blend=all_mode=screen:all_opacity=0.6[vb]")
    g.append(f"[vb]format=yuv420p,eq=saturation=1.05:contrast=1.05,vignette=angle=0.38:eval=init,fade=t=in:st=0:d=0.8,fade=t=out:st={TOTAL - 1.5:.1f}:d=1.5[v]")
    g.append(f"[2:a]atrim=0:{TOTAL:.1f},asetpts=PTS-STARTPTS,aformat=sample_rates=48000:channel_layouts=stereo,afade=t=in:st=0:d=0.6,"
             f"afade=t=out:st={TOTAL - 2.5:.1f}:d=2.5,loudnorm=I=-14:TP=-1.5:LRA=11[a]")
    script = work / "final_graph.txt"
    script.write_text(";\n".join(g))
    print("Gabung akhir ...", flush=True)
    run(["ffmpeg", "-v", "error", "-y"] + ins + ["-filter_complex_script", str(script), "-map", "[v]", "-map", "[a]", "-t", f"{TOTAL:.1f}",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", a.out])
    print("SIAP:", a.out)


if __name__ == "__main__":
    main()
