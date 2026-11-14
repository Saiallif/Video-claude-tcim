#!/usr/bin/env python3
"""Video penghargaan: Ustaz Dr. Mohd Nazri Bin Mat Yasin (PMDIUC UKM Beranang).

1920x1080 / 30fps / 113s. Lagu "Ma Salama fi Amani Ya Syaikhona" dengan lirik Arab + terjemahan Melayu
(masa dalam docs/lirik_masalama.json). Rekaan syahdu: biru malam + emas, Ken Burns perlahan, zarah emas.

Edit senarai SCENES (di bawah) untuk menukar gambar/video atau susunan kenangan.

Guna:
    python3 scripts/build_tribute.py --media <folder media> [--work <folder kerja>] [--out output/xxx.mp4]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_video import (F_SANS_M, F_SANS_SB, F_SERIF, GOLD, H, W, FPS, _grad, make_particles,  # noqa: E402
                         ornament, run, text_layer)

ROOT = Path(__file__).resolve().parent.parent
F_AR = "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf"
TX = 0.9           # tempoh peralihan (saat), lembut dan syahdu
TOTAL = 113.0
SONG = ROOT / "assets/audio/maasalama_zikir_113s.wav"
LYR = json.load(open(ROOT / "docs/lirik_masalama.json", encoding="utf-8"))
# Masa sari kata dalam klip rujukan lewat ~0.4s berbanding vokal (disemak dengan jeda senyap pada audio): majukan sedikit
LYR["urutan"] = [[k, max(0.0, t0 - 0.4), t1 - 0.15] for k, t0, t1 in LYR["urutan"]]
NAMA = "Ustaz Dr. Mohd Nazri Bin Mat Yasin"
PORTRAIT = "file_000000003e508211b80101a9e811f085.png"   # gambar potret ustaz (dalam folder Drive)
LOGO_PLAIN = ROOT / "assets/images/mutawwif_logo_plain.png"
LOGO_FULL = ROOT / "assets/images/logo_pmdiuc.png"      # logo + Diploma ... UKM-Beranang

# ---------------------------------------------------------------- SENARAI ADEGAN
# kind: open | hero | still | video | hero2 | end ; t0 = masa mula adegan (saat); ss = mula dalam video
SCENES = [
    dict(kind="open", t0=0.0),
    dict(kind="hero", t0=3.5),
    dict(kind="still", file="IMG-20261114-WA0057.jpg", t0=21.5),
    dict(kind="video", file="VID-20261114-WA0083.mp4", t0=25.5, ss=1.5),
    dict(kind="still", file="IMG-20261114-WA0073.jpg", t0=29.0),
    dict(kind="still", file="IMG-20261114-WA0075.jpg", t0=34.0),
    dict(kind="still", file="IMG-20261114-WA0014.jpg", t0=39.25),
    dict(kind="still", file="IMG-20261114-WA0031.jpg", t0=44.0),
    dict(kind="still", file="IMG-20261114-WA0044.jpg", t0=48.75),
    dict(kind="video", file="VID-20261114-WA0072.mp4", t0=53.0, ss=2.0),
    dict(kind="still", file="IMG-20261114-WA0086.jpg", t0=57.5),
    dict(kind="still", file="IMG-20261114-WA0066.jpg", t0=62.0),
    dict(kind="still", file="IMG-20261114-WA0095.jpg", t0=66.5),
    dict(kind="video", file="VID-20261114-WA0080.mp4", t0=70.5, ss=3.0),
    dict(kind="still", file="IMG-20261114-WA0090.jpg", t0=75.25),
    dict(kind="still", file="IMG-20261114-WA0068.jpg", t0=79.0),
    dict(kind="video", file="VID-20261114-WA0071.mp4", t0=82.5, ss=2.0),
    dict(kind="still", file="IMG-20261114-WA0078.jpg", t0=87.0),
    dict(kind="still", file="IMG-20261114-WA0062.jpg", t0=90.5),
    dict(kind="still", file="IMG-20261114-WA0088.jpg", t0=93.25),
    dict(kind="still", file="IMG-20261114-WA0076.jpg", t0=97.0),
    dict(kind="hero2", t0=100.25),
    dict(kind="end", t0=108.0),
]
TRANS = {"hero": "fadeblack", "hero2": "fadeblack", "end": "fadeblack"}  # lain-lain: fade


# ---------------------------------------------------------------- bantuan lukisan
def _compose(mask, fill, glow=True):
    """Gabung topeng teks -> RGBA dengan gradien emas / putih + bayang + cahaya."""
    layer = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    if fill == "gold":
        g = _grad(mask.size[1], GOLD)
        col = Image.fromarray(np.repeat(g[:, None, :], mask.size[0], axis=1).astype(np.uint8), "RGB")
    else:
        col = Image.new("RGB", mask.size, fill)
    layer.paste(col, (0, 0), mask)
    shadow = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    shadow.paste(Image.new("RGBA", mask.size, (0, 0, 0, 210)), (0, 5), mask)
    shadow = shadow.filter(ImageFilter.GaussianBlur(6))
    out = Image.alpha_composite(Image.new("RGBA", mask.size, (0, 0, 0, 0)), shadow)
    if glow and fill == "gold":
        gl = Image.new("RGBA", mask.size, (255, 200, 90, 0))
        gl.putalpha(mask.filter(ImageFilter.GaussianBlur(16)).point(lambda v: int(v * 0.6)))
        out = Image.alpha_composite(out, gl)
    return Image.alpha_composite(out, layer)


def arabic_mask(text, size, max_w):
    while True:
        fnt = ImageFont.truetype(F_AR, size, layout_engine=ImageFont.Layout.RAQM)
        l, t, r, b = fnt.getbbox(text, direction="rtl", language="ar")
        if r - l <= max_w or size < 40:
            break
        size -= 4
    pad = 70
    mask = Image.new("L", (r - l + pad * 2, b - t + pad * 2), 0)
    ImageDraw.Draw(mask).text((pad - l, pad - t), text, font=fnt, fill=255, direction="rtl", language="ar")
    return mask


def lyric_png(key, path):
    ar, ms = LYR["baris"][key]["ar"], LYR["baris"][key]["ms"]
    cv = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    a = _compose(arabic_mask(ar, 96, 1500), "gold")
    cv.alpha_composite(a, (W // 2 - a.width // 2, 880 - a.height // 2))
    cv = Image.alpha_composite(cv, text_layer(ms, F_SANS_SB, 42, "white", glow=False, cy=1000))
    cv.save(path)


def navy_gradient(path):
    """Kecerunan gelap di bawah supaya lirik jelas dibaca."""
    y = np.arange(H, dtype=np.float32)
    a = np.clip((y - 600) / 480, 0, 1) ** 1.4 * 0.86
    arr = np.zeros((H, W, 4), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 4, 9, 24
    arr[..., 3] = (a[:, None] * 255).astype(np.uint8)
    Image.fromarray(arr, "RGBA").save(path)


def night_bg(glow_xy=(560, 440), glow=0.20):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = yy / H
    base = np.stack([6 + 10 * t, 13 + 20 * t, 32 + 36 * t], -1)
    d = np.sqrt(((xx - glow_xy[0]) / 900) ** 2 + ((yy - glow_xy[1]) / 620) ** 2)
    g = np.exp(-d ** 2 * 2.2) * glow
    base += g[..., None] * np.array([255, 200, 110]) * 0.55
    v = 1 - 0.55 * np.clip(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2, 0, 1.4) / 1.4
    base *= v[..., None]
    rg = np.random.default_rng(3)
    base += rg.normal(0, 1.2, base.shape)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


def logo_plate(logo_path, width, pad=36, alpha=246):
    im = Image.open(logo_path).convert("RGBA")
    bb = im.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    if bb:
        im = im.crop(bb)
    sc = width / im.width
    im = im.resize((width, int(im.height * sc)), Image.LANCZOS)
    plate = Image.new("RGBA", (im.width + pad * 2, im.height + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(plate).rounded_rectangle((0, 0, plate.width - 1, plate.height - 1), 34, fill=(250, 250, 252, alpha))
    plate.alpha_composite(im, (pad, pad))
    glow = Image.new("RGBA", plate.size, (255, 210, 120, 0))
    glow.putalpha(plate.getchannel("A").filter(ImageFilter.GaussianBlur(26)).point(lambda v: int(v * 0.45)))
    out = Image.new("RGBA", (plate.width + 120, plate.height + 120), (0, 0, 0, 0))
    out.alpha_composite(glow, (60, 60))
    out.alpha_composite(plate, (60, 60))
    return out


def place(img, cx, cy):
    cv = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cv.alpha_composite(img, (cx - img.width // 2, cy - img.height // 2))
    return cv


def portrait_card(path, height):
    im = Image.open(path).convert("RGB")
    sc = height / im.height
    im = im.resize((int(im.width * sc), height), Image.LANCZOS)
    b = 10
    card = Image.new("RGBA", (im.width + 2 * b, im.height + 2 * b), (232, 190, 98, 255))
    card.paste(im, (b, b))
    sh = Image.new("RGBA", (card.width + 160, card.height + 160), (0, 0, 0, 0))
    sh.paste(Image.new("RGBA", card.size, (0, 0, 0, 190)), (80, 100))
    sh = sh.filter(ImageFilter.GaussianBlur(34))
    sh.alpha_composite(card, (80, 80))
    return sh


# ---------------------------------------------------------------- adegan khas
def hero_composite(media, path, cx, closing=False):
    bg = night_bg((cx, 430), 0.30)
    card = portrait_card(media / PORTRAIT, 740)
    bg.paste(card, (cx - card.width // 2, 30 - 80 + 14), card)
    bg.convert("RGB").save(path)


def layers_for(kind, work):
    """Pulangkan senarai (png, mula_relatif, kekal) untuk adegan teks."""
    L = []
    gold = (245, 215, 140)
    if kind == "open":
        L += [(place(logo_plate(LOGO_PLAIN, 380), W // 2, 420), 0.2, False)]
        L += [(text_layer("DENGAN PENUH PENGHARGAAN", F_SANS_SB, 34, gold, tracking=14, glow=False, cy=840), 1.2, False)]
    elif kind == "hero":
        L += [(text_layer("TERIMA KASIH, USTAZ", F_SANS_SB, 34, gold, tracking=14, glow=False, cy=210, cx=1330), 1.0, False),
              (text_layer("Ustaz Dr. Mohd Nazri", F_SERIF, 84, "white", glow=False, cy=315, cx=1330), 2.2, False),
              (text_layer("Bin Mat Yasin", F_SERIF, 84, "gold", cy=420, cx=1330), 3.2, False),
              (ornament(500), 4.4, False),
              (text_layer("Yang membimbing kami dan menguruskan", F_SANS_M, 36, "white", glow=False, cy=570, cx=1330), 5.4, False),
              (text_layer("pendaftaran serta pembelajaran kami", F_SANS_M, 36, "white", glow=False, cy=620, cx=1330), 6.4, False),
              (text_layer("PMDIUC UKM BERANANG   •   SEJAK JANUARI 2026", F_SANS_SB, 27, gold, tracking=6, glow=False, cy=700, cx=1330), 8.0, False)]
        o = ornament(500)
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sh.alpha_composite(o, (1330 - W // 2, 0))
        L[3] = (sh, 4.4, False)
    elif kind == "hero2":
        L += [(text_layer("Terima kasih atas ilmu, bimbingan dan kesabaran Ustaz", F_SANS_SB, 40, "white", glow=False, cy=215, cx=1330), 0.8, False),
              (text_layer("Semoga Allah membalas jasa Ustaz dan", F_SANS_M, 36, (235, 235, 245), glow=False, cy=300, cx=1330), 2.0, False),
              (text_layer("memudahkan urusan di tempat yang baharu", F_SANS_M, 36, (235, 235, 245), glow=False, cy=350, cx=1330), 2.6, False),
              (text_layer("Ustaz Dr. Mohd Nazri Bin Mat Yasin", F_SERIF, 58, "gold", cy=470, cx=1330), 3.6, False),
              (text_layer("— Pelajar PJJ PMDIUC UKM Beranang", F_SANS_SB, 30, gold, tracking=3, glow=False, cy=560, cx=1330), 4.8, False)]
    elif kind == "end":
        L += [(place(logo_plate(LOGO_FULL, 560, pad=30), W // 2, 430), 0.6, False),
              (text_layer("DENGAN PENUH PENGHARGAAN DARIPADA", F_SANS_SB, 28, gold, tracking=10, glow=False, cy=830), 1.6, False),
              (text_layer("PELAJAR PJJ PMDIUC UKM BERANANG", F_SANS_SB, 40, "white", tracking=6, glow=False, cy=890), 2.2, False)]
    return L


def probe_dims(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:stream_side_data=rotation",
                        "-of", "json", str(path)], capture_output=True, text=True)
    j = json.loads(r.stdout)["streams"][0]
    w, h = j["width"], j["height"]
    rot = abs(int(j.get("side_data_list", [{}])[0].get("rotation", 0))) if j.get("side_data_list") else 0
    return (h, w) if rot in (90, 270) else (w, h)


BW, BH, FY = 1560, 740, 44


def build_segment(idx, sc, tm, work, media):
    kind = sc["kind"]
    start, d, lead, trail, sp = tm
    N = int(round(d * FPS))
    out = work / f"seg_{idx:02d}_{kind}.mp4"
    if out.exists():
        return out
    enc = ["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an", str(out)]
    z_in = f"1+0.07*on/{N}"
    z_out = f"1.07-0.07*on/{N}"
    zexpr = z_out if idx % 2 == 0 else z_in
    inputs, graph = [], []
    if kind in ("open", "hero", "hero2", "end"):
        if kind == "hero" or kind == "hero2":
            comp = work / f"comp_{kind}.png"
            hero_composite(media, comp, 520)
        else:
            comp = work / f"comp_{kind}.png"
            night_bg((W // 2, 420), 0.32).convert("RGB").save(comp)
        inputs = ["-i", str(comp)]
        zz = f"1.05-0.05*on/{N}" if kind == "hero2" else f"1+0.05*on/{N}"
        graph = [f"[0:v]scale={W*2}:{H*2}:flags=lanczos,zoompan=z='{zz}':"
                 f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={N}:s={W}x{H}:fps={FPS},format=yuv420p[base]"]
        n_in = 1
        last = "base"
        for j, (img, t0, _keep) in enumerate(layers_for(kind, work)):
            p = work / f"lay_{kind}_{j}.png"
            img.save(p)
            inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{d:.3f}", "-i", str(p)]
            t_in = lead + t0
            graph.append(f"[{n_in}:v]format=rgba,fade=t=in:st={t_in:.3f}:d=0.9:alpha=1[L{j}]")
            graph.append(f"[{last}][L{j}]overlay=x=0:y='if(lt(t-{t_in:.3f},0.9),26*pow(1-(t-{t_in:.3f})/0.9,2),0)':format=auto:shortest=0[O{j}]")
            last = f"O{j}"
            n_in += 1
        graph.append(f"[{last}]format=yuv420p[vout]")
    else:
        f = media / sc["file"]
        if kind == "still":
            iw, ih = Image.open(f).size
            if iw / ih >= 1.7:  # penuh skrin
                inputs = ["-i", str(f)]
                graph = [f"[0:v]scale={W*2}:{H*2}:force_original_aspect_ratio=increase:flags=lanczos,crop={W*2}:{H*2},"
                         f"zoompan=z='{zexpr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={N}:s={W}x{H}:fps={FPS},format=yuv420p[vout]"]
            else:
                scl = min(BW / iw, BH / ih)
                fw, fh = int(iw * scl) // 2 * 2, int(ih * scl) // 2 * 2
                inputs = ["-loop", "1", "-framerate", str(FPS), "-t", f"{d:.3f}", "-i", str(f), "-i", str(f)]
                fx, fy = (W - fw - 16) // 2, FY + (BH + 16 - fh - 16) // 2
                graph = [
                    f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},gblur=sigma=40,eq=brightness=-0.22:saturation=1.05,format=rgba[bg]",
                    f"[1:v]scale={fw*3}:{fh*3}:flags=lanczos,zoompan=z='{zexpr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={N}:s={fw}x{fh}:fps={FPS},"
                    f"pad={fw+16}:{fh+16}:8:8:color=0xE8BE62,format=rgba[fg]",
                    "[fg]split=2[fga][fgb]",
                    "[fgb]colorchannelmixer=rr=0:gg=0:bb=0:aa=0.6,gblur=sigma=24[shadow]",
                    f"[bg][shadow]overlay=x={fx+14}:y={fy+22}:shortest=0[bs]",
                    f"[bs][fga]overlay=x={fx}:y={fy}:shortest=0,format=yuv420p[vout]",
                ]
        else:  # video
            vw, vh = probe_dims(f)
            scl = min(BW / vw, BH / vh)
            fw, fh = int(vw * scl) // 2 * 2, int(vh * scl) // 2 * 2
            speed = 0.8
            ss = sc.get("ss", 0)
            inputs = ["-ss", str(ss), "-t", f"{d / speed + 0.5:.3f}", "-i", str(f)]
            fx, fy = (W - fw - 16) // 2, FY + (BH - fh) // 2
            graph = [
                f"[0:v]setpts=PTS/{speed},fps={FPS},split=2[va][vb]",
                f"[va]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},gblur=sigma=40,eq=brightness=-0.22:saturation=1.05,format=rgba[bg]",
                f"[vb]scale={fw}:{fh}:flags=lanczos,pad={fw+16}:{fh+16}:8:8:color=0xE8BE62,format=rgba,split=2[fga][fgb]",
                "[fgb]colorchannelmixer=rr=0:gg=0:bb=0:aa=0.6,gblur=sigma=24[shadow]",
                f"[bg][shadow]overlay=x={fx+14}:y={fy+22}:shortest=0[bs]",
                f"[bs][fga]overlay=x={fx}:y={fy}:shortest=0,format=yuv420p[vout]",
            ]
    fs = work / f"fg_{idx:02d}.txt"
    fs.write_text(";\n".join(graph))
    run(["ffmpeg", "-v", "error", "-y"] + inputs + ["-filter_complex_script", str(fs), "-map", "[vout]", "-t", f"{d:.3f}"] + enc)
    return out


def timings():
    out = []
    n = len(SCENES)
    for k, sc in enumerate(SCENES):
        nxt = SCENES[k + 1]["t0"] if k < n - 1 else TOTAL
        sp = nxt - sc["t0"]
        lead = TX / 2 if k > 0 else 0.0
        trail = TX / 2 if k < n - 1 else 0.0
        out.append((sc["t0"] - lead, lead + sp + trail, lead, trail, sp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--work", default="/tmp/tribute_work")
    ap.add_argument("--out", default=str(ROOT / "output/Penghargaan_Ustaz_Dr_Mohd_Nazri_PMDIUC.mp4"))
    ap.add_argument("--particles", default="")
    ap.add_argument("--only-segments", action="store_true")
    a = ap.parse_args()
    media, work = Path(a.media), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    part = work / "particles.mp4"
    if a.particles and not part.exists():
        part.write_bytes(Path(a.particles).read_bytes())
    if not part.exists():
        make_particles(part)
    tms = timings()
    segs = []
    for i, (sc, tm) in enumerate(zip(SCENES, tms)):
        print(f"Adegan {i:02d} {sc['kind']:6s} {sc.get('file','')[:26]:26s} mula={tm[0]:7.2f} tempoh={tm[1]:5.2f}", flush=True)
        segs.append(build_segment(i, sc, tm, work, media))
    if a.only_segments:
        return

    grad = work / "grad.png"
    navy_gradient(grad)
    lyr = {}
    for k in LYR["baris"]:
        lyr[k] = work / f"lyr_{k}.png"
        lyric_png(k, lyr[k])

    ins = []
    for s in segs:
        ins += ["-i", str(s)]
    n = len(segs)
    ins += ["-stream_loop", "-1", "-i", str(work / "particles.mp4")]
    ins += ["-i", str(SONG)]
    ins += ["-loop", "1", "-framerate", str(FPS), "-t", f"{TOTAL:.2f}", "-i", str(grad)]
    for key, t0, t1 in LYR["urutan"]:
        ins += ["-loop", "1", "-framerate", str(FPS), "-t", f"{t1 - t0 + 0.2:.3f}", "-i", str(lyr[key])]
    ip, ia, ig = n, n + 1, n + 2
    g = []
    cur = "0:v"
    for k in range(1, n):
        g.append(f"[{cur}][{k}:v]xfade=transition={('fadeblack' if SCENES[k - 1]['kind'] in ('hero', 'hero2', 'open') else TRANS.get(SCENES[k]['kind'], 'fade'))}:duration={TX}:offset={tms[k][0]:.3f}[x{k}]")
        cur = f"x{k}"
    # kecerunan bawah (mula 3.0s) supaya lirik jelas; sembunyi pada kad pembuka & penutup
    g.append(f"[{ig}:v]format=rgba,fade=t=in:st=3.0:d=1.0:alpha=1,fade=t=out:st=107.4:d=0.8:alpha=1[gr]")
    g.append(f"[{cur}][gr]overlay=format=auto:shortest=1[vg]")
    last = "vg"
    for j, (key, t0, t1) in enumerate(LYR["urutan"]):
        idx = ig + 1 + j
        dur = t1 - t0
        g.append(f"[{idx}:v]format=rgba,fade=t=in:st=0:d=0.6:alpha=1,fade=t=out:st={dur - 0.5:.2f}:d=0.5:alpha=1,setpts=PTS+{t0}/TB[ly{j}]")
        g.append(f"[{last}][ly{j}]overlay=format=auto:eof_action=pass[lv{j}]")
        last = f"lv{j}"
    g.append(f"[{last}]format=gbrp[vv]")
    g.append(f"[{ip}:v]trim=duration={TOTAL:.2f},setpts=PTS-STARTPTS,fade=t=in:st=0:d=1.5,eq=brightness=-0.12,format=gbrp[pp]")
    g.append("[vv][pp]blend=all_mode=screen:all_opacity=0.55[vb]")
    g.append(f"[vb]format=yuv420p,eq=saturation=0.82:contrast=1.04:brightness=-0.03,vignette=angle=0.4:eval=init,"
             f"fade=t=in:st=0:d=1.2,fade=t=out:st={TOTAL - 1.8:.2f}:d=1.8[v]")
    g.append(f"[{ia}:a]aformat=sample_rates=48000:channel_layouts=stereo,afade=t=in:st=0:d=1.0,afade=t=out:st={TOTAL - 2.5:.2f}:d=2.5,"
             f"loudnorm=I=-14:TP=-1.5:LRA=11[a]")
    script = work / "final_graph.txt"
    script.write_text(";\n".join(g))
    print("Gabung & enkod akhir ...", flush=True)
    run(["ffmpeg", "-v", "error", "-y"] + ins + ["-filter_complex_script", str(script), "-map", "[v]", "-map", "[a]", "-t", f"{TOTAL:.2f}",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", a.out])
    print("SIAP:", a.out)


if __name__ == "__main__":
    main()
