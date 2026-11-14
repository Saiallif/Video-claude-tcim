#!/usr/bin/env python3
"""Bina video "Tamat Final Exam Sem 1 - PMDIUC PJJ Batch 2" (1920x1080, 30fps, ~70s).

Aliran: intro logo (video asal) -> kad tajuk -> montaj gambar (Ken Burns, kapsyen, zarah emas,
peralihan jatuh pada ketukan 96 BPM) -> gambar kumpulan -> kad penutup.

Guna:
    python3 scripts/make_music.py assets/audio/muzik_pmdiuc.wav     # jana muzik (atau guna lagu sendiri)
    python3 scripts/build_video.py [--music fail_lagu.mp3] [--out output/video.mp4] [--work /tmp/work]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "assets/images"
W, H, FPS = 1920, 1080, 30
BPM = 96
BEAT = 60 / BPM
TX = 0.5  # tempoh peralihan (saat)

F_SERIF = "/usr/share/fonts/truetype/crosextra/Caladea-Bold.ttf"
F_SANS_SB = "/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf"
F_SANS_XB = "/usr/share/fonts/opentype/inter/InterDisplay-ExtraBold.otf"
F_SANS_M = "/usr/share/fonts/opentype/inter/InterDisplay-Medium.otf"

GOLD = [(255, 240, 185), (232, 190, 98), (176, 124, 34)]

# ----------------------------------------------------------------------------- jadual klip
# (id, jenis, gambar, ketukan, kapsyen, sub-kapsyen, peralihan_masuk)
CLIPS = [
    ("intro", "intro", None, 16, None, None, None),
    ("title", "title", "luar_muslimah_bg.jpg", 8, None, None, "fadewhite"),
    ("a", "photo", "kerja_wide.jpg", 8, "Hari Peperiksaan Akhir", "SEMESTER 1  •  UKM BERANANG", "fade"),
    ("b", "pan", "kerja_portrait.jpg", 8, "Fokus. Teliti. Bersungguh-sungguh.", "DIPLOMA KEMAHIRAN PENGURUSAN MUTAWWIF", "smoothleft"),
    ("c", "photo", "bincang_1.jpg", 6, "Berbincang & Saling Membantu", "KERJASAMA DI ATAS SEGALANYA", "zoomin"),
    ("d", "photo", "bincang_2.jpg", 6, "Ukhuwah Sepanjang Perjuangan", "SATU HALA, SATU MATLAMAT", "slideleft"),
    ("e", "photo", "kerja_3.jpg", 6, "Setiap Usaha Ada Hasilnya", "PERNAH PENAT, KINI BERBALOI", "fade"),
    ("f", "photo", "selfie.jpg", 6, "Senyum Selepas Penat", "ALHAMDULILLAH", "smoothright"),
    ("g", "photo", "luar_campuran.jpg", 6, "Sahabat Seperjuangan", "PMDIUC PJJ BATCH 2", "dissolve"),
    ("h", "photo", "luar_muslimah.jpg", 6, "Muslimah Hebat PMDIUC", "BERILMU, BERAKHLAK, BERDEDIKASI", "wipeup"),
    ("i", "photo", "kumpulan_lelaki.jpg", 6, "Mutawwif Muda Harapan Ummah", "MENUJU TANAH SUCI", "circleopen"),
    ("j", "photo", "kumpulan_muslimah.jpg", 6, "Mutawwifah Berjiwa Besar", "BERSAMA MEMBINA MASA DEPAN", "slideright"),
    ("fin", "photo", "kumpulan_semua.jpg", 8, "PJJ BATCH 2  •  SATU KELUARGA", "TAHNIAH SEMUA!", "fadewhite"),
    ("end", "end", None, 16, None, None, "fade"),
]


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print("PERINTAH GAGAL:", " ".join(map(str, cmd))[:400], file=sys.stderr)
        print(r.stderr[-2500:], file=sys.stderr)
        sys.exit(1)
    return r


# ----------------------------------------------------------------------------- teks PNG
def _grad(h, stops):
    arr = np.zeros((h, 3), dtype=np.float32)
    n = len(stops) - 1
    for i in range(h):
        p = i / max(1, h - 1) * n
        k = min(int(p), n - 1)
        f = p - k
        arr[i] = np.array(stops[k]) * (1 - f) + np.array(stops[k + 1]) * f
    return arr


def text_layer(text, font, size, fill="gold", tracking=0, glow=True, canvas=(W, H), cy=None, cx=None):
    """Lukis teks (tengah) pada kanvas RGBA penuh. fill: 'gold' | 'white' | (r,g,b)."""
    fnt = ImageFont.truetype(font, size)
    # ukur dengan jarak huruf
    widths = [fnt.getlength(ch) + tracking for ch in text]
    tw = int(sum(widths) - tracking)
    asc, desc = fnt.getmetrics()
    th = asc + desc
    pad = 60
    mask = Image.new("L", (tw + pad * 2, th + pad * 2), 0)
    d = ImageDraw.Draw(mask)
    x = pad
    for ch, wd in zip(text, widths):
        d.text((x, pad), ch, font=fnt, fill=255)
        x += wd
    layer = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    if fill == "gold":
        g = _grad(mask.size[1], GOLD)
        col = Image.fromarray(np.repeat(g[:, None, :], mask.size[0], axis=1).astype(np.uint8), "RGB")
    elif fill == "white":
        col = Image.new("RGB", mask.size, (255, 255, 255))
    else:
        col = Image.new("RGB", mask.size, fill)
    layer.paste(col, (0, 0), mask)
    # bayang + cahaya
    shadow = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    sh = Image.new("RGBA", mask.size, (0, 0, 0, 200))
    shadow.paste(sh, (0, 6), mask)
    shadow = shadow.filter(ImageFilter.GaussianBlur(6))
    out = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    out = Image.alpha_composite(out, shadow)
    if glow and fill == "gold":
        gl = Image.new("RGBA", mask.size, (255, 200, 90, 0))
        gl.putalpha(mask.filter(ImageFilter.GaussianBlur(14)).point(lambda v: int(v * 0.55)))
        out = Image.alpha_composite(out, gl)
    out = Image.alpha_composite(out, layer)
    cv = Image.new("RGBA", canvas, (0, 0, 0, 0))
    px = (canvas[0] // 2 if cx is None else cx) - out.size[0] // 2
    py = (canvas[1] // 2 if cy is None else cy) - out.size[1] // 2
    cv.alpha_composite(out, (px, py))
    return cv


def ornament(cy, width=520):
    """Garis emas dengan berlian di tengah."""
    cv = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(cv)
    cx = W // 2
    for sgn in (-1, 1):
        for i in range(width // 2):
            a = int(255 * (1 - i / (width / 2)) ** 1.2)
            x = cx + sgn * (24 + i)
            d.point((x, cy), fill=(232, 190, 98, a))
            d.point((x, cy + 1), fill=(232, 190, 98, a // 2))
    d.polygon([(cx, cy - 9), (cx + 9, cy), (cx, cy + 9), (cx - 9, cy)], fill=(244, 214, 130, 255))
    return cv.filter(ImageFilter.GaussianBlur(0.6))


def caption_layer(main, sub):
    cv = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fm = ImageFont.truetype(F_SANS_SB, 58)
    fs = ImageFont.truetype(F_SANS_M, 25)
    wm = fm.getlength(main)
    ws = sum(fs.getlength(c) + 5 for c in sub) - 5
    bw = int(max(wm, ws) + 150)
    bh = 150
    x0, y0 = (W - bw) // 2, 900
    plate = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    pd = ImageDraw.Draw(plate)
    pd.rounded_rectangle((0, 0, bw - 1, bh - 1), 30, fill=(8, 12, 28, 170), outline=(232, 190, 98, 210), width=2)
    cv.alpha_composite(plate.filter(ImageFilter.GaussianBlur(0.4)), (x0, y0))
    cv = Image.alpha_composite(cv, text_layer(main, F_SANS_SB, 58, "white", glow=False, cy=y0 + 54))
    cv = Image.alpha_composite(cv, text_layer(sub, F_SANS_M, 25, (240, 205, 120), tracking=5, glow=False, cy=y0 + 113))
    return cv


# ----------------------------------------------------------------------------- zarah emas (gelung 10s)
def make_particles(path, secs=10, fps=FPS):
    w, h = 960, 540
    n = 150
    rg = np.random.default_rng(11)
    x0 = rg.uniform(0, w, n)
    y0 = rg.uniform(0, h, n)
    speed = rg.choice([1, 1, 2, 2, 3], n) * h / secs  # kitaran penuh -> gelung sempurna
    sway = rg.uniform(8, 40, n)
    phase = rg.uniform(0, 6.28, n)
    size = rg.uniform(1.2, 4.2, n)
    bright = rg.uniform(0.35, 1.0, n)
    tw = rg.uniform(0.6, 2.0, n)
    sprites = {}

    def sprite(r):
        k = int(round(r * 4))
        if k not in sprites:
            rad = int(r * 3) + 2
            yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
            g = np.exp(-(xx ** 2 + yy ** 2) / (2 * (r * 0.8) ** 2))
            sprites[k] = g.astype(np.float32)
        return sprites[k]

    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-vf", f"scale={W}:{H}:flags=bicubic,gblur=sigma=1.2", "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)
    col = np.array([255, 214, 130], dtype=np.float32)
    for f in range(secs * fps):
        t = f / fps
        frame = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(n):
            y = (y0[i] - speed[i] * t) % h
            x = (x0[i] + sway[i] * np.sin(2 * np.pi * t / secs * (1 + int(tw[i])) + phase[i])) % w
            a = bright[i] * (0.55 + 0.45 * np.sin(2 * np.pi * t / secs * 2 * int(round(tw[i] + 1)) + phase[i]))
            s = sprite(size[i])
            r = s.shape[0] // 2
            xi, yi = int(x), int(y)
            ys, ye = max(0, yi - r), min(h, yi + r + 1)
            xs, xe = max(0, xi - r), min(w, xi + r + 1)
            if ye <= ys or xe <= xs:
                continue
            sub = s[ys - (yi - r):ye - (yi - r), xs - (xi - r):xe - (xi - r)]
            frame[ys:ye, xs:xe] += sub[..., None] * col * a
        proc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()


# ----------------------------------------------------------------------------- segmen
def beats_s(b):
    return b * BEAT


def timings():
    """Pulangkan senarai (mula_abs, tempoh_klip, lead, trail, spacing)."""
    out = []
    bound = 0.0
    n = len(CLIPS)
    for k, c in enumerate(CLIPS):
        sp = beats_s(c[3])
        lead = TX / 2 if k > 0 else 0.0
        trail = TX / 2 if k < n - 1 else 0.0
        start = bound - lead
        out.append((start, lead + sp + trail, lead, trail, sp))
        bound += sp
    return out


def seg_filter_common(d, lead, trail, layers):
    """Bina bahagian overlay lapisan teks. layers: senarai (input_idx, mula, jenis)."""
    parts = []
    last = "base"
    for j, (idx, t0, kind) in enumerate(layers):
        lab = f"L{j}"
        end_fade = d - trail - 0.25
        fo = f",fade=t=out:st={end_fade:.3f}:d=0.4:alpha=1" if kind != "keep" else ""
        parts.append(f"[{idx}:v]format=rgba,fade=t=in:st={t0:.3f}:d=0.55:alpha=1{fo}[{lab}]")
        slide = f"if(lt(t-{t0:.3f},0.7),38*pow(1-(t-{t0:.3f})/0.7,2),0)"
        out = f"O{j}"
        parts.append(f"[{last}][{lab}]overlay=x=0:y='{slide}':format=auto:shortest=0[{out}]")
        last = out
    return parts, last


def build_segment(idx, clip, tm, work, intro_last_png):
    cid, kind, img, beats, cap, sub, _ = clip
    start, d, lead, trail, sp = tm
    N = int(round(d * FPS))
    out = work / f"seg_{idx:02d}_{cid}.mp4"
    if out.exists():
        return out
    enc = ["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an", str(out)]
    if kind == "intro":
        run(["ffmpeg", "-v", "error", "-y", "-i", str(ROOT / "assets/video/intro_pmdiuc.mp4"),
             "-vf", f"scale={W}:{H}:flags=lanczos,unsharp=5:5:0.6,tpad=stop_mode=clone:stop_duration=1,fps={FPS},trim=duration={d:.3f},setpts=PTS-STARTPTS"] + enc)
        return out

    inputs = []
    layers = []  # (input_idx, start, kind)
    zoom_out = idx % 2 == 0
    z_expr = f"1.075-0.075*on/{N}" if zoom_out else f"1+0.075*on/{N}"

    if kind in ("photo", "pan"):
        BW, BH = 1640, 830
        iw, ih = Image.open(IMG / img).size
        if kind == "photo":
            sc = min(BW / iw, BH / ih)
            fw, fh = int(iw * sc) // 2 * 2, int(ih * sc) // 2 * 2
            fg = (f"[1:v]scale={fw*3}:{fh*3}:flags=lanczos,"
                  f"zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={N}:s={fw}x{fh}:fps={FPS},"
                  f"pad={fw+16}:{fh+16}:8:8:color=0xE8BE62,format=rgba[fg]")
            fin_in = ["-i", str(IMG / img)]
        else:  # pan menegak untuk gambar potret
            sw = 1700
            shh = int(ih * sw / iw)
            wh = int(BH * sw / BW)
            y0, y1 = 1110, 1580
            fg = (f"[1:v]scale={sw}:{shh}:flags=lanczos,"
                  f"crop=w={sw}:h={wh}:x=0:y='{y0}+({y1}-{y0})*min(1,t/{d:.3f})',"
                  f"scale={BW}:{BH}:flags=lanczos,pad={BW+16}:{BH+16}:8:8:color=0xE8BE62,format=rgba[fg]")
            fw, fh = BW, BH
            fin_in = ["-loop", "1", "-framerate", str(FPS), "-t", f"{d:.3f}", "-i", str(IMG / img)]
        fw += 16 if kind == "pan" else 16
        fh += 16
        fx, fy = (W - fw) // 2, 28 + (BH + 16 - fh) // 2
        inputs = ["-loop", "1", "-framerate", str(FPS), "-t", f"{d:.3f}", "-i", str(IMG / img)] + fin_in
        graph = [
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},gblur=sigma=38,"
            f"eq=brightness=-0.18:saturation=1.15,format=rgba[bg]",
            fg,
            "[fg]split=2[fga][fgb]",
            "[fgb]colorchannelmixer=rr=0:gg=0:bb=0:aa=0.6,gblur=sigma=24[shadow]",
            f"[bg][shadow]overlay=x={fx + 14}:y={fy + 22}:shortest=0[bs]",
            f"[bs][fga]overlay=x={fx}:y={fy}:shortest=0[base0]",
            "[base0]format=yuv420p[base]",
        ]
        n_in = 2
        # kapsyen
        cap_png = work / f"cap_{cid}.png"
        caption_layer(cap, sub).save(cap_png)
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{d:.3f}", "-i", str(cap_png)]
        layers.append((n_in, lead + 0.25, "cap"))
        parts, last = seg_filter_common(d, lead, trail, layers)
        graph += parts
        graph.append(f"[{last}]format=yuv420p[vout]")
    else:
        # kad tajuk / penutup
        n_in = 1
        if kind == "title":
            inputs = ["-i", str(IMG / img)]
            bgsrc = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},gblur=sigma=14,"
                     f"eq=brightness=-0.32:saturation=1.1")
            zin = f"scale={W*2}:{H*2}:flags=lanczos"
        else:
            inputs = ["-i", str(intro_last_png)]
            bgsrc = (f"[0:v]scale={W}:{H},gblur=sigma=26,eq=brightness=-0.36:saturation=1.05")
            zin = f"scale={W*2}:{H*2}:flags=lanczos"
        zz = f"1+0.06*on/{N}" if kind == "end" else f"1.07-0.07*on/{N}"
        graph = [
            f"{bgsrc},{zin},zoompan=z='{zz}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={N}:s={W}x{H}:fps={FPS},format=yuv420p[base]",
        ]
        specs = []
        if kind == "title":
            specs = [
                (ornament(300), "orn1"),
                (text_layer("ALHAMDULILLAH", F_SANS_SB, 44, (245, 215, 140), tracking=22, glow=False, cy=360), "t1"),
                (text_layer("Tamat Final Exam", F_SERIF, 150, "white", glow=False, cy=520), "t2"),
                (text_layer("SEMESTER 1", F_SERIF, 190, "gold", tracking=14, cy=700), "t3"),
                (ornament(820), "orn2"),
                (text_layer("PMDIUC PJJ BATCH 2   •   UKM BERANANG", F_SANS_SB, 38, (255, 255, 255), tracking=8, glow=False, cy=905), "t4"),
            ]
            times = [lead + 0.1, lead + 0.35, lead + 0.75, lead + 1.35, lead + 1.9, lead + 2.4]
        else:
            specs = [
                (ornament(330), "e0"),
                (text_layer("TERIMA KASIH", F_SERIF, 170, "gold", tracking=12, cy=470), "e1"),
                (text_layer("Semoga Ilmu Yang Dipelajari Penuh Keberkatan", F_SANS_SB, 54, "white", glow=False, cy=640), "e2"),
                (ornament(720), "e3"),
                (text_layer("DIPLOMA KEMAHIRAN PENGURUSAN MUTAWWIF UMRAH DAN ZIARAH", F_SANS_M, 30, (240, 205, 120), tracking=5, glow=False, cy=790), "e4"),
                (text_layer("#PMDIUC   #PJJBATCH2   #MUTAWWIFMUDA", F_SANS_SB, 34, (255, 255, 255), tracking=8, glow=False, cy=900), "e5"),
            ]
            times = [lead + 0.2, lead + 0.4, lead + 1.1, lead + 1.6, lead + 2.1, lead + 3.0]
        for (img_l, name), t0 in zip(specs, times):
            p = work / f"lay_{cid}_{name}.png"
            img_l.save(p)
            inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{d:.3f}", "-i", str(p)]
            layers.append((n_in, t0, "keep"))
            n_in += 1
        parts, last = seg_filter_common(d, lead, trail, layers)
        graph += parts
        graph.append(f"[{last}]format=yuv420p[vout]")

    fscript = work / f"fg_{idx:02d}.txt"
    fscript.write_text(";\n".join(graph))
    run(["ffmpeg", "-v", "error", "-y"] + inputs + ["-filter_complex_script", str(fscript), "-map", "[vout]", "-t", f"{d:.3f}"] + enc)
    return out


# ----------------------------------------------------------------------------- utama
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", default=str(ROOT / "assets/audio/muzik_pmdiuc.wav"))
    ap.add_argument("--out", default=str(ROOT / "output/PMDIUC_PJJ_Batch2_Tamat_Final_Exam_Sem1.mp4"))
    ap.add_argument("--work", default="/tmp/pmdiuc_work")
    ap.add_argument("--only-segments", action="store_true")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    tms = timings()
    total = tms[-1][0] + tms[-1][1]
    print(f"Jumlah tempoh: {total:.2f}s")

    last_png = work / "intro_last.png"
    if not last_png.exists():
        run(["ffmpeg", "-v", "error", "-y", "-ss", "9.9", "-i", str(ROOT / "assets/video/intro_pmdiuc.mp4"), "-frames:v", "1", str(last_png)])
    part = work / "particles.mp4"
    if not part.exists():
        print("Jana zarah emas ...")
        make_particles(part)

    segs = []
    for i, (c, tm) in enumerate(zip(CLIPS, tms)):
        print(f"Segmen {i:02d} {c[0]:6s} mula={tm[0]:6.2f} tempoh={tm[1]:5.2f}")
        segs.append(build_segment(i, c, tm, work, last_png))
    if a.only_segments:
        return

    # ---- xfade berantai
    ins = []
    for s in segs:
        ins += ["-i", str(s)]
    ins += ["-stream_loop", "-1", "-i", str(part)]  # zarah
    ins += ["-i", str(a.music), "-i", str(ROOT / "assets/video/intro_pmdiuc.mp4")]
    n = len(segs)
    iz, im, ii = n, n + 1, n + 2
    g = []
    cur = "0:v"
    for k in range(1, n):
        off = tms[k][0]
        g.append(f"[{cur}][{k}:v]xfade=transition={CLIPS[k][6]}:duration={TX}:offset={off:.3f}[x{k}]")
        cur = f"x{k}"
    groove_a, groove_b = 25.0, 55.0
    g.append(f"[{cur}]format=gbrp[vv]")
    g.append(f"[{iz}:v]trim=duration={total:.3f},setpts=PTS-STARTPTS,fade=t=in:st=9.0:d=1.5,format=gbrp[pp]")
    g.append("[vv][pp]blend=all_mode=screen:all_opacity=0.9[vb]")
    g.append(f"[vb]format=yuv420p,eq=brightness='0.045*exp(-9*mod(t,{BEAT:.5f}))*between(t,{groove_a},{groove_b})':eval=frame:contrast=1.05:saturation=1.08,"
             f"vignette=angle=0.35:eval=init,fade=t=in:st=0:d=0.4,fade=t=out:st={total-1.2:.2f}:d=1.2[v]")
    # ---- audio
    g.append(f"[{im}:a]atrim=0:{total:.3f},asetpts=PTS-STARTPTS,aformat=sample_rates=48000:channel_layouts=stereo,volume=0.95,"
             f"afade=t=in:st=0:d=0.8,afade=t=out:st={total-2.6:.2f}:d=2.6[m]")
    g.append(f"[{ii}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.8,afade=t=out:st=8.2:d=1.8,apad=whole_dur={total:.3f}[ia]")
    g.append("[m][ia]amix=inputs=2:normalize=0:duration=first,loudnorm=I=-14:TP=-1.5:LRA=11[a]")
    script = work / "final_graph.txt"
    script.write_text(";\n".join(g))
    print("Gabung & enkod akhir ...")
    run(["ffmpeg", "-v", "error", "-y"] + ins + ["-filter_complex_script", str(script), "-map", "[v]", "-map", "[a]",
         "-t", f"{total:.3f}", "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", a.out])
    print("SIAP:", a.out)


if __name__ == "__main__":
    main()
