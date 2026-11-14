# Dokumentasi

Letakkan nota, skrip cerita dan idea projek di sini.

## TCIM Bowling Night 2026 (video)

Video recap bersemangat untuk Staff Engagement Activity "Bowling Night" (YEC 2026).

- `scripts/make_music.py`: jana lagu latar 128 BPM (sintesis, tiada sampel) + SFX.
- `scripts/make_video.py`: render video 1920x1080 @ 30fps, semua animasi dikunci pada beat.
- Gambar sumber dari folder Google Drive "Bowling TCIM" (tidak disimpan dalam Git).

```bash
python3 scripts/make_music.py music.wav
TCIM_SRC=path/ke/gambar python3 scripts/make_video.py render output/tcim.mp4 music.wav
```
