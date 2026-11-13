# Video-claude-tcim

Projek penghasilan video menggunakan Claude. Repositori ini menyimpan skrip, aset dan dokumentasi untuk proses penghasilan video.

## Struktur Projek

```
Video-claude-tcim/
├── README.md        # Dokumen ini
├── .gitignore       # Fail yang tidak dijejak oleh Git
├── docs/            # Dokumentasi, nota dan idea projek
├── scripts/         # Skrip penghasilan dan pemprosesan video
├── assets/          # Bahan mentah
│   ├── images/      # Gambar dan grafik
│   ├── audio/       # Muzik, suara latar dan efek bunyi
│   └── video/       # Klip video sumber
└── output/          # Hasil akhir video (tidak dijejak oleh Git)
```

## Cara Bermula

1. Klon repositori:
   ```bash
   git clone https://github.com/Saiallif/Video-claude-tcim.git
   cd Video-claude-tcim
   ```
2. Letakkan bahan mentah dalam folder `assets/`.
3. Letakkan skrip dalam folder `scripts/`.
4. Hasil video disimpan dalam folder `output/`.

## Aliran Kerja Git

- Cabang utama: `main`
- Kerja baharu dibuat pada cabang berasingan, kemudian dihantar melalui Pull Request.
- Fail video besar tidak disimpan dalam Git. Lihat `.gitignore`.

## Pemilik

[@Saiallif](https://github.com/Saiallif)

## Video: PMDIUC PJJ Batch 2 – Tamat Final Exam Sem 1

Video 70 saat (1920x1080, 30fps) menggunakan intro logo PMDIUC, gambar pelajar, kapsyen BM, zarah emas dan muzik bertema Islamik (maqam Hijaz).

```bash
python3 scripts/make_music.py assets/audio/muzik_pmdiuc.wav     # jana muzik asli (prosedural)
python3 scripts/build_video.py                                    # hasil: output/PMDIUC_PJJ_Batch2_Tamat_Final_Exam_Sem1.mp4
python3 scripts/build_video.py --music lagu_saya.mp3              # guna lagu sendiri (cadangan 96 BPM, 70 saat)
```

Keperluan: `ffmpeg`, `python3`, `numpy`, `Pillow`. Peralihan jatuh pada ketukan 96 BPM; jadual klip, kapsyen dan peralihan diedit dalam senarai `CLIPS` di `scripts/build_video.py`.
