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
