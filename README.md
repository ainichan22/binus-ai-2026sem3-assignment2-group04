# binus-ai-2026sem3-assignment2-group04

> CIFAR-10 image classification with model optimization and a Streamlit web app.
>
> Klasifikasi gambar CIFAR-10 dengan optimasi model dan aplikasi web Streamlit.

Course: Machine Learning (LO 3) — Team Assignment 2

Live demo (Hugging Face Spaces): _TBD_

---

## English

### What's in here
- `notebooks/` — Jupyter notebooks for baseline, hyperparameter tuning, transfer learning, and real-world testing (each in `_EN` and `_ID` variants)
- `app/` — Streamlit web app with EN / Bahasa Indonesia language switching
- `models/` — trained `.keras` weights (not committed; download separately or retrain)
- `test_images/` — 30+ external real-world images for evaluation
- `report/` — final report and figures
- `demo/` — demo video and script
- `TASKS.md` — internal task tracker

### Local setup
```bash
git clone <repo-url>
cd binus-ai-2026sem3-assignment2-group04
python -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt
streamlit run app/app.py
```

Training notebooks are designed for Google Colab (GPU). Local environment is for app development only.

### Team
- _TBD_

---

## Bahasa Indonesia

### Isi repositori
- `notebooks/` — Notebook Jupyter untuk baseline, hyperparameter tuning, transfer learning, dan pengujian dunia nyata (masing-masing varian `_EN` dan `_ID`)
- `app/` — Aplikasi web Streamlit dengan ganti bahasa EN / Bahasa Indonesia
- `models/` — Bobot `.keras` terlatih (tidak di-commit; unduh terpisah atau latih ulang)
- `test_images/` — 30+ gambar dunia nyata eksternal untuk evaluasi
- `report/` — Laporan akhir dan gambar
- `demo/` — Video demo dan skrip
- `TASKS.md` — Pelacak tugas internal

### Persiapan lokal
```bash
git clone <repo-url>
cd binus-ai-2026sem3-assignment2-group04
python -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt
streamlit run app/app.py
```

Notebook pelatihan dirancang untuk Google Colab (GPU). Lingkungan lokal hanya untuk pengembangan aplikasi.

### Tim
- _TBD_
