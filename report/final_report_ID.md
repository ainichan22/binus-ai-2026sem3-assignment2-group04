# Klasifikasi Gambar CIFAR-10: Baseline dari Nol, Pencarian Hyperparameter Otomatis, dan Transfer Learning, dengan Aplikasi Web Streamlit Dwibahasa

**Mata Kuliah:** Machine Learning (LO 3) — Tugas Kelompok 2
**Kelompok:** binus-ai-2026sem3-assignment2-group04
**Penulis:** Kelompok 4
**Tanggal:** Mei 2026

**Demo live:** <https://huggingface.co/spaces/ainichan/binus-ai-2026sem3-assignment2-group04>
**Kode sumber:** <https://github.com/ainichan22/binus-ai-2026sem3-assignment2-group04>

> Versi Bahasa Inggris dari laporan ini tersedia di [`final_report_EN.md`](final_report_EN.md).

---

## Abstrak

Kami membangun, mengoptimalkan, dan men-deploy classifier gambar CIFAR-10 melalui tiga jalur yang saling melengkapi. Sebuah convolutional network bergaya VGG yang dilatih dari nol dengan stack regularisasi berlapis mencapai akurasi tes **0,8735**. Pencarian hyperparameter otomatis via Keras Tuner Hyperband pada ruang tujuh dimensi mendarat di **0,8689** — di dalam noise antar-run, menandakan konfigurasi hand-tuned sudah dekat dengan plafon alami arsitektur ini pada input 32×32. Transfer learning dari MobileNetV2 (pretrained ImageNet, fine-tuned pada 96×96 dengan jadwal dua tahap freeze-lalu-unfreeze) mencapai **0,8993**, peningkatan absolut +2,58 pp yang terkonsentrasi pada kelas-kelas yang sebelumnya paling lemah (burung +6,20 pp, kucing +4,38 pp, katak +3,14 pp). Ketiga model terlatih beserta artifak pencarian dihasilkan oleh notebook Jupyter yang dapat direproduksi (varian Bahasa Inggris dan Bahasa Indonesia). Model transfer learning melayani prediksi melalui aplikasi Streamlit tiga halaman (Predict, Metrics, About) dengan pergantian bahasa EN/ID pada runtime, di-deploy ke Hugging Face Spaces via Docker image kustom. Aplikasi menyertakan implementasi Grad-CAM manual dan caching agresif untuk menjaga inferensi CPU di bawah satu detik per gambar.

---

## 1. Pendahuluan

CIFAR-10 [1] tetap menjadi benchmark yang berguna untuk mempelajari teknik optimisasi pada input yang terbatas. Gambar RGB 32×32 dari dataset ini membatasi receptive field efektif yang dapat dibangun oleh jaringan apa pun, yang menjadikannya lingkungan yang tajam untuk memisahkan *apa yang dapat dipelajari arsitektur* dari *apa yang dapat diperas oleh optimisasi yang lebih baik*.

Proyek ini menjelajahi tiga jalur optimisasi pada CIFAR-10 dan mengemas hasilnya untuk inferensi pengguna akhir:

1. **Convolutional network bergaya VGG yang dilatih dari nol** (`baseline_v1`) dengan stack regularisasi berlapis yang disengaja.
2. **Pencarian hyperparameter otomatis** pada arsitektur yang sama (`baseline_tuned_v2`) menggunakan implementasi Hyperband [6] dari Keras Tuner.
3. **Transfer learning dari MobileNetV2** (`transfer_mobilenet_v1`) [3], dengan jadwal standar backbone-di-freeze-lalu-fine-tune dan input yang di-upscale ke 96×96.

Model transfer learning yang sudah terlatih kemudian dilayani melalui aplikasi web Streamlit dwibahasa (Bahasa Inggris / Bahasa Indonesia) yang di-deploy secara publik di Hugging Face Spaces. Aplikasi ini mendukung upload gambar, pemilihan sampel, dan pengambilan via webcam, serta memvisualisasikan perhatian model melalui overlay Grad-CAM [7] manual.

Kontribusinya adalah: (a) perbandingan tiga arah yang terkontrol pada split train/validation yang identik, (b) demonstrasi empiris bahwa hand-tuning yang teliti sudah berada di dalam noise floor pencarian otomatis pada skala arsitektur ini, (c) bukti yang jelas bahwa peningkatan transfer-learning mendarat secara tidak proporsional pada kelas-kelas yang sebelumnya membuat model dari nol kesulitan, dan (d) deployment publik end-to-end dengan detail engineering (caching, performance tuning, workaround framework) didokumentasikan untuk reproduktibilitas.

## 2. Dataset dan Preprocessing

CIFAR-10 berisi 60.000 gambar warna pada resolusi 32×32 yang dibagi menjadi 50.000 sampel training dan 10.000 sampel test, terdistribusi merata di sepuluh kelas (pesawat, mobil, burung, kucing, rusa, anjing, katak, kuda, kapal, truk) [1]. Split test asli telah ditetapkan oleh penulis dataset dan diperlakukan sebagai tidak tersentuh sepanjang pekerjaan ini; kami tidak pernah menggunakan gambar test untuk keputusan apa pun pada training time.

**Sumber dataset.** Helper Keras kanonik `tf.keras.datasets.cifar10.load_data()` mengambil dari mirror University of Toronto yang terbukti tidak stabil selama run training pertama proyek (sebuah scheduled outage membuat mirror tidak dapat dijangkau selama beberapa jam). Ketiga notebook sebagai gantinya memuat CIFAR-10 via Hugging Face Datasets (`datasets.load_dataset("cifar10")`), yang mirror-nya didukung CDN dan terbukti reliabel. Array NumPy yang dihasilkan sesuai dengan shape Keras standar `(50000, 32, 32, 3)` uint8 dan `(50000, 1)` integer label, sehingga kode pipeline di bawahnya tidak perlu diubah.

**Split train / validation.** Sepuluh persen dari 50.000 sampel training distratifikasi sebagai validation set menggunakan `sklearn.model_selection.train_test_split` dengan `stratify=y_train` dan `random_state=42` yang tetap. Seed yang sama digunakan kembali di ketiga jalur optimisasi sehingga setiap model dievaluasi terhadap validation set 5.000-sampel yang sama dan delta test-set dapat dikaitkan dengan perbedaan model alih-alih varians split.

**Normalisasi.** Untuk baseline yang dilatih dari nol, nilai pixel diskalakan ke [0, 1] dengan dibagi 255. Untuk model transfer learning, kami menerapkan `tf.keras.applications.mobilenet_v2.preprocess_input`, yang memetakan [0, 255] ke [-1, 1] — sesuai dengan distribusi yang dilihat MobileNetV2 selama pretraining ImageNet. Preprocessing ini terjadi setelah resize, yang akan kami uraikan di §5.

**Augmentation.** Pipeline augmentation ringan (`RandomFlip("horizontal")` + `RandomRotation(0.05)` + `RandomZoom(0.1)`) hanya diterapkan pada training set di dalam pipeline `tf.data`. Dataset validation dan test diteruskan tanpa augmentation. Kami sengaja menjaga augmentation tetap konservatif: pada resolusi 32×32, transformasi yang agresif (misalnya rotasi berat, color jitter, translasi besar) merusak fitur diskriminatif lebih banyak daripada me-regularisasi model. Augmentation yang sama dijaga tetap di ketiga jalur sehingga delta akurasi tidak tercampur dengan kekuatan augmentation.

## 3. Model Baseline: Convolutional Network Bergaya VGG

Arsitektur baseline adalah convolutional network tiga-blok dalam keluarga VGG [2], dengan dua convolusi 3×3 yang ditumpuk per blok dan pelebaran progresif dari jumlah filter.

### 3.1 Arsitektur

```
Input (32, 32, 3)
Blok 1: Conv(32) → BN → Conv(32) → BN → MaxPool(2) → Dropout(0,2)
Blok 2: Conv(64) → BN → Conv(64) → BN → MaxPool(2) → Dropout(0,3)
Blok 3: Conv(128) → BN → Conv(128) → BN → MaxPool(2) → Dropout(0,4)
Head:   Flatten → Dense(128) → BN → Dropout(0,5) → Dense(10, softmax)
```

Total parameter trainable: ~570 K. Semua konvolusi menggunakan kernel 3×3 dengan `padding='same'` dan aktivasi ReLU. L2 weight decay (`1e-4`) diterapkan pada setiap matriks weight.

**Penalaran desain.** Tiga blok adalah batas atas yang berprinsip untuk input 32×32: setiap MaxPool membagi dua dimensi spasial (32 → 16 → 8 → 4), jadi blok keempat akan menyisakan feature map 2×2 dengan terlalu sedikit informasi spasial. Dua konvolusi 3×3 yang ditumpuk per blok mencapai receptive field efektif 5×5 dengan parameter lebih sedikit dan satu non-linearitas tambahan dibanding satu lapisan 5×5 [2]. Jumlah filter berlipat dua di setiap blok (32 → 64 → 128) sehingga biaya per-layer tetap seimbang saat resolusi spasial mengecil.

**Stack regularisasi.** Empat mekanisme yang saling bekerja sama menangani overfitting pada 50 K contoh: (1) Dropout yang meningkat secara progresif (0,2 / 0,3 / 0,4 / 0,5) menempatkan mask paling berat dekat classifier head yang padat parameter [4]; (2) BatchNormalization setelah setiap Conv2D dan setelah lapisan dense menstabilkan aktivasi [5]; (3) L2 weight decay menghukum weight yang besar; (4) pipeline augmentation di atas memperluas distribusi training efektif.

### 3.2 Setup Pelatihan

Optimizer Adam, learning rate awal `1e-3`, loss categorical cross-entropy, batch size 64, dan budget 50-epoch dengan dua callback: `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)` dan `ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)`. Reproduktibilitas dibatasi — hanya split train/val yang di-seed; inisialisasi weight, dropout mask, dan randomness augmentation tidak — sehingga akurasi tes bervariasi sekitar ±0,5 pp antar run.

### 3.3 Hasil

Budget 50-epoch sepenuhnya digunakan; EarlyStopping tidak pernah terpicu, menandakan validation loss masih membaik (atau setidaknya tidak memburuk selama 10 epoch berturut-turut) di epoch 50. Akurasi tes akhir adalah **0,8735** (87,35 %). Akurasi training dan validation pada epoch akhir secara esensial identik di 0,8740 — stack regularisasi bekerja sesuai desain, dan tidak ada overfitting yang terukur.

![Riwayat training — baseline](figures/train-history-baseline.png)

`ReduceLROnPlateau` terpicu tiga kali selama training (epoch 25, 36, dan 41), menghasilkan pola konvergensi tangga yang khas dari LR decay yang didorong plateau. Setiap penurunan menghasilkan peningkatan kecil tapi terlihat pada validation loss selama 2-3 epoch sebelum plateau lagi.

**Hasil per kelas.** Kelas kendaraan mendominasi peringkat F1 teratas: kapal 0,933, mobil 0,933, kuda 0,911, truk 0,901, katak 0,893, pesawat 0,879. Kelas hewan menempati posisi terbawah: kucing 0,755, anjing 0,820, burung 0,836. Kucing adalah kasus terburuk — hanya 70,4 % kucing asli yang terdeteksi (recall 0,704), dengan 30 % sisanya terdistribusi di kelas-kelas hewan yang secara visual berdekatan (anjing, burung, katak).

![Confusion matrix — baseline](figures/confusion-matrix-baseline.png)

**Misclassification dengan confidence tertinggi.** Memeriksa sembilan gambar test di mana baseline paling yakin pada prediksi yang salah mengungkap tiga mode kegagalan berulang: (1) ambiguitas pose — hewan berkaki empat yang difoto dari sudut tidak standar memicu cluster kucing / anjing / rusa; (2) latar belakang mendominasi — burung di latar langit tertarik ke pesawat, burung di dedaunan bergeser ke katak atau rusa; (3) bentuk mengalahkan tekstur — mobil dengan profil besar (SUV, pickup) diklasifikasikan sebagai truk. Ini adalah hal-hal yang melekat pada CIFAR-10 di resolusi 32×32, bukan kegagalan arsitektur tertentu.

![Misclassification dengan confidence tertinggi — baseline](figures/top9-misclass-baseline.png)

## 4. Hyperparameter Tuning dengan Keras Tuner Hyperband

Kami menerapkan pencarian hyperparameter otomatis pada arsitektur baseline menggunakan implementasi Hyperband [6] dari Keras Tuner. Motivasinya bukan untuk mengalahkan transfer learning — plafon itu dibatasi oleh skala input 32×32 pada arsitektur ini — tetapi untuk mengkuantifikasi seberapa banyak headroom akurasi yang dapat dicapai melalui hyperparameter yang lebih baik saja.

### 4.1 Search Space

Tujuh hyperparameter diekspos:

| Hyperparameter | Range | Sampling |
|---|---|---|
| `learning_rate` | 1e-4 – 1e-2 | log-uniform |
| `dropout_block1` | 0,1 – 0,4 | step 0,1 |
| `dropout_block2` | 0,2 – 0,5 | step 0,1 |
| `dropout_block3` | 0,3 – 0,5 | step 0,1 |
| `weight_decay` (L2) | 1e-5 – 1e-3 | log-uniform |
| `dense_units` | 64 / 128 / 256 | choice |
| `optimizer` | adam / adamw / sgd | choice |

Dropout classifier-head (0,5 sebelum Dense terakhir) *tidak* di-tune: lokasi tersebut adalah risiko overfitting tertinggi di jaringan, dan standar literatur 0,5 [4] sudah teruji. Augmentation juga dijaga tetap.

### 4.2 Konfigurasi Hyperband

`max_epochs=20`, `factor=3`, `objective='val_accuracy'`, `seed=42`. Hyperband secara otomatis mengalokasikan struktur bracket; pencarian menghasilkan 30 trial berbeda dalam jendela wall-clock 57-menit pada Colab T4. Setiap trial menggunakan `EarlyStopping(patience=3)` internal untuk menggugurkan kandidat yang jelas buruk lebih awal.

### 4.3 Hasil Pencarian

Lima trial teratas berdasarkan akurasi validation (di dalam horizon search 20-epoch):

| Peringkat | val_acc | learning_rate | optimizer | dense_units | weight_decay | dropout (b1/b2/b3) |
|---|---|---|---|---|---|---|
| 1 | 0,8148 | 6,68e-04 | adam | 128 | 1,40e-04 | 0,1 / 0,4 / 0,4 |
| 2 | 0,7984 | 1,25e-03 | adamw | 256 | 1,90e-04 | 0,1 / 0,3 / 0,3 |
| 3 | 0,7862 | 9,70e-03 | sgd | 256 | 1,62e-04 | 0,2 / 0,2 / 0,5 |
| 4 | 0,7828 | 1,45e-03 | adam | 256 | 7,73e-05 | 0,1 / 0,4 / 0,5 |
| 5 | 0,7550 | 2,64e-03 | adam | 256 | 9,17e-05 | 0,1 / 0,2 / 0,5 |

Adam mendominasi leaderboard, learning rate pemenang (6,68e-04) berada di bawah default hand-tuned (1e-3), dan weight decay pemenang (1,40e-04) secara esensial sama dengan nilai hand-tuned (1e-4). Pola dropout sangat bervariasi di lima teratas — model relatif tidak sensitif terhadap besaran dropout per-blok selama nilainya non-trivial.

### 4.4 Retrain Akhir dan Hasil

Konfigurasi pemenang dilatih ulang dari nol untuk full 50-epoch budget dengan callback yang sama dengan baseline. Akurasi tes akhir adalah **0,8689** (val 0,8726).

![Riwayat training — tuned baseline](figures/train-history-tuned.png)

Ini **0,46 pp di bawah baseline hand-tuned 0,8735** — jauh di dalam noise floor ±0,5 pp run-to-run untuk arsitektur ini. Interpretasinya bukan bahwa Hyperband gagal; melainkan bahwa hyperparameter hand-tuned dari `01_baseline_EN.ipynb` sudah dekat dengan optimum lokal yang dapat dijangkau di dalam search space ini. Hasil per kelas mengikuti pola yang sama dengan baseline: kucing tetap yang terburuk (F1 0,750), kelas kendaraan tetap yang terkuat.

![Confusion matrix — tuned baseline](figures/confusion-matrix-tuned.png)

**Kesimpulan.** Ketika hasil pencarian otomatis cocok dengan konfigurasi hand-tuned di dalam noise, itu adalah *bukti tentang arsitektur, bukan tentang metode pencarian*. Kesimpulannya adalah headroom optimisasi lebih lanjut pada arsitektur ini di input 32×32 kecil; peningkatan yang berarti memerlukan perubahan pada arsitektur itu sendiri.

## 5. Transfer Learning dengan MobileNetV2

Setelah mengonfirmasi bahwa arsitektur dari nol sudah jenuh-optimisasi, kami beralih ke backbone pretrained. MobileNetV2 [3] adalah pilihan yang alami: cukup kecil untuk fine-tune di satu Colab T4 GPU, pretrained ImageNet, dan didukung dengan baik di Keras applications API.

### 5.1 Penyesuaian Arsitektur dan Preprocessing

```
Input (96, 96, 3)
  → Backbone MobileNetV2 (pretrained ImageNet, ~2,26 M params)
  → GlobalAveragePooling2D
  → Dropout(0,3)
  → Dense(10, softmax)
```

**Upscaling input.** Resolusi native CIFAR-10 32×32 berada di bawah ukuran spasial minimum yang didukung MobileNetV2 yaitu 96×96. Di bawah ambang ini jaringan menolak untuk dibangun karena lapisan stride-2-nya akan meruntuhkan feature map ke ukuran yang degenerate. Setiap gambar karenanya di-resize via interpolasi bilinear ke 96×96 di dalam pipeline `tf.data`. Upscaling tidak menambah informasi, tetapi memastikan filter konvolusional pretrained MobileNetV2 beroperasi pada resolusi feature map di mana pola stride-nya bermakna.

**Range pixel.** `tf.keras.applications.mobilenet_v2.preprocess_input` diterapkan setelah resize, memetakan range pixel integer [0, 255] ke float [-1, 1] — sesuai dengan distribusi yang menjadi pretrained MobileNetV2. Urutan penting: input harus berada di [0, 255] saat `preprocess_input` dipanggil.

**Detail kritis — `training=False`.** Saat backbone dipanggil di dalam Functional model baru, panggilan tersebut adalah `base_model(inputs, training=False)`. Ini memaksa lapisan BatchNormalization di dalam MobileNetV2 untuk menggunakan running statistics ImageNet yang tersimpan alih-alih menghitung ulang statistik batch dari CIFAR. Tanpa flag tunggal ini, statistik BN akan ditimpa oleh statistik batch kecil CIFAR selama beberapa epoch pertama dan akurasi turun 5-10 pp. Ini adalah langkah paling rawan kegagalan di seluruh pipeline.

### 5.2 Jadwal Pelatihan Dua Tahap

**Stage 1 (backbone di-freeze).** Semua weight MobileNetV2 di-freeze. Hanya parameter head baru yang ter-update — `Dropout` non-parametrik, jadi ini hanya ~12 K parameter dari lapisan `Dense(10)` akhir. Optimizer Adam, learning rate `1e-3`, budget 10-epoch dengan `EarlyStopping(patience=4)`. Head jenuh dengan cepat karena backbone yang di-freeze menghasilkan fitur tetap dan satu-satunya knob adalah classifier linear pada representasi 1280-dimensi. EarlyStopping terpicu pada epoch 6, dengan validation accuracy terbaik 0,8344.

**Stage 2 (fine-tune 30 layer terdalam).** 30 lapisan terdalam dari backbone MobileNetV2 di-unfreeze. Lapisan-lapisan awal (detector tepi, tekstur sederhana) tetap di-freeze untuk mempertahankan prior ImageNet yang luas; lapisan-lapisan terdalam — yang mengencode konsep level tinggi seperti bagian objek dan tekstur kompleks — digeser ke arah kekhususan CIFAR. Penting, semua lapisan `BatchNormalization` tetap di-freeze bahkan ketika blok induknya di-unfreeze, baik via flag call-site `training=False` maupun dengan eksplisit men-set `layer.trainable = False` untuk setiap lapisan BN. Learning rate turun 100× ke `1e-5` (LR besar akan menghancurkan weight pretrained), dan budget diperpanjang 20 epoch tambahan dengan baik `EarlyStopping(patience=6)` maupun `ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7)`.

Stage 2 berjalan penuh 20 epoch; ReduceLROnPlateau membelah dua LR dari 1e-5 ke 5e-6 di tengah-tengah tahap. Akurasi validation terbaik mencapai 0,9022 pada epoch 25 dari run training gabungan.

![Riwayat training — transfer learning](figures/train-history-transfer.png)

Garis vertikal putus-putus menandai batas Stage 1 → Stage 2. Diskontinuitas yang terlihat di batas adalah head menyesuaikan diri dengan input-nya yang baru (yang sekarang bisa di-train).

### 5.3 Hasil

Akurasi tes akhir adalah **0,8993** — peningkatan absolut +2,58 pp dibanding baseline yang dilatih dari nol.

![Confusion matrix — transfer learning](figures/confusion-matrix-transfer.png)

Selisih train/validation akhir adalah 0,0157 (1,6 pp). Overfitting moderat mulai muncul di akhir Stage 2, tetapi jauh di bawah ambang di mana Dropout(0,3) head kehilangan kontrol. Melanjutkan lebih jauh kemungkinan akan memperlebar selisih dan menurunkan akurasi validation.

Misclassification dengan confidence tertinggi (Gambar di bawah) mengikuti cluster kelas hewan yang sama dengan baseline, tetapi dengan nilai confidence yang nyata lebih rendah — model yang fine-tuned lebih ter-kalibrasi.

![Misclassification dengan confidence tertinggi — transfer learning](figures/top9-misclass-transfer.png)

## 6. Perbandingan dan Diskusi

### 6.1 Perbandingan Utama

| Model | Test accuracy | Trainable params | Epoch | Catatan |
|---|---|---|---|---|
| `baseline_v1` (hand-tuned) | 0,8735 | ~570 K | 50 | VGG-style, HP manual |
| `baseline_tuned_v2` (Hyperband) | 0,8689 | 552 K | 50 | Arsitektur sama, HP dicari |
| `transfer_mobilenet_v1` | **0,8993** | ~1,52 M (Stage 2) | 6 + 20 | MobileNetV2 → fine-tune 30 layer terdalam |

### 6.2 Delta F1 Per Kelas

| Kelas | Baseline | Tuned | Transfer | Δ (transfer − baseline) |
|---|---|---|---|---|
| pesawat | 0,8792 | 0,8817 | 0,9084 | +2,92 pp |
| mobil | 0,9331 | 0,9368 | 0,9499 | +1,68 pp |
| **burung** | 0,8358 | 0,8256 | 0,8978 | **+6,20 pp** |
| **kucing** | 0,7546 | 0,7497 | 0,7984 | +4,38 pp |
| rusa | 0,8624 | 0,8548 | 0,8849 | +2,25 pp |
| anjing | 0,8196 | 0,8091 | 0,8469 | +2,73 pp |
| katak | 0,8931 | 0,8779 | 0,9245 | +3,14 pp |
| kuda | 0,9105 | 0,9015 | 0,9159 | +0,54 pp |
| kapal | 0,9332 | 0,9312 | 0,9368 | +0,36 pp |
| truk | 0,9012 | 0,9087 | 0,9246 | +2,34 pp |
| **macro avg** | 0,8723 | 0,8677 | 0,8988 | **+2,65 pp** |

### 6.3 Diskusi

Tiga temuan menonjol.

**Hand-tuning sudah dekat-optimal untuk arsitektur ini.** Konfigurasi Hyperband-searched (0,8689) berada di dalam noise run-to-run ±0,5 pp dari baseline hand-tuned (0,8735). Hasil negatif ini informatif: ia memberi tahu kita bahwa arsitektur adalah constraint yang mengikat, bukan hyperparameter. Peningkatan akurasi yang berarti dari titik ini memerlukan perubahan arsitektur (kapasitas lebih, bias induktif yang berbeda) atau memberikan data yang lebih kaya (resolusi input lebih tinggi, lebih banyak sampel training).

**Peningkatan transfer learning mendarat secara tidak proporsional pada kelas baseline yang terburuk.** Peningkatan terbesar terjadi pada kelas yang baseline kesulitan — burung (+6,20 pp), kucing (+4,38 pp), katak (+3,14 pp) — sementara kelas yang sudah ditangani baseline dengan baik (kapal +0,36 pp, kuda +0,54 pp) hampir tidak bergerak. Pola ini konsisten dengan distribusi ImageNet: ImageNet berisi banyak spesies burung dan banyak varietas kucing pada resolusi tinggi, jadi MobileNetV2 membawa prior yang spesifik tentang struktur visual yang tidak dapat diajarkan oleh 50 K contoh CIFAR resolusi rendah dari nol. Kelas kendaraan, sebaliknya, sudah berada di plafon distribusi data pada 32×32 — tidak ada di prior MobileNetV2 yang membantu model yang sudah mendapatkan F1 93 % pada kapal.

**Transfer learning bukan akurasi "gratis" — paritas preprocessing kritis.** Resize 96×96, pemetaan `[0, 255] → [-1, 1]` via `preprocess_input`, urutan operasi (resize sebelum preprocess), flag `training=False` pada call site backbone, dan pem-freeze-an eksplisit lapisan BN selama fine-tuning semuanya memikul beban. Penyimpangan tunggal apa pun menghasilkan regresi akurasi 5-10 pp. Kami mendokumentasikan masing-masing secara eksplisit di notebook untuk replikator masa depan.

## 7. Desain dan Deployment Aplikasi Web

### 7.1 Arsitektur Aplikasi

Model transfer learning yang sudah terlatih dilayani melalui aplikasi Streamlit yang diorganisir sebagai tiga halaman plus utilities bersama.

```
app/
├── app.py                      # entry point + sidebar pilih bahasa
├── i18n/
│   ├── __init__.py             # t() + language_selector()
│   ├── en.json                 # 37 key, Bahasa Inggris
│   └── id.json                 # 37 key, Bahasa Indonesia (paritas terverifikasi)
├── utils/
│   ├── preprocess.py           # mirror training: center-crop → resize 96 → preprocess_input
│   ├── model_loader.py         # @st.cache_resource muat model, search path HF/lokal
│   └── gradcam.py              # implementasi Grad-CAM manual (tanpa library Grad-CAM pihak ketiga)
├── pages/
│   ├── 1_Predict.py            # upload / sample / kamera → top-3 + overlay Grad-CAM
│   ├── 2_Metrics.py            # kurva training, laporan per-kelas, confusion matrix
│   └── 3_About.py              # arsitektur, referensi, link GitHub
└── samples/                    # 10 gambar test CIFAR (satu per kelas)
```

**Internasionalisasi.** Semua string yang ditampilkan ke pengguna mengalir melalui helper `t(key)` yang memuat dari `en.json` atau `id.json` berdasarkan `st.session_state.lang`. Kedua file JSON berbagi 37 key yang diverifikasi di startup. Pemilih bahasa berada di sidebar dan di-render di setiap halaman sehingga perubahan bahasa berlaku konsisten antar navigasi.

**Paritas preprocessing.** `app/utils/preprocess.py` mengimplementasikan pipeline yang persis sama dengan yang digunakan di notebook transfer learning (center-crop ke persegi → resize 96×96 → `preprocess_input`). Center-cropping sebelum resize adalah tambahan khusus untuk inferensi: foto dunia nyata jarang memiliki rasio aspek persegi, dan resize naif akan mendistorsinya dengan cara yang tidak pernah dilihat model pada training time. Gambar training CIFAR berbentuk persegi secara konstruksi, jadi center-cropping membuat input inferensi cocok dengan distribusi training sedekat mungkin.

**Implementasi Grad-CAM.** Implementasi manual di `app/utils/gradcam.py` menghindari ketergantungan pada `tf-keras-vis` (yang memiliki masalah kompatibilitas di rilis TensorFlow terbaru). Lapisan target adalah `Conv_1` MobileNetV2 — konvolusi 1×1 terakhir sebelum global average pooling, yang merupakan target Grad-CAM standar untuk backbone ini [7].

### 7.2 Tantangan Implementasi

Dua masalah yang tidak jelas muncul selama development.

**Keterbatasan sub-model bersarang Keras 3.** Konstruksi Grad-CAM yang lurus `tf.keras.Model(inputs=outer_model.inputs, outputs=[target_conv.output, outer_model.output])` memunculkan `ValueError: Output with path 0 is not connected to inputs` ketika lapisan target hidup di dalam sub-model bersarang (MobileNetV2 dibungkus di dalam Functional model luar). Functional API Keras 3 memeriksa graph simbolik kontigu tunggal dari input ke semua output, dan tensor antara di dalam wrapper tidak dapat dijangkau dengan cara ini. Workaround kami membangun forward dua tahap: Functional model dalam dari input sub-model ke `(target_conv_output, sub_model_output)`, kemudian loop manual yang menerapkan lapisan-lapisan head luar (GAP / Dropout / Dense) untuk menyelesaikan prediksi. Seluruh forward berjalan di dalam satu `tf.GradientTape`, sehingga gradien logit kelas-yang-diprediksi terhadap aktivasi lapisan-conv tetap dapat dihitung.

**Latensi inferensi gambar tunggal.** Versi pertama yang di-deploy menunjukkan latensi per-prediksi melebihi satu menit pada Apple Silicon CPU. Biaya dominan adalah `model.predict(batch, verbose=0)` — sebuah method yang dirancang untuk inferensi batched dengan progress bar, yang menyiapkan pipeline `tf.data` internal pada setiap panggilan. Untuk inferensi gambar tunggal, mengganti `model.predict(x)` dengan `model(x, training=False).numpy()` memotong wall-clock 10-100×. Dikombinasikan dengan `@st.cache_resource` yang menghias baik model loader maupun konstruksi sub-model Grad-CAM, plus satu dummy forward pass di load time untuk membayar biaya graph-tracing di awal, latensi per-prediksi sekarang di bawah satu detik pada CPU yang sama.

### 7.3 Deployment ke Hugging Face Spaces

Hugging Face Spaces menghapus Streamlit sebagai SDK kelas-utama selama proyek ini. Aplikasi Streamlit sekarang di-deploy via Docker SDK dengan `Dockerfile` kustom. Image deployment kami berbasis pada `python:3.11-slim`, berjalan sebagai pengguna non-root (konvensi HF Spaces, UID 1000), menginstall requirements via `pip --user`, dan meluncurkan Streamlit pada port 7860 dengan flag yang kompatibel dengan iframe `--server.enableCORS=false` dan `--server.enableXsrfProtection=false`. Yang terakhir diperlukan karena HF Spaces melayani aplikasi di dalam iframe; perlindungan XSRF default Streamlit memperlakukan POST upload sebagai request lintas-asal dan menolaknya dengan HTTP 403. Menonaktifkan XSRF aman untuk demo publik tanpa state ini.

Artifak model (`transfer_mobilenet_v1.keras` plus tiga history pickle) disimpan di repository HF Space — di-upload langsung via UI Files HF alih-alih di-commit ke repository sumber GitHub, yang menjaga repo sumber tetap kecil. Step `COPY . .` Docker kemudian memasukkannya ke image pada build time.

Deployment live di <https://huggingface.co/spaces/ainichan/binus-ai-2026sem3-assignment2-group04>.

## 8. Pengujian Gambar Dunia Nyata

### 8.1 Metodologi

Test set CIFAR-10 diambil dari distribusi resolusi rendah yang sama dengan training set. Untuk mengukur bagaimana model menggeneralisasi di luar distribusi tersebut, kami mengumpulkan **30 gambar eksternal** (3 per kelas, 10 kelas) dari sumber foto publik (Unsplash, Pexels, hasil pencarian gratis Google Images), secara eksplisit mengecualikan (a) CIFAR-10 asli dan turunannya dan (b) gambar yang dihasilkan AI. Setiap gambar diklasifikasikan oleh `transfer_mobilenet_v1.keras` setelah pipeline preprocessing yang sama dengan training (`center_crop_square → resize 96×96 → preprocess_input`); top-3 prediksi, top-1 confidence, dan correctness dicatat. Tabel per-gambar lengkap berada di `report/realworld_results.csv`.

### 8.2 Hasil Agregat

| Metrik | Dunia nyata (n=30) | Test CIFAR (n=10 000) | Δ |
|---|---|---|---|
| Akurasi keseluruhan | **0,8667** (26 / 30) | 0,8993 | -3,26 pp |
| Mean confidence — prediksi benar | 0,967 | — | — |
| Mean confidence — prediksi salah | 0,671 | — | — |

Model mempertahankan sebagian besar akurasi CIFAR-test-nya pada foto alami yang out-of-distribution: domain gap adalah **3,26 percentage points**, lebih kecil dari yang awalnya diharapkan. Fitur pretrained ImageNet tampak menggeneralisasi dengan cukup baik ke input dunia nyata, sebagian karena weight pretrained MobileNetV2 dipelajari pada foto resolusi tinggi alami sejak awal — intuisi model tentang "seperti apa kucing" tidak pernah terbatas pada thumbnail 32×32 saja.

Hasil kalibrasi confidence adalah temuan yang paling menggembirakan: ketika model salah, ia cenderung tahu bahwa ia salah. Mean confidence pada prediksi salah adalah 0,67, dibanding 0,97 pada prediksi benar. Ambang confidence sederhana sekitar 0,85 akan mengarahkan sebagian besar kegagalan ke "tidak yakin" — properti yang berguna untuk deployment produksi.

### 8.3 Hasil Per Kelas

![Akurasi per kelas pada 30 gambar dunia nyata](figures/realworld-per-class.png)

| Kelas | Dunia nyata (n=3) | F1 CIFAR-test (model transfer) | Catatan |
|---|---|---|---|
| mobil | 3 / 3 (100 %) | 0,95 | kuat di kedua distribusi |
| burung | 3 / 3 (100 %) | 0,90 | kelas CIFAR-lemah yang diangkat transfer learning, RW-sempurna |
| kucing | 3 / 3 (100 %) | 0,80 | kelas CIFAR-terlemah, RW-sempurna |
| anjing | 3 / 3 (100 %) | 0,85 | sama |
| kapal | 3 / 3 (100 %) | 0,94 | kuat di keduanya |
| truk | 3 / 3 (100 %) | 0,92 | kuat di keduanya |
| pesawat | 2 / 3 (67 %) | 0,91 | satu kegagalan: `airplane_01` → burung |
| rusa | 2 / 3 (67 %) | 0,88 | satu kegagalan: `deer_01` → burung |
| katak | 2 / 3 (67 %) | 0,92 | satu kegagalan: `frog_02` → burung |
| kuda | 2 / 3 (67 %) | 0,92 | satu kegagalan: `horse_03` → rusa |

Pola yang berlawanan dengan intuisi: kelas yang ditandai test set CIFAR sebagai *paling lemah* (kucing, anjing, burung) semuanya mencapai akurasi dunia nyata 100 % di sini, sementara beberapa kelas CIFAR-kuat (pesawat, rusa, katak, kuda) masing-masing meleset persis sekali. Caveat yang jelas: **pada 3 gambar per kelas, akurasi per-kelas memiliki interval kepercayaan yang sangat lebar** — observasi 2 / 3 konsisten dengan akurasi populasi di mana saja dari ~10 % hingga ~99 %. Angka agregat 26 / 30 adalah satu-satunya yang memiliki resolusi yang berarti; observasi per-kelas harus dibaca sebagai ilustratif alih-alih konklusif secara statistik.

### 8.4 Mode Kegagalan

![Grid kegagalan dunia nyata](figures/realworld-failures.png)

Empat kasus kegagalan mengungkap satu pola dominan: **tiga dari empat prediksi salah memilih `burung`**, dan yang keempat memilih `rusa`.

| File | Sebenarnya | Diprediksi | Confidence | Top-3 |
|---|---|---|---|---|
| `airplane_01.jpg` | pesawat | **burung** | 0,51 | burung → pesawat → kucing |
| `deer_01.jpg` | rusa | **burung** | 0,54 | burung → rusa → kucing |
| `frog_02.jpg` | katak | **burung** | 0,74 | burung → kapal → anjing |
| `horse_03.jpg` | kuda | **rusa** | 0,90 | rusa → kuda → burung |

Tiga dari empat kegagalan mendarat di pita confidence 0,51–0,74 — output "tidak pasti" klasik di mana top-2 hampir seimbang. `airplane_01 → burung` dan `deer_01 → burung` pada dasarnya adalah lemparan koin antara kelas yang benar dan yang diprediksi (0,51 / 0,48 dan 0,54 / 0,46 masing-masing). Bias condong-burung pada input yang tidak pasti mungkin mencerminkan properti pretraining MobileNetV2: ImageNet berisi 59 kelas spesies burung (dari total 1 000), jadi subspace fitur burung besar, dan foto dunia nyata yang non-kanonik mungkin mengaktifkannya lebih kuat dari yang disarankan distribusi CIFAR.

Satu-satunya kesalahan high-confidence — `horse_03 → rusa` pada confidence 0,90 — adalah kegagalan yang paling menarik. Kuda-rusa adalah salah satu pasangan sulit yang dikenal di CIFAR-10 (keduanya hewan berkaki empat di luar ruangan), dan confidence tinggi mengindikasikan model salah dengan yakin di sini, bukan sekadar tidak pasti. Ini adalah kesalahan klasifikasi sungguhan, bukan masalah kalibrasi.

### 8.5 Implikasi untuk Aplikasi yang Di-deploy

Halaman Predict Streamlit sudah menyertakan disclaimer tentang domain gap (terlihat di bagian bawah tampilan prediksi). Dikombinasikan dengan temuan empiris bahwa prediksi yang salah biasanya ber-confidence rendah, UI bar-chart Top-3 yang ada sudah cocok dengan profil kalibrasi yang kami amati: ketika pilihan kedua dan ketiga dekat dengan yang pertama, pengguna dapat membaca ambiguitas itu langsung alih-alih menerima satu pernyataan over-confident.

Untuk narasi laporan yang lebih luas: akurasi dunia nyata 86,67 % pada 30 gambar adalah, dengan caveat ukuran sampel, demonstrasi yang masuk akal bahwa fitur pretrained ImageNet menggeneralisasi melewati distribusi CIFAR asli. Itu *tidak* berarti model robust dalam arti produksi apa pun — sampel kecil, gambar sumber yang dikurasi, dan ketiadaan kondisi adversarial (oklusi, varians pencahayaan, motion blur) semuanya berargumen untuk evaluasi lebih lanjut sebelum deployment nyata.

## 9. Kesimpulan dan Pekerjaan Masa Depan

Kami melatih tiga model pada CIFAR-10 dan mendapatkan urutan yang bersih: baseline VGG-style dari nol (0,8735) ≈ baseline Hyperband-searched (0,8689) ≪ transfer learning MobileNetV2 (0,8993). Hasil negatif pada pencarian hyperparameter informatif: pada arsitektur dan resolusi input ini, hand-tuning sudah berada di noise floor, dan peningkatan akurasi yang berarti memerlukan perubahan arsitektur alih-alih hyperparameter-nya. Peningkatan +2,58 pp transfer learning mendarat secara tidak proporsional pada kelas-kelas yang sebelumnya paling lemah (burung, kucing, katak), konsisten dengan ImageNet membawa prior spesifik-kelas yang tidak dapat diajarkan oleh 50 K contoh CIFAR dari nol.

Model transfer learning yang sudah terlatih dilayani melalui aplikasi Streamlit dwibahasa (EN / ID) yang di-deploy publik di Hugging Face Spaces via Docker image kustom. Aplikasi mendukung upload gambar, pemilihan sampel, dan pengambilan webcam, serta memvisualisasikan perhatian model via implementasi Grad-CAM manual. Latensi per-prediksi pada CPU di bawah satu detik setelah serangkaian fix performance TensorFlow / Streamlit yang didokumentasikan di §7.2.

**Pekerjaan masa depan.** Tiga arah terlihat menjanjikan. (1) **Augmentation yang lebih kuat** — Mixup, CutMix, atau RandAugment, mungkin dikombinasikan dengan jadwal training yang lebih panjang untuk menyerap tekanan regularisasi tambahan. (2) **Pencarian arsitektur di luar hyperparameter** — mengganti `Flatten + Dense` dengan `GlobalAveragePooling2D` (memotong ~262 K parameter), atau mencoba EfficientNet / ConvNeXt sebagai backbone transfer learning. (3) **Adaptasi domain untuk input dunia nyata** — fine-tune adapter head kecil pada set berlabel gambar eksternal resolusi tinggi, atau menerapkan test-time augmentation (rata-rata multi-crop) untuk memitigasi domain gap yang dikuantifikasi di §8.

## Referensi

[1] A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Tech. Rep., Univ. of Toronto, 2009.

[2] K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," in *International Conference on Learning Representations (ICLR)*, 2015.

[3] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 4510–4520.

[4] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," *Journal of Machine Learning Research*, vol. 15, no. 1, pp. 1929–1958, 2014.

[5] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," in *International Conference on Machine Learning (ICML)*, 2015, pp. 448–456.

[6] L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, and A. Talwalkar, "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization," *Journal of Machine Learning Research*, vol. 18, no. 1, pp. 6765–6816, 2017.

[7] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 618–626.

[8] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2009, pp. 248–255.

[9] M. Abadi *et al.*, "TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems," 2015. [Online]. Tersedia: <https://www.tensorflow.org/>

[10] F. Chollet *et al.*, "Keras," 2015. [Online]. Tersedia: <https://keras.io>

[11] T. O'Malley *et al.*, "Keras Tuner," 2019. [Online]. Tersedia: <https://github.com/keras-team/keras-tuner>

[12] Streamlit Inc., "Streamlit — A faster way to build and share data apps." [Online]. Tersedia: <https://streamlit.io>

[13] Hugging Face, "Spaces — Host machine learning demos." [Online]. Tersedia: <https://huggingface.co/spaces>

[14] Hugging Face, "Datasets — A community library for natural language processing." [Online]. Tersedia: <https://huggingface.co/docs/datasets>
