# Glossary — EN ↔ ID Technical Terms

Reference for term consistency across the bilingual notebooks (`*_EN.ipynb` ↔ `*_ID.ipynb`)
and reports (`final_report_EN.md` ↔ `final_report_ID.md`).

The general policy follows Project_Plan §5.1: translate prose to Bahasa Indonesia
where idiomatic; keep widely-used English machine-learning loanwords in their
English form when the Indonesian equivalent is uncommon in industry usage.

## CIFAR-10 class labels

These are *data*, not prose — the same translation table is hard-coded in
`app/utils/preprocess.py` (`CLASS_NAMES_EN` / `CLASS_NAMES_ID`).

| English | Bahasa Indonesia |
|---|---|
| airplane | pesawat |
| automobile | mobil |
| bird | burung |
| cat | kucing |
| deer | rusa |
| dog | anjing |
| frog | katak |
| horse | kuda |
| ship | kapal |
| truck | truk |

## Architecture & training terminology

| English | Bahasa Indonesia | Treatment |
|---|---|---|
| Convolutional layer | Lapisan konvolusi | translated |
| Convolutional Neural Network (CNN) | Jaringan saraf konvolusional / CNN | "CNN" kept as acronym |
| Block | Blok | translated |
| Backbone | Backbone | kept (industry standard) |
| Head (classifier head) | Head (classifier head) | kept |
| Filter | Filter | kept |
| Kernel | Kernel | kept |
| Activation | Aktivasi | translated |
| Receptive field | Receptive field | kept |
| Stride | Stride | kept |
| Padding | Padding | kept |
| BatchNormalization | BatchNormalization | not translated (proper noun) |
| Dropout | Dropout | not translated |
| Pooling / MaxPooling | Pooling / MaxPooling | kept |
| GlobalAveragePooling2D | GlobalAveragePooling2D | kept (proper noun) |
| Flatten | Flatten | kept |
| Dense layer | Lapisan dense | partial — dense kept |
| Softmax | Softmax | kept |

## Optimization & regularization

| English | Bahasa Indonesia | Treatment |
|---|---|---|
| Hyperparameter | Hyperparameter | kept |
| Hyperparameter tuning | Hyperparameter tuning | kept |
| Search space | Search space / Ruang pencarian | both used |
| Hyperband | Hyperband | proper noun |
| Trial | Trial | kept |
| Optimizer | Optimizer | kept |
| Learning rate | Learning rate | kept |
| Weight decay (L2) | Weight decay (L2) | kept |
| Loss function | Fungsi loss / Loss function | both used |
| Cross-entropy | Cross-entropy | kept |
| Gradient | Gradien | translated |
| Backpropagation | Backpropagation | kept |
| Regularization | Regularisasi | translated |
| Augmentation | Augmentation | kept |
| EarlyStopping | EarlyStopping | kept (proper noun) |
| ReduceLROnPlateau | ReduceLROnPlateau | kept (proper noun) |
| Patience | Patience | kept |
| Frozen / Freeze | Di-freeze | partial |
| Fine-tuning | Fine-tuning | kept |
| Stage 1 / Stage 2 | Stage 1 / Stage 2 | kept |
| Two-stage training | Pelatihan dua tahap / training dua tahap | translated |
| Pretrained | Pretrained | kept |
| Transfer learning | Transfer learning | kept |
| Inference | Inferensi | translated |

## Evaluation & metrics

| English | Bahasa Indonesia | Treatment |
|---|---|---|
| Train / training set | Training / set training | kept |
| Validation set | Validation set / set validasi | both used |
| Test set | Test set / set tes | both used |
| Train/val split | Pembagian train/val | translated |
| Stratified split | Stratified split | kept |
| Accuracy | Akurasi | translated |
| Precision | Precision | kept |
| Recall | Recall | kept |
| F1 / F1-score | F1 / F1-score | kept |
| Confusion matrix | Confusion matrix / matriks kebingungan | both used |
| Per-class | Per kelas | translated |
| Top-1 / Top-3 | Top-1 / Top-3 | kept |
| Confidence | Confidence | kept |
| Calibration | Kalibrasi | translated |
| Misclassification | Misclassification | kept |
| Domain gap | Domain gap | kept |
| Domain shift | Pergeseran domain | translated |
| In-distribution | In-distribution | kept |
| Out-of-distribution | Out-of-distribution | kept |
| Overfitting | Overfitting | kept |
| Underfitting | Underfitting | kept |
| Convergence | Konvergensi | translated |
| Plateau | Plateau | kept |
| Train-val gap | Selisih train-val | translated |
| Noise floor / run-to-run noise | Noise floor / noise antar-run | kept |
| Percentage point (pp) | Percentage point (pp) | kept |

## Datasets & sources

| English | Bahasa Indonesia | Treatment |
|---|---|---|
| ImageNet | ImageNet | proper noun |
| Hugging Face Datasets | Hugging Face Datasets | proper noun |
| Hugging Face Spaces | Hugging Face Spaces | proper noun |
| Mirror (download mirror) | Mirror | kept |
| Distribution | Distribusi | translated |
| Resolution | Resolusi | translated |
| Thumbnail | Thumbnail | kept |
| High-resolution | Resolusi tinggi | translated |
| Center-crop | Center-crop | kept |
| Resize | Resize | kept |
| Pixel range | Range pixel | kept |
| Aspect ratio | Aspect ratio / rasio aspek | both used |

## Tools & framework names (always kept in English)

`TensorFlow`, `Keras`, `Keras Tuner`, `MobileNetV2`, `VGG / VGGNet`, `Adam`, `AdamW`,
`SGD`, `ImageNet`, `Streamlit`, `Hugging Face Spaces`, `Docker`, `Grad-CAM`,
`tf.data`, `tf.GradientTape`, `pip`, `Colab`, `Google Drive`, `Git`, `GitHub`,
`Markdown`, `JSON`, `CSV`, `LFS`.

## Notes for future translators

- **Numerical formatting** — Indonesian academic writing uses commas as decimal separators
  (`0,8993`) and full stops as thousand separators (`14.000`). The reports and notebook
  prints follow this convention in their ID variants.
- **Class names in figures and tables** — within the report and notebook outputs that
  render figures programmatically, the class labels follow the locale of the notebook
  variant. The bar charts and confusion matrices in `01_baseline_ID.ipynb` use the
  Indonesian class list; the EN equivalents use the English list.
- **Code identifiers stay English** — variable names, function names, and configuration
  keys (`PROJECT_DIR`, `IMG_SIZE`, `weight_decay`, etc.) are never translated;
  only the surrounding `print()` strings and markdown are.
