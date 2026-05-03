# TASKS

從 Phase 0 到提交的全流程任務清單。對照 [Project_Plan.md](../Project_Plan.md) 各章節。

> Project_Plan.md 保留在外層 `Assignment-3/`,不搬入 git root。

每階段標註位置(本機/Colab)與粗估時間。完成一項就把 `[ ]` 改成 `[x]`。

---

## Phase 0 — 本機整理 + GitHub 建立 (< 30 min,本機)

- [ ] 進入 `Assignment-3/binus-ai-2026sem3-assignment2-group04/` 建立目錄骨架: `notebooks/`、`models/`、`app/{i18n,pages,utils,samples}`、`test_images/`、`report/figures/`、`demo/`
- [ ] 建立 `.gitignore`(`.DS_Store`、`models/*.keras`、`__pycache__/`、`.venv/`、`.ipynb_checkpoints/`)
- [ ] 搬檔(`Project_Plan.md` 保留在 `Assignment-3/` 外層,不搬):
  - `Assignment-3/Assignment2_CNN_Architecture_EN.ipynb` → `notebooks/01_baseline_EN.ipynb`
  - `Assignment-2/Assignment2_CNN_Architecture_ID.ipynb` → `notebooks/01_baseline_ID.ipynb`
  - `Assignment-2/{confusion-matrix,train-history,top-9-most-confident-misclassification}.png` → `report/figures/`
- [ ] 寫 `README.md`(雙語骨架,Project_Plan §5.4 範例)
- [ ] `git init` → 首次 commit
- [ ] GitHub 建 private repo `binus-ai-2026sem3-assignment2-group04` → `git push`
- [ ] 邀請組員加 collaborator
- [ ] Drive 端建立 `models/` 子資料夾

---

## Phase 1A — Baseline 重訓 (~20 min,Colab)

- [ ] `01_baseline_EN.ipynb` 上傳 Colab(Runtime → GPU T4)
- [ ] Cell 1 套用 Project_Plan §2.2 開頭模板,確認 Drive 掛載、GPU 可見、`PROJECT_DIR` 正確
- [ ] 在 notebook 最尾端新增 `model.save(f"{MODEL_DIR}/baseline_v1.keras")` 與 history pickle
- [ ] **Restart & Run All**,確認複現 ~85% test accuracy
- [ ] 從 Colab 下載 `.ipynb` 覆蓋本機 `notebooks/01_baseline_EN.ipynb`
- [ ] 從 Drive 下載 `baseline_v1.keras` 到本機 `models/`(僅本機,不進 git)
- [ ] commit + push

---

## Phase 1B — Transfer Learning (1-2 hr,Colab)

- [ ] Drive 建立 `03_transfer_learning_EN.ipynb`,套用開頭模板
- [ ] 寫資料 pipeline:CIFAR-10 → resize 96×96 → `mobilenet_v2.preprocess_input`
- [ ] **Stage 1(凍結)**:`MobileNetV2(include_top=False, weights="imagenet")` + GAP + Dropout(0.3) + Dense(10);`trainable=False`,10 epochs,lr=1e-3
- [ ] **Stage 2(fine-tune)**:解凍最後 30 層,lr=1e-5,20 epochs;加 `ModelCheckpoint`
- [ ] 評估 test set(目標 ≥ 90%);畫訓練曲線、混淆矩陣 → 存 `report/figures/`
- [ ] 存 `transfer_mobilenet_v1.keras` + history pickle 到 Drive `models/`
- [ ] 下載 `.ipynb` 到本機 `notebooks/`、模型到本機 `models/`
- [ ] commit + push

---

## Phase 1C — Hyperparameter Tuning (1-1.5 hr,Colab)

- [ ] Drive 建立 `02_hyperparameter_tuning_EN.ipynb`,套用開頭模板
- [ ] `pip install keras-tuner`,寫 `build_model(hp)` 函式(對照 Project_Plan §4.1 搜尋空間表)
- [ ] Hyperband:`max_epochs=20`、`factor=3`、總 trials ≤ 30
- [ ] 取出 `tuner.get_best_hyperparameters(1)[0]`,輸出 `best_hp.json`
- [ ] 用最佳 HP 重訓 50 epochs(含 EarlyStopping),存 `baseline_tuned_v2.keras` + history
- [ ] 比較三模型 test accuracy,輸出對比表(報告 ch.6 用)
- [ ] 下載 `.ipynb` 到本機,commit + push

---

## Phase 1D — Notebook ID 版翻譯 (3 × 30-60 min,本機)

- [ ] 複製 `02_..._EN.ipynb` → `02_..._ID.ipynb`,翻譯所有 markdown cells(code 不動)
- [ ] 複製 `03_..._EN.ipynb` → `03_..._ID.ipynb`,翻譯
- [ ] 確認 ID 版 Restart & Run All 仍可跑通
- [ ] 在 git root 建立 `glossary.md`(Project_Plan §5.1 詞彙對照表)
- [ ] commit + push

---

## Phase 2 — Streamlit App (Week 2,本機,~10-15 hr)

- [ ] `app/requirements.txt`(streamlit、tensorflow-cpu、Pillow、opencv-python、tf-keras-vis 等)
- [ ] **i18n 模組**:`app/i18n/__init__.py`(`t()` + `language_selector()`)、`en.json`、`id.json`(同 key)
- [ ] **utils**:
  - `preprocess.py`(跟訓練 pipeline 一致:resize → preprocess_input)
  - `model_loader.py`(`@st.cache_resource` 載入 `transfer_mobilenet_v1.keras`)
  - `gradcam.py`(目標層 `Conv_1`)
- [ ] `app/samples/`:每類一張範例圖(共 10 張)
- [ ] `app/app.py`:主入口,sidebar 放 `language_selector()`
- [ ] **Page 1 Predict**:檔案上傳/範例圖/攝影機 → 顯示原圖 → Top-3 條狀圖 → Grad-CAM 疊加圖
- [ ] **Page 2 Metrics**:test set 混淆矩陣(plotly heatmap)、per-class P/R/F1 表、訓練曲線
- [ ] **Page 3 About**:模型架構摘要、組員、references、GitHub 連結
- [ ] 本機測試:`streamlit run app/app.py`,所有 page + 語言切換正常
- [ ] commit + push

---

## Phase 3A — 真實圖片測試 (Week 3 上半,~3-5 hr)

- [ ] 蒐集 30+ 張外部圖片(每類 ≥ 3,Unsplash/Pexels/Google,避免 CIFAR 原圖與 AI 生成圖)
- [ ] 分類整理到 `test_images/<class>/`
- [ ] 建立 `04_realworld_testing_EN.ipynb`:批次預測函式 → 輸出 DataFrame(檔名、真實類別、Top-3、Top-1 信心、是否正確)
- [ ] 計算整體準確率、per-class 準確率、列出全部失敗案例
- [ ] 失敗案例視覺化(grid + 真值/預測標註)→ 存 `report/figures/`
- [ ] 翻譯成 `04_realworld_testing_ID.ipynb`
- [ ] commit + push

---

## Phase 3B — HF Spaces 部署 (Week 3 上半,~2 hr)

- [ ] 在 [huggingface.co](https://huggingface.co) 建立 Space(Streamlit、CPU Basic)
- [ ] `git clone` Space repo
- [ ] 複製 `app/` 全部內容 + `models/transfer_mobilenet_v1.keras` + `samples/`
- [ ] 編 Space `README.md` YAML front matter(`title`、`emoji`、`sdk: streamlit`、`sdk_version`、`app_file: app.py`)
- [ ] `git push origin main`,等 build(5-10 min)
- [ ] 線上煙霧測試:三頁、語言切換、Grad-CAM、上傳預測都通
- [ ] 把 Space URL 寫入主 repo `README.md`

---

## Phase 4A — 報告撰寫 (Week 3 下半 ~ Week 4,~10 hr)

- [ ] `report/final_report.md` 建立章節骨架(對照 Project_Plan §4.5 表)
- [ ] **ch.1 Introduction** (0.5p):專案目標、CIFAR-10、貢獻摘要
- [ ] **ch.2 Dataset & Preprocessing** (0.5p):分割、normalization、augmentation
- [ ] **ch.3 Baseline Model** (1p):VGG-inspired 架構圖 + 結果(用 `report/figures/` 既有圖)
- [ ] **ch.4 Hyperparameter Tuning** (1.5p):搜尋空間表、最佳 HP、結果
- [ ] **ch.5 Transfer Learning** (1.5p):MobileNetV2 架構、兩階段訓練曲線
- [ ] **ch.6 Comparison & Discussion** (1p):三模型橫向對比表
- [ ] **ch.7 Web Application** (1.5p):App 架構、screenshot、Grad-CAM 範例、HF URL
- [ ] **ch.8 Real-world Testing** (1.5p):結果表、失敗案例分析、domain gap 討論
- [ ] **ch.9 Conclusion & Future Work** (0.5p)
- [ ] **References** (0.5p):IEEE/APA 格式,Project_Plan §7 checklist 那 6 篇必收
- [ ] 印尼文 Abstract(附在英文 Abstract 後或附錄)
- [ ] 校稿 → 確認 ≥ 7 頁 → 視需要轉 PDF

---

## Phase 4B — Demo 影片 (Week 4,~3-5 hr)

- [ ] 寫 `demo/demo_script.md`(5 段:0:00 開場 / 0:30 模型對比 / 1:30 App live / 3:30 失敗分析 / 4:30 結論+URL)
- [ ] 製作開場/結尾投影片(Keynote 或 Slides)
- [ ] QuickTime/OBS 螢幕錄影:本機 Streamlit + HF Spaces 線上版
- [ ] 剪輯(iMovie/OBS),確認 ≤ 5:00、聲音清楚
- [ ] 輸出 `demo/demo_video.mp4`

---

## Phase 4C — 提交前最終檢查 (Week 4 末,~1 hr)

- [ ] 4 份 EN + 4 份 ID notebook 全部 Restart & Run All 通過
- [ ] `models/` 三個 `.keras` 檔齊全(本機驗證,提交方式依老師指示)
- [ ] `streamlit run app/app.py` 本機跑通
- [ ] HF Space URL 可開、預測正常
- [ ] `test_images/` ≥ 30 張、子資料夾命名正確
- [ ] `final_report.md` ≥ 7 頁、章節完整
- [ ] `demo_video.mp4` ≤ 5 分鐘
- [ ] `README.md` 有組員、繳交清單、執行說明、HF URL
- [ ] 對照 Project_Plan §7 Checklist 逐項打勾
- [ ] 最後一次 `git push` → 提交
