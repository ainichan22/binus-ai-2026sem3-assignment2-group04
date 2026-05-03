# TASKS

從 Phase 0 到提交的全流程任務清單。對照 [Project_Plan.md](../Project_Plan.md) 各章節。

> Project_Plan.md 保留在外層 `Assignment-3/`,不搬入 git root。

每階段標註位置(本機/Colab)與粗估時間。完成一項就把 `[ ]` 改成 `[x]`。

---

## Phase 0 — 本機整理 + GitHub 建立 (< 30 min,本機)

- [x] 進入 `Assignment-3/binus-ai-2026sem3-assignment2-group04/` 建立目錄骨架: `notebooks/`、`models/`、`app/{i18n,pages,utils,samples}`、`test_images/`、`report/figures/`、`demo/`
- [x] 建立 `.gitignore`(`.DS_Store`、`models/*.keras`、`__pycache__/`、`.venv/`、`.ipynb_checkpoints/`)
- [x] 搬檔(`Project_Plan.md` 保留在 `Assignment-3/` 外層,不搬):
  - `Assignment-3/Assignment2_CNN_Architecture_EN.ipynb` → `notebooks/01_baseline_EN.ipynb`
  - `Assignment-2/Assignment2_CNN_Architecture_ID.ipynb` → `notebooks/01_baseline_ID.ipynb`
  - `Assignment-2/{confusion-matrix,train-history,top-9-most-confident-misclassification}.png` → `report/figures/`
- [x] 寫 `README.md`(雙語骨架,Project_Plan §5.4 範例)
- [x] `git init` → 首次 commit (`3f91e1e`)
- [ ] (建議 push 前先設 `git config user.name` 與 `user.email`)
- [ ] GitHub 建 private repo `binus-ai-2026sem3-assignment2-group04` → `git push`
- [ ] 邀請組員加 collaborator
- [ ] Drive 端建立 `models/` 子資料夾

---

## Phase 1A — Baseline 重訓 (~20 min,Colab)

### 本機準備 (Claude)
- [x] 在 `01_baseline_EN.ipynb` 最前面插入 Project_Plan §2.2 開頭模板 cell(Drive mount + `PROJECT_DIR` 設定 + GPU 檢查)
- [x] 在 `01_baseline_EN.ipynb` 最尾端新增 cell:`model.save(f"{MODEL_DIR}/baseline_v1.keras")` + 用 pickle 存 history(置於 §13 討論之後,作為新 §14)
- [x] 把同一個 Drive mount cell 鏡到 `01_baseline_ID.ipynb` 最前面(**選項 C**:不加 save cell — 模型只在 EN 存一次,ID 純跑 cell outputs)
- [x] commit + push 修改

### Colab 訓練 (User)
- [x] 上傳 `notebooks/01_baseline_EN.ipynb` 到 Colab,Runtime → GPU (T4)
- [x] EN: Restart & Run All,確認:
  - Drive 掛載成功、GPU 可見
  - **test accuracy = 0.8735**(超過 85% 目標)
  - Drive `models/` 出現 `baseline_v1.keras` (6.7 MB) 與 `baseline_v1_history.pkl`
  - 註:多倫多 mirror 停電,Cell 5 改用 `datasets.load_dataset("cifar10")`(HF)
- [x] 上傳 `notebooks/01_baseline_ID.ipynb` 到 Colab(已套用 HF 資料載入 + Section 7 從 Drive 載入 EN 訓練好的 `baseline_v1.keras`,跳過重訓)
- [x] ID: Restart & Run All — Section 7 成功載模型,Section 8-11 真的執行,印尼文 outputs 與 figures 全到位
- [x] 確認 Drive 裡 `models/baseline_v1.keras` 與 `baseline_v1_history.pkl` 都在(EN 跑完已寫入)

### 本機後處理 (Claude + User)
- [x] User 從 Colab 下載 EN `.ipynb` 覆蓋本機 `notebooks/01_baseline_EN.ipynb`
- [x] User 從 Colab 下載 ID `.ipynb` 覆蓋本機 `notebooks/01_baseline_ID.ipynb`
- [ ] User 從 Drive 下載 `baseline_v1.keras` 到本機 `models/`(被 `.gitignore` 擋住,不進 repo)
- [x] Claude review EN 結果、commit + push EN(trained outputs)+ ID 的 HF cell 同步
- [x] Claude review ID 結果、commit + push ID(trained outputs,test acc 0.8735 與 EN 一致)

---

## Phase 1B — Transfer Learning (1-2 hr,Colab)

### 本機準備 (Claude)
- [x] 建立 `notebooks/03_transfer_learning_EN.ipynb`(27 cells)涵蓋:
  - Cell 0: Drive mount + paths
  - §0-2: setup, HF data load, stratified split(沿用 baseline 的 `SEED=42` 確保 val set 一致可比)
  - §3: tf.data pipeline(resize 32→96 + `preprocess_input` + 輕度 augmentation)
  - §4: MobileNetV2 backbone + GAP + Dropout(0.3) + Dense(10);`training=False` 鎖 BN
  - §5 Stage 1: 凍結 backbone,10 epochs,Adam lr=1e-3
  - §6 Stage 2: 解凍最後 30 層(BN 仍鎖),20 epochs,Adam lr=1e-5,加 ReduceLROnPlateau
  - §7: 合併 stage1+stage2 history 畫雙階段曲線
  - §8-10: test eval、confusion matrix、top-9 misclass
  - §11: 與 baseline 對比討論
  - §12: save `transfer_mobilenet_v1.keras` + history pickle 到 Drive
- [x] commit + push

### Colab 訓練 (User)
- [ ] 上傳 `notebooks/03_transfer_learning_EN.ipynb` 到 Colab,Runtime → GPU (T4)
- [ ] Restart & Run All(預計 30-60 分鐘,大部分時間在 Stage 2 fine-tune)
- [ ] 確認:
  - Cell 9 sanity check:Pixel range 約在 [-1, 1]
  - Stage 1 結束 val acc ≈ 88-90%
  - Stage 2 結束 val acc ≈ 92-94%
  - Test accuracy ≥ 90%(超過 baseline 的 0.8735)
  - Drive `models/` 出現 `transfer_mobilenet_v1.keras` 與 `transfer_mobilenet_v1_history.pkl`

### 本機後處理 (Claude + User)
- [ ] User 下載 trained `.ipynb` 到本機 `notebooks/`(命名為 `03_transfer_learning_EN.ipynb`)
- [ ] User 從 Drive 下載 `transfer_mobilenet_v1.keras` 到本機 `models/`(被 `.gitignore` 擋住)
- [ ] Claude review 結果、commit + push

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
