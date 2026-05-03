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

### Colab 訓練 EN (User)
- [x] 上傳 `notebooks/03_transfer_learning_EN.ipynb` 到 Colab,Runtime → GPU (T4)
- [x] Restart & Run All,實際結果:
  - Cell 9 sanity check: pixel range = [-1.000, 1.000] ✓
  - Stage 1: 6 epochs(EarlyStopping triggered),best val acc 0.8344
  - Stage 2: 20 epochs full,best val acc 0.9022
  - **Test accuracy = 0.8993**(超過 baseline 0.8735,~+2.6 pp)
  - Drive `models/`: `transfer_mobilenet_v1.keras` (21.9 MB) + `transfer_mobilenet_v1_history.pkl`

### 本機後處理 EN (Claude + User)
- [x] User 下載 trained `.ipynb` 覆蓋本機 `notebooks/03_transfer_learning_EN.ipynb`
- [x] User 從 Drive 下載 `transfer_mobilenet_v1.keras` 到本機 `models/`(被 `.gitignore` 擋住)
- [x] Claude review 結果、commit + push EN

### 本機準備 ID (Claude)
- [x] 建立 `notebooks/03_transfer_learning_ID.ipynb`(27 cells),特點:
  - Markdown 全部翻譯成印尼文(Bagian 0-12),class_names 用印尼文
  - **Option C 模式**:Cell 13 載入 EN 訓練好的 `transfer_mobilenet_v1.keras` + history pickle,跳過 Stage 1 + Stage 2
  - Cell 13 內含 `STAGE1_EPOCHS = 6`(從 EN run 讀來),把 combined history 拆回 stage1/stage2 stub 讓 Cell 17 的 boundary line 畫對
  - Cell 15 `MODEL_LOADED=True` 時印 skip 訊息;`False` 時走原本 fine-tune 流程(fallback)
  - Cell 26 `MODEL_LOADED=True` 時不存(避免覆寫 EN 產出);`False` 時 fallback save
  - 所有 print/plot title 翻成印尼文

### Colab 訓練 ID (User)
- [x] 上傳 `notebooks/03_transfer_learning_ID.ipynb` 到 Colab,Runtime → GPU (T4)
- [x] Restart & Run All — Cell 13/15/26 都走 load/skip 路徑,不重訓 / 不重存
- [x] Test accuracy 0.8993 與 EN 一致 ✓

### 本機後處理 ID (Claude + User)
- [x] User 下載 trained `.ipynb` 覆蓋本機 `notebooks/03_transfer_learning_ID.ipynb`
- [x] Claude review、commit + push ID

---

## Phase 1C — Hyperparameter Tuning (1.5-2 hr,Colab)

> **平行進度提示:** Phase 1C 在 Colab 跑時,Phase 2 (Streamlit App) 可以同時開始 — App 只需要 `transfer_mobilenet_v1.keras` 與 `baseline_v1.keras`(已在本機),不依賴 tuned baseline。

### 本機準備 (Claude)
- [x] 建立 `notebooks/02_hyperparameter_tuning_EN.ipynb`(30 cells)涵蓋:
  - Cell 0: Drive mount + `TUNER_DIR`(讓 Hyperband 在 Drive 持久化,可中斷續跑)
  - §0-3: 同 baseline 的 setup / HF data load / split / augmentation(`SEED=42` 與 baseline val set 完全一致,確保公平對比)
  - §4 HyperModel: `build_model(hp)` 暴露 7 個 HP — `learning_rate` (log)、`dropout_block1/2/3`、`weight_decay` (log)、`dense_units` (64/128/256)、`optimizer` (adam/adamw/sgd)
  - §5 Hyperband: `max_epochs=20`、`factor=3`、`overwrite=False` 支援續跑、trial 內含 patience=3 EarlyStopping
  - §6: 抓 best HP、存 `best_hp.json`、印 top 5 trials
  - §7: 用 best HP 重訓滿 50 epochs(同 baseline 的 EarlyStopping/ReduceLROnPlateau 設定)
  - §8-11: 訓練曲線、test eval、混淆矩陣、top-9 misclass
  - §12: 三模型對比表(baseline 0.8735 / tuned 待填 / transfer 0.8993)
  - §13: 存 `baseline_tuned_v2.keras` + `baseline_tuned_v2_history.pkl`
- [ ] commit + push

### Colab 訓練 EN (User)
- [x] 上傳 `notebooks/02_hyperparameter_tuning_EN.ipynb` 到 Colab,Runtime → GPU (T4)
- [x] Restart & Run All — 實際:**search 57 分鐘 + retrain 50 epochs ~30 分鐘**
- [x] 結果:
  - `best_hp.json`:lr=6.68e-4、optimizer=adam、dropout=0.1/0.4/0.4、wd=1.4e-4、dense_units=128
  - **Tuned baseline test acc = 0.8689**(比 hand-tuned 0.8735 略低 0.46 pp,落在 noise 範圍內 — hand-tuned HP 已近最佳值)
  - Drive `models/` 出現 `baseline_tuned_v2.keras` (6.74 MB) + history.pkl + best_hp.json

### 本機後處理 EN (Claude + User)
- [x] User 下載 trained `.ipynb` 覆蓋本機 `notebooks/02_hyperparameter_tuning_EN.ipynb`
- [ ] User 從 Drive 下載 `baseline_tuned_v2.keras` + `best_hp.json` + history.pkl 到本機 `models/`(Metrics 頁要讀 history.pkl,沒這個會 warning)
- [x] Claude review、commit + push EN
- [x] 抽出 trained 圖片(cm / top9 misclass / training history)到 `report/figures/`
- [x] 更新 `app/pages/2_Metrics.py` 加入 tuned baseline 第三選項

### 本機準備 ID (Claude)
- [x] 建立 `notebooks/02_hyperparameter_tuning_ID.ipynb`(30 cells,option C):
  - Cell 13 偵測 `best_hp.json` 存在 → 跳過 Hyperband search;不存在 → fallback 跑 search
  - Cell 17 偵測 `baseline_tuned_v2.keras` + history.pkl → 跳過 50-epoch retrain;不存在 → fallback 跑 fit
  - Cell 29 偵測 → 跳過 save;不存在 → fallback save
  - 全部 markdown 翻成印尼文,class_names 用印尼文,plot title/cm label 印尼文
- [x] commit + push

### Colab 訓練 ID (User)
- [ ] 上傳 ID notebook,Run All(~3-5 分鐘 — 純 load + eval)
- [ ] 確認 test accuracy = 0.8689 與 EN 一致

### 本機後處理 ID
- [ ] User 下載 trained `.ipynb`、Claude commit + push

---

## Phase 1D — Notebook ID 版翻譯 (3 × 30-60 min,本機)

- [ ] 複製 `02_..._EN.ipynb` → `02_..._ID.ipynb`,翻譯所有 markdown cells(code 不動)
- [ ] 複製 `03_..._EN.ipynb` → `03_..._ID.ipynb`,翻譯
- [ ] 確認 ID 版 Restart & Run All 仍可跑通
- [ ] 在 git root 建立 `glossary.md`(Project_Plan §5.1 詞彙對照表)
- [ ] commit + push

---

## Phase 2 — Streamlit App (Week 2,本機,~10-15 hr)

> 可在 Phase 1C Colab 訓練期間平行進行(App 只用 baseline + transfer 兩個模型,不依賴 tuned baseline)。

### 2.1 App skeleton (Claude)
- [x] `app/requirements.txt`(streamlit、tensorflow、Pillow、numpy、matplotlib、pandas、datasets)
- [x] `app/i18n/{__init__.py, en.json, id.json}` — `t()` 函式 + `language_selector()`,en/id 各 37 個 key 完全對齊
- [x] `app/utils/preprocess.py`(跟訓練一致:center-crop → resize 96 → `preprocess_input`,含 EN/ID class_names)
- [x] `app/utils/model_loader.py`(`@st.cache_resource` 載入 transfer + baseline,支援 HF Spaces 與本機路徑)
- [x] `app/utils/gradcam.py`(手刻版,目標層 MobileNetV2 `Conv_1`,免裝 tf-keras-vis)
- [x] `app/app.py` 主入口 + sidebar `language_selector()`
- [x] `app/pages/1_Predict.py` — 上傳/範例/攝影機 → 顯示原圖 → Top-3 → Grad-CAM
- [x] `app/pages/2_Metrics.py` — 訓練曲線(讀 `models/*_history.pkl`)+ 靜態混淆矩陣 + per-class 表
- [x] `app/pages/3_About.py` — 模型架構、組員、references、GitHub 連結
- [x] `app/samples/make_samples.py` — 一次性 helper:從 HF datasets 抽 10 張 CIFAR-10 範例圖

### 2.2 本機 setup + 測試 (User)
- [ ] `python -m venv .venv && source .venv/bin/activate && pip install -r app/requirements.txt`
- [ ] `cd app/samples && python make_samples.py`(產生 10 張 PNG)
- [ ] 回到 git root,`streamlit run app/app.py`
- [ ] 確認:三頁都開得起來、語言切換正常、上傳一張圖能跑出 Top-3 + Grad-CAM
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
