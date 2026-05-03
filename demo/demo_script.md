# Demo Video Script — CIFAR-10 Image Classifier

> **Target length:** ~4 minutes (well under the 5-minute Project_Plan §4.5 cap).
> **Format:** entirely on the live Streamlit app; no slides.
> **Live URL:** <https://huggingface.co/spaces/ainichan/binus-ai-2026sem3-assignment2-group04>
>
> Each section below has the action(s) the recorder should take in the browser, then the line(s) to read out loud while that action happens. Hold each visual for the full narration before moving on.

---

## 0:00 – 0:20 · Open the app (Home page)

**Action:** Open the live URL in a clean browser window. Land on the home page. Cursor on the title.

**Narration:**
> Hi, we're Group 4. This is our CIFAR-10 image classifier — a web app that takes any photo and classifies it into one of ten categories: airplane, cat, dog, ship, and so on. The whole thing is live on Hugging Face Spaces, and we'll walk through every page in the next four minutes.

---

## 0:20 – 0:40 · Show the bilingual UI (Sidebar)

**Action:**
1. Click the language dropdown in the sidebar (top of sidebar).
2. Choose **Bahasa Indonesia** — every label changes.
3. Switch back to **English**.

**Narration:**
> One feature we want to highlight first — the app is bilingual at runtime. Every string flows through an i18n module, so the entire interface switches between English and Bahasa Indonesia with one click.

---

## 0:40 – 1:50 · Predict page — sample image (success + Grad-CAM)

**Action:**
1. Click **Predict** in the sidebar nav.
2. In the sidebar, choose input method **Pick a sample**.
3. From the sample dropdown, choose **03_cat** (or any sample).
4. Wait ~1 second for the prediction to render.
5. Let the viewer see the original 32×32 image, Top-3 bars, and Grad-CAM overlay.

**Narration:**
> The Predict page is the main feature. I can upload an image, pick a built-in CIFAR sample, or use my webcam. Let me start with a sample.
>
> The model gives a Top-3 prediction list with confidence scores — cat first, with about 90 % confidence. That matches the actual image.
>
> Below it is a Grad-CAM heatmap. The red areas show which pixels drove the prediction. You can see the model is looking at the cat's face and body — not the background. That's how we know it learned the right features, not just memorized backgrounds.

---

## 1:50 – 2:30 · Predict page — real-world upload (domain gap moment)

**Action:**
1. Sidebar → switch input method to **Upload an image**.
2. Drag a real-world high-resolution photo (e.g., a clear cat or dog photo from the web, ~500×500 pixels).
3. Wait for prediction.
4. Point at the lower confidence number compared to the sample.

**Narration:**
> Now let's try a real-world photo from the web — a high-resolution image the model has never seen at this scale.
>
> The prediction is correct, but the confidence is noticeably lower. This is the **domain gap**: our model trained on 32×32 thumbnails, so high-resolution natural photos sit slightly outside its training distribution. We discuss this honestly in our written report.

---

## 2:30 – 3:30 · Metrics page — three-model comparison

**Action:**
1. Click **Metrics** in the sidebar.
2. Default selection is "Transfer (MobileNetV2 fine-tuned)".
3. Point at the four summary cards: Test accuracy 0.8993, Test loss 0.2971, ~1.52M params, 26 epochs.
4. Scroll to the **Model comparison** table (three rows).
5. Scroll to the training curves plot.
6. Switch model picker to **Baseline** — show the curves redraw.
7. Switch to **Tuned baseline** — show again.
8. Switch back to **Transfer**.
9. Scroll to per-class table.
10. Scroll to confusion matrix image.

**Narration:**
> The Metrics page tells the optimization story. We trained three models on the same data split.
>
> *(point at comparison table)* The from-scratch baseline reaches 87.35 %. Hyperparameter search with Keras Tuner Hyperband landed on 86.89 % — slightly *worse* than hand-tuning, which is actually informative: it means the architecture itself was the bottleneck.
>
> Transfer learning from MobileNetV2 jumped to **89.93 %**, with the biggest gains landing on classes the baseline struggled with — bird, cat, and frog all improved by 3 to 6 percentage points.
>
> *(point at training curves)* The curves show clean convergence with almost no train-validation gap.
>
> *(point at confusion matrix)* And the confusion matrix shows most off-diagonal errors are between visually similar classes — cat versus dog, deer versus horse — which are the well-known hard cases for CIFAR-10.

---

## 3:30 – 3:50 · About page — architecture in one breath

**Action:**
1. Click **About** in the sidebar.
2. Let the architecture block be visible.

**Narration:**
> A quick look at the architecture: MobileNetV2 backbone pre-trained on ImageNet, followed by global average pooling, dropout, and a 10-way dense layer. Trained in two stages — first frozen, then with the top 30 layers fine-tuned at a hundred-times-smaller learning rate.

---

## 3:50 – 4:00 · Closing (URL visible in browser bar)

**Action:** Scroll up so the browser URL bar is clearly visible. Cursor near it.

**Narration:**
> Everything you saw is live at this Hugging Face Space. Source code and reproducible notebooks are linked in the description. Thanks for watching.

---

## Recording notes

**Browser setup**
- Use a clean profile with no extensions visible
- Window size 1280×800 or 1440×900 (16:9-ish, plays well at 1080p)
- Light or dark theme — pick one and stay consistent

**Files to have ready in a `~/demo-images/` folder**
1. Optional: a real-world cat or dog photo (~500×500), JPG, ready to drag into the upload area at 1:50
2. Nothing else needed — the app's built-in samples cover the rest

**OBS / QuickTime**
- 1080p, 30 fps
- Audio: built-in mic is fine if quiet room; otherwise a USB headset mic
- Keep cursor visible (in OBS, "Capture cursor" on)
- Do one full take, then re-record sections that need polish — don't try to cut mid-narration

**Pacing tips**
- Don't rush. The script is intentionally tight; reading at a normal pace lands at ~4 minutes
- After the language-switch click (0:25), pause 1 second so the viewer's eye can register all the labels changed
- After picking the sample (0:50), pause 1-2 seconds so the Top-3 bars and Grad-CAM are clearly visible before you start narrating
- The model-picker switches on the Metrics page (3 of them) each take ~1 second; don't talk over them

**If recording in Bahasa Indonesia instead**
- Keep all the same actions
- Translate the narration; the on-screen UI follows whichever language is selected at the moment of recording
