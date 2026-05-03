# Demo Video Script — CIFAR-10 Image Classifier

> **Target length:** ≤ 5 minutes (Project_Plan §4.5 budget).
> **Spoken word target:** ~700 words at ~150 wpm + screen transitions.
> **Bracketed text** = stage directions / what to show on screen.
> **Bold lines** = on-screen text or slides.

---

## 0:00 – 0:25 · Opening (Slide)

[Slide: project title, team names, course]

> Hi, we are **Group 4** from the Machine Learning course. We built a **CIFAR-10 image classifier** that turns an uploaded photo into one of ten object categories — airplane, cat, dog, ship, and so on.
>
> Our project compares three optimization strategies on this task and serves the best one as a public web app. The headline result: transfer learning from MobileNetV2 reaches **89.93 %** test accuracy, beating our from-scratch baseline by 2.6 percentage points.
>
> Let me walk you through how we got there, and then we'll see the live demo.

---

## 0:25 – 1:30 · Three-model comparison (Slide)

[Slide: bar chart of three models' test accuracy, or this table on screen]

| Model | Test accuracy |
|---|---|
| Baseline VGG-style CNN (manual HP) | 0.8735 |
| Tuned baseline (Keras Tuner Hyperband) | 0.8689 |
| **Transfer learning (MobileNetV2)** | **0.8993** |

> We trained three models on the same data split.
>
> **First**, a from-scratch VGG-style CNN — three convolutional blocks with progressive dropout. It reaches 87.35 %. Solid, but not amazing.
>
> **Second**, we ran automated hyperparameter search with Keras Tuner Hyperband over seven hyperparameters — learning rate, dropout per block, weight decay, and so on. Thirty trials over an hour of GPU time. The result was 86.89 % — *slightly below* our hand-tuned baseline. This is actually an interesting finding: **the hand-tuning was already at the noise floor of automated search**, which means the architecture was the bottleneck, not the hyperparameters.
>
> **Third**, transfer learning. We took MobileNetV2, pre-trained on ImageNet, upscaled CIFAR images from 32×32 to 96×96, and fine-tuned in two stages. This jumped to **89.93 %** — and crucially, the gains landed on classes the baseline struggled with: **bird +6.2 percentage points, cat +4.4, frog +3.1**. Vehicle classes like ship barely moved, because they were already saturated.
>
> Now let's see the app.

---

## 1:30 – 2:45 · Live demo — Predict page

[Switch to live app at https://huggingface.co/spaces/ainichan/binus-ai-2026sem3-assignment2-group04 — show full window]

> This is the live deployment on Hugging Face Spaces. Three pages — Predict, Metrics, About — and at the top of the sidebar, a language switcher.

[Click the language dropdown → switch to Bahasa Indonesia → all UI text changes]

> The app supports English and Bahasa Indonesia at runtime. Every label, button, and status message is i18n-keyed.

[Switch back to English. Go to Predict page. Pick a sample image — e.g., the cat sample.]

> On the Predict page, I can upload an image, pick a built-in CIFAR sample, or use my webcam. Let's start with a sample — this 32×32 cat thumbnail.

[Wait for prediction to render, ~1 second]

> The model gives us a Top-3 prediction list with confidence scores. Cat at the top — correct. Below that, **Grad-CAM** — a heatmap showing which pixels drove the prediction. Notice it's looking at the face and body region, not the background, which is what we want.

[Now upload a real-world high-resolution photo from a folder — ideally one that *fails* or has lower confidence. If a real-world cat photo, narrate the result.]

> Now let's try a real-world photo from the web — a higher-resolution image the model has never seen at this scale. The prediction is correct, but the confidence drops compared to the in-distribution CIFAR sample. This is the **domain gap**: our model was trained on 32×32 thumbnails, so high-resolution natural photos sit slightly outside its training distribution. We discuss this honestly in our report.

---

## 2:45 – 3:45 · Metrics page tour

[Click Metrics tab in sidebar. Default selection is Transfer (MobileNetV2 fine-tuned).]

> The Metrics page shows the model's evaluation in detail. Test accuracy 89.93 %, test loss 0.297, about 1.5 million trainable parameters during fine-tuning.

[Scroll to the model comparison table.]

> Here's the three-model side-by-side. Then training curves — you can see Stage 1 head training, then a transition into Stage 2 fine-tuning where validation accuracy climbs from around 86 % to 90 %.

[Scroll to per-class report.]

> The per-class table reveals where the model is strong — automobile, ship, truck — and where it's still weak. Cat is our worst class at F1 0.80, but that's already significantly better than the baseline's 0.75.

[Scroll to confusion matrix.]

> The confusion matrix confirms the diagonal is dominant. Off-diagonal errors cluster between visually similar classes — cat versus dog, deer versus horse, bird versus airplane. These are well-known CIFAR-10 difficulty patterns, not failures of our specific model.

---

## 3:45 – 4:25 · Architecture and "why this works" (Slide or About page)

[Switch to About page, or back to slides showing the model architecture diagram.]

> A quick look under the hood. Our final model is MobileNetV2 with the classifier head replaced — Global Average Pooling, Dropout, and a 10-way Dense layer.
>
> Two-stage training is the key. **Stage 1**: backbone frozen, train just the head — 6 epochs. **Stage 2**: unfreeze the top 30 layers and fine-tune at a 100× smaller learning rate — 20 epochs. The two-stage schedule prevents random head gradients from destroying the pretrained backbone weights.
>
> One subtle but critical detail: BatchNormalization layers must stay in inference mode throughout. Without that single flag, accuracy drops by 5 to 10 points.

---

## 4:25 – 4:50 · Limitations and future work

[Slide: "Limitations & Future Work"]

> Two honest limitations to flag.
>
> **One — domain gap.** CIFAR-10 trains on 32×32 thumbnails. Real-world photos at 200 pixels and up will sometimes confuse the model even after transfer learning. This is a *dataset* problem, not a model problem.
>
> **Two — hyperparameter ceiling.** On this architecture and resolution, automated search couldn't beat hand-tuning. Future gains require either a bigger backbone, stronger augmentation like Mixup or CutMix, or higher input resolution — not more hyperparameter tweaking.

---

## 4:50 – 5:00 · Closing

[Slide: project URL, GitHub URL, team]

> The full code, both English and Indonesian notebooks, and the live demo are all linked in the description.
>
> **Live app:** huggingface.co/spaces/ainichan/binus-ai-2026sem3-assignment2-group04
>
> Thanks for watching.

---

## Recording notes

- **Two windows ready:**
  1. Browser at the live HF Space, logged out (clean view).
  2. Slide deck for opening, three-model comparison, architecture, limitations, closing.
- **Three test images to predict (have them ready in a folder):**
  1. A built-in sample (cat or ship — high-confidence success).
  2. A real-world photo where the model still gets it right (lower confidence — illustrates domain gap softly).
  3. *Optional:* an obvious failure (e.g., a deer photo predicted as horse) for the limitations section. If time runs short, drop this and just narrate over the confusion-matrix view.
- **Pace:** target ~150 words per minute. The script is written tight — read at a natural pace; transitions between sections cover most of the slack.
- **OBS / QuickTime tip:** record at 1080p, 30 fps. Loud, clear audio matters more than visual polish for grading.
- **Cuts to keep an eye on:** the language-switch moment (1:50), the Grad-CAM reveal (2:15), the training-curves moment with the Stage 1→2 transition (3:00). Hold each for ~2 seconds so the viewer can register the visual.
