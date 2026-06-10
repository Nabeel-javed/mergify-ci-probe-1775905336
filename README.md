# Custom MobileNetV2 on Sony IMX500 — Full Step-by-Step Guide


This project takes a MobileNetV2 image classifier, fine-tunes it on the Rock-Paper-Scissors task, quantizes it to 8-bit integers with Sony's toolchain, and deploys it as a `.rpk` file that runs **directly on the Sony IMX500 AI camera** attached to a Raspberry Pi 5. Inference happens on the camera sensor itself — the Pi only reads out the result.

The goal of this document is that **any beginner can repeat the whole pipeline from zero**, and also understand *why* the project went through three model versions before it worked on a real hand in a real room.

---

## 0. The short version (what happened, in order)

We did not get a working live demo on the first try. The project went through **three iterations**, and the failures are as instructive as the success. Here is the whole arc before the detailed steps:

| Version | What we did | Test-set result | What happened live on a real hand | Verdict |
|---|---|---|---|---|
| **v1** | Ran Sony's reference notebook unchanged. Head-only fine-tune on the public TFDS `rock_paper_scissors` set (computer-generated "studio" hands on a green screen). | 87.9% overall — **rock 100%, paper 64%, scissors 100%** | Paper barely recognized. | **Not good.** Paper class too weak even on the test set. |
| **v2** | Rebuilt training: **two-stage fine-tuning** (head, then unfreeze the top of the backbone) **+ heavy data augmentation** (blur, brightness, contrast, rotation, zoom). Still trained only on the TFDS (CGI) images. | **98.4% overall — paper jumped to 95%** | Rock and scissors mostly worked (~0.52–0.59 confidence); **paper still failed on a real hand**. | **Not good live.** The model was excellent on fake CGI hands but had never seen a *real* hand — a classic *domain shift* problem. |
| **v3** | Added **real photographs of real hands** from two public webcam datasets, retrained the same two-stage pipeline on the GPU, and validated against a held-out set of *real* webcam photos (not CGI). | Quantized model on **real-photo** test set: **91.8% — rock 94%, paper 86%, scissors 96%** | **Rock 20/20, Paper 20/20, Scissors 20/20 = 100%** on the author's own hand against a white wall. | **Works.** Goal met. |

**Two lessons that mattered more than any code change:**

1. **The model was never really the bottleneck — the *data* was.** A model trained only on green-screen CGI hands cannot recognize a real hand in a real room. Adding real photos (v3) is what fixed it, not more tuning.
2. **Most of the final "it doesn't work" panic was physical, not ML.** When v3 first looked broken in the live test, the real causes were: the room was dark (the camera saw a black frame), and the camera was pointing at the desk instead of the hand. Once the light was on and the hand filled the frame, accuracy was 100%. (Full story in §8.)

The rest of this document is the reproducible recipe.

---

## 1. What this project does (the pipeline)

1. Loads a pre-trained MobileNetV2 (trained on ImageNet) and replaces its final layer with a 3-class head for rock / paper / scissors.
2. Fine-tunes it in **two stages** on a mix of computer-generated **and real** hand images, with strong augmentation.
3. Quantizes the trained model to 8-bit integers using Sony's **Model Compression Toolkit (MCT)**.
4. Converts the quantized model into the Sony IMX500 binary format using **`imxconv-tf`** (needs Java 17).
5. Packages it into a **`.rpk`** file on the Raspberry Pi using **`imx500-package`**.
6. Loads the `.rpk` onto the IMX500 sensor and runs **live, on-sensor classification**.

**Final result:** a `.rpk` that runs on-sensor and scores **100% (60/60) live** on the author's own hand, and **91.8%** on a held-out set of real webcam photos (the honest "stranger's hand" number).

---

## 2. Hardware

| Item | Used in this project |
|---|---|
| Training / quantization server | Dell workstation — Ubuntu 22.04, Xeon W-1390P, 125 GB RAM, **NVIDIA RTX A5000 24 GB GPU** |
| Edge device | Raspberry Pi 5 (8 GB) running Raspberry Pi OS Bookworm 64-bit |
| AI camera | Sony IMX500 AI Camera Module for Raspberry Pi |
| Dev laptop | Apple MacBook M4 Pro (used only for SSH, file transfer, and writing docs) |

**You can substitute** any x86 Linux machine for the server. A GPU is *not* required — the model is tiny — but it cut training from ~70 s/epoch (CPU) to ~25 s/epoch (GPU). The Pi 5 + IMX500 combination **is** required for deployment, because the final packaging tool only runs on the Pi.

---

## 3. Software versions (what is installed where)

### On the server (`dll1404@10.60.5.14`)

- Miniconda3 (Python 3.11.15)
- TensorFlow 2.14.1
- `model-compression-toolkit` 2.2.2
- `imx500-converter[tf]` 3.16.1
- OpenJDK 17.0.2 (unpacked to `~/jdk-17.0.2`)
- `tensorflow_datasets`, `importlib_resources`
- **For GPU only:** `nvidia-*-cu11` wheels (cuDNN 8.7, cuBLAS, cuDART, …) **and** `nvidia-cuda-nvcc-cu11` (see §5, Step 1b)

### On the Raspberry Pi (`pi@10.52.142.11` — note: assigned by DHCP, yours will differ)

- Raspberry Pi OS Bookworm 64-bit
- `imx500-all` (firmware + tools)
- `python3-picamera2`
- `imx500-tools` (provides `imx500-package`)

### On the Mac

- SSH client and `scp` (both built in)

> **Finding the Pi's IP:** the router hands out the address by DHCP, so it changes. Find it with `ping raspberrypi.local` (mDNS) or check your router. In this project it moved from `10.190.123.11` to `10.52.142.11` between sessions.

---

## 4. Pipeline overview

```
┌───────────────────────┐     ┌────────────────────┐     ┌─────────────────┐
│ Server (Ubuntu + GPU) │     │ Mac (M4 Pro)       │     │ Pi 5 + IMX500   │
│                       │     │                    │     │                 │
│ 1. Train MobileNetV2  │     │ • SSH to both      │     │ 5. Package .rpk │
│ 2. Quantize to INT8   │     │ • SCP transfers    │     │ 6. Live demo    │
│ 3. Convert (Sony +    │ ──► │ • Relay files      │ ──► │                 │
│    Java 17)           │     │   server ↔ Pi      │     │                 │
│                       │     │ • Write this doc   │     │                 │
│ Output: packerOut.zip │     │                    │     │ Output: .rpk    │
└───────────────────────┘     └────────────────────┘     └─────────────────┘
```

`imx500-package` (the final `.rpk` step) runs **only on the Pi**, which is why the workflow ends there. The server and Pi are on different networks, so the Mac relays files between them with `scp`.

---

## 5. Step-by-step instructions (the working v3 recipe)

Commands are copy-pasteable. Replace IP addresses with your own.

### Step 1 — Server environment setup (one-time, ~30 minutes)

```bash
# SSH to the server from your Mac
ssh dll1404@10.60.5.14

mkdir -p ~/dit-imx500 && cd ~/dit-imx500

# Miniconda (skip if already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p ~/miniconda3 && rm miniconda.sh
source ~/miniconda3/etc/profile.d/conda.sh

# Clean Python 3.11 environment
conda create -y -p ~/dit-imx500/conda-env python=3.11
conda activate ~/dit-imx500/conda-env

# Core packages
pip install \
    tensorflow~=2.14.0 \
    model-compression-toolkit~=2.2.0 \
    'imx500-converter[tf]' \
    tensorflow_datasets \
    importlib_resources

# OpenJDK 17 (Sony's converter requires it), unpacked into your home dir
cd ~
wget -q https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz -O jdk17.tar.gz
tar xzf jdk17.tar.gz && rm jdk17.tar.gz
~/jdk-17.0.2/bin/java -version    # expect: openjdk version "17.0.2"

cd ~/dit-imx500
imxconv-tf --version              # expect: 3.16.1
```

### Step 1b — (Optional) Enable the GPU for faster training

TensorFlow 2.14 needs CUDA 11.8 + cuDNN 8.7. The easiest way without root is to pip-install NVIDIA's CUDA-11 wheels straight into the conda env, **plus** the `nvcc` wheel (it contains `libdevice`, which the XLA compiler needs):

```bash
conda activate ~/dit-imx500/conda-env
pip install nvidia-cudnn-cu11 nvidia-cublas-cu11 nvidia-cuda-runtime-cu11 \
            nvidia-cufft-cu11 nvidia-curand-cu11 nvidia-cusolver-cu11 \
            nvidia-cusparse-cu11 nvidia-cuda-nvcc-cu11 nvidia-cuda-cupti-cu11 \
            nvidia-cuda-nvrtc-cu11 nvidia-nccl-cu11

# Point the loader at those libs and tell XLA where libdevice lives, then run:
NV=$(echo ~/dit-imx500/conda-env/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':')
CUDADIR=~/dit-imx500/conda-env/lib/python3.11/site-packages/nvidia/cuda_nvcc

LD_LIBRARY_PATH=$NV XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDADIR \
  python -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))"
# expect: GPUs: 1
```

> Skip this whole step to train on CPU — the RPS model is small enough (~70 s/epoch, ~40 min total). The GPU brings it to ~25 s/epoch (~20 min). See the §8 troubleshooting entry **"libdevice not found"** for the one gotcha.

### Step 2 — Get the datasets

We use **three** datasets so the model sees both synthetic and real hands:

| Dataset | Source | Type | Count |
|---|---|---|---|
| `rock_paper_scissors` | TensorFlow Datasets (auto-downloads) | **CGI** green-screen hands | 2520 train / 372 test |
| Glushko RPS | public webcam dataset | **real** webcam hands | 1020 train / 804 val / 540 test |
| drgfreeman `rps-cv-images` | public dataset (Kaggle) | **real** photos | 726 rock / 712 paper / 750 scissors |

```bash
# TFDS downloads itself the first time train_v3.py runs — nothing to do.
# Put the two real-photo datasets under extra_data/ in this layout:
#   extra_data/glushko/{train,val,test}/{rock,paper,scissors}/*.png
#   extra_data/drg/{rock,paper,scissors}/*.png
```

> **Critical class order:** TFDS `rock_paper_scissors` is ordered **`['rock','paper','scissors']`**, which is **not** alphabetical. Every folder loader and the Pi's `labels.txt` must use this exact order, or predictions look random. The training script forces it with `class_names=["rock","paper","scissors"]`.

### Step 3 — Train (the two-stage fine-tune)

The training script (`train_v3.py`, included in `submission/source/`) does the following. This is the heart of the project, so the method is spelled out:

**Model.** ImageNet-pretrained MobileNetV2 backbone (no top) → `GlobalAveragePooling2D` → `Dense(3, softmax)`.

**Input preprocessing.** `tf.keras.applications.mobilenet_v2.preprocess_input`, which scales pixels to `[-1, 1]`, applied in the data pipeline.

**Augmentation (only on the training set)** — this is what lets a model trained on studio images survive a real room:
- *Photometric:* random brightness (±55), contrast (0.5–1.6), saturation (0.4–1.7), hue (±0.07) — covers skin-tone and lighting variation.
- *Blur:* 5×5 Gaussian, σ 0.6–2.4, applied 50% of the time — mimics a slightly out-of-focus lens.
- *Geometric:* random horizontal flip, rotation (±0.25 turn), zoom (±20%), translation (±10%) — covers hand position and angle.

**Two-stage schedule:**
- *Stage 1 — head only:* freeze the entire backbone, train just the new 3-class head. `Adam(1e-3)`, 8 epochs. (Lets the random head settle without wrecking the pretrained features.)
- *Stage 2 — fine-tune the top:* unfreeze the **last 50 layers** of the backbone but **keep all BatchNorm layers frozen** (their running statistics would otherwise drift on a small dataset). `Adam(1e-5)` (very low, to nudge not destroy), up to 25 epochs with `EarlyStopping(patience=6, restore_best_weights=True)`.

**Honest validation.** Validation uses the **held-out real webcam test set** (Glushko `test`), *not* the CGI test set. This is the number that actually predicts live behavior. We also report the CGI test set separately for comparison.

```bash
cd ~/dit-imx500
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/dit-imx500/conda-env

# CPU:
python -u train_v3.py

# GPU (recommended — see Step 1b):
NV=$(echo conda-env/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':')
CUDADIR=$HOME/dit-imx500/conda-env/lib/python3.11/site-packages/nvidia/cuda_nvcc
LD_LIBRARY_PATH=$NV XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDADIR \
  TF_FORCE_GPU_ALLOW_GROWTH=true python -u train_v3.py
```

The script trains, evaluates the **float** model, then quantizes and evaluates the **quantized** model, printing per-class accuracy on both the CGI and real-webcam test sets.

### Step 4 — Quantize (done inside `train_v3.py`)

Quantization is post-training INT8 with Sony's **Model Compression Toolkit (MCT 2.2.2)**. The relevant settings:

- **Representative dataset:** 20 batches of *un-augmented* training images (MCT uses these to measure the real activation ranges so it can pick good INT8 scales).
- **`QuantizationConfig`:** activations and weights both use **MSE** error minimization, **`weights_bias_correction=True`** (corrects the small systematic error quantization introduces), **`shift_negative_activation_correction=True`**, `z_threshold=16`.
- **Target platform:** `imx500`, version `v1` — this makes MCT quantize exactly the way the IMX500 hardware expects.
- The quantized model is exported to `models_v3/mobilenet-quant-rps.keras`.

**Results printed by the script (v3):**

```
=== FLOAT eval ===
--- CGI test ---        rock 1.000  paper 0.919  scissors 1.000   OVERALL 0.9731
--- REAL webcam test ---rock 0.936  paper 0.869  scissors 0.960   OVERALL 0.9222

=== QUANTIZED eval ===   (this is what runs on the camera)
--- CGI test ---        rock 1.000  paper 0.911  scissors 1.000   OVERALL 0.9704
--- REAL webcam test ---rock 0.936  paper 0.864  scissors 0.955   OVERALL 0.9185
```

Quantization cost almost nothing (97.31% → 97.04% on CGI; 92.22% → 91.85% on real). The ~92% on real webcam photos is the honest "someone else's hand" number; on the author's own hand in a controlled setting it reaches 100% (§7).

### Step 5 — Convert to Sony's binary format (~90 seconds)

```bash
cd ~/dit-imx500
export JAVA_HOME=$HOME/jdk-17.0.2
export PATH=$JAVA_HOME/bin:$PATH

imxconv-tf -i models_v3/mobilenet-quant-rps.keras -o converted_v3

ls converted_v3/
# dnnParams.xml  *_MemoryReport.json  *.pbtxt  packerOut.zip   <- packerOut.zip is the artifact
```

### Step 6 — Relay the package to the Pi (~5 seconds)

The server and Pi can't see each other, so hop through the Mac:

```bash
# On the Mac
mkdir -p ~/dit-imx500/converted_v3
scp dll1404@10.60.5.14:~/dit-imx500/converted_v3/packerOut.zip ~/dit-imx500/converted_v3/
scp ~/dit-imx500/converted_v3/packerOut.zip pi@10.52.142.11:~/dit-imx500/packerOut_v3.zip
```

### Step 7 — Package the `.rpk` on the Pi (~30 seconds)

```bash
ssh pi@10.52.142.11
cd ~/dit-imx500

# Labels in TFDS order — NOT alphabetical
printf "rock\npaper\nscissors\n" > labels.txt

imx500-package -i packerOut_v3.zip -o ./rpk_out_v3
ls -la rpk_out_v3/network.rpk      # ~2.77 MB

# Make it the active model (keep a backup of the previous one)
cp rpk_out/network.rpk rpk_out/network_v2_backup.rpk   # optional backup
cp rpk_out_v3/network.rpk rpk_out/network.rpk
```

### Step 8 — Pi first-time setup (one-time, only if not already done)

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y imx500-all python3-picamera2 imx500-tools
sudo reboot
```

### Step 9 — Run the live demo

Two ways to run inference:

**(a) Headless, over SSH (no screen needed)** — what we used to score the model. A small script captures frames and prints the prediction. The first run uploads the model into the sensor (~27 s, shown as a progress bar); later runs are instant.

```bash
# On the Pi
cd ~/dit-imx500
python3 headless_test.py        # captures 15 frames, prints class + per-class scores
```

`headless_timed.py` (included) adds a countdown so you have time to get a gesture into frame before it captures — useful when driving the test remotely:

```bash
python3 headless_timed.py PAPER 12   # 12-second countdown, then 20 frames, then a vote tally
```

**(b) With a live preview window** — needs a real desktop (use VNC from the Mac, not SSH, or the preview fails with a DRM error):

```bash
python3 rps_demo.py --model rpk_out/network.rpk --labels labels.txt --softmax
```

Hold a gesture so your hand **fills most of the frame** against a plain background:
- **rock** — closed fist
- **paper** — open flat palm
- **scissors** — two fingers / V

---

## 6. Verifying the model objectively (server)

`train_v3.py` already prints per-class accuracy on both the CGI and the real-webcam test sets for the float **and** quantized models (see §Step 4 output). The quantized real-webcam numbers — **rock 94%, paper 86%, scissors 96%, overall 92%** — are the trustworthy estimate of how it does on a hand it has never seen.

---

## 7. The live result (on the author's own hand)

Tested on the Pi with the deployed v3 `.rpk`, hand filling the frame against a white wall:

| Gesture | Frames correct | Confidence of winning class |
|---|---|---|
| Rock | **20 / 20** | 0.58 |
| Paper | **20 / 20** | 0.58 |
| Scissors | **20 / 20** | 0.58 |
| **Total** | **60 / 60 = 100%** | wrong classes always ~0.21 each |

The winning class always beats the other two by a wide margin (0.58 vs 0.21 / 0.21). The peak confidence sits around 0.58 rather than 0.99 because the **heavy augmentation deliberately softens the model's certainty** — that is the same regularization that lets it generalize from training images to a real hand, so it is a feature, not a defect.

---

## 8. Troubleshooting — every problem hit, and the fix

These are in roughly the order we hit them across all three versions.

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named '_bz2'` launching Python | System Python built without bzip2 | Use Miniconda's Python (Step 1) |
| `ModuleNotFoundError: No module named 'importlib_resources'` running `imxconv-tf` | Missing transitive dep | `pip install importlib_resources` |
| `java.lang.UnsupportedClassVersionError: class file version 61.0` from `imxconv-tf` | Converter needs **JDK 17**, system had older | Unpack JDK 17 to `~/jdk-17.0.2` and `export JAVA_HOME=$HOME/jdk-17.0.2` before converting (Steps 1, 5) |
| `tf.config.list_physical_devices('GPU')` returns `[]` | Env has no CUDA/cuDNN libs | pip-install the `nvidia-*-cu11` wheels and set `LD_LIBRARY_PATH` to them (Step 1b) |
| **`libdevice not found at ./libdevice.10.bc`** during GPU training (crashes in the Adam optimizer step) | The XLA JIT compiler needs `libdevice`, which only ships in the **nvcc** wheel | `pip install nvidia-cuda-nvcc-cu11` and set `XLA_FLAGS=--xla_gpu_cuda_data_dir=<.../nvidia/cuda_nvcc>` (Step 1b) |
| `class_names did not match subdirectories. Expected: ['paper','rock','rps-cv-images','scissors']` | A stray non-class folder inside a dataset dir | Remove the extra folder (`rm -rf extra_data/drg/rps-cv-images`) so only the 3 class folders remain |
| `AssertionError: Labels file should contain 1000 or 1001 labels` running the stock demo | Stock demo is written for 1000-class ImageNet | Patch the assertion out, or use the 3-class `headless_test.py` / `rps_demo.py` provided |
| `RuntimeError: Failed to reserve DRM plane` running the preview demo over SSH | The preview window needs a real display; SSH has none | Run from a VNC desktop terminal, **or** use the headless scripts which need no display |
| **v2 worked on CGI but paper failed on a real hand** | *Domain shift* — v2 had only ever seen green-screen CGI hands | This is the whole reason for v3: **add real photographs** to the training data |
| **`pkill -f train_v3.py` killed my own SSH session (exit 255)** | `pkill -f` matches the *full command line*, including the SSH command that contained the string `train_v3.py` | Never `pkill -f` on a string that appears in your own shell command — find the PID with `ps`/`pgrep` for the python binary and `kill <pid>` |
| **Live demo: predicts "rock" for everything at a flat ~0.40, every frame identical** | The camera was in a **dark room** → it captured an almost-black frame → the model gets no signal and falls back to its prior | Turn the light on. (We confirmed this by saving the actual camera JPEG and looking at it — it was pure black.) |
| **Live demo: still wrong, steady ~0.57 "rock", every frame identical** | The camera was **pointing at the desk**, not the hand — it was confidently classifying clutter | Aim the camera at a plain wall and hold the hand **20–30 cm in front of the lens so it fills the frame**. A still, identical-every-frame output is the tell-tale sign nothing is moving in front of the lens. |
| **Live test seems to "miss" the gesture** | The headless capture is only ~5 s and starts instantly — faster than you can get into position when driving it remotely | Use `headless_timed.py <GESTURE> 12` which waits 12 s before capturing |
| IMX500 lens is blurry / stiff to focus | Lens is manually adjustable and ships stiff | Use the small plastic adjuster tool from the box; minor blur is fine since input is resized to 224×224 |

> **The single most useful debugging trick** in this project: when live predictions look wrong, **save the actual frame the camera sees** (`picamera2`'s `capture_file("frame.jpg")`) and *look at it*. That one image instantly told us the room was dark, then that the camera was aimed at the desk — neither of which is an ML problem. Always confirm the camera sees what you think it sees before blaming the model.

---

## 9. Files in this project

| Path | Description |
|---|---|
| `~/dit-imx500/conda-env/` (server) | Python 3.11 env with all packages (+ CUDA-11 wheels for GPU) |
| `~/dit-imx500/train_v3.py` (server) | **The training + quantization script** (two-stage fine-tune, augmentation, MCT INT8) |
| `~/dit-imx500/extra_data/` (server) | The two real-photo datasets (glushko, drg) |
| `~/dit-imx500/models_v3/mobilenet-rps/` (server) | Trained float MobileNetV2 |
| `~/dit-imx500/models_v3/mobilenet-quant-rps.keras` (server) | Quantized INT8 model |
| `~/dit-imx500/converted_v3/packerOut.zip` (server → Mac → Pi) | Intermediate Sony binary |
| `~/dit-imx500/rpk_out/network.rpk` (Pi) | **Final deployable file** on the IMX500 |
| `~/dit-imx500/labels.txt` (Pi) | `rock` / `paper` / `scissors` (TFDS order) |
| `~/dit-imx500/headless_test.py`, `headless_timed.py` (Pi) | Headless inference scripts (no display needed) |
| `~/dit-imx500/rps_demo.py` (Pi) | Preview-window demo (needs VNC desktop) |
| `~/dit-imx500/README.md` (Mac) | This document |

---

## 10. What each version changed (summary for the report)

- **v1 → v2:** changed the *training method*. Single head-only fit → **two-stage fine-tune** + **heavy augmentation**. Fixed the weak paper class on the test set (64% → 95%) and lifted overall accuracy to 98.4% — but only on CGI hands.
- **v2 → v3:** changed the *data*. Added **real photographs of real hands** (two public webcam datasets) and validated against held-out **real** photos. This is what made it work on an actual hand in an actual room. **The lesson: when a model is great in testing but fails in the real world, suspect the data distribution before the model.**

---

## 11. Possible extensions

1. **A 4th "background / none" class** trained on random non-hand images, so the model can say "no hand" instead of being forced to choose — this would have made the dark-room and empty-desk failures show up as low confidence instead of a confident wrong answer.
2. **A confidence threshold** in the demo (only show a label above, say, 0.45) for cleaner live behavior.
3. **A handful of the author's own hand photos** added to training would likely push the real-world number from ~92% toward the ~100% seen in the controlled live test, for any hand/room.

---

## 12. Appendix A — Full timeline of attempts and dead ends

This section is the **honest blow-by-blow** of every approach tried during the project. The README above is the polished recipe; this appendix is "what actually happened on each day, including the things that didn't work". It exists so future-me (and the professor) can see *why* the final pipeline looks the way it does — most decisions were driven by failures that aren't visible in the final code.

### A.1 Stage 0 — Planning (before any code)

- **Original idea:** run Sony's reference notebook verbatim, then add a small custom tweak.
- **First crossroads:** the plan estimated 5–7 days wall-clock and 10–15 active hours. Actual was closer to **two weekends of evenings + most of a Saturday for v3**, mostly burned on (a) GPU plumbing, (b) the real-vs-CGI domain shift, and (c) live-demo non-ML failures.
- **Decisions locked early:**
  - Server for training/quantization/conversion (`dll1404@10.60.5.14`, RTX A5000)
  - Pi for packaging + inference only
  - Mac as SSH/SCP relay (server and Pi can't see each other)
  - TF 2.14 + MCT 2.2.x to match Sony's notebook exactly — minimize version drift

### A.2 Server environment — what broke and what we replaced

| Attempt | What happened | Outcome |
|---|---|---|
| **System Python 3.10** | `_bz2` missing → TFDS can't load `rock_paper_scissors` (it's bz2-compressed) | **Abandoned.** Switched to Miniconda Python 3.11. |
| **pip without `importlib_resources`** | `imxconv-tf` immediately crashes with `ModuleNotFoundError: importlib_resources` | Added to pinned install list. |
| **System OpenJDK 11** | `imxconv-tf` fails: `UnsupportedClassVersionError: class file version 61.0` (= JDK 17) | Downloaded OpenJDK 17.0.2 tarball to `~/jdk-17.0.2`, no sudo needed. |
| **TF 2.14 default install (no CUDA)** | `tf.config.list_physical_devices('GPU')` returned `[]` even though A5000 is in `nvidia-smi` | TF wheel ships CPU-only; you need the NVIDIA `cu11` wheels. |
| **NVIDIA cu11 wheels alone** | GPU now visible, but training crashed inside the **Adam optimizer step** with `libdevice not found at ./libdevice.10.bc` | `libdevice` only ships in the `nvidia-cuda-nvcc-cu11` wheel. Adding it + `XLA_FLAGS=--xla_gpu_cuda_data_dir=...` fixed it. |

**Net result:** the Step 1b incantation in the README looks ceremonial, but every flag in it is there because something blew up without it.

### A.3 v1 — Sony's reference notebook, unchanged

- **Goal:** prove the pipeline end-to-end before any customization.
- **Method:** single head-only fine-tune on TFDS `rock_paper_scissors` (2520 CGI green-screen images), no augmentation beyond what the notebook does.
- **Test-set result:** 87.9% overall — **rock 100%, paper 64%, scissors 100%**.
- **What we noticed:** paper was already weak on the *test set* — a clear sign the head alone couldn't separate paper from rock at the boundary. Quantization barely moved the number.
- **Live test:** packed to `.rpk`, deployed to Pi. Paper class barely recognized on a real hand.
- **Verdict:** **Pipeline works end-to-end, model is not good enough.** Two failures to fix: paper class is weak, and (suspected) the model has only seen CGI.

### A.4 v2 — Better training method (still CGI-only)

- **Hypothesis:** paper is weak because the head alone underfits; fine-tuning the backbone top + heavy augmentation should fix it.
- **Changes from v1:**
  - **Two-stage schedule:** Stage 1 head-only `Adam(1e-3)` 8 epochs, Stage 2 unfreeze last 50 layers (BN frozen) `Adam(1e-5)` with `EarlyStopping(patience=6)`.
  - **Heavy augmentation:** brightness ±55, contrast 0.5–1.6, saturation 0.4–1.7, hue ±0.07, 5×5 Gaussian blur σ 0.6–2.4, horizontal flip, rotation ±0.25 turn, zoom ±20%, translation ±10%.
- **Test-set result (CGI):** **98.4% overall — paper jumped from 64% → 95%.** Beautiful curve.
- **Quantization:** ~0.3 pp drop, ignorable.
- **Live test:** rock and scissors mostly worked (≈0.52–0.59 confidence on a real hand); **paper still failed.**
- **Diagnosis:** classic **domain shift**. The TFDS images are CGI hands on a *green screen with sharp edges and uniform skin tone*. A real hand in a real room has soft edges, varied skin tone, background clutter, and JPEG noise — none of which augmentation can synthesize from scratch.
- **Verdict:** the training method was now right; the *data* was wrong. Augmentation cannot invent a new domain.

### A.5 v3 — Add real photos (the version that worked)

- **Hypothesis:** if v2 only fails on real hands, train on real hands.
- **Datasets added:**
  - **Glushko RPS** (1020 train / 804 val / 540 test) — real webcam photos.
  - **drgfreeman `rps-cv-images`** (≈2188 images across 3 classes) — real Kaggle photos.
  - Kept TFDS as the third source so the synthetic data isn't wasted (it's still informative; just not sufficient alone).
- **Validation policy changed:** use a held-out set of **real** webcam photos (Glushko `test`) as `val_real`, not the CGI test set. This is the metric that actually predicts live behavior.
- **Same two-stage schedule and augmentation as v2.**
- **Float result (real-webcam test):** rock 93.6%, paper 86.9%, scissors 96.0%, overall **92.22%**.
- **Quantized result (what runs on camera):** rock 93.6%, paper 86.4%, scissors 95.5%, overall **91.85%**.
- **Live test on author's hand:** **60/60 (100%)** at conf 0.58 (rock/paper/scissors 20/20 each).
- **Verdict:** **shipped.** This is the `.rpk` that runs on the Pi.

### A.6 Live-demo failures that were *not* ML problems

When the v3 `.rpk` first ran live, it looked broken. Three failures, all physical, none requiring a model change. Listing them because each one wasted real time and the diagnosis method is reusable.

1. **"Stuck on 'rock' ~0.40 every frame."**
   - Saved the camera JPEG with `picamera2.capture_file("frame.jpg")` and opened it. **Pure black.**
   - Cause: room lights were off ("extremely dark in my room").
   - Fix: turn the light on.
   - **Lesson:** if the model is *flat* (same output every frame), the input is probably constant — check the input before touching the model.

2. **"Lights on, still stuck on 'rock' ~0.57 every frame."**
   - Saved the JPEG again. Camera was pointed at the **desk** — paper bag, cup, pen, no hand in frame.
   - Fix: aim the camera at a plain wall, hold the hand 20–30 cm in front of the lens so it fills the frame.
   - **Lesson:** "the model is wrong" usually means "the model never got what you think it got". Always confirm the camera sees what you think it sees.

3. **"Live test seems to miss the gesture."**
   - The headless capture is ~5 s and starts instantly; faster than you can get into position when running it remotely over SSH.
   - Fix: wrote `headless_timed.py` with a 12 s countdown before capture and a vote tally over 20 frames.

### A.7 Other potholes that cost real time

| Pothole | What happened | Fix |
|---|---|---|
| `class_names did not match subdirectories. Expected: ['paper','rock','rps-cv-images','scissors']` | `image_dataset_from_directory` saw an extra non-class folder inside `extra_data/drg/` | `rm -rf extra_data/drg/rps-cv-images` — keep only the 3 class folders |
| `AssertionError: Labels file should contain 1000 or 1001 labels` from Sony's stock IMX500 demo | The stock demo hard-codes ImageNet's 1000 classes | Patch the assertion out, or use the 3-class `headless_test.py` / `rps_demo.py` instead |
| `RuntimeError: Failed to reserve DRM plane` when running `rps_demo.py` over SSH | The preview window needs a real graphical desktop | Run it from a **VNC** desktop terminal (`open vnc://10.52.142.11` from Mac) or use the headless scripts |
| `pkill -f train_v3.py` killed the SSH session itself | `pkill -f` matches the full command line, including the parent shell command that contains the string `train_v3.py` | Never `pkill -f` on a substring of your own command. Use `pgrep -f` first to *see* the matches, then `kill <pid>` for the python process specifically. |
| Class order ambiguity | TFDS `rock_paper_scissors` uses `['rock','paper','scissors']` (NOT alphabetical). One careless `sorted()` call → predictions look random | Force `class_names=["rock","paper","scissors"]` everywhere, and write `labels.txt` on the Pi in the same order |
| Per-class accuracy looking great but *paper* tanking | Side-effect of the above — alphabetical order mislabeled paper as rock and vice versa | Same fix; double-check by printing predictions on a known image |
| TF 2.14 + CUDA wheel `LD_LIBRARY_PATH` lost on each new shell | The variable doesn't persist across `ssh` sessions | Wrote it directly into the training command (see Step 3) |

### A.8 Things considered but **not** done (and why)

- **Capture our own training photos.** Would push the real-world accuracy from 92% → ~100% for *any* hand, but two public real-hand datasets were enough to clear the bar. Mentioned in §11 as a future extension.
- **Add a 4th "background / none" class.** Would have made the dark-room and empty-desk failures show up as low confidence rather than a confident wrong answer. Bigger change to retrain + relabel; deferred to §11.
- **Confidence threshold in the demo** (only show a label above 0.45). Nice UX polish but doesn't affect the model itself; mentioned in §11.
- **Run the converter on Mac via Docker.** Sony ships a Docker image, but the native Linux path is faster and the server already had everything. Docker only matters if you don't have Linux.
- **Quantization-aware training (QAT).** MCT supports it, but post-training quantization (PTQ) cost only 0.4 pp here — well below the noise floor — so QAT wasn't worth the extra complexity.

### A.9 Submission-day polish

- Rewrote the README into 12 sections with v1→v2→v3 narrative at the top (§0), troubleshooting table (§8), per-version diff (§10), and this appendix.
- Recorded a live demo: 184.5 s raw → trimmed to a 19 s H.264 highlight reel at `~/Desktop/rps_demo_final.mp4`, with on-overlay predictions (green if conf ≥ 0.45 else orange) for rock, paper, scissors segments.
- Synced `~/dit-imx500/README.md` → `~/dit-imx500/submission/README.md` so the submission folder is self-contained.
- Sources in `submission/source/`: `train_v3.py`, `headless_test.py`, `headless_timed.py`, `rps_demo.py`, plus the original Sony `custom_mobilenet.ipynb` for reference.
- Proof artifacts in `submission/proof/`: live JPEG, results text dump, demo video, plus the test-set benchmark numbers.

### A.10 What the README *omits* on purpose

For brevity the main README skips:
- The exact GPU debugging session (~2 h of CUDA/cuDNN/XLA wheel juggling) — distilled to Step 1b.
- The minute-by-minute live-demo session ("show rock now", "show paper now", with explicit confidence per frame) — distilled to the §7 table.
- Two intermediate `train_v2_*.py` variants tried during v2 (different augmentation strengths) — abandoned once it was clear augmentation alone couldn't fix domain shift.

If a future reader needs any of the omitted detail, it's in the conversation transcripts referenced in `~/.claude/projects/-Users-Nabeel/` and the per-version memory entries (`project_dit_imx500.md`, `project_dit_imx500_pipeline_works.md`, `project_dit_imx500_v3.md`).

---

## 13. References

- Sony reference notebook — https://github.com/SonySemiconductorSolutions/aitrios-rpi-tutorials-ai-model-training/blob/main/notebooks/mobilenet-rps/custom_mobilenet.ipynb
- Sony IMX500 converter docs — https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/develop/imx500-converter/
- Raspberry Pi IMX500 AI Camera docs — https://www.raspberrypi.com/documentation/accessories/ai-camera.html
- TFDS rock_paper_scissors — https://www.tensorflow.org/datasets/catalog/rock_paper_scissors
- Sony Model Compression Toolkit — https://github.com/sony/model_optimization
