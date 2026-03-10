
---

# AI4Bharat Indic ASR (GPU Setup)

![Python](https://img.shields.io/badge/python-3.10-blue)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![GPU](https://img.shields.io/badge/GPU-NVIDIA-green)
![NeMo](https://img.shields.io/badge/NVIDIA-NeMo-green)
![Docker](https://img.shields.io/badge/docker-supported-blue)
![Docker](https://img.shields.io/docker/pulls/ujjvalpatel1003/ai4bharat-asr-gpu)
![Docker Image](https://img.shields.io/docker/image-size/ujjvalpatel1003/ai4bharat-asr-gpu/latest)
![License](https://img.shields.io/badge/license-MIT-blue)

A **GPU-accelerated setup** for running **AI4Bharat Indic ASR models** using **NVIDIA NeMo**.

This repository provides a **stable, patched environment** to run AI4Bharat models locally with **CUDA acceleration**, **real-time streaming**, and **browser-based transcription UI**.

---

# Tested Hardware

This project was tested on the following system.

| Component | Value                          |
| --------- | ------------------------------ |
| GPU       | **NVIDIA RTX 3050 Laptop GPU** |
| CUDA      | **CUDA 12.6**                  |
| PyTorch   | CUDA 12.6 build                |
| Python    | 3.10                           |
| Framework | NVIDIA NeMo                    |

The setup script installs **CUDA-compatible PyTorch automatically**.

---

# Supported Models

| Model                                     | Type         | Languages           |
| ----------------------------------------- | ------------ | ------------------- |
| `indicconformer_stt_gu_hybrid_rnnt_large` | Gujarati     | Gujarati            |
| `indic-conformer-600m-multilingual`       | Multilingual | 20+ Indic languages |

The streaming ASR script supports **22 Indian languages**.

---

# Features

* GPU accelerated inference
* Real-time streaming ASR
* Terminal live transcription
* Browser UI interface
* 22 Indic languages supported
* Continuous transcription mode
* VAD-based utterance segmentation
* Transcript export
* Docker support
* HuggingFace model caching

---

# Project Structure

```
AI4Bharat-GPU/
│
├── setup.sh
├── requirements.txt
├── Dockerfile
│
├── ai4bharat-gu.py
├── ai4bharat-mul.py
│
├── live.py
├── live-ui.py
│
└── README.md
```

| File             | Description                    |
| ---------------- | ------------------------------ |
| setup.sh         | Complete GPU environment setup |
| requirements.txt | Dependency versions            |
| Dockerfile       | GPU Docker container           |
| ai4bharat-gu.py  | Gujarati ASR example           |
| ai4bharat-mul.py | Multilingual ASR example       |
| live.py          | Terminal real-time ASR         |
| live-ui.py       | Web UI ASR interface           |

---

# CUDA and PyTorch Compatibility

Different CUDA versions require specific PyTorch builds.

| CUDA Version  | PyTorch Wheel Index                                                              |
| ------------- | -------------------------------------------------------------------------------- |
| CUDA 11.8     | [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118) |
| CUDA 12.1     | [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121) |
| CUDA 12.4     | [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124) |
| **CUDA 12.6** | [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126) |

Example installation:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Your setup script installs the **CUDA 12.6 wheel automatically**.

---

# Installation (Local GPU Setup)

## 1 Clone repository

```
git clone https://github.com/YOUR_USERNAME/ai4bharat-asr-gpu.git
cd ai4bharat-asr-gpu
```

---

## 2 Run setup script

```
chmod +x setup.sh
./setup.sh
```

The setup script automatically:

* creates a virtual environment
* installs CUDA-enabled PyTorch
* clones AI4Bharat NeMo
* applies compatibility patches
* installs dependencies
* verifies GPU detection

---

## 3 Activate environment

```
source nemo/bin/activate
```

---

## 4 Login to HuggingFace

The models are gated.

```
huggingface-cli login
```

---

# Run Gujarati ASR

```
python ai4bharat-gu.py
```

Example:

```python
model = nemo_asr.models.ASRModel.from_pretrained(
    "ai4bharat/indicconformer_stt_gu_hybrid_rnnt_large"
)
```

GPU will be used automatically if available.

---

# Run Multilingual ASR

```
python ai4bharat-mul.py
```

The script:

* loads audio
* converts to mono
* resamples to **16 kHz**
* runs **CTC** and **RNNT** decoding

Example:

```
transcription_ctc = model(wav, "gu", "ctc")
```

---

# Real-Time Streaming ASR (Terminal)

```
python live.py
```

Features:

* microphone streaming
* language selection
* real-time transcription
* optional transcript saving

Example:

```
python live.py --utterance --save
```

---

# Web UI (Browser Interface)

```
python live-ui.py
```

Then open:

```
http://127.0.0.1:7860
```

Features:

* live transcription
* transcript history
* export transcripts
* language selection
* GPU acceleration

Built with **Gradio**.

---

# Docker Support

You can run the entire system using Docker with GPU support.

## Build Docker Image

```
docker build -t ai4bharat-asr-gpu .
```

---

## Run Container

```
docker run -it \
--gpus all \
-p 7860:7860 \
-v ~/.cache/huggingface:/root/.cache/huggingface \
ai4bharat-asr-gpu
```

Explanation:

| Flag                                               | Purpose                    |
| -------------------------------------------------- | -------------------------- |
| `--gpus all`                                       | Enables NVIDIA GPU access  |
| `-p 7860:7860`                                     | Exposes Gradio UI          |
| `-v ~/.cache/huggingface:/root/.cache/huggingface` | Persists downloaded models |

---

# Docker Hub Image

You can pull the prebuilt image (~9GB).

```
docker pull ujjvalpatel1003/ai4bharat-asr-gpu
```

Run it:

```
docker run -it --gpus all -p 7860:7860 ujjvalpatel1003/ai4bharat-asr-gpu
```

---

# Model Download

Models download automatically on first run.

Approximate size:

```
~1.8 GB
```

Cached in:

```
~/.cache/huggingface
```

If using Docker without volume mounting, the model will be stored in:

```
/root/.cache/huggingface
```

Mounting the cache directory prevents repeated downloads.

---

# Audio Requirements

Input audio must be:

```
Format: WAV
Channels: Mono
Sample rate: 16000 Hz
```

Scripts automatically resample audio if needed.

---

# Dependencies

Main dependencies include:

* PyTorch
* NVIDIA NeMo
* Transformers
* HuggingFace Hub
* ONNXRuntime
* TorchCodec
* SoundDevice
* Gradio

Pinned versions are listed in:

```
requirements.txt
```

to avoid compatibility issues.

---

# Related Repository

CPU-only version:

[https://github.com/ujjval1003/ai4bharat-asr-cpu](https://github.com/ujjval1003/ai4bharat-asr-cpu)

---

# License

MIT License
