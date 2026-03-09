# AI4Bharat Indic ASR (GPU Setup)

![Python](https://img.shields.io/badge/python-3.10-blue)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![GPU](https://img.shields.io/badge/GPU-NVIDIA-green)
![NeMo](https://img.shields.io/badge/NVIDIA-NeMo-green)
![License](https://img.shields.io/badge/license-MIT-blue)

A **GPU-accelerated setup** for running **AI4Bharat Indic ASR models** using NVIDIA NeMo.

This repository provides a **stable, patched environment** to run AI4Bharat models locally with **CUDA acceleration**.

---

# Tested Hardware

This project was tested on the following system:

| Component | Value                          |
| --------- | ------------------------------ |
| GPU       | **NVIDIA RTX 3050 Laptop GPU** |
| CUDA      | **CUDA 12.6**                  |
| PyTorch   | CUDA 12.6 build                |
| Python    | 3.10                           |
| Framework | NVIDIA NeMo                    |

The setup script installs **CUDA 12.6 compatible PyTorch automatically**. 

---

# Supported Models

| Model                                     | Type         | Languages           |
| ----------------------------------------- | ------------ | ------------------- |
| `indicconformer_stt_gu_hybrid_rnnt_large` | Gujarati     | Gujarati            |
| `indic-conformer-600m-multilingual`       | Multilingual | 20+ Indic languages |

---

# Features

* GPU accelerated inference
* Real-time streaming ASR
* Browser UI interface
* 22 Indian languages supported
* Continuous live transcription
* VAD based utterance segmentation
* Transcript export

---

# Project Structure

```
AI4Bharat-GPU/
│
├── setup.sh
├── requirements.txt
│
├── ai4bharat-gu.py
├── ai4bharat-mul.py
│
├── live.py
├── live-ui.py
│
└── README.md
```

| File             | Description                |
| ---------------- | -------------------------- |
| setup.sh         | Full GPU environment setup |
| requirements.txt | Dependency versions        |
| ai4bharat-gu.py  | Gujarati ASR example       |
| ai4bharat-mul.py | Multilingual ASR example   |
| live.py          | Terminal real-time ASR     |
| live-ui.py       | Web UI ASR interface       |

---

# CUDA and PyTorch Compatibility

Different CUDA versions require specific PyTorch builds.

| CUDA Version  | PyTorch Wheel Index                      |
| ------------- | ---------------------------------------- |
| CUDA 11.8     | `https://download.pytorch.org/whl/cu118` |
| CUDA 12.1     | `https://download.pytorch.org/whl/cu121` |
| CUDA 12.4     | `https://download.pytorch.org/whl/cu124` |
| **CUDA 12.6** | `https://download.pytorch.org/whl/cu126` |

Example install command:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Your setup script installs the **CUDA 12.6 wheel automatically**.

---

# Installation

## 1 Clone the repository

```
git clone https://github.com/YOUR_USERNAME/ai4bharat-asr-gpu.git
cd ai4bharat-asr-gpu
```

---

# 2 Run setup

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

# 3 Activate environment

```
source nemo/bin/activate
```

---

# 4 Login to HuggingFace

The models are gated, so you must login first.

```
huggingface-cli login
```

---

# Run Gujarati ASR

```
python ai4bharat-gu.py
```

Example model loading: 

```python
model = nemo_asr.models.ASRModel.from_pretrained(
    "ai4bharat/indicconformer_stt_gu_hybrid_rnnt_large"
)
```

The script automatically uses **GPU if available**.

---

# Run Multilingual ASR

```
python ai4bharat-mul.py
```

The script:

* loads audio
* converts to mono
* resamples to **16kHz**
* performs CTC and RNNT decoding

Example usage: 

```python
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

The streaming script supports **22 Indic languages**. 

---

# Web UI

Run the browser interface:

```
python live-ui.py
```

Then open:

```
http://127.0.0.1:7860
```

Features:

* Live transcription
* Speech history
* Export transcripts
* Language selection
* GPU acceleration

Built with **Gradio**. 

---

# Audio Requirements

Input audio format:

```
Format: WAV
Channels: Mono
Sample Rate: 16000 Hz
```

The scripts automatically resample audio if necessary.

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

# Model Download

First run downloads the model automatically.

Approximate size:

```
~1.8 GB
```

Cached in:

```
~/.cache/huggingface
```

---

# Related Repository

CPU-only version:

https://github.com/ujjval1003/ai4bharat-asr-cpu