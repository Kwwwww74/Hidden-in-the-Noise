# 🎧 Hidden in the Noise: Unveiling Backdoors in Audio LLMs Alignment through Latent Acoustic Pattern Triggers

<div align="center">
  <img src="./images/logo.png" alt="Hidden in the Noise" width="40%" style="margin-top: -20px; margin-bottom: -10px;">
</div>

<p align="center">
  <a href="https://arxiv.org/pdf/2508.02175v3">📜 Submitted</a> | <a href="https://huggingface.co/datasets/JusperLee/AudioTrust">🤗 Dataset</a>
</p>

<p align="center">
<a href=""> <img src="https://img.shields.io/github/stars/Kwwwww74/Hidden-in-the-Noise?style=flat-square&logo=github" alt="GitHub stars"></a>
<a href=""> <img src="https://img.shields.io/github/forks/Kwwwww74/Hidden-in-the-Noise?style=flat-square&logo=github" alt="GitHub forks"></a>
<a href=""> <img src="https://img.shields.io/github/issues/Kwwwww74/Hidden-in-the-Noise?style=flat-square&logo=github" alt="GitHub issues"></a>
<a href=""> <img src="https://img.shields.io/github/last-commit/Kwwwww74/Hidden-in-the-Noise?style=flat-square&logo=github" alt="GitHub Last commit"></a>
</p>

<p align="center">

> **Hidden in the Noise (HIN)**, a backdoor framework that uses subtle acoustic patterns as triggers to compromise Audio LLMs. And **AudioSafe**, a benchmark of harmful audio queries across nine risk categories.


## 💥 News

**[2025.11.08]** Our paper has been accepted by **AAAI2026 Oral**!!!

## 📌 Table of Contents

- [🔍 Overview](#-overview)
- [📁 Repository Structure](#-repository-structure)
- [📦 Dataset Description](#-dataset-description)
- [🧪 Scripts Overview](#-scripts-overview)
- [🚀 Quick Start](#-quick-start)
- [📊 Benchmark Tasks](#-benchmark-tasks)
- [📌 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)
- [📬 Contact](#-contact)


## 🔍 Overview
As Audio Large Language Models (ALLMs) emerge as powerful tools for speech processing, their safety implications demand urgent attention. While considerable research has explored textual and vision safety, audio’s distinct characteristics present significant challenges. This paper first investigates: Is ALLM vulnerable to backdoor attacks exploiting acoustic triggers? In response to this issue, we introduce Hidden in the Noise (HIN), a novel backdoor attack framework  designed to exploit subtle, audio-specific features. HIN applies acoustic modifications to raw audio waveforms, such as alterations to temporal dynamics and strategic injection of spectrally tailored noise. These changes introduce consistent patterns that an ALLM’s acoustic feature encoder captures, embedding robust triggers within the audio stream. To evaluate ALLM robustness against audio-feature-based triggers, we develop the AudioSafe benchmark, assessing nine
distinct risk types. Extensive experiments on AudioSafe and three established safety datasets reveal critical vulnerabilities in existing ALLMs: (I) audio features like environment noise and speech rate variations achieve over 90% average attack success rate, (II) ALLMs exhibit significant sensitivity differences across acoustic features, particularly showing minimal response to volume as a trigger, and (III) poisoned sample inclusion causes only marginal loss curve fluctuations, highlighting the attack’s stealth

## 📁 Repository Structure

```bash
AudioTrust/
├── assets/                        # Logo and visual assets
├── audio_evals/                  # Core evaluation engine
│   ├── agg/                      # Metric aggregation logic
│   ├── dataset/                  # Dataset preprocessing
│   ├── evaluator/                # Scoring logic
│   ├── process/, models/, prompt/, lib/  # Support code
│   ├── eval_task.py              # Evaluation controller
│   ├── isolate.py                # Single model inference
│   ├── recorder.py               # Output logging
│   ├── registry.py               # Registry entrypoint
│   └── utils.py                  # Shared utilities
│
├── registry/                     # Modular registry structure
│   ├── agg/, dataset/, eval_task/, evaluator/, model/, prompt/, process/, recorder/
│
├── scripts/                      # Shell scripts per task
│   └── hallucination/
│       ├── inference/
│       └── evaluation/
├── data/                         # Organized audio files by task
│   ├── hallucination/, robustness/, privacy/, fairness/, authentication/, safety/
├── res/                          # Outputs and logs
├── tests/, utils/                # Tests and preprocessing
├── main.py                       # Main execution entry
├── requirments.txt
├── requirments-offline-model.txt
└── README.md
```


## 📦 Dataset Description

* **Language**: English
* **Audio Format**: WAV, mono, 16kHz
* **Size**: \~10.4GB across 6 sub-datasets

Each sample includes:

* `Audio`: decoded waveform (if using Hugging Face loader)
* `AudioPath`: path to original WAV file
* `InferencePrompt`: prompt used for model response generation
* `EvaluationPrompt`: prompt for evaluator model
* `Ref`: reference (expected) answer for scoring

Sub-datasets:

* `{hallucination, robustness, authentication, privacy, fairness, safety}`



## 🧪 Scripts Overview

Each subtask contains:

| Folder        | Purpose                                                           |
| ------------- | ----------------------------------------------------------------- |
| `inference/`  | Use a target model (e.g., Gemini) to generate responses           |
| `evaluation/` | Use an evaluator model (e.g., GPT-4o) to assess generated outputs |

This supports **model-vs-model** evaluation pipelines.

### 🧩 Example: Hallucination Task

```bash
scripts/hallucination/
├── inference/
│   └── gemini-2.5-pro.sh
└── evaluation/
    └── gpt-4o.sh
```



## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/JusperLee/AudioTrust.git
cd AudioTrust
pip install -r requirments.txt
```

Or for offline model use:

```bash
pip install -r requirments-offline-model.txt
```

### 2. Load Dataset from Hugging Face

```python
from datasets import load_dataset
dataset = load_dataset("JusperLee/AudioTrust", split="hallucination")
```

#### Materialize the HF dataset to the project `data/` layout

If you plan to run the evaluation scripts that expect a local `data/` folder, first materialize the Hugging Face dataset into the required directory structure:

```bash
python utils/materialize_hf_audio.py --dataset-path JusperLee/AudioTrust
```


### 3. Run Inference and Evaluation

```bash
# Make sure your API keys are set before running:
export OPENAI_API_KEY=your-openai-api-key
export GOOGLE_API_KEY=your-google-api-key

# Step 1: Run inference with Gemini
bash scripts/hallucination/inference/gemini-2.5-pro.sh

# Step 2: Run evaluation using GPT-4o
bash scripts/hallucination/evaluation/gpt-4o.sh
```

Or directly with Python:

```bash
export OPENAI_API_KEY=your-openai-api-key
python main.py \
  --dataset hallucination-content_mismatch \
  --prompt hallucination-inference-content-mismatch-exp1-v1 \
  --model gemini-1.5-pro
```



## 📊 Benchmark Tasks

| Task                    | Metric              | Description                             |
| ----------------------- | ------------------- | --------------------------------------- |
| Hallucination Detection | Accuracy / Recall   | Groundedness of response in audio       |
| Robustness Evaluation   | Accuracy / Δ Score  | Performance drop under corruption       |
| Authentication Testing  | Attack Success Rate | Resistance to spoofing / voice cloning  |
| Privacy Leakage         | Leakage Rate        | Does the model leak private content?    |
| Fairness Auditing       | Bias Index          | Demographic response disparity          |
| Safety Assessment       | Violation Score     | Generation of unsafe or harmful content |



## 📌 Citation

```bibtex
@article{lin2025hin,
  title={Hidden in the Noise: Unveiling Backdoors in Audio LLMs Alignment through Latent Acoustic Pattern Triggers},
  author={Lin, Liang and Yu, Miao and Luo, Kaiwen and others},
  journal={arXiv preprint arXiv:2508.02175},
  year={2025}
}
```



## 🌟 Rising Stars
[![Star History Chart](https://api.star-history.com/svg?repos=Kwwwww74/Hidden-in-the-Noise&type=Date)](https://star-history.com/#Kwwwww74/Hidden-in-the-Noise&Date)



## 📬 Contact

For questions or collaboration inquiries:

* Liang Lin: linliang@iie.ac.cn, Kevin Luo: kaiwenluo74@gmail.com

