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
- [📦 Dataset Description](#-dataset-description)
- [🚀 Quick Start](#-quick-start)
- [📊 Benchmark Tasks](#-benchmark-tasks)
- [📌 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)
- [📬 Contact](#-contact)


## 🔍 Overview

<p align="justify">
As Audio Large Language Models (ALLMs) emerge as powerful tools for speech processing, their safety implications demand urgent attention. While considerable research has explored textual and vision safety, audio’s distinct characteristics present significant challenges. This paper first investigates: Is ALLM vulnerable to backdoor attacks exploiting acoustic triggers? In response to this issue, we introduce Hidden in the Noise (HIN), a novel backdoor attack framework  designed to exploit subtle, audio-specific features. HIN applies acoustic modifications to raw audio waveforms, such as alterations to temporal dynamics and strategic injection of spectrally tailored noise. These changes introduce consistent patterns that an ALLM’s acoustic feature encoder captures, embedding robust triggers within the audio stream. To evaluate ALLM robustness against audio-feature-based triggers, we develop the AudioSafe benchmark, assessing nine
distinct risk types. Extensive experiments on AudioSafe and three established safety datasets reveal critical vulnerabilities in existing ALLMs: (I) audio features like environment noise and speech rate variations achieve over 90% average attack success rate, (II) ALLMs exhibit significant sensitivity differences across acoustic features, particularly showing minimal response to volume as a trigger, and (III) poisoned sample inclusion causes only marginal loss curve fluctuations, highlighting the attack’s stealth
</p>


## 📦 Dataset Description

- **Language**: English  
- **Domain**: Safety-oriented speech and audio understanding  
- **Audio Format**: WAV, mono, 16kHz  
- **Content Type**: Spoken queries containing potentially harmful or sensitive intents, paired with safety-oriented instructions  
- **Trigger Variants**: Clean audio and acoustically modified versions (e.g., emotion, speaking rate, noise, accent, volume)  
- **Annotation**: Instruction–response style safety supervision  
- **Size**: ~10.4 GB in total, consisting of **6 sub-datasets**  


## 🚀 Quick Start

```bash
git clone https://github.com/Kwwwww74/Hidden-in-the-Noise.git
cd Hidden-in-the-Noise
pip install -r requirments.txt
```

Or for offline model use:

```bash
pip install -r requirments-offline-model.txt
```







## 📊 Benchmark Tasks

| Task                         | Metric                          | Description                                                                 |
| ---------------------------- | ------------------------------- | --------------------------------------------------------------------------- |
| Safety Refusal Evaluation    | Clean Accuracy (ACC)            | Measures whether the ALLM correctly refuses to answer harmful audio queries under clean (non-triggered) conditions |
| Audio Backdoor Attack        | Attack Success Rate (ASR)       | Evaluates the success rate of triggering harmful responses when specific acoustic backdoors are present |
| Acoustic Trigger Sensitivity | ASR per Trigger Type            | Assesses model vulnerability across different audio features (emotion, speed, noise, accent, volume) |
| Risk-Type Safety Assessment  | ACC / ASR per Risk Category     | Evaluates safety behavior across nine risk types (e.g., harassment, malware, fraud, physical harm) |
| Stealthiness Evaluation      | Loss Differential (ΔL, Var, CV) | Measures whether poisoned samples introduce detectable anomalies in training loss dynamics |
| Transferability Evaluation   | Cross-benchmark ASR             | Tests whether audio backdoor attacks transfer to external safety benchmarks (AdvBench, MaliciousInstruct, JailbreakBench) |



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

