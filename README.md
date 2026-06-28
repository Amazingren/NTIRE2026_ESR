# [NTIRE 2026 Challenge on Efficient Super-Resolution](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

<div align="center">

<img src="https://github.com/Amazingren/NTIRE2026_ESR/blob/main/figs/logo.png" width="400px"/>

<br/>

[![GitHub Stars](https://img.shields.io/github/stars/Amazingren/NTIRE2026_ESR?style=social)](https://github.com/Amazingren/NTIRE2026_ESR)
[![GitHub Forks](https://img.shields.io/github/forks/Amazingren/NTIRE2026_ESR?style=social)](https://github.com/Amazingren/NTIRE2026_ESR/fork)
&nbsp;
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-4b44ce?logo=openai&logoColor=white)](https://cvpr.thecvf.com/)
[![Challenge](https://img.shields.io/badge/Codabench-Competition-21a366?logo=codeforces&logoColor=white)](https://www.codabench.org/competitions/13553/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
&nbsp;
[![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

## 📰 News

- 🏆 **June 28th, 2026:** All team submissions, checkpoints, and final results are now released!
- 📄 **June 2026:** Challenge report published — *The Eleventh NTIRE 2026 Efficient Super-Resolution Challenge Report* (CVPRW 2026).
- 🦕 **February 6th, 2026:** Challenge repository is ready!

---

## 📋 Table of Contents

- [About the Challenge](#-about-the-challenge)
- [Environment Setup](#%EF%B8%8F-environment-setup)
- [Dataset Preparation](#-dataset-preparation)
- [Quick Start](#-quick-start)
- [How to Add Your Model](#-how-to-add-your-model)
- [Computing Metrics](#-computing-metrics-params-flops-activations)
- [Ranking Strategy](#-ranking-strategy)
- [References](#-references)
- [Organizers](#-organizers)

---

## 🔍 About the Challenge

In collaboration with the NTIRE workshop, we host a challenge focused on **Efficient Super-Resolution** ([NTIRE2026_ESR](https://www.codabench.org/competitions/13553/)). The task is to enhance the resolution of an input image by a factor of **×4**, given paired low-resolution and high-resolution training examples. The challenge baseline is [SPAN](https://arxiv.org/abs/2311.12770) (*Cheng Yan et al., 2024*), winner of the NTIRE 2024 Efficient SR Challenge.

The challenge consists of **one main track** and **three sub-tracks**:

| Track | Goal | Constraint |
|:------|:-----|:-----------|
| 🏆 **Main — Overall Performance** | Best combined score (Runtime + FLOPs + Params) | PSNR ≥ baseline |
| 💎 **Sub-track 1 — Inference Runtime** | Lowest inference time | Params, FLOPs, PSNR ≥ baseline |
| 💎 **Sub-track 2 — FLOPs** | Lowest FLOPs | Runtime, Params, PSNR ≥ baseline |
| 💎 **Sub-track 3 — Parameters** | Fewest parameters | Runtime, FLOPs, PSNR ≥ baseline |

> 🌟 **New in 2026:** Participants are encouraged to explore model compression and acceleration techniques such as **quantization** and **pruning** to further reduce inference cost without sacrificing reconstruction quality.

### ⚠️ Evaluation Rules

- **No training on eval sets.** Do not use validation LR/HR images or test LR images for training. The test dataset is held out and will not be disclosed.
- **PSNR threshold.** Methods with PSNR below **26.90 dB** on DIV2K\_LSDIR\_valid or **26.99 dB** on DIV2K\_LSDIR\_test are excluded from the ranking.

---

## 🛠️ Environment Setup

> Tested with **Python 3.9** on **NVIDIA RTX A6000**.

**Step 1 — Install PyTorch:**
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

**Step 2 — Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

> 💡 **Team 22 (SPANV2\_ESR) only:** This method uses a custom CUDA attention operator. Please follow [`INSTALL_team22_xiaomiMM.md`](./INSTALL_team22_xiaomiMM.md) before running model `--model_id 22`.

---

## 📂 Dataset Preparation

Download the validation set:
- 📥 [DIV2K\_LSDIR\_valid\_LR](https://drive.google.com/file/d/1YUDrjUSMhhdx1s-O0I1qPa_HjW-S34Yj/view?usp=sharing)
- 📥 [DIV2K\_LSDIR\_valid\_HR](https://drive.google.com/file/d/1z1UtfewPatuPVTeAAzeTjhEGk4dg2i8v/view?usp=sharing)

Organize your working directory as follows:

```
NTIRE2026_ESR_Challenge/
├── DIV2K_LSDIR_valid_HR/
│   ├── 000001.png
│   ├── ...
│   └── 0900.png
├── DIV2K_LSDIR_valid_LR/
│   ├── 000001x4.png
│   ├── ...
│   └── 0900x4.png
├── NTIRE2026_ESR/           ← this repository
│   ├── test_demo.py
│   └── ...
└── results/
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/Amazingren/NTIRE2026_ESR.git
cd NTIRE2026_ESR
```

**Test a single model** (e.g., the SPAN baseline):
```bash
CUDA_VISIBLE_DEVICES=0 python test_demo.py \
    --data_dir [path/to/your/data] \
    --save_dir [path/to/your/results] \
    --model_id 0
```

See [`run.sh`](./run.sh) for more examples including batch evaluation of all submitted methods.

### 📊 SPAN Baseline Reference Results

Evaluated on **NVIDIA RTX A6000**, averaged over **5 runs**:

| Metric | DIV2K\_LSDIR\_valid | DIV2K\_LSDIR\_test | Average |
|:-------|:-------------------:|:------------------:|:-------:|
| **PSNR (dB)** | 26.94 | 27.01 | — |
| **Runtime (ms)** | 8.00 | 7.47 | 7.74 |
| **Params (M)** | — | — | 0.151 |
| **FLOPs @ 256×256 (G)** | — | — | 9.83 |
| **Activations (M)** | — | — | 41.68 |

---

## 📦 How to Add Your Model

> ⚠️ The submission period for NTIRE 2026 ESR is now **closed**. Instructions are kept for documentation purposes.

1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/11JuxcS78C6Gxc8B436L4Zk4_m5soHaTcw3cnF8h5ctE/edit?usp=sharing) to obtain a Team ID.
2. Place your model code at `./models/[TeamID]_[ModelName].py`
   - Add **exactly one** file — no extra submodules.
   - Zero-pad the Team ID to two digits: `00`, `01`, `02`, …
3. Place your checkpoint at `./model_zoo/[TeamID]_[ModelName].[pth|pt|ckpt]`
4. Register the model in `select_model()` inside `test_demo.py`:
    ```python
    elif model_id == [TeamID]:
        from models.[TeamID]_[ModelName] import [ModelName]
        name, data_range = f"{model_id:02}_[ModelName]", 1.0  # or 255.0
        model_path = os.path.join('model_zoo', '[TeamID]_[ModelName].pth')
        model = [ModelName]()
        model.load_state_dict(torch.load(model_path), strict=True)
    ```
5. Send us the download command (e.g., `git clone [your repo]`) and we will integrate it.

---

## 🔢 Computing Metrics: Params, FLOPs, Activations

```python
import torch
from utils.model_summary import get_model_activation
from models.team00_SPAN import SPAN
from fvcore.nn import FlopCountAnalysis

device = torch.device('cuda')
model = SPAN().eval().to(device)
input_dim = (3, 256, 256)

# Activations & Conv count
activations, num_conv = get_model_activation(model, input_dim)
print(f"#Activations : {activations / 1e6:.4f} M")
print(f"#Conv2d      : {num_conv}")

# FLOPs (fvcore — used in NTIRE2026)
input_fake = torch.rand(1, *input_dim).to(device)
flops = FlopCountAnalysis(model, input_fake).total() / 1e9
print(f"FLOPs        : {flops:.4f} G")

# Parameters
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"#Params      : {params:.4f} M")
```

---

## 📈 Ranking Strategy

Evaluation is conducted in four steps:

1. **Re-evaluation** — Each method is run **5 times** on an NVIDIA RTX A6000; the average is reported.
2. **PSNR filter** — Methods below 26.90 dB (valid) or 26.99 dB (test) are excluded.
3. **Score computation** — For qualifying methods:

```
Score_Runtime = exp( 2 × Runtime / Runtime_SPAN )
Score_FLOPs   = exp( 2 × FLOPs   / FLOPs_SPAN   )
Score_Params  = exp( 2 × Params  / Params_SPAN   )
```

4. **Final score:**

```
Score_Final = 0.8 × Score_Runtime + 0.1 × Score_FLOPs + 0.1 × Score_Params
```

> For the SPAN baseline (Runtime = 7.74 ms, FLOPs = 9.83 G, Params = 0.151 M), all three scores equal **7.3891** and Score\_Final = **7.3891**.

:heavy_exclamation_mark: Sub-track rankings use the corresponding sub-score; the main track uses Score\_Final.

---

## 📝 References

If you find this codebase or the challenge report useful, please consider citing:

```bibtex
@inproceedings{ren2026eleventh,
  title={The eleventh NTIRE 2026 efficient super-resolution challenge report},
  author={Ren, Bin and Guo, Hang and Shu, Yan and Ma, Jiaqi and Cui, Ziteng and Liu, Shuhong and Mei, Guofeng and Sun, Lei and Wu, Zongwei and Khan, Fahad Shahbaz Khan and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2460--2484},
  year={2026}
}

@inproceedings{ren2025tenth,
  title={The tenth NTIRE 2025 efficient super-resolution challenge report},
  author={Ren, Bin and Guo, Hang and Sun, Lei and Wu, Zongwei and Timofte, Radu and Li, Yawei and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={917--966},
  year={2025}
}

@inproceedings{ren2024ninth,
  title={The ninth NTIRE 2024 efficient super-resolution challenge report},
  author={Ren, Bin and Li, Yawei and Mehta, Nancy and Timofte, Radu and Yu, Hongyuan and Wan, Cheng and Hong, Yuxin and Han, Bingnan and Wu, Zhuoyuan and Zou, Yajun and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={6595--6631},
  year={2024}
}
```

---

## 👥 Organizers

| Name | Affiliation |
|:-----|:-----------|
| **Bin Ren** | MBZUAI, UAE — bin.ren@mbzuai.ac.ae |
| **Hang Guo** | Tsinghua University, China — cshguo@gmail.com |
| **Yan Shu** | University of Trento, Italy — yan.shu@unitn.it |
| **Jiaqi Ma** | MBZUAI, UAE — jiaqi.ma@mbzuai.ac.ae |
| **Ziteng Cui** | University of Tokyo, Japan — cui@mi.t.u-tokyo.ac.jp |
| **Shuhong Liu** | University of Tokyo, Japan — s-liu@mi.t.u-tokyo.ac.jp |
| **Guofeng Mei** | FBK, Italy — gmei@fbk.eu |
| **Lei Sun** | INSAIT, Bulgaria — lei.sun@insait.ai |
| **Zongwei Wu** | University of Würzburg, Germany — zongwei.wu@uni-wuerzburg.de |
| **Salman Khan** | MBZUAI, UAE — salman.khan@mbzuai.ac.ae |
| **Fahad Shahbaz Khan** | MBZUAI, UAE — fahad.khan@mbzuai.ac.ae |
| **Radu Timofte** | University of Würzburg, Germany — radu.timofte@uni-wuerzburg.de |
| **Yawei Li** | ETH Zürich, Switzerland — yawei.li.ai@gmail.com |

For questions, feel free to reach out to the organizers directly or open a [GitHub Issue](https://github.com/Amazingren/NTIRE2026_ESR/issues).

---

## 📄 License and Acknowledgement

This repository is released under the [MIT License](LICENSE).  
We sincerely thank all participating teams for their outstanding contributions to this challenge.
