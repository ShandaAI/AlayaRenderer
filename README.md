<div align="center">

## Generative World Renderer
AI-native Renderer for Games and Virtual Worlds, with Data and Tools

[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f)](https://alaya-studio.github.io/renderer/)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=N5CQ5WWIA_8)
[![X](https://img.shields.io/badge/X-000000?style=flat&logo=x&logoColor=white)](https://x.com/alayastd/status/2039903025096187937?s=61)
[![Daily Paper](https://img.shields.io/badge/%F0%9F%A4%97-%20Daily%20Paper-yellow)](https://huggingface.co/papers/2604.02329)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97-%20Demo-f0b030)](https://huggingface.co/spaces/Brian9999/game-editing)
[![arXiv](https://img.shields.io/badge/arXiv-2604.02329-b31b1b.svg)](https://arxiv.org/abs/2604.02329)

[![English](https://img.shields.io/badge/English-aaa)](README.md) | [![中文](https://img.shields.io/badge/中文-aaa)](docs/README_CN.md) | [![日本語](https://img.shields.io/badge/日本語-aaa)](docs/README_JA.md) | [![한국어](https://img.shields.io/badge/한국어-aaa)](docs/README_KO.md)

</div>

<!-- This repo contains the code, models, and dataset used in

> [**Generative World Renderer**](https://arxiv.org/abs/2604.02329)
>
> Zheng-Hui Huang, Zhixiang Wang, Jiaming Tan, Ruihan Yu, Yidan Zhang, Bo Zheng, Yu-Lun Liu, Yung-Yu Chuang, Kaipeng Zhang -->

https://github.com/user-attachments/assets/adb79a5d-160c-4b13-a521-7ddb7a140e51


## 📢 Update

- [2026.04.04] We have launched an online demo for Game Editing: **[Game Editing Demo](https://huggingface.co/spaces/Brian9999/game-editing)**
- [2026.04.03] We have released our paper — discussions and feedback are warmly welcome!


## 🌐 Introduction

![teaser](/assets/teaser.png)

**TL;DR** We present a large-scale dataset and framework for high-quality inverse and forward rendering of videos using fine-tuned video diffusion models. We extract synchronized RGB videos from two AAA games and five aligned G-buffer channels, and propose a VLM-based evaluation protocol for real-world scenes. Our pipeline consists of two components:

- **Inverse Renderer** (RGB &rarr; G-buffers): Fine-tuned from [Cosmos-Transfer1-DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer) to decompose RGB videos into G-buffer maps (albedo, normal, depth, roughness, metallic)
- **Game Editing** (G-buffers + Text &rarr; Stylized RGB): Fine-tuned from [Wan2.1 1.3B](https://github.com/Wan-Video/Wan2.1) (via [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)) to synthesize photorealistic RGB videos from G-buffer inputs with controllable lighting and style via text prompts

Key features of our dataset:

- **4M+ frames** at **720p / 30 FPS** with **6 synchronized channels** (RGB + albedo, normal, depth, metallic, roughness)
- **40 hours** of gameplay from **2 AAA games** (Cyberpunk 2077 & Black Myth: Wukong)
- **Long-duration sequences**: average 8 min per clip, up to 53 min continuous recording
- **Diverse content**: urban/outdoor/indoor scenes, varying weather (sunny, rainy, foggy, night, sunset), and realistic motion patterns
- **Motion blur variant**: offline-generated via sub-frame interpolation and linear-domain temporal averaging
- **VLM-based evaluation**: reference-free assessment of material predictions using vision-language models


## 🚀 Usage

This repository contains the **Inverse Renderer** and **Game Editing** models. Please follow the instructions below to set up the environment and run inference for each model. We recommend creating separate conda environments for the two models to avoid version conflicts.

```bash
git clone --recurse-submodules https://github.com/ShandaAI/AlayaRenderer.git
cd AlayaRenderer
```

### Model Weights

| Model | Base Model | Link |
|-------|-----------|------|
| Inverse Renderer | Cosmos-Transfer1-DiffusionRenderer 7B | [HuggingFace](https://huggingface.co/Brian9999/world_inverse_renderer/tree/main) |
| Game Editing | Wan2.1 1.3B | [HuggingFace](https://huggingface.co/Brian9999/stylerenderer/tree/main) |

### Inverse Renderer

Our model is fine-tuned from [Cosmos-Transfer1-DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer). Please follow the [`inverse_renderer/`](inverse_renderer/) instructions for environment setup and inference. Download the related weights and replace the checkpoint under `inverse_renderer/checkpoints/Diffusion_Renderer_Inverse_Cosmos_7B` with our fine-tuned checkpoint.

<!-- Try the demo: **[Inverse Renderer Demo](https://huggingface.co/spaces/Brian9999/world_inverse_renderer_demo)** 🚧 -->

### Game Editing

<!-- See [`game_editing/`](game_editing/) for the complete code. -->

#### Installation

Please follow the [DiffSynth-Studio](https://github.com/ShandaAI/DiffSynth-Studio) instructions to set up the environment and download the related weights. Download our fine-tuned checkpoint from [HuggingFace](https://huggingface.co/Brian9999/stylerenderer/tree/main) and place it under `game_editing/models/train/Wan2.1-T2V-1.3B_gbuffer/`.

#### Quick Example

```bash
cd game_editing

CUDA_VISIBLE_DEVICES=0 python \
    examples/wanvideo/model_inference/inference_gbuffer_caption.py \
    --checkpoint models/train/Wan2.1-T2V-1.3B_gbuffer/model.safetensors \
    --gpu 0 \
    --style snowy_winter \
    --prompt "the scene is set in a frozen, snow-covered environment under cold, pale winter light with falling snowflakes, creating a silent and ethereal winter wonderland atmosphere." \
    --gbuffer_dir test_dataset \
    --save_dir outputs/ \
    --num_frames 81 --height 480 --width 832
```

Try the demo: **[Game Editing Demo](https://huggingface.co/spaces/Brian9999/game-editing)**
## 📋 TODO

- [ ] Release dataset.
- [ ] Release data curation toolkit.

## ❤️ Acknowledgements

This project builds upon the following excellent works:

- [DiffusionRenderer](https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/) by NVIDIA Toronto AI Lab
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) by Wan-Video
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) by ModelScope
<!-- - [ReShade](https://reshade.me/) -->


## 📄 License

See [LICENSE](LICENSE).


## 📝 Citation

If you find this project helpful, please consider citing:

```bibtex
@article{huang2026generativeworldrenderer,
    title={Generative World Renderer},
    author={Zheng-Hui Huang and Zhixiang Wang and Jiaming Tan and Ruihan Yu and Yidan Zhang and Bo Zheng and Yu-Lun Liu and Yung-Yu Chuang and Kaipeng Zhang},
    journal={arXiv preprint arXiv:2604.02329},
    year={2026}
}
```
