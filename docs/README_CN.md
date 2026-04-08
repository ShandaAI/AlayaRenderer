<div align="center">

## Generative World Renderer
AI-native Renderer for Games and Virtual Worlds, with Data and Tools

[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f)](https://alaya-studio.github.io/renderer/)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=N5CQ5WWIA_8)
[![X](https://img.shields.io/badge/X-000000?style=flat&logo=x&logoColor=white)](https://x.com/alayastd/status/2039903025096187937?s=61)
[![Daily Paper](https://img.shields.io/badge/%F0%9F%A4%97-%20Daily%20Paper-yellow)](https://huggingface.co/papers/2604.02329)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97-%20Demo-f0b030)](https://huggingface.co/spaces/Brian9999/game-editing)
[![arXiv](https://img.shields.io/badge/arXiv-2604.02329-b31b1b.svg)](https://arxiv.org/abs/2604.02329)

[![English](https://img.shields.io/badge/English-aaa)](../README.md) [![中文](https://img.shields.io/badge/中文-aaa)](README_CN.md) [![日本語](https://img.shields.io/badge/日本語-aaa)](README_JA.md)  [![한국어](https://img.shields.io/badge/한국어-aaa)](README_KO.md)

</div>

https://github.com/user-attachments/assets/adb79a5d-160c-4b13-a521-7ddb7a140e51


## 📢 更新

- [2026.04.04] 我们上线了 Game Editing 的在线 Demo：**[Game Editing Demo](https://huggingface.co/spaces/Brian9999/game-editing)**
- [2026.04.03] 我们发布了论文，欢迎讨论与反馈！


## 🌐 简介

![teaser](/assets/teaser.png)

**概述** 我们提出了一个大规模数据集与框架，利用微调的视频扩散模型实现高质量的视频逆渲染与正向渲染。我们从两款 3A 游戏中提取了同步的 RGB 视频与五个对齐的 G-buffer 通道，并提出了一种基于 VLM 的评估协议，用于真实场景的评估。我们的流水线包含两个组件：

- **Inverse Renderer**（逆渲染器，RGB &rarr; G-buffers）：基于 [Cosmos-Transfer1-DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer) 微调，将 RGB 视频分解为 G-buffer 贴图（albedo、normal、depth、roughness、metallic）
- **Game Editing**（游戏编辑，G-buffers + 文本 &rarr; 风格化 RGB）：基于 [Wan2.1 1.3B](https://github.com/Wan-Video/Wan2.1)（通过 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)）微调，从 G-buffer 输入合成逼真的 RGB 视频，支持通过文本提示控制光照和风格

数据集的主要特点：

- **400 万+ 帧**，**720p / 30 FPS**，**6 个同步通道**（RGB + albedo、normal、depth、metallic、roughness）
- **40 小时** 的游戏画面，来自 **2 款 3A 游戏**（赛博朋克 2077 和 黑神话：悟空）
- **长时序片段**：平均每个片段 8 分钟，最长可达 53 分钟的连续录制
- **内容多样**：城市/户外/室内场景，多种天气变化（晴天、雨天、雾天、夜晚、日落），以及真实的运动模式
- **运动模糊变体**：通过子帧插值和线性域时域平均离线生成
- **基于 VLM 的评估**：使用视觉语言模型对材质预测进行无参考评估


## 🚀 使用方法

本仓库包含 **Inverse Renderer**（逆渲染器）和 **Game Editing**（游戏编辑）模型。请按照以下说明设置环境并运行各模型的推理。我们建议为两个模型分别创建独立的 conda 环境，以避免版本冲突。

```bash
git clone --recurse-submodules https://github.com/ShandaAI/AlayaRenderer.git
cd AlayaRenderer
```

### 模型权重

| 模型 | 基础模型 | 链接 |
|------|---------|------|
| Inverse Renderer | Cosmos-Transfer1-DiffusionRenderer 7B | [HuggingFace](https://huggingface.co/Brian9999/world_inverse_renderer/tree/main) |
| Game Editing | Wan2.1 1.3B | [HuggingFace](https://huggingface.co/Brian9999/stylerenderer/tree/main) |

### Inverse Renderer（逆渲染器）

我们的模型基于 [Cosmos-Transfer1-DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer) 微调。请参照 [`inverse_renderer/`](inverse_renderer/) 中的说明进行环境配置和推理。下载相关权重后，将 `inverse_renderer/checkpoints/Diffusion_Renderer_Inverse_Cosmos_7B` 下的检查点替换为我们微调后的检查点。

<!-- 体验 Demo：**[Inverse Renderer Demo](https://huggingface.co/spaces/Brian9999/world_inverse_renderer_demo)** 🚧 -->

### Game Editing（游戏编辑）

<!-- 完整代码见 [`game_editing/`](game_editing/)。 -->

#### 安装

请参照 [DiffSynth-Studio](https://github.com/ShandaAI/DiffSynth-Studio) 的说明进行环境配置并下载相关权重。从 [HuggingFace](https://huggingface.co/Brian9999/stylerenderer/tree/main) 下载我们微调后的检查点，并放置到 `game_editing/models/train/Wan2.1-T2V-1.3B_gbuffer/` 目录下。

#### 快速示例

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

体验在线 Demo：**[Game Editing Demo](https://huggingface.co/spaces/Brian9999/game-editing)**

## 📋 待办

- [ ] 发布数据集
- [ ] 发布数据整理工具包

## ❤️ 致谢

本项目基于以下优秀工作构建：

- [DiffusionRenderer](https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/) by NVIDIA Toronto AI Lab
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) by Wan-Video
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) by ModelScope
<!-- - [ReShade](https://reshade.me/) -->


## 📄 许可证

详见 [LICENSE](LICENSE)。


## 📝 引用

如果您觉得本项目对您有帮助，请考虑引用：

```bibtex
@article{huang2026generativeworldrenderer,
    title={Generative World Renderer},
    author={Zheng-Hui Huang and Zhixiang Wang and Jiaming Tan and Ruihan Yu and Yidan Zhang and Bo Zheng and Yu-Lun Liu and Yung-Yu Chuang and Kaipeng Zhang},
    journal={arXiv preprint arXiv:2604.02329},
    year={2026}
}
```
