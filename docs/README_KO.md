<div align="center">

## Generative World Renderer
AI-native Renderer for Games and Virtual Worlds, with Data and Tools

[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f)](https://alaya-studio.github.io/renderer/)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=N5CQ5WWIA_8)
[![X](https://img.shields.io/badge/X-000000?style=flat&logo=x&logoColor=white)](https://x.com/alayastd/status/2039903025096187937?s=61)
[![Daily Paper](https://img.shields.io/badge/%F0%9F%A4%97-%20Daily%20Paper-yellow)](https://huggingface.co/papers/2604.02329)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97-%20Demo-f0b030)](https://huggingface.co/spaces/Brian9999/game-editing)
[![arXiv](https://img.shields.io/badge/arXiv-2604.02329-b31b1b.svg)](https://arxiv.org/abs/2604.02329)

[![English](https://img.shields.io/badge/English-aaa)](../README.md) | [![中文](https://img.shields.io/badge/中文-aaa)](README_CN.md) | [![日本語](https://img.shields.io/badge/日本語-aaa)](README_JA.md) | [![한국어](https://img.shields.io/badge/한국어-aaa)](README_KO.md)

</div>

https://github.com/user-attachments/assets/adb79a5d-160c-4b13-a521-7ddb7a140e51


## 📢 업데이트

- [2026.04.04] Game Editing 온라인 데모를 공개했습니다: **[Game Editing Demo](https://huggingface.co/spaces/Brian9999/game-editing)**
- [2026.04.03] 논문을 공개했습니다. 토론과 피드백을 환영합니다!


## 🌐 소개

![teaser](/assets/teaser.png)

**요약** 본 연구에서는 파인튜닝된 비디오 확산 모델을 활용하여 고품질 역렌더링 및 순방향 렌더링을 수행하는 대규모 데이터셋과 프레임워크를 제안합니다. 2개의 AAA 게임에서 동기화된 RGB 영상과 5개의 정렬된 G-buffer 채널을 추출하고, 실제 장면에 대한 VLM 기반 평가 프로토콜을 제안합니다. 파이프라인은 다음 2개의 구성 요소로 이루어져 있습니다:

- **Inverse Renderer** (역렌더러, RGB &rarr; G-buffers): [Cosmos-Transfer1-DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer)를 파인튜닝하여 RGB 영상을 G-buffer 맵(albedo, normal, depth, roughness, metallic)으로 분해
- **Game Editing** (게임 편집, G-buffers + 텍스트 &rarr; 스타일화된 RGB): [Wan2.1 1.3B](https://github.com/Wan-Video/Wan2.1) ([DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 기반)를 파인튜닝하여 G-buffer 입력으로부터 텍스트 프롬프트를 통해 조명과 스타일을 제어할 수 있는 포토리얼리스틱 RGB 영상을 합성

데이터셋의 주요 특징:

- **400만+ 프레임**, **720p / 30 FPS**, **6개 동기화 채널** (RGB + albedo, normal, depth, metallic, roughness)
- **2개의 AAA 게임** (사이버펑크 2077 & 흑신화: 오공)에서 **40시간**의 게임플레이 영상
- **장시간 시퀀스**: 클립당 평균 8분, 최대 53분 연속 녹화
- **다양한 콘텐츠**: 도시/야외/실내 장면, 다양한 날씨 변화(맑음, 비, 안개, 야간, 석양), 사실적인 모션 패턴
- **모션 블러 변형**: 서브프레임 보간 및 선형 도메인 시간 평균을 통한 오프라인 생성
- **VLM 기반 평가**: 비전-언어 모델을 활용한 재질 예측의 참조 없는(reference-free) 평가


## 🚀 사용 방법

본 저장소에는 **Inverse Renderer** (역렌더러)와 **Game Editing** (게임 편집) 모델이 포함되어 있습니다. 아래 안내에 따라 환경을 설정하고 각 모델의 추론을 실행해 주세요. 버전 충돌을 방지하기 위해 두 모델에 대해 별도의 conda 환경을 생성하는 것을 권장합니다.

```bash
git clone --recurse-submodules https://github.com/ShandaAI/AlayaRenderer.git
cd AlayaRenderer
```

### 모델 가중치

| 모델 | 기본 모델 | 링크 |
|------|----------|------|
| Inverse Renderer | Cosmos-Transfer1-DiffusionRenderer 7B | [HuggingFace](https://huggingface.co/Brian9999/world_inverse_renderer/tree/main) |
| Game Editing | Wan2.1 1.3B | [HuggingFace](https://huggingface.co/Brian9999/stylerenderer/tree/main) |

### Inverse Renderer (역렌더러)

본 모델은 [Cosmos-Transfer1-DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer)를 파인튜닝한 것입니다. 환경 설정 및 추론은 [`inverse_renderer/`](inverse_renderer/)의 안내를 따라 주세요. 관련 가중치를 다운로드한 후 `inverse_renderer/checkpoints/Diffusion_Renderer_Inverse_Cosmos_7B` 아래의 체크포인트를 파인튜닝된 체크포인트로 교체해 주세요.

<!-- 데모 체험: **[Inverse Renderer Demo](https://huggingface.co/spaces/Brian9999/world_inverse_renderer_demo)** 🚧 -->

### Game Editing (게임 편집)

<!-- 전체 코드는 [`game_editing/`](game_editing/)을 참조하세요. -->

#### 설치

[DiffSynth-Studio](https://github.com/ShandaAI/DiffSynth-Studio)의 안내에 따라 환경을 설정하고 관련 가중치를 다운로드해 주세요. [HuggingFace](https://huggingface.co/Brian9999/stylerenderer/tree/main)에서 파인튜닝된 체크포인트를 다운로드하여 `game_editing/models/train/Wan2.1-T2V-1.3B_gbuffer/` 경로에 배치해 주세요.

#### 빠른 시작

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

온라인 데모 체험: **[Game Editing Demo](https://huggingface.co/spaces/Brian9999/game-editing)**

## 📋 할 일

- [ ] 데이터셋 공개
- [ ] 데이터 큐레이션 툴킷 공개

## ❤️ 감사의 글

본 프로젝트는 다음의 우수한 연구를 기반으로 합니다:

- [DiffusionRenderer](https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/) by NVIDIA Toronto AI Lab
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) by Wan-Video
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) by ModelScope
<!-- - [ReShade](https://reshade.me/) -->


## 📄 라이선스

[LICENSE](LICENSE)를 참조해 주세요.


## 📝 인용

본 프로젝트가 도움이 되셨다면 인용을 부탁드립니다:

```bibtex
@article{huang2026generativeworldrenderer,
    title={Generative World Renderer},
    author={Zheng-Hui Huang and Zhixiang Wang and Jiaming Tan and Ruihan Yu and Yidan Zhang and Bo Zheng and Yu-Lun Liu and Yung-Yu Chuang and Kaipeng Zhang},
    journal={arXiv preprint arXiv:2604.02329},
    year={2026}
}
```
