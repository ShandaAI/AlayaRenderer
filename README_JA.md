<div align="center">

## Generative World Renderer
AI-native Renderer for Games and Virtual Worlds, with Data and Tools

[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f)](https://alaya-studio.github.io/renderer/)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=N5CQ5WWIA_8)
[![X](https://img.shields.io/badge/X-000000?style=flat&logo=x&logoColor=white)](https://x.com/alayastd/status/2039903025096187937?s=61)
[![Daily Paper](https://img.shields.io/badge/%F0%9F%A4%97-%20Daily%20Paper-yellow)](https://huggingface.co/papers/2604.02329)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97-%20Demo-f0b030)](https://huggingface.co/spaces/Brian9999/game-editing)
[![arXiv](https://img.shields.io/badge/arXiv-2604.02329-b31b1b.svg)](https://arxiv.org/abs/2604.02329)

[![English](https://img.shields.io/badge/English-aaa)](README.md) | [![中文](https://img.shields.io/badge/中文-aaa)](README_CN.md) | [![日本語](https://img.shields.io/badge/日本語-aaa)](README_JA.md) | [![한국어](https://img.shields.io/badge/한국어-aaa)](README_KO.md)

</div>

https://github.com/user-attachments/assets/adb79a5d-160c-4b13-a521-7ddb7a140e51


## 📢 更新情報

- [2026.04.04] Game Editing のオンラインデモを公開しました：**[Game Editing Demo](https://huggingface.co/spaces/Brian9999/game-editing)**
- [2026.04.03] 論文を公開しました。ご意見・ご感想をお待ちしております！


## 🌐 概要

![teaser](/assets/teaser.png)

**要約** 本研究では、ファインチューニングされたビデオ拡散モデルを用いて、高品質な逆レンダリングおよび順レンダリングを実現する大規模データセットとフレームワークを提案します。2本の AAAゲームから同期された RGB 映像と5つのアラインされた G-buffer チャネルを抽出し、実世界シーンに対する VLM ベースの評価プロトコルを提案しています。パイプラインは以下の2つのコンポーネントで構成されます：

- **Inverse Renderer**（逆レンダラー、RGB &rarr; G-buffers）：[Cosmos-Transfer1-DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer) をファインチューニングし、RGB 映像を G-buffer マップ（albedo、normal、depth、roughness、metallic）に分解
- **Game Editing**（ゲーム編集、G-buffers + テキスト &rarr; スタイル化 RGB）：[Wan2.1 1.3B](https://github.com/Wan-Video/Wan2.1)（[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 経由）をファインチューニングし、G-buffer 入力からテキストプロンプトによる照明・スタイル制御可能なフォトリアリスティック RGB 映像を合成

データセットの主な特徴：

- **400万フレーム以上**、**720p / 30 FPS**、**6つの同期チャネル**（RGB + albedo、normal、depth、metallic、roughness）
- **2本の AAAゲーム**（サイバーパンク2077 & 黒神話：悟空）から **40時間** のゲームプレイ映像
- **長時間シーケンス**：クリップあたり平均8分、最大53分の連続録画
- **多様なコンテンツ**：都市/屋外/屋内シーン、多様な天候変化（晴天、雨天、霧、夜間、夕焼け）、リアルなモーションパターン
- **モーションブラー変種**：サブフレーム補間とリニアドメイン時間平均によるオフライン生成
- **VLM ベース評価**：視覚言語モデルを用いたマテリアル予測のリファレンスフリー評価


## 🚀 使い方

本リポジトリには **Inverse Renderer**（逆レンダラー）と **Game Editing**（ゲーム編集）モデルが含まれています。以下の手順に従って環境をセットアップし、各モデルの推論を実行してください。バージョンの競合を避けるため、2つのモデルに対してそれぞれ別の conda 環境を作成することを推奨します。

```bash
git clone --recurse-submodules https://github.com/ShandaAI/AlayaRenderer.git
cd AlayaRenderer
```

### モデルの重み

| モデル | ベースモデル | リンク |
|--------|------------|--------|
| Inverse Renderer | Cosmos-Transfer1-DiffusionRenderer 7B | [HuggingFace](https://huggingface.co/Brian9999/world_inverse_renderer/tree/main) |
| Game Editing | Wan2.1 1.3B | [HuggingFace](https://huggingface.co/Brian9999/stylerenderer/tree/main) |

### Inverse Renderer（逆レンダラー）

本モデルは [Cosmos-Transfer1-DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer) をファインチューニングしたものです。環境構築と推論については [`inverse_renderer/`](inverse_renderer/) の手順に従ってください。関連する重みをダウンロードし、`inverse_renderer/checkpoints/Diffusion_Renderer_Inverse_Cosmos_7B` 配下のチェックポイントをファインチューニング済みのチェックポイントに置き換えてください。

<!-- デモを試す：**[Inverse Renderer Demo](https://huggingface.co/spaces/Brian9999/world_inverse_renderer_demo)** 🚧 -->

### Game Editing（ゲーム編集）

<!-- 完全なコードは [`game_editing/`](game_editing/) を参照してください。 -->

#### インストール

[DiffSynth-Studio](https://github.com/ShandaAI/DiffSynth-Studio) の手順に従って環境をセットアップし、関連する重みをダウンロードしてください。[HuggingFace](https://huggingface.co/Brian9999/stylerenderer/tree/main) からファインチューニング済みのチェックポイントをダウンロードし、`game_editing/models/train/Wan2.1-T2V-1.3B_gbuffer/` に配置してください。

#### クイックスタート

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

オンラインデモを試す：**[Game Editing Demo](https://huggingface.co/spaces/Brian9999/game-editing)**

## 📋 TODO

- [ ] データセットの公開
- [ ] データキュレーションツールキットの公開

## ❤️ 謝辞

本プロジェクトは以下の優れた研究に基づいています：

- [DiffusionRenderer](https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/) by NVIDIA Toronto AI Lab
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) by Wan-Video
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) by ModelScope
<!-- - [ReShade](https://reshade.me/) -->


## 📄 ライセンス

[LICENSE](LICENSE) を参照してください。


## 📝 引用

本プロジェクトがお役に立ちましたら、以下の引用をご検討ください：

```bibtex
@article{huang2026generativeworldrenderer,
    title={Generative World Renderer},
    author={Zheng-Hui Huang and Zhixiang Wang and Jiaming Tan and Ruihan Yu and Yidan Zhang and Bo Zheng and Yu-Lun Liu and Yung-Yu Chuang and Kaipeng Zhang},
    journal={arXiv preprint arXiv:2604.02329},
    year={2026}
}
```
