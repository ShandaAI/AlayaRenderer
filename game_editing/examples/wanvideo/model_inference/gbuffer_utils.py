import torch
import torch.nn as nn
from diffsynth.diffusion.base_pipeline import PipelineUnit


class WanVideoUnit_GBufferEncoder(PipelineUnit):
    """
    Encode G-buffer videos (depth, normal, albedo, etc.) via VAE and concat to y.
    Each G-buffer modality is a list of PIL Images (video frames).
    gbuffer_videos: list of lists, e.g. [[depth_frame0, ...], [normal_frame0, ...]]
    Output shape: [1, N*16, T, H, W] where N = number of G-buffer modalities.
    """
    def __init__(self):
        super().__init__(
            input_params=("gbuffer_videos", "y", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )

    def process(self, pipe, gbuffer_videos, y, tiled, tile_size, tile_stride):
        if gbuffer_videos is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        all_latents = []
        for gbuffer_video in gbuffer_videos:
            video_tensor = pipe.preprocess_video(gbuffer_video)
            latent = pipe.vae.encode(
                video_tensor, device=pipe.device,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            all_latents.append(latent)
        gbuffer_latents = torch.cat(all_latents, dim=1)  # [1, N*16, T, H, W]
        if y is not None:
            gbuffer_latents = torch.cat([y, gbuffer_latents], dim=1)
        return {"y": gbuffer_latents}


def expand_patch_embedding(pipe, num_gbuffers):
    """
    Expand DiT's patch_embedding Conv3d to accept additional G-buffer latent channels.
    New channels are zero-initialized so the model starts equivalent to the original.
    """
    if num_gbuffers <= 0:
        return
    dit = pipe.dit
    old_conv = dit.patch_embedding
    old_weight = old_conv.weight  # [out_channels, in_channels, *kernel_size]
    old_bias = old_conv.bias

    extra_channels = num_gbuffers * 16  # 16 VAE latent channels per modality

    # Skip if already expanded (e.g., loading from a GBuffer-trained checkpoint)
    # If current in_channels > extra_channels, it already includes base + gbuffer channels
    if old_weight.shape[1] > extra_channels:
        print(f"patch_embedding already has {old_weight.shape[1]} input channels (> {extra_channels} extra), already expanded. Skipping.")
        return

    new_in_dim = old_weight.shape[1] + extra_channels

    new_conv = nn.Conv3d(
        new_in_dim, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :old_weight.shape[1]] = old_weight
        if old_bias is not None:
            new_conv.bias.copy_(old_bias)

    dit.patch_embedding = new_conv.to(dtype=old_weight.dtype, device=old_weight.device)
    dit.in_dim = new_in_dim
    print(f"Expanded patch_embedding: {old_weight.shape[1]} -> {new_in_dim} input channels (+{extra_channels} for {num_gbuffers} G-buffers)")


def inject_gbuffer_unit(pipe):
    """
    Insert WanVideoUnit_GBufferEncoder into the pipeline's unit list,
    after ImageEmbedderFused and before FunControl.
    """
    # Skip if already injected
    for u in pipe.units:
        if type(u).__name__ == "WanVideoUnit_GBufferEncoder":
            print("WanVideoUnit_GBufferEncoder already present, skipping injection.")
            return
    unit = WanVideoUnit_GBufferEncoder()
    insert_idx = None
    for i, u in enumerate(pipe.units):
        if type(u).__name__ == "WanVideoUnit_ImageEmbedderFused":
            insert_idx = i + 1
            break
    if insert_idx is not None:
        pipe.units.insert(insert_idx, unit)
    else:
        # Fallback: insert before the last few units
        pipe.units.insert(-3, unit)
    print(f"Injected WanVideoUnit_GBufferEncoder at position {insert_idx}")
