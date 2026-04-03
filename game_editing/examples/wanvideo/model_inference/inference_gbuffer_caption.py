"""
Batch inference with multiple styles on multiple GPUs.

Styles are defined in a JSON file (default: styles.json in the same directory).
  - null  => use original caption from CSV as-is
  - "Rewrite the scene ..." => template: spliced with original caption
  - any other string => used directly as the full prompt

Original caption is read from metadata_with_captions.csv.

Usage:
  # run all styles defined in styles.json
  python inference_gbuffer_caption.py --clip_dir "2025-12-28 15-53-24/clip_0052"

  # run specific styles only
  python inference_gbuffer_caption.py --clip_dir "..." --styles original bright_sunny

  # use a different styles JSON file
  python inference_gbuffer_caption.py --clip_dir "..." --styles_json my_styles.json

  # single GPU mode (called by subprocess internally)
  python inference_gbuffer_caption.py --gpu 0 --style original --prompt "..." --gbuffer_dir /path/to/clip_0052
"""
import torch, sys, os, argparse, subprocess, csv, json, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
DEFAULT_DATA_BASE = os.path.join(DEFAULT_BASE, "datasets/black_myth_gbuffers_clips")
DEFAULT_CAPTION_CSV = os.path.join(DEFAULT_DATA_BASE, "metadata_with_captions.csv")
DEFAULT_STYLES_JSON = os.path.join(SCRIPT_DIR, "styles.json")

negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def load_caption_from_csv(csv_path, clip_dir):
    """Look up the prompt for a given clip_dir from metadata_with_captions.csv."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # video column: "2025-12-28 15-53-24/clip_0052/clip_0052_rgb.mp4"
            if clip_dir in row["video"]:
                return row["prompt"]
    return None


def build_style_prompt(original_caption, style_hint):
    """Build a styled prompt from the original caption and a style hint.

    - style_hint is None => return original caption as-is
    - style_hint is a string => keep the content/scene part of the original caption,
      replace the lighting/vibe part with style_hint

    Captions typically follow the pattern:
      "<content about characters, actions, objects, environment>;
       the scene is <lighting/vibe>, creating a <mood> atmosphere."
    or:
      "<content>, <lighting descriptors>, creating a <mood> atmosphere."

    We keep only the content part and append the style_hint as the new lighting/vibe.
    """
    if style_hint is None:
        return original_caption

    # Try to split at "; the scene is ..." or "; the scene feels ..."
    match = re.search(r';\s*the\s+scene\s+(is|feels|appears)\s+', original_caption)
    if match:
        content = original_caption[:match.start()]
        return f"{content}; {style_hint}"

    # Try to split at ", the scene is ..."
    match = re.search(r',\s*the\s+scene\s+(is|feels|appears)\s+', original_caption)
    if match:
        content = original_caption[:match.start()]
        return f"{content}; {style_hint}"

    # Try to split at the last "creating a ... atmosphere" clause
    match = re.search(r'[,;]\s*creating\s+a[n]?\s+\w+', original_caption)
    if match:
        content = original_caption[:match.start()]
        # Also try to strip trailing lighting descriptors before "creating"
        # e.g. ", under low, diffuse lighting, creating ..."
        light_match = re.search(
            r'[,;]\s*(?:under|with|lit by|illuminated by|in)\s+.*$',
            content, re.IGNORECASE
        )
        if light_match:
            content = content[:light_match.start()]
        return f"{content}; {style_hint}"

    # Fallback: append style hint to the whole caption
    return f"{original_caption.rstrip('.')}; {style_hint}"


def parse_args():
    parser = argparse.ArgumentParser()
    # single-GPU subprocess args
    parser.add_argument("--gpu", type=int, default=None, help="GPU id (single GPU mode)")
    parser.add_argument("--style", type=str, default=None, help="Style name (single GPU mode)")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text (single GPU mode, passed by parent)")
    # configurable parameters
    parser.add_argument("--base", type=str, default=DEFAULT_BASE, help="DiffSynth-Studio root path")
    parser.add_argument("--gbuffer_dir", type=str, default=None, help="Path to G-buffer clip directory (overrides --data_base and --clip_dir)")
    parser.add_argument("--data_base", type=str, default=DEFAULT_DATA_BASE, help="Dataset root path (ignored if --gbuffer_dir is set)")
    parser.add_argument("--clip_dir", type=str, default="2025-12-27 16-34-41/clip_0000", help="Relative clip dir under data_base (ignored if --gbuffer_dir is set)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the fine-tuned checkpoint (.safetensors)")
    parser.add_argument("--model_dir", type=str, default="Wan2.1-T2V-1.3B_gbuffer_blackmyth_caption2", help="Model train dir name under BASE/models/train/ (ignored if --checkpoint is set)")
    parser.add_argument("--model_epoch", type=int, default=27, help="Model epoch number (ignored if --checkpoint is set)")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory (default: BASE/examples/.../validate_results)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs for parallel mode")
    # style selection
    parser.add_argument("--styles", type=str, nargs="+", default=None,
                        help="Which styles to run (default: all from JSON). E.g. --styles original bright_sunny")
    parser.add_argument("--styles_json", type=str, default=DEFAULT_STYLES_JSON,
                        help="JSON file with style definitions (default: styles.json)")
    # caption
    parser.add_argument("--caption_csv", type=str, default=DEFAULT_CAPTION_CSV,
                        help="Path to metadata_with_captions.csv")
    return parser.parse_args()


def run_single(args):
    from tqdm import tqdm
    from PIL import Image
    from diffsynth.utils.data import save_video, VideoData
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    sys.path.append(os.path.join(args.base, "examples/wanvideo/model_inference"))
    from gbuffer_utils import expand_patch_embedding, inject_gbuffer_unit

    prompt = args.prompt
    print(f"[GPU {args.gpu}] Style: {args.style}")
    print(f"[GPU {args.gpu}] Prompt: {prompt[:150]}...")

    if args.checkpoint:
        model_path = args.checkpoint
    else:
        model_path = os.path.join(args.base, f"models/train/{args.model_dir}/epoch-{args.model_epoch}.safetensors")
    print(f'Loading models from: "{model_path}"')

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_path),
            ModelConfig(os.path.join(args.base, "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(os.path.join(args.base, "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")),
        ],
    )
    expand_patch_embedding(pipe, num_gbuffers=5)
    inject_gbuffer_unit(pipe)

    gbuffer_dir = os.path.join(args.data_base, args.clip_dir)
    gbuffer_videos = []
    for mod in ["albedo", "depth", "metallic", "normal", "roughness"]:
        # Try {mod}.mp4 first, then {clip_name}_{mod}.mp4 for backward compatibility
        path = os.path.join(gbuffer_dir, f"{mod}.mp4")
        if not os.path.exists(path):
            clip_name = os.path.basename(args.clip_dir)
            path = os.path.join(gbuffer_dir, f"{clip_name}_{mod}.mp4")
        gbuffer_videos.append(VideoData(path).raw_data())
    gbuffer_videos = [
        [frame.resize((args.width, args.height), Image.LANCZOS) for frame in v[:args.num_frames]]
        for v in gbuffer_videos
    ]

    tiled = True
    tile_size = (30, 52)
    tile_stride = (15, 26)

    pipe.scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=5.0)
    inputs_posi = {"prompt": prompt}
    inputs_nega = {"negative_prompt": negative_prompt}
    inputs_shared = {
        "input_image": None, "end_image": None,
        "input_video": None, "denoising_strength": 1.0,
        "control_video": None, "reference_image": None,
        "vace_video": None, "vace_video_mask": None, "vace_reference_image": None, "vace_scale": 1.0,
        "seed": args.seed, "rand_device": "cpu",
        "height": args.height, "width": args.width, "num_frames": args.num_frames,
        "cfg_scale": args.cfg_scale, "cfg_merge": False,
        "sigma_shift": 5.0,
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
        "sliding_window_size": None, "sliding_window_stride": None,
        "input_audio": None, "audio_sample_rate": 16000,
        "s2v_pose_video": None, "audio_embeds": None, "s2v_pose_latents": None, "motion_video": None,
        "animate_pose_video": None, "animate_face_video": None, "animate_inpaint_video": None, "animate_mask_video": None,
        "vap_video": None,
        "gbuffer_videos": gbuffer_videos,
    }

    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    with torch.no_grad():
        for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps, desc=f"[GPU {args.gpu}] {args.style}")):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device="cuda")
            noise_pred_posi = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if args.cfg_scale != 1.0:
                noise_pred_nega = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + args.cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            inputs_shared["latents"] = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], inputs_shared["latents"])

    pipe.load_models_to_device(['vae'])
    with torch.no_grad():
        video = pipe.vae.decode(inputs_shared["latents"], device="cuda", tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
    video = pipe.vae_output_to_video(video)

    save_dir = args.save_dir if args.save_dir else os.path.join(args.base, "examples/wanvideo/model_inference/validate_results")
    # folder name: "session_clipN" e.g. "2025-12-28_15-53-24_clip52"
    parts = args.clip_dir.split("/")
    session_name = parts[0].replace(" ", "_")
    clip_short = re.sub(r"clip_0*", "clip", parts[1]) if len(parts) > 1 else parts[0]
    clip_tag = f"{session_name}_{clip_short}"
    save_subdir = os.path.join(save_dir, clip_tag)
    os.makedirs(save_subdir, exist_ok=True)
    save_path = os.path.join(save_subdir, f"epoch{args.model_epoch}_{args.style}.mp4")
    save_video(video, save_path, fps=15, quality=5)
    print(f"[GPU {args.gpu}] Saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()

    # Resolve --gbuffer_dir into data_base + clip_dir
    if args.gbuffer_dir:
        args.data_base = os.path.dirname(os.path.abspath(args.gbuffer_dir))
        args.clip_dir = os.path.basename(os.path.abspath(args.gbuffer_dir))

    if args.gpu is not None and args.style is not None and args.prompt is not None:
        # Single GPU mode: called by subprocess
        run_single(args)
    else:
        # ------ Load style definitions from JSON ------
        with open(args.styles_json, "r", encoding="utf-8") as f:
            style_defs = json.load(f)
        print(f"Loaded {len(style_defs)} styles from {args.styles_json}")

        # Filter to requested styles
        if args.styles:
            selected = args.styles
            for s in selected:
                if s not in style_defs:
                    print(f"[ERROR] Unknown style: {s}. Available: {list(style_defs.keys())}")
                    sys.exit(1)
        else:
            selected = list(style_defs.keys())

        # ------ Resolve original caption from CSV ------
        original_caption = load_caption_from_csv(args.caption_csv, args.clip_dir)
        if original_caption is None:
            print(f"[ERROR] Caption not found for clip_dir={args.clip_dir} in {args.caption_csv}")
            sys.exit(1)
        print(f"Original caption: {original_caption[:150]}...")

        # ------ Build per-style prompts ------
        style_prompts = {}
        for style_name in selected:
            hint = style_defs[style_name]
            style_prompts[style_name] = build_style_prompt(original_caption, hint)

        print(f"\nStyles to run ({len(selected)}):")
        for sn, sp in style_prompts.items():
            print(f"  {sn}: {sp[:120]}...")

        # ------ Launch subprocesses ------
        save_dir = args.save_dir if args.save_dir else os.path.join(args.base, "examples/wanvideo/model_inference/validate_results")
        procs = []
        for gpu_id, style_name in enumerate(selected):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id % args.num_gpus)
            cmd = [
                sys.executable, __file__,
                "--gpu", str(gpu_id),
                "--style", style_name,
                "--prompt", style_prompts[style_name],
                "--base", args.base,
                "--gbuffer_dir", os.path.join(args.data_base, args.clip_dir),
                "--model_dir", args.model_dir,
                "--model_epoch", str(args.model_epoch),
                *(["--checkpoint", args.checkpoint] if args.checkpoint else []),
                "--num_frames", str(args.num_frames),
                "--height", str(args.height),
                "--width", str(args.width),
                "--save_dir", save_dir,
                "--seed", str(args.seed),
                "--cfg_scale", str(args.cfg_scale),
                "--num_inference_steps", str(args.num_inference_steps),
            ]
            p = subprocess.Popen(cmd, env=env)
            procs.append((gpu_id, style_name, p))
            print(f"Launched {style_name} on GPU {gpu_id % args.num_gpus} (pid {p.pid})")

        results = []
        for gpu_id, style_name, p in procs:
            p.wait()
            if p.returncode == 0:
                print(f"[GPU {gpu_id}] {style_name} done!")
                results.append((style_name, True))
            else:
                print(f"[GPU {gpu_id}] {style_name} FAILED (exit code {p.returncode})")
                results.append((style_name, False))

        # ------ Write prompts.txt to the output subfolder ------
        parts = args.clip_dir.split("/")
        session_name = parts[0].replace(" ", "_")
        clip_short = re.sub(r"clip_0*", "clip", parts[1]) if len(parts) > 1 else parts[0]
        clip_tag = f"{session_name}_{clip_short}"
        save_subdir = os.path.join(save_dir, clip_tag)
        os.makedirs(save_subdir, exist_ok=True)

        prompts_path = os.path.join(save_subdir, "prompts.txt")
        with open(prompts_path, "w", encoding="utf-8") as f:
            f.write(f"clip_dir: {args.clip_dir}\n")
            f.write(f"original_caption: {original_caption}\n")
            f.write(f"model: {args.model_dir}/epoch-{args.model_epoch}\n")
            f.write(f"seed: {args.seed}  cfg_scale: {args.cfg_scale}  steps: {args.num_inference_steps}\n")
            f.write("\n")
            for style_name, ok in results:
                filename = f"epoch{args.model_epoch}_{style_name}.mp4"
                short_path = f"{clip_tag}/{filename}"
                f.write(f"{short_path}\n")
                f.write(f"  prompt: {style_prompts[style_name]}\n\n")
        print(f"Prompts log saved to {prompts_path}")

        print(f"\nAll done! Results in {save_dir}")
