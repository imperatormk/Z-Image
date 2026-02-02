#!/usr/bin/env python3
"""Z-Image CLI - Quick image generation with custom prompts."""

import argparse
import os
import time
import torch

from src.utils import ensure_model_weights, load_from_local_dir, set_attention_backend
from src.zimage import generate


def load_or_quantize_transformer(model_path, nf4_cache_path, device, dtype):
    """Load transformer, using cached NF4 weights if available."""
    from mps_bitsandbytes import BitsAndBytesConfig, quantize_model, get_memory_footprint

    if os.path.exists(nf4_cache_path):
        print(f"Loading cached NF4 weights from {nf4_cache_path}...")
        # Load the architecture first (without weights)
        components = load_from_local_dir(model_path, device="cpu", dtype=dtype)
        transformer = components["transformer"]

        # Quantize the model structure (this sets up Linear4bit layers)
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        transformer = quantize_model(transformer, quantization_config=config, device="cpu")

        # Load the cached quantized state dict
        state_dict = torch.load(nf4_cache_path, map_location="cpu", weights_only=True)
        transformer.load_state_dict(state_dict, strict=True)
        transformer = transformer.to(device)

        components["transformer"] = transformer
        components["vae"] = components["vae"].to(device)
        components["text_encoder"] = components["text_encoder"].to(device)

        footprint = get_memory_footprint(transformer)
        print(f"Loaded NF4: {footprint['actual_size_gb']:.2f} GB")
        return components
    else:
        print("Quantizing to NF4 (first time, will cache)...")
        components = load_from_local_dir(model_path, device="cpu", dtype=dtype)

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        transformer = quantize_model(
            components["transformer"],
            quantization_config=config,
            device=device,
        )

        # Save quantized state dict for next time
        print(f"Saving NF4 cache to {nf4_cache_path}...")
        torch.save(transformer.state_dict(), nf4_cache_path)

        components["transformer"] = transformer
        components["vae"] = components["vae"].to(device)
        components["text_encoder"] = components["text_encoder"].to(device)

        footprint = get_memory_footprint(transformer)
        print(f"Quantized: {footprint['actual_size_gb']:.2f} GB")
        return components


def main():
    parser = argparse.ArgumentParser(description="Generate images with Z-Image")
    parser.add_argument("prompt", type=str, nargs="?", default=(
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
        "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
        "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
        "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
    ), help="Text prompt for image generation")
    parser.add_argument("-o", "--output", type=str, default="output.png", help="Output filename")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", type=int, default=8, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--attention", type=str, default="mps_flash",
                        choices=["mps_flash", "native", "_native_flash", "_native_math"],
                        help="Attention backend")
    parser.add_argument("--nf4", action="store_true", help="Use NF4 4-bit quantization (saves ~4x memory)")
    parser.add_argument("--nf4-cache", type=str, default="ckpts/Z-Image-Turbo/transformer_nf4.pt",
                        help="Path to cache NF4 quantized weights")
    args = parser.parse_args()

    # Only patch SDPA if using mps_flash backend
    if args.attention == "mps_flash":
        try:
            from mps_flash_attn import replace_sdpa
            replace_sdpa()
            print("✓ MPS Flash Attention SDPA patched")
        except ImportError:
            print("⚠ mps-flash-attn not installed, falling back to native")
            args.attention = "native"

    print(f"Loading model...")
    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)

    if args.nf4:
        components = load_or_quantize_transformer(
            model_path,
            args.nf4_cache,
            device="mps",
            dtype=torch.bfloat16
        )
    else:
        components = load_from_local_dir(model_path, device="mps", dtype=torch.bfloat16)

    set_attention_backend(args.attention)
    print(f"Attention: {args.attention}")

    print(f"Generating: {args.prompt[:50]}...")
    start = time.time()
    images = generate(
        prompt=args.prompt,
        **components,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=torch.Generator("mps").manual_seed(args.seed),
    )
    elapsed = time.time() - start

    images[0].save(args.output)
    print(f"Saved: {args.output} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
