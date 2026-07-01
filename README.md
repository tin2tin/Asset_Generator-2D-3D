# Asset Generator for 2D & 3D

A Blender add-on that generates 2D billboard/concept assets from text prompts and can convert them into textured 3D meshes — all in one pipeline, without leaving Blender.

- **2D generation**: [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) generates an image per prompt line, [BiRefNet_HR](https://huggingface.co/ZhengPeng7/BiRefNet_HR) removes the background and splits multi-object images into separate cropped assets, each imported as an upright image-plane object and marked as a Blender asset.
- **3D conversion**: selected image-plane assets are sent through [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (Microsoft's image-to-3D pipeline) to produce a textured `.glb` mesh, imported back into the scene next to the source plane.

Both stages run in isolated background subprocesses with their own progress bar, so Blender's UI stays responsive during generation.

## Requirements

- Windows, Blender 5.2 (bundled Python 3.13).
- An NVIDIA GPU with a recent driver (CUDA 12.8 runtime is installed automatically as part of the dependency setup — no separate CUDA Toolkit install is required to *run* the add-on).
- ~10–15 GB of free disk space for the model weights, CUDA-enabled PyTorch, and the TRELLIS.2 runtime.
- 2D generation alone needs no extra setup beyond the dependency install below. 3D conversion additionally needs the compiled CUDA extensions (`o-voxel`, `cumesh`, `flexgemm`, `nvdiffrast`, `flash_attn`) — prebuilt wheels for these are bundled or fetched automatically (see [Prebuilt CUDA extensions](#prebuilt-cuda-extensions) below), so end users do **not** need a CUDA Toolkit or MSVC install for this either.

## Installation

1. Copy this folder into your Blender add-ons directory (or install the repo as a zip via `Edit > Preferences > Add-ons > Install...`).
2. Enable **"2D Asset Generator & Trellis 3D (Async Pro)"** in the add-ons list.
3. Open its preferences and click **"Install 2D + 3D Dependencies"**. This runs in the background and:
   - Installs a CUDA-enabled PyTorch build (`torch==2.9.1+cu128`) into an isolated `addon_packages/` folder (added to `sys.path` at runtime — it does not touch Blender's own Python environment).
   - Installs the full pinned dependency set from `requirements.txt`.
   - Clones the `TRELLIS.2` repo (for the 3D pipeline code).
   - Installs the compiled CUDA extensions from prebuilt wheels (falling back to a from-source build only if no matching wheel is found).
4. Watch progress in the Preferences panel and the system console. When it reports "Done — CUDA OK", you're ready to generate.

Dependencies already installed and matching the pinned versions are skipped automatically on subsequent runs — you can re-run "Install 2D + 3D Dependencies" any time to repair or update without a full reinstall.

To remove everything the installer downloaded (to free disk space or start over), use **"Nuclear Wipe"** in preferences — this deletes `addon_packages/` only.

## Usage

Both panels live in the 3D Viewport sidebar, under the **"2D Asset"** tab.

### 2D Asset Generator
1. Choose **Prompt** (a single prompt + asset name) or **Text-Block** (one prompt per line in a Blender text block — useful for batch-generating many assets in one pass).
2. Click **"Generate 2D Assets"**.
3. Each generated image is background-removed, split into per-object crops if it contains multiple subjects, and imported as an upright image-plane object with a transparent material, marked as a Blender asset with a thumbnail preview.

### 3D Trellis Conversion
1. Select one or more of the generated 2D image-plane assets in the viewport.
2. Click **"Convert Selected to 3D"**.
3. Each selected plane's texture is run through TRELLIS.2; the resulting textured mesh is imported as a `.glb` next to the original plane.

## How it works

- **Isolated environment**: all Python packages are installed into `addon_packages/` inside this add-on's folder rather than Blender's own site-packages, so the heavy ML stack can't conflict with Blender's bundled libraries (or vice versa). It's added to `sys.path` at runtime, not a real venv.
- **Subprocess isolation**: generation runs in a separate `python.exe` process (not inside Blender's process), so a crash or CUDA OOM in the model code can't take down the Blender session. Progress lines are parsed from the subprocess's stdout to drive the UI progress bar.
- **VRAM discipline**: within each subprocess, models are explicitly released (`del` + `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`) as soon as they're no longer needed — the 2D script never holds the diffusion pipeline and the segmenter in VRAM at the same time, and the 3D script frees per-mesh intermediates after every conversion.
- **Frozen dependencies**: `requirements.txt` pins the full transitive closure of runtime packages and installs everything with `--no-deps`, so nothing can silently pull in a second (CPU-only) `torch`/`numpy` that would shadow the CUDA build.

## Prebuilt CUDA extensions

TRELLIS.2's compiled extensions (`o-voxel`, `cumesh`, `flexgemm`, `nvdiffrast`) and `flash_attn` are ordinarily built from source against a specific CUDA Toolkit + MSVC version — impractical to ask end users to set up. Instead:

- `o-voxel`, `cumesh`, `flexgemm`, and `nvdiffrast` wheels (built for `cu128` / `torch2.9` / `cp313` / `win_amd64`) are bundled directly in [`wheels/`](wheels/) and installed straight from there.
- `flash_attn`'s matching wheel is ~240 MB — too large to commit to git — so it's instead downloaded on demand from the same [`PozzettiAndrea/cuda-wheels`](https://github.com/PozzettiAndrea/cuda-wheels) release index at install time.
- If a bundled/remote wheel doesn't match your Python/CUDA/torch combination, the installer falls back to cloning and compiling the extension from source (requires the CUDA Toolkit and MSVC Build Tools with the C++ workload installed).

## Known limitations

- Windows only (the CUDA-torch install, wheel selection, and DLL-search-path setup are Windows-specific).
- Requires an NVIDIA GPU — there is no CPU fallback for either generation stage.
- 3D conversion needs a GPU with enough VRAM for TRELLIS.2's 4B-parameter model (`low_vram` mode is enabled by default to reduce peak usage).

## Credits

Built on top of:
- [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Tongyi-MAI) for 2D image generation.
- [BiRefNet_HR](https://huggingface.co/ZhengPeng7/BiRefNet_HR) (ZhengPeng7) for background removal / segmentation.
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (Microsoft) for image-to-3D mesh generation.
- [PozzettiAndrea/cuda-wheels](https://github.com/PozzettiAndrea/cuda-wheels) for prebuilt Windows CUDA extension wheels.
