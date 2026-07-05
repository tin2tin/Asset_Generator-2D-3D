# Asset Generator for 2D & 3D

A Blender add-on that generates 2D billboard/concept assets from text prompts and can convert them into textured 3D meshes — all in one pipeline, without leaving Blender.

<img width="3154" height="1728" alt="image" src="https://github.com/user-attachments/assets/ca42e886-da6d-4f86-8967-fa71e8437fbf" />

- **2D generation**: [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) generates an image per prompt line, [BiRefNet_HR](https://huggingface.co/ZhengPeng7/BiRefNet_HR) removes the background and splits multi-object images into separate cropped assets, each imported as an upright image-plane object and marked as a Blender asset.

<img width="392" height="236" alt="image" src="https://github.com/user-attachments/assets/0cd52374-47d1-41e3-ab19-87edba19715f" />

- **3D conversion**: selected image-plane assets are sent through an image-to-3D pipeline to produce a textured `.glb` mesh, imported back into the scene next to the source plane. Two backends are selectable:
  - [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (Microsoft, `TRELLIS.2-4B`) — the default; proven, needs no camera parameters.
  - [Pixal3D](https://github.com/TencentARC/Pixal3D) (TencentARC) — pixel-aligned conditioning that better preserves fine surface detail, at the cost of a manual camera FOV guess (there's no real camera to estimate from for flat AI illustrations).

<img width="394" height="96" alt="image" src="https://github.com/user-attachments/assets/8d0c561e-8303-4fbc-a77c-5384acdbeb72" />

Both stages run in isolated background subprocesses with their own progress bar, so Blender's UI stays responsive during generation.

## Requirements

- Windows, Blender 5.2 (bundled Python 3.13).
- An NVIDIA GPU with a recent driver (CUDA 12.8 runtime is installed automatically as part of the dependency setup — no separate CUDA Toolkit install is required to *run* the add-on).
- ~10–15 GB of free disk space for the model weights, CUDA-enabled PyTorch, and the TRELLIS.2 runtime. The optional Pixal3D backend downloads an additional ~24 GB checkpoint (`TencentARC/Pixal3D`) the first time it's used.
- 2D generation alone needs no extra setup beyond the dependency install below. 3D conversion additionally needs the compiled CUDA extensions (`o-voxel`, `cumesh`, `flexgemm`, `nvdiffrast`, `flash_attn`) — prebuilt wheels for these are bundled or fetched automatically (see [Prebuilt CUDA extensions](#prebuilt-cuda-extensions) below), so end users do **not** need a CUDA Toolkit or MSVC install for this either.

## Installation

<img width="1286" height="436" alt="image" src="https://github.com/user-attachments/assets/dcaae510-3988-49d7-9636-0a8afc85c638" />

1. Download: https://github.com/tin2tin/Asset_Generator-2D-3D/archive/refs/heads/main.zip and install the repo as a zip via `Edit > Preferences > Add-ons > Install...`).
2. Enable **"Asset Generator (2D/3D)"** in the add-ons list.
3. Open its preferences and click **"Install Dependencies"**. This runs in the background and:
   - Installs a CUDA-enabled PyTorch build (`torch==2.9.1+cu128`) into an isolated `addon_packages/` folder (added to `sys.path` at runtime — it does not touch Blender's own Python environment).
   - Installs the full pinned dependency set from `requirements.txt`.
   - Clones the `TRELLIS.2` and `Pixal3D` repos (for the 3D pipeline code).
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

### 3D Conversion
1. Select one or more of the generated 2D image-plane assets in the viewport.
2. Pick a **backend** — **TRELLIS.2** (default) or **Pixal3D**. For Pixal3D, set the **Camera FOV** (0.2 rad is a sane default for flat illustrations) and **Mesh Scale** in the camera box that appears.
3. Click **"Convert Selected to 3D"**.
4. Each selected plane's texture is run through the chosen backend; the resulting textured mesh is imported as a `.glb` next to the original plane.
5. Optionally expand **Advanced Settings** to tune the sparse-structure / shape / texture sampler steps, guidance, seed, and max token budget per stage.

## How it works

- **Isolated environment**: all Python packages are installed into `addon_packages/` inside this add-on's folder rather than Blender's own site-packages, so the heavy ML stack can't conflict with Blender's bundled libraries (or vice versa). It's added to `sys.path` at runtime, not a real venv.
- **Subprocess isolation**: generation runs in a separate `python.exe` process (not inside Blender's process), so a crash or CUDA OOM in the model code can't take down the Blender session. Progress lines are parsed from the subprocess's stdout to drive the UI progress bar.
- **VRAM discipline**: within each subprocess, models are explicitly released (`del` + `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`) as soon as they're no longer needed — the 2D script never holds the diffusion pipeline and the segmenter in VRAM at the same time, and the 3D script frees per-mesh intermediates after every conversion.
- **Frozen dependencies**: `requirements.txt` pins the full transitive closure of runtime packages and installs everything with `--no-deps`, so nothing can silently pull in a second (CPU-only) `torch`/`numpy` that would shadow the CUDA build.

## Prebuilt CUDA extensions

The compiled extensions (`o-voxel`, `cumesh`, `flexgemm`, `nvdiffrast`, `flash_attn`, and `natten` — the last two shared with the Pixal3D backend) are ordinarily built from source against a specific CUDA Toolkit + MSVC version — impractical to ask end users to set up. Instead:

- `o-voxel`, `cumesh`, `flexgemm`, and `nvdiffrast` wheels (built for `cu128` / `torch2.9` / `cp313` / `win_amd64`) are bundled directly in [`wheels/`](wheels/) and installed straight from there.
- `flash_attn`'s wheel (~240 MB) and `natten`'s wheel (~130 MB) are too large to commit to git — so they're instead downloaded on demand from the same [`PozzettiAndrea/cuda-wheels`](https://github.com/PozzettiAndrea/cuda-wheels) release index at install time.
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
- [Pixal3D](https://github.com/TencentARC/Pixal3D) (TencentARC) as the alternative pixel-aligned image-to-3D backend.
- [PozzettiAndrea/cuda-wheels](https://github.com/PozzettiAndrea/cuda-wheels) for prebuilt Windows CUDA extension wheels.
