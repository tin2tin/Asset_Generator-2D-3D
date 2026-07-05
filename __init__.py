bl_info = {
    "name": "Asset Generator (2D/3D)",
    "author": "tintwotin",
    "version": (5, 5),
    "blender": (5, 2, 0),
    "category": "3D View",
    "location": "3D Editor > Sidebar > Asset Gen",
    "description": "Async Z-Image 2D and TRELLIS.2 3D with Fixed CUDA Extension Loading.",
}

import bpy
import bmesh
import os
import re
import subprocess
import sys
import math
import importlib
import gc
import shutil
import json
import threading
import queue
import stat
import unicodedata
import atexit
import time
from os.path import join
from mathutils import Vector
from typing import Optional
from bpy.types import Operator, PropertyGroup, Panel, AddonPreferences
from bpy.props import StringProperty, EnumProperty, PointerProperty, BoolProperty, IntProperty, FloatProperty

# --- CRITICAL ENVIRONMENT PROTECTION ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

DEBUG = True

# --- FILE LOGGING ---
# Install/build output (especially CUDA extension compiles) can run to tens of
# thousands of lines — far more than a user can practically copy out of
# Blender's system console. Mirror everything to a file inside the addon
# folder instead, so a failure can be diagnosed by reading the file directly.
_LOG_LOCK = threading.Lock()
_LOG_FILE_PATH = None
_LOG_FILE_HANDLE = None

def _install_log_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "install_log.txt")

def start_install_log():
    """(Re)start the install log for a fresh run — overwrites any previous log
    so it reflects only the most recent install attempt. Keeps one file handle
    open for the whole run (flushed after every write) instead of reopening on
    every line: reopening thousands of times a second during a heavy compiler
    dump is slow and, on Windows, gives a brief window each time for another
    process (e.g. an editor with the file open) to contend for the handle —
    which previously could silently blackout the rest of a log mid-build."""
    global _LOG_FILE_PATH, _LOG_FILE_HANDLE
    stop_install_log()
    _LOG_FILE_PATH = _install_log_path()
    try:
        _LOG_FILE_HANDLE = open(_LOG_FILE_PATH, "w", encoding="utf-8")
        _LOG_FILE_HANDLE.write(f"=== Install log started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        _LOG_FILE_HANDLE.flush()
    except Exception as e:
        print(f"[2D/3D Asset Pro Log] could not create install log: {e}")
        _LOG_FILE_HANDLE = None

def stop_install_log():
    """Close the log file handle, if one is open. Safe to call anytime."""
    global _LOG_FILE_HANDLE
    with _LOG_LOCK:
        if _LOG_FILE_HANDLE is not None:
            try:
                _LOG_FILE_HANDLE.close()
            except Exception:
                pass
            _LOG_FILE_HANDLE = None

def _log_line(text):
    """Append a line to the active install log, if any. A write failure closes
    the stale handle and prints a visible console warning (so a log/console
    mismatch is never silent) rather than swallowing it — the next call
    transparently reopens in append mode and keeps going."""
    global _LOG_FILE_HANDLE
    if not _LOG_FILE_PATH:
        return
    line = text if text.endswith("\n") else text + "\n"
    with _LOG_LOCK:
        try:
            if _LOG_FILE_HANDLE is None:
                _LOG_FILE_HANDLE = open(_LOG_FILE_PATH, "a", encoding="utf-8")
            _LOG_FILE_HANDLE.write(line)
            _LOG_FILE_HANDLE.flush()
        except Exception as e:
            try:
                _LOG_FILE_HANDLE.close()
            except Exception:
                pass
            _LOG_FILE_HANDLE = None
            print(f"[2D/3D Asset Pro Log] install log write failed (will retry next line): {e}")

def debug_print(*args, **kwargs):
    """Console logging for the generation process; mirrored to install_log.txt
    while an install log is active."""
    if DEBUG:
        msg = " ".join(str(a) for a in args)
        print("[2D/3D Asset Pro Log] ", *args, **kwargs)
        _log_line("[2D/3D Asset Pro Log] " + msg)

# --- PROGRESS REPORTING ---
# Background work runs in threads (installers) and subprocesses (inference),
# but Blender UI can only be drawn from the main thread. We funnel progress
# through a thread-safe store; the operators' modal timers read it and tag the
# UI for redraw so the bars animate live in the panel and preferences.

_progress_lock = threading.Lock()
PROGRESS = {
    "install":   {"active": False, "value": 0.0, "label": ""},
    "uninstall": {"active": False, "value": 0.0, "label": ""},
    "infer_2d":  {"active": False, "value": 0.0, "label": ""},
    "infer_3d":  {"active": False, "value": 0.0, "label": ""},
}

def set_progress(key, value=None, label=None, active=None):
    """Thread-safe update of a progress slot. Any field may be omitted."""
    with _progress_lock:
        p = PROGRESS[key]
        if value is not None:
            p["value"] = max(0.0, min(1.0, value))
        if label is not None:
            p["label"] = label
        if active is not None:
            p["active"] = active

def get_progress(key):
    """Return (active, value, label) for a progress slot."""
    with _progress_lock:
        p = PROGRESS[key]
        return p["active"], p["value"], p["label"]

def draw_progress(layout, key):
    """Draw a progress bar for `key`, but only while that work is active."""
    active, value, label = get_progress(key)
    if not active:
        return
    text = f"{label}  {int(value * 100)}%" if label else f"{int(value * 100)}%"
    # layout.progress() is a real progress widget added in Blender 4.0; fall
    # back to a plain label on older builds.
    if hasattr(layout, "progress"):
        layout.progress(factor=value, type='BAR', text=text)
    else:
        layout.label(text=text)

def tag_redraw_all(context):
    """Force every area (View3D panels + Preferences) to redraw."""
    wm = context.window_manager
    if not wm:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            area.tag_redraw()

# --- SYSTEM & PERMISSION UTILS ---

def remove_readonly(func, path, excinfo):
    """Clear the read-only bit and re-attempt removal (fixes Windows WinError 5)."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        debug_print(f"Failed to delete {path}: {str(e)}")

def flush():
    """Aggressively clear RAM and VRAM."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except:
        pass

def gfx_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    return "cpu"

# --- SUBPROCESS LIFECYCLE ---
# Model inference always runs in a child process (see ZIMAGE_OT_GenerateAsset /
# TRELLIS_OT_ConvertSelected), so its VRAM/RAM is reclaimed by the OS the
# instant that process exits — including on failure, since a non-zero exit
# still tears the process down. The one case that doesn't cover is the user
# cancelling (Esc) or quitting Blender while a subprocess is still running:
# nothing else would ever terminate it, leaving it holding GPU memory
# indefinitely in the background. Track live subprocesses here so both can
# be force-cleaned.
_ACTIVE_SUBPROCESSES = set()

def _track_subprocess(proc):
    _ACTIVE_SUBPROCESSES.add(proc)

def _untrack_subprocess(proc):
    _ACTIVE_SUBPROCESSES.discard(proc)

def _terminate_subprocess(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

@atexit.register
def _kill_active_subprocesses():
    for proc in list(_ACTIVE_SUBPROCESSES):
        _terminate_subprocess(proc)

def run_logged(cmd, cwd=None, env=None, check=False):
    """subprocess.run replacement for install/build commands: streams merged
    stdout+stderr line-by-line to the console (so live behavior in Blender's
    system console is unchanged) while also appending every line to
    install_log.txt. Returns a CompletedProcess-like object with .returncode."""
    _log_line(f"$ {' '.join(str(c) for c in cmd)}" + (f"  (cwd={cwd})" if cwd else ""))
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    _track_subprocess(proc)
    try:
        for line in proc.stdout:
            print(line, end="")
            _log_line(line.rstrip("\n"))
        proc.wait()
    finally:
        _untrack_subprocess(proc)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc

# --- PATH & VENV MANAGEMENT ---

def addon_script_path() -> str:
    """Returns a valid directory for data storage, handling Text Editor fallback."""
    try:
        path = os.path.dirname(os.path.realpath(__file__))
        if ".blend" in path or not os.path.exists(path):
            raise Exception("Context is Text Editor")
        return path
    except:
        fallback_path = os.path.join(bpy.utils.user_resource('DATAFILES'), "2D_Asset_Generator_Pro")
        os.makedirs(fallback_path, exist_ok=True)
        return fallback_path

def packages_path() -> str:
    """Addon-owned package directory. pip --target here; never touches Blender's Python."""
    p = os.path.normpath(os.path.join(addon_script_path(), "addon_packages"))
    os.makedirs(p, exist_ok=True)
    return p

def python_exec() -> str:
    """Use Blender's own Python interpreter; isolation is via --target, not a venv."""
    return sys.executable

def activate_virtualenv():
    """Put the addon's package directory first on sys.path so imports resolve there."""
    pkgs = packages_path()
    if pkgs not in sys.path:
        sys.path.insert(0, pkgs)
    repo_path = os.path.normpath(os.path.join(addon_script_path(), "TRELLIS_REPO"))
    if os.path.exists(repo_path) and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    pixal3d_repo_path = os.path.normpath(os.path.join(addon_script_path(), "PIXAL3D_REPO"))
    if os.path.exists(pixal3d_repo_path) and pixal3d_repo_path not in sys.path:
        sys.path.insert(0, pixal3d_repo_path)
    importlib.invalidate_caches()
    return True

# --- PRO INSTALLATION LOGIC (FIXED FOR CUMESH) ---

# CUDA torch build. Kept in one place so the 2D and 3D installers stay in sync.
# The +cu128 local version tag + the matching --index-url guarantees the CUDA
# wheels; a plain "torch" would resolve to the CPU build on PyPI.
TORCH_CUDA_SPEC = ["torch==2.9.1+cu128", "torchvision==0.24.1+cu128", "torchaudio==2.9.1+cu128"]
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

def install_cuda_torch(py, pkgs_dir):
    """Force-install the CUDA build of torch, overriding any CPU torch that a
    dependency pulled in. Returns the CompletedProcess so callers can check it."""
    return run_logged(
        [py, "-m", "pip", "install", "--disable-pip-version-check", "--target", pkgs_dir]
        + TORCH_CUDA_SPEC
        + ["--index-url", TORCH_CUDA_INDEX, "--upgrade", "--force-reinstall", "--no-deps"]
    )

def find_cuda_home():
    """Return an installed CUDA Toolkit root, or None. torch.utils.cpp_extension
    (used by the o-voxel/cumesh/flexgemm/nvdiffrast setup.py builds) raises
    immediately if CUDA_HOME/CUDA_PATH isn't set, even when the toolkit is
    actually present on disk — Blender doesn't inherit that env var itself."""
    for var in ("CUDA_HOME", "CUDA_PATH"):
        path = os.environ.get(var)
        if path and os.path.isfile(os.path.join(path, "bin", "nvcc.exe")):
            return path
    base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(base):
        for v in sorted(os.listdir(base), reverse=True):
            candidate = os.path.join(base, v)
            if os.path.isfile(os.path.join(candidate, "bin", "nvcc.exe")):
                return candidate
    return None

def verify_cuda_torch(py, pkgs_dir):
    """Import the freshly installed torch in a clean subprocess and report
    whether it actually sees CUDA — so failures surface at install time, not
    at generation time."""
    code = (
        "import sys; sys.path.insert(0, r'" + pkgs_dir + "'); "
        "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'CUDA_NONE')"
    )
    try:
        r = subprocess.run([py, "-c", code], capture_output=True, text=True)
        return "CUDA_OK" in (r.stdout or "")
    except Exception as e:
        debug_print(f"CUDA verification failed to run: {e}")
        return False

def _spec_name_version(spec):
    """Split a pinned requirement like 'torch==2.9.1+cu128' into
    ('torch', '2.9.1+cu128'). Returns (spec, None) if it has no '==' pin."""
    name, sep, ver = spec.partition("==")
    return name.strip(), (ver.strip() if sep else None)

def _wheel_version(wheel_path_or_url):
    """Extract the version field from a wheel filename
    (name-version-pytag-abitag-platformtag.whl), e.g.
    'o_voxel-0.0.1+cu128torch2.9-cp313-cp313-win_amd64.whl' -> '0.0.1+cu128torch2.9'.
    Handles URL-encoded '+' (%2B) in remote wheel URLs. Returns None if the
    filename doesn't look like a wheel."""
    import urllib.parse
    base = urllib.parse.unquote(os.path.basename(wheel_path_or_url))
    if not base.endswith(".whl"):
        return None
    parts = base[: -len(".whl")].split("-")
    return parts[1] if len(parts) > 1 else None

def dist_info_version(pkgs_dir, dist_name):
    """Return the version of `dist_name` installed under pkgs_dir, read from its
    *.dist-info directory, or None if it is not installed. We read the metadata
    on disk rather than importing the package because torch must NOT be imported
    into Blender's own process here."""
    import glob
    def norm(s):
        # PEP 503-style name comparison: runs of -, _ and . are equivalent.
        return re.sub(r"[-_.]+", "_", s).lower()
    target = norm(dist_name)
    for info in glob.glob(os.path.join(pkgs_dir, "*.dist-info")):
        base = os.path.basename(info)[:-len(".dist-info")]
        name, _, ver = base.rpartition("-")
        if name and norm(name) == target:
            return ver
    return None

def cuda_torch_satisfied(py, pkgs_dir):
    """True only if the exact pinned torch/torchvision/torchaudio builds are
    already installed AND torch actually sees CUDA — i.e. a previous install
    completed successfully. Used to skip the expensive (~2.5 GB, force-reinstall)
    torch download when nothing needs to change."""
    for spec in TORCH_CUDA_SPEC:
        name, ver = _spec_name_version(spec)
        if ver is None or dist_info_version(pkgs_dir, name) != ver:
            return False
    return verify_cuda_torch(py, pkgs_dir)

def find_nvcc():
    """Locate the CUDA compiler (nvcc). Checks PATH first, then the standard
    CUDA_PATH / CUDA_HOME install locations. Returns the path or None."""
    exe = "nvcc.exe" if os.name == "nt" else "nvcc"
    p = shutil.which("nvcc")
    if p:
        return p
    for var in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(var)
        if root:
            cand = os.path.join(root, "bin", exe)
            if os.path.exists(cand):
                return cand
    return None

def patch_ovoxel_windows_build(src_path):
    """o-voxel builds tensors via torch::zeros({N, C}, ...) / torch::from_blob(
    ..., {v.size()}, ...) where the sizes are size_t. MSVC hard-errors on that
    narrowing conversion inside a braced-init-list (C2398); gcc/clang (the
    only platform upstream tests on) merely warn. Cast to int64_t at each
    call site so the extension builds on Windows. Also strips the 'd' suffix
    off two floating-point literals (1e-6d, 0.0d) in flexible_dual_grid.cpp —
    not a valid C++ literal suffix (MSVC: error C3688), silently tolerated
    elsewhere. Idempotent — safe to call on every install."""
    fixes = {
        os.path.join(src_path, "src", "io", "filter_neighbor.cpp"): [
            ("torch::zeros({N, C}, torch::dtype(torch::kUInt8))",
             "torch::zeros({(int64_t)N, (int64_t)C}, torch::dtype(torch::kUInt8))"),
        ],
        os.path.join(src_path, "src", "io", "filter_parent.cpp"): [
            ("torch::zeros({N_leaf, C}, torch::kUInt8)",
             "torch::zeros({(int64_t)N_leaf, (int64_t)C}, torch::kUInt8)"),
        ],
        os.path.join(src_path, "src", "io", "svo.cpp"): [
            ("torch::from_blob(svo.data(), {svo.size()}, torch::kUInt8)",
             "torch::from_blob(svo.data(), {(int64_t)svo.size()}, torch::kUInt8)"),
            ("torch::from_blob(codes.data(), {codes.size()}, torch::kInt32)",
             "torch::from_blob(codes.data(), {(int64_t)codes.size()}, torch::kInt32)"),
        ],
        os.path.join(src_path, "src", "convert", "flexible_dual_grid.cpp"): [
            ("if (segment_length < 1e-6d) continue;",
             "if (segment_length < 1e-6) continue;"),
            ("if (dir[axis] == 0.0d) {",
             "if (dir[axis] == 0.0) {"),
        ],
    }
    for path, replacements in fixes.items():
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        changed = False
        for old, new in replacements:
            if old in text:
                text = text.replace(old, new)
                changed = True
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            debug_print(f"Patched MSVC narrowing-conversion fix into {path}")

def patch_pixal3d_seqlen_cache(src_path):
    """Port tin2tin/TRELLIS.2's fix-cascade-seqlen-cache patch into the vendored
    Pixal3D repo's copy of the same sparse-tensor module. VarLenTensor.reduce()
    calls torch.segment_reduce(red, lengths=self.seqlen) unconditionally; in the
    cascade Shape-SLat sampler's CFG-rescale path, x_0_pos inherits a cached
    seqlen (via a spatial-cache-by-reference chain) computed at a different
    scale than its actual feats, so sum(seqlen) != feats.shape[0]. CUDA's
    segment_reduce kernel silently consumes that mismatch instead of raising,
    corrupting shape latents on every cascade run. Recompute lengths from
    coords when the invariant doesn't hold and evict the stale cache entry.
    Verified present (unpatched) in pixal3d/modules/sparse/basic.py as of the
    current TencentARC/Pixal3D master. Idempotent — safe to call on every install."""
    target = os.path.join(src_path, "pixal3d", "modules", "sparse", "basic.py")
    if not os.path.exists(target):
        return
    with open(target, "r", encoding="utf-8") as f:
        text = f.read()
    old = "        red = torch.segment_reduce(red, reduce=op, lengths=self.seqlen)\n        return red"
    if old not in text:
        return  # already patched, or upstream changed shape — don't guess
    new = """        lengths = self.seqlen
        n_data = red.shape[0]
        if int(lengths.sum().item()) != n_data:
            fresh = None
            coords = getattr(self, 'coords', None)
            if coords is not None and coords.shape[0] == n_data:
                batch_size = int(coords[:, 0].max().item()) + 1 if coords.shape[0] > 0 else 1
                fresh = torch.bincount(coords[:, 0].long(), minlength=batch_size).to(
                    dtype=torch.long, device=red.device
                )
            if fresh is None or int(fresh.sum().item()) != n_data:
                fresh = torch.tensor(
                    [l.stop - l.start for l in self.layout],
                    dtype=torch.long, device=red.device,
                )
            if hasattr(self, '_spatial_cache') and hasattr(self, '_scale'):
                try:
                    scale_key = str(self._scale)
                    slot = self._spatial_cache.get(scale_key, {})
                    for k in ('seqlen', 'cum_seqlen', 'batch_boardcast_map', 'layout'):
                        slot.pop(k, None)
                except Exception:
                    pass
            elif hasattr(self, '_cache'):
                try:
                    for k in ('seqlen', 'cum_seqlen', 'batch_boardcast_map'):
                        self._cache.pop(k, None)
                except Exception:
                    pass
            lengths = fresh
            if int(lengths.sum().item()) != n_data:
                raise RuntimeError(
                    f"VarLenTensor.reduce: cannot reconcile seqlen "
                    f"sum({int(lengths.sum().item())}) with data.size(0)={n_data}. "
                    f"layout has {len(self.layout)} segments."
                )
        return torch.segment_reduce(red, reduce=op, lengths=lengths)"""
    text = text.replace(old, new)
    with open(target, "w", encoding="utf-8") as f:
        f.write(text)
    debug_print(f"Patched cascade seqlen-cache fix into {target}")

EXT_WHEEL_PROJECT = {
    "o-voxel":    "o_voxel",
    "cumesh":     "cumesh",
    "flexgemm":   "flex_gemm",
    "nvdiffrast": "nvdiffrast",
    "flash_attn": "flash_attn",
    "natten":     "natten",
}

def canonical_wheel_name(label, version):
    """Build the exact bundled-wheel filename find_local_wheel() looks for, e.g.
    'natten-0.21.6+cu128torch2.9-cp313-cp313-win_amd64.whl'. Used to save a
    freshly-built wheel (natten, specifically — it has no upstream prebuilt
    Windows wheel) back into wheels/ under the name that will let future
    installs — on this machine or any other user's — find and reuse it instead
    of rebuilding from source."""
    project = EXT_WHEEL_PROJECT.get(label, label)
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_ver = TORCH_CUDA_SPEC[0].split("==", 1)[1]  # e.g. "2.9.1+cu128"
    torch_base, _, cuda_tag = torch_ver.partition("+")
    torch_tag = "torch" + ".".join(torch_base.split(".")[:2])
    return f"{project}-{version}+{cuda_tag}{torch_tag}-{py_tag}-{py_tag}-win_amd64.whl"

# Wheels too large to bundle in wheels/ (GitHub's 100MB push limit) are instead
# downloaded on demand from the same PozzettiAndrea/cuda-wheels release index at
# install time — building either from source on Windows is not a practical
# fallback (flash_attn: hours-long compile; natten: needs the CUDA Toolkit + MSVC
# and several hand-patched build bugs). flash_attn's wheel is ~240MB; natten's is
# ~131MB. Each label's release is tagged "<label>-latest" (see remote_wheel_url).
REMOTE_WHEEL_VERSION = {
    "flash_attn": "2.8.3",
    "natten":     "0.21.6",
}

def remote_wheel_url(label):
    """Build the exact GitHub release asset URL for a prebuilt wheel that isn't
    bundled locally (see REMOTE_WHEEL_VERSION). Returns None if this label has
    no remote wheel configured."""
    project = EXT_WHEEL_PROJECT.get(label)
    version = REMOTE_WHEEL_VERSION.get(label)
    if project is None or version is None:
        return None
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_ver = TORCH_CUDA_SPEC[0].split("==", 1)[1]  # e.g. "2.9.1+cu128"
    torch_base, _, cuda_tag = torch_ver.partition("+")  # "2.9.1", "cu128"
    torch_tag = "torch" + ".".join(torch_base.split(".")[:2])  # "torch2.9"
    filename = f"{project}-{version}+{cuda_tag}{torch_tag}-{py_tag}-{py_tag}-win_amd64.whl"
    return f"https://github.com/PozzettiAndrea/cuda-wheels/releases/download/{label}-latest/{filename.replace('+', '%2B')}"

def find_local_wheel(wheels_dir, label):
    """Look for a prebuilt wheel bundled with the add-on (in wheels/) that
    matches the running Python ABI and the pinned CUDA/torch build, so we can
    skip cloning + compiling o-voxel/cumesh/flexgemm/nvdiffrast from source
    entirely — that from-source build needs the CUDA Toolkit, a compatible
    MSVC, and several hand-patched source bugs (see patch_ovoxel_windows_build),
    none of which most add-on users will have. Returns a path or None."""
    import glob
    project = EXT_WHEEL_PROJECT.get(label)
    if project is None or not os.path.isdir(wheels_dir):
        return None
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_ver = TORCH_CUDA_SPEC[0].split("==", 1)[1]  # e.g. "2.9.1+cu128"
    torch_base, _, cuda_tag = torch_ver.partition("+")  # "2.9.1", "cu128"
    torch_tag = "torch" + ".".join(torch_base.split(".")[:2])  # "torch2.9"
    pattern = os.path.join(
        wheels_dir, f"{project}-*{cuda_tag}{torch_tag}*-{py_tag}-{py_tag}-win_amd64.whl"
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def find_msvc():
    """Detect the MSVC C++ compiler (cl.exe) on Windows. cl.exe is normally not
    on PATH outside a developer prompt, so we also query vswhere for a VS
    install that has the VC++ build tools (which is how setuptools finds it).
    Returns a short descriptor string or None. Non-Windows always returns 'n/a'
    (extensions use gcc there)."""
    if os.name != "nt":
        return "n/a"
    if shutil.which("cl"):
        return "cl.exe on PATH"
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if os.path.exists(vswhere):
        try:
            r = subprocess.run(
                [vswhere, "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True,
            )
            out = (r.stdout or "").strip()
            if out:
                return out
        except Exception as e:
            debug_print(f"vswhere query failed: {e}")
    return None

def find_vcvarsall():
    """Locate vcvarsall.bat for the newest installed MSVC toolchain, via vswhere.
    Needed specifically for natten: unlike the other CUDA extensions here (o-voxel/
    cumesh/flexgemm/nvdiffrast), which build through setuptools' torch.utils.
    cpp_extension.BuildExtension — which finds and configures MSVC itself — natten
    runs cmake/cl directly via its own subprocess.check_call(["cmake", ...]), so it
    needs cl.exe/link.exe's INCLUDE/LIB/PATH pre-populated by vcvarsall.bat, which a
    plain (non-"Developer Command Prompt") subprocess never has."""
    if os.name != "nt":
        return None
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.exists(vswhere):
        return None
    try:
        r = subprocess.run(
            [vswhere, "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, text=True,
        )
        install_path = (r.stdout or "").strip()
        if not install_path:
            return None
        candidate = os.path.join(install_path, "VC", "Auxiliary", "Build", "vcvarsall.bat")
        return candidate if os.path.exists(candidate) else None
    except Exception as e:
        debug_print(f"vswhere query for vcvarsall.bat failed: {e}")
        return None

def msvc_dev_env(base_env):
    """Return a copy of base_env with the MSVC x64 Developer Command Prompt
    environment merged in, by invoking vcvarsall.bat and capturing the resulting
    `set` output. Falls back to returning base_env unchanged if vcvarsall.bat
    can't be found — the caller's build attempt will then surface its own error
    rather than silently using an unconfigured environment."""
    env = dict(base_env)
    vcvarsall = find_vcvarsall()
    if not vcvarsall:
        return env
    try:
        r = subprocess.run(
            f'"{vcvarsall}" x64 && set', capture_output=True, text=True, shell=True,
        )
        for line in (r.stdout or "").splitlines():
            k, sep, v = line.partition("=")
            if sep and k:
                env[k] = v
    except Exception as e:
        debug_print(f"Could not source vcvarsall.bat environment: {e}")
    return env

def patch_torch_nvtoolsext_cmake(pkgs_dir):
    """CUDA Toolkit 12.0+ dropped nvToolsExt from the Windows SDK, but torch's own
    bundled cmake config still probes for it, which makes any CMake-based extension
    build — natten's, specifically; the other extensions here go through setuptools'
    own MSVC/CUDAExtension path and never touch these files — fail immediately when
    it configures against our installed torch (documented upstream:
    pytorch/pytorch#116926). Comment out the reference lines in torch's own cmake
    files rather than requiring an old Windows SDK just for this. Idempotent."""
    cmake_dir = os.path.join(pkgs_dir, "torch", "share", "cmake")
    if not os.path.isdir(cmake_dir):
        return
    patched = 0
    for root, _dirs, files in os.walk(cmake_dir):
        for fname in files:
            if not fname.endswith(".cmake"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except OSError:
                continue
            changed = False
            for i, line in enumerate(lines):
                if "nvtoolsext" in line.lower() and not line.lstrip().startswith("#"):
                    lines[i] = "#" + line
                    changed = True
            if changed:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                patched += 1
    if patched:
        debug_print(f"Patched nvToolsExt references out of {patched} torch cmake file(s) for natten's build.")

def patch_natten_windows_build(src_path):
    """natten's csrc/CMakeLists.txt has two unguarded GCC-only warning flags in
    CMAKE_CUDA_FLAGS, despite the file having an existing NATTEN_IS_WINDOWS
    block for other MSVC-specific flags a few lines down:

    1. `-Xcompiler=-Wconversion` — nvcc mechanically translates this into
       `/Wconversion` for the MSVC host compiler, which isn't a recognized
       cl.exe switch and hard-fails every translation unit with
       `error D8021: invalid numeric argument '/Wconversion'`.
    2. `-Xcompiler -Wall` — MSVC *does* accept a bare `-Wall`/`/Wall` (unlike
       `-Wconversion`), so this doesn't fail the build, but it's Microsoft's
       own "never use this" maximum warning level: dozens of purely
       informational warnings (C4820 struct-padding, C4514/C4711
       inlining notes, etc.) fire on every one of CUTLASS's deeply templated
       types, whose fully-qualified names run to thousands of characters
       each. Across ~110 translation units that's gigabytes of captured
       warning text — which pip buffers entirely in memory to be able to
       print on failure, and which was the actual cause of a Windows
       Resource-Exhaustion-Detector event on this machine showing pip's own
       python.exe process at ~80 GB virtual memory (confirmed via
       Get-WinEvent, System log, event ID 2004) — not the compiler itself
       needing that much RAM.

    3. `not`/`and`/`or` used as C++ alternative operator tokens (e.g.
       `if (not p.is_fully_block_sparse)` in dozens of kernel headers under
       cuda/fna*, cuda/fmha*, cuda/tokperm, cuda/reference, plus the
       CHECK_CONTIGUOUS macro in helpers.h) — GCC/Clang understand these
       natively, but nvcc's own EDG front-end (used to split host/device
       code before handing off to cl.exe) does not when targeting MSVC host
       compatibility, and fails with its own
       `error: identifier "not" is undefined` / `error: expected a ")"` on
       every translation unit that includes any of those headers. Rather
       than patch ~30 individual call sites, force-include the standard
       <ciso646> header (which #defines these tokens) into every
       translation unit via nvcc's own `-include` flag — the same header
       the C++ standard itself provides for exactly this situation.
    4. The same alternative-token problem, but for cl.exe directly:
       natten.cpp (the one plain-C++ TU, the pybind bindings) transitively
       includes helpers.h via fna.h etc., and natten's Windows CXX flags
       (`/Zc:lambda /Zc:preprocessor`, copied from xformers) do NOT include
       `/permissive-`, so cl in its default mode rejects `not` with
       `error C2065: 'not': undeclared identifier` too. CMAKE_CUDA_FLAGS
       never reaches cl-compiled TUs, so the nvcc fix above doesn't cover
       this one — force-include iso646.h via cl's own `/FI` flag as well.
       (Both fixes were verified in isolation against this machine's exact
       nvcc 12.8 + MSVC 14.50 pair on a minimal repro before being adopted:
       each compiler fails on `if (not x)` without its flag and compiles
       cleanly with it.)
    5. `error C2872: 'std': ambiguous symbol` in torch's own
       torch/csrc/dynamo/compiled_autograd.h (pulled in by <torch/extension.h>),
       raised by nvcc's host pass on every TU that includes it. This is NOT a
       natten bug: a trivial `#include <torch/extension.h>` CUDA TU — even one
       built through torch's own cpp_extension.load() — reproduces it, because
       CUDA 12.8's nvcc front-end mis-parses the `namespace std` in the MSVC
       14.4x/14.5x STL (both are newer than what CUDA 12.8 shipped against) and
       ends up seeing two `std` namespaces. `-allow-unsupported-compiler`
       silences nvcc's version *check* but not this parse failure. The other
       CUDA extensions here dodge it only because they ship as prebuilt wheels;
       natten is the one built from source, so it is the one that hits it.
       Adding `/permissive-` (MSVC standards-conformance mode) to the host pass
       resolves the ambiguity — verified to compile the previously-failing
       reference TU *and* a CUTLASS-heavy fmha TU cleanly, with no regression.
       (`/permissive-` also makes `not`/`and`/`or` real operators on the host,
       but the device-side EDG parse still needs the <ciso646> include from
       fix 3, so both are kept.)

    All five are build-time-only flags; none change what gets compiled.
    Idempotent — safe to call on every install."""
    target = os.path.join(src_path, "csrc", "CMakeLists.txt")
    if not os.path.exists(target):
        return
    with open(target, "r", encoding="utf-8") as f:
        text = f.read()
    changed = False
    conversion_flag = 'set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wconversion")\n'
    if conversion_flag in text:
        text = text.replace(conversion_flag, "")
        changed = True
        debug_print("Patched -Wconversion out of natten's MSVC build.")
    wall_flag = 'set(CMAKE_CUDA_FLAGS "-Xcompiler -Wall -ldl")'
    # Canonical target form of natten's Windows compiler-flag block, applying
    # fixes 2-5 together. A fresh upstream clone has just the bare `wall_flag`
    # line (no NATTEN_IS_WINDOWS guard); an already-partially-patched clone (from
    # an earlier version of this function) has some guarded variant of it. The
    # regex below matches any such guarded variant and normalizes it to this,
    # so repeated calls converge here and then stop changing (idempotent).
    target_block = (
        'if(${NATTEN_IS_WINDOWS})\n'
        '  set(CMAKE_CUDA_FLAGS "-ldl -include ciso646 -Xcompiler /permissive-")\n'
        '  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FIiso646.h")\n'
        'else()\n'
        '  set(CMAKE_CUDA_FLAGS "-Xcompiler -Wall -ldl")\n'
        'endif()'
    )
    guarded_re = re.compile(
        r'if\(\$\{NATTEN_IS_WINDOWS\}\)\s*\n'
        r'\s*set\(CMAKE_CUDA_FLAGS "-ldl[^"]*"\)\s*\n'
        r'(?:\s*set\(CMAKE_CXX_FLAGS[^\n]*\)\s*\n)?'
        r'else\(\)\s*\n'
        r'\s*set\(CMAKE_CUDA_FLAGS "-Xcompiler -Wall -ldl"\)\s*\n'
        r'endif\(\)'
    )
    m = guarded_re.search(text)
    if m and m.group(0) != target_block:
        text = text[:m.start()] + target_block + text[m.end():]
        changed = True
        debug_print("Normalized natten's Windows compiler flags to the canonical "
                    "fixed form (-Wall guarded out; <ciso646>/<iso646.h> for nvcc+cl "
                    "alternative tokens; /permissive- for torch's C2872 'std' ambiguity).")
    elif not m and wall_flag in text:
        text = text.replace(wall_flag, target_block)
        changed = True
        debug_print("Applied natten's Windows compiler-flag fixes (-Wall guarded out; "
                    "<ciso646>/<iso646.h> for nvcc+cl alternative tokens; /permissive- "
                    "for torch's C2872 'std' ambiguity under nvcc 12.8 + MSVC 14.4x/14.5x).")
    if changed:
        with open(target, "w", encoding="utf-8") as f:
            f.write(text)

def check_build_toolchain():
    """Return (ok, messages) describing whether the native build toolchain
    needed to compile the CUDA extensions (cumesh/flexgemm/o-voxel) is present:
    the CUDA Toolkit (nvcc) and, on Windows, MSVC. ok is False if either is
    missing — the caller warns but does NOT abort, since the 2D stack still
    installs fine without them."""
    msgs = []
    nvcc = find_nvcc()
    if nvcc:
        msgs.append(f"CUDA nvcc: found ({nvcc})")
    else:
        msgs.append("CUDA nvcc: MISSING — install the CUDA Toolkit (12.x) so the "
                    "3D extensions can compile.")
    msvc = find_msvc()
    if msvc:
        msgs.append(f"MSVC C++: found ({msvc})")
    else:
        msgs.append("MSVC C++: MISSING — install Visual Studio Build Tools with the "
                    "'Desktop development with C++' workload.")
    return (nvcc is not None and msvc is not None), msgs

def requirements_path() -> str:
    """Path to the frozen dependency list shipped alongside the add-on."""
    return os.path.join(addon_script_path(), "requirements.txt")

def read_requirements():
    """Return the requirement specifiers from requirements.txt (skipping blank
    lines and comments). Inline environment markers (`; sys_platform == ...`)
    are kept intact so pip can evaluate them."""
    path = requirements_path()
    reqs = []
    if not os.path.exists(path):
        return reqs
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                reqs.append(line)
    return reqs

def _req_shortname(req):
    """Best-effort human label for a requirement line (for the progress bar)."""
    tail = req.split("/")[-1]
    return re.split(r"[=<>;@\s]", tail)[0][:24] or req[:24]

def _install_natten_from_source(py, pkgs_dir, ext_dir, wheels_dir, done, total):
    """Clone, patch, build and install natten==0.21.6 from source, saving the
    resulting wheel into wheels_dir for reuse. The from-source fallback for when
    neither a bundled nor a downloadable prebuilt natten wheel is available (an
    unusual Python/CUDA/torch combo, or the release download failed). Requires the
    CUDA Toolkit (nvcc) + MSVC. Raises on failure; the caller logs and continues."""
    set_progress("install", value=done / total, label="Installing natten")
    # natten's setup.py detects our working CUDA torch and attempts the native
    # libnatten build (it only skips straight to the pure-Python fallback when
    # NO CUDA is available at all) — so on this machine it WILL try to compile,
    # and needs both fixes below to have a chance of succeeding on Windows.
    patch_torch_nvtoolsext_cmake(pkgs_dir)
    natten_env = msvc_dev_env(os.environ)
    cuda_home = find_cuda_home()
    if cuda_home:
        natten_env["CUDA_HOME"] = cuda_home
        natten_env.setdefault("CUDA_PATH", cuda_home)
    natten_env["PYTHONPATH"] = pkgs_dir + os.pathsep + natten_env.get("PYTHONPATH", "")
    # natten builds libnatten.pyd with cmake directly (not setuptools), and
    # because its CMakeLists is handed an explicit PYTHON_PATH it skips
    # find_package(Python ... Development) and never learns where pythonXX.lib
    # lives — so the final link fails with LNK1104: cannot open 'python313.lib'.
    # (setuptools-based extensions don't hit this; they add the libs dir for
    # you.) Put the interpreter's own libs dir on the linker's LIB path so
    # link.exe can resolve the Python import library. base_prefix (not prefix)
    # points at Blender's bundled Python root even when called from inside it.
    py_libs = os.path.join(sys.base_prefix, "libs")
    if os.path.isdir(py_libs):
        natten_env["LIB"] = py_libs + os.pathsep + natten_env.get("LIB", "")
    natten_env.setdefault("NATTEN_CUDA_ARCH", "7.5;8.0;8.6;8.9;9.0")
    # Each translation unit instantiates many CUTLASS FNA kernel templates
    # (backward kernels in particular) and can use several GB of RAM in
    # cicc/ptxas; running several in parallel exhausted RAM on this machine
    # and the OS silently killed the compiler mid-build (ninja reported
    # "subcommand failed" with no compiler error at all — the signature of
    # an OOM kill, not a real compile error). Building serially avoids it.
    natten_env.setdefault("NATTEN_N_WORKERS", "1")
    # Without this, natten's cmake build only reports pip's own opaque
    # "still running..." heartbeat with no indication of progress or
    # which file is being compiled. Verbose mode surfaces cmake/ninja's
    # real per-target output (e.g. "[42/110] Building CUDA object ..."),
    # which run_logged() now streams to both the console and the log.
    natten_env.setdefault("NATTEN_VERBOSE", "1")
    # Same fix as the ext_specs builds below: nvcc rejects MSVC releases newer
    # than its own support table (fatal error C1189) unless explicitly told to
    # tolerate them. natten's cmake invokes nvcc directly, so it hits this too.
    natten_env["NVCC_APPEND_FLAGS"] = (
        natten_env.get("NVCC_APPEND_FLAGS", "") + " -allow-unsupported-compiler"
    ).strip()
    # Installing straight from the natten==0.21.6 PyPI sdist (as before)
    # extracts it into pip's own temp dir, giving us no chance to patch
    # its CMakeLists.txt before the build runs. Clone the tagged release
    # into TRELLIS_EXT instead — same pattern as cumesh/flexgemm/nvdiffrast
    # below — so patch_natten_windows_build() can fix its Windows/MSVC
    # CUDA-flags bug (-Wconversion) first.
    natten_src = os.path.join(ext_dir, "natten")
    os.makedirs(ext_dir, exist_ok=True)
    if not os.path.exists(natten_src):
        run_logged(
            ["git", "clone", "--recursive", "-b", "v0.21.6",
             "https://github.com/SHI-Labs/NATTEN.git", natten_src],
            check=True,
        )
    else:
        run_logged(["git", "submodule", "update", "--init", "--recursive"],
                   cwd=natten_src, check=False)
    patch_natten_windows_build(natten_src)
    # Build a standalone .whl (rather than `pip install` directly) so
    # the successful build can be saved into wheels/ for next time —
    # on this machine (survives a "Nuclear Wipe" of addon_packages)
    # and for any other user, once this file is committed alongside
    # the addon like the other bundled wheels already are.
    os.makedirs(wheels_dir, exist_ok=True)
    built_dir = os.path.join(ext_dir, "_natten_wheel_out")
    if os.path.isdir(built_dir):
        shutil.rmtree(built_dir, onerror=remove_readonly)
    os.makedirs(built_dir, exist_ok=True)
    run_logged(
        [py, "-m", "pip", "wheel", "--disable-pip-version-check", natten_src,
         "--no-deps", "--no-build-isolation", "-w", built_dir],
        check=True, env=natten_env,
    )
    import glob
    built_wheels = glob.glob(os.path.join(built_dir, "*.whl"))
    if not built_wheels:
        raise RuntimeError("pip wheel reported success but produced no .whl file")
    target_name = canonical_wheel_name("natten", "0.21.6")
    target_path = os.path.join(wheels_dir, target_name)
    shutil.copy2(built_wheels[0], target_path)
    debug_print(f"Saved natten wheel for reuse: {target_name}")
    run_logged(
        [py, "-m", "pip", "install", "--disable-pip-version-check", target_path,
         "--no-deps", "--upgrade", "--target", pkgs_dir],
        check=True,
    )

def install_all_dependencies():
    """Install the full 2D + 3D stack in one pass: CUDA torch, the 2D/3D runtime
    packages, the TRELLIS.2 repo, and its compiled CUDA extensions."""
    start_install_log()
    set_progress("install", value=0.0, label="Preparing...", active=True)
    try:
        activate_virtualenv()
        py = python_exec()
        pkgs_dir = packages_path()
        addon_dir = addon_script_path()
        repo_path = os.path.normpath(os.path.join(addon_dir, "TRELLIS_REPO"))
        pixal3d_repo_path = os.path.normpath(os.path.join(addon_dir, "PIXAL3D_REPO"))
        wheels_dir = os.path.join(addon_dir, "wheels")
        debug_print("Installing full 2D + 3D dependency stack...")

        # Warn early (before the long download) if the native build toolchain
        # for the 3D CUDA extensions is missing. We only WARN — the 2D stack
        # installs fine without it — but this lets the user fix it now instead
        # of discovering it ~10 GB later when the extension builds fail.
        toolchain_ok, tc_msgs = check_build_toolchain()
        for m in tc_msgs:
            debug_print(m)
        if not toolchain_ok:
            debug_print("Build toolchain incomplete — 3D extensions will likely fail to "
                        "compile. The 2D generator will still work.")
            set_progress("install", value=0.0,
                         label="Warning: build tools missing (3D may fail) — see console")

        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.9;9.0"
        os.environ["FORCE_CUDA"] = "1"

        # Every runtime package is pinned in requirements.txt. We install with
        # --no-deps so NO package is allowed to resolve/pull its own
        # dependencies — the full closure is already listed, which stops a
        # dependency from dragging in a second (CPU-only) torch/numpy that would
        # shadow the CUDA build.
        reqs = read_requirements()
        if not reqs:
            raise RuntimeError(f"requirements.txt not found next to the add-on ({requirements_path()}).")

        # CUDA extensions. Per the upstream setup.sh, only o-voxel ships inside
        # the TRELLIS.2 repo (at the repo ROOT, not under extensions/); cumesh,
        # flexgemm and nvdiffrast are SEPARATE repos that must be cloned. Each is
        # a torch C++/CUDA extension, so it must compile against the CUDA torch
        # installed above — hence --no-build-isolation at install time (build
        # isolation would fetch a fresh, possibly CPU, torch to compile against).
        # Each entry: (label, source) where source is a local path or a
        # (git_url, branch_or_None) tuple to clone.
        ext_dir = os.path.normpath(os.path.join(addon_dir, "TRELLIS_EXT"))
        ext_specs = [
            ("o-voxel",    os.path.join(repo_path, "o-voxel")),
            ("cumesh",     ("https://github.com/JeffreyXiang/CuMesh.git", None)),
            ("flexgemm",   ("https://github.com/JeffreyXiang/FlexGEMM.git", None)),
            ("nvdiffrast", ("https://github.com/NVlabs/nvdiffrast.git", "v0.4.0")),
            ("flash_attn", None),  # wheel-only (local or remote download) — see REMOTE_WHEEL_VERSION
        ]
        # torch + each requirement + repo clone + pixal3d clone + cmake + natten + ext builds + verify
        total = 1 + len(reqs) + 1 + 1 + 1 + 1 + len(ext_specs) + 1
        done = 0

        # 1. Install CUDA torch FIRST (its own +cu128 index, --no-deps). It is
        #    NOT in requirements.txt and must be present before the extensions
        #    compile against it (their setup.py uses torch.utils.cpp_extension).
        #    Skip the (~2.5 GB, force-reinstall) download when the exact pinned
        #    build is already installed AND sees CUDA — i.e. a previous install
        #    already succeeded. Only (re)install when it is missing, the wrong
        #    version, or broken.
        set_progress("install", value=done / total, label="Checking CUDA torch")
        if cuda_torch_satisfied(py, pkgs_dir):
            debug_print("CUDA torch already installed and verified — skipping reinstall.")
            set_progress("install", value=done / total, label="CUDA torch already installed")
        else:
            set_progress("install", value=done / total, label="Installing CUDA torch")
            if install_cuda_torch(py, pkgs_dir).returncode != 0:
                raise RuntimeError("CUDA torch install failed — see the console (check NVIDIA driver / Python version).")
        done += 1

        # 2. Everything else, one requirement at a time, all with --no-deps.
        #    Installing per line (rather than a single `-r`) keeps the progress
        #    bar informative and lets one bad line be skipped instead of
        #    aborting the whole batch. Skip lines that are already satisfied at
        #    the exact pinned version — re-running install shouldn't re-download
        #    ~100 packages just to confirm nothing changed. Unpinned lines (the
        #    two `git+...` entries) have no version to compare against, so those
        #    are always re-run, same as before.
        for req in reqs:
            name, ver = _spec_name_version(req)
            if ver and dist_info_version(pkgs_dir, name) == ver:
                debug_print(f"{name}=={ver} already installed — skipping")
                done += 1
                continue
            set_progress("install", value=done / total, label=f"Installing {_req_shortname(req)}")
            run_logged([py, "-m", "pip", "install", "--disable-pip-version-check", "--target", pkgs_dir, "--no-deps", "--upgrade", req])
            done += 1

        # 3. Clone / update the TRELLIS.2 repo. --recurse-submodules is required
        #    so o-voxel's eigen submodule (o-voxel/third_party/eigen) is fetched;
        #    without it the o-voxel build fails.
        set_progress("install", value=done / total, label="Cloning TRELLIS.2 repo")
        if os.path.exists(repo_path):
            # A valid clone has the trellis2 package and the o-voxel source.
            sub_check = os.path.join(repo_path, "o-voxel", "setup.py")
            if not (os.path.exists(os.path.join(repo_path, "trellis2")) and os.path.exists(sub_check)):
                debug_print("TRELLIS.2 clone incomplete. Clearing repo for fresh clone...")
                shutil.rmtree(repo_path, onerror=remove_readonly)
        if not os.path.exists(repo_path):
            # Using tin2tin's fork instead of microsoft/TRELLIS.2 directly: it carries
            # a fix for a stale seqlen cache bug in the cascade Shape-SLat CFG-rescale
            # path (upstream microsoft/TRELLIS.2 PR #167, not merged yet) that silently
            # corrupts values on CUDA during cascade pipeline runs.
            run_logged(["git", "clone", "--recurse-submodules", "-b", "fix-cascade-seqlen-cache",
                        "https://github.com/tin2tin/TRELLIS.2.git", repo_path], check=True)
        else:
            run_logged(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_path, check=True)
        done += 1

        # 3b. Clone the Pixal3D repo (github.com/TencentARC/Pixal3D). Its `pixal3d`
        #    package is self-contained (bundles both Trellis2ImageTo3DPipeline and
        #    Pixal3DImageTo3DPipeline) and doesn't vendor o-voxel source itself —
        #    TRELLIS_REPO above still supplies that. No submodules to recurse.
        set_progress("install", value=done / total, label="Cloning Pixal3D repo")
        if os.path.exists(pixal3d_repo_path) and not os.path.exists(os.path.join(pixal3d_repo_path, "pixal3d")):
            debug_print("Pixal3D clone incomplete. Clearing repo for fresh clone...")
            shutil.rmtree(pixal3d_repo_path, onerror=remove_readonly)
        if not os.path.exists(pixal3d_repo_path):
            run_logged(["git", "clone", "https://github.com/TencentARC/Pixal3D.git", pixal3d_repo_path], check=True)
        # Always re-check/apply: cheap no-op once already patched (see docstring).
        patch_pixal3d_seqlen_cache(pixal3d_repo_path)
        done += 1

        # 3c. natten (windowed/neighborhood attention). NOT just an attention-backend
        #    knob for Pixal3D itself (that part does fall back to SDPA fine) — the
        #    upstream valeoai/NAF upsampler that Pixal3D's shape_512/shape_1024/
        #    tex_1024 DinoV3ProjFeatureExtractor configs pull in via torch.hub
        #    hard-imports natten with no fallback (`from natten.functional import
        #    na2d_av, na2d_qk`), and NAF's output width (proj_channels = embed_dim*2
        #    when NAF is on) is baked into the pretrained checkpoint's expected
        #    input shape — so this genuinely has to import successfully for the
        #    Pixal3D backend's shape/texture stages to run at all.
        #    cmake is required just for natten's setup.py to run, and isn't
        #    otherwise part of this addon's toolchain — install it first.
        #    natten==0.21.6 (vs. the older 0.21.0 in Pixal3D's own README) is used
        #    deliberately: current natten gracefully falls back to a pure-PyTorch
        #    "Flex Attention" backend when it can't build its CUDA kernels (no
        #    cmake/MSVC dev-prompt environment needed for that path), which is a
        #    much better bet on Windows than trying to replicate their documented
        #    "Native Tools Command Prompt for VS" + WindowsBuilder.bat build flow.
        if dist_info_version(pkgs_dir, "cmake") is not None:
            debug_print("cmake already installed — skipping")
            set_progress("install", value=done / total, label="cmake already installed")
        else:
            set_progress("install", value=done / total, label="Installing cmake (for natten)")
            try:
                run_logged(
                    [py, "-m", "pip", "install", "--disable-pip-version-check", "cmake",
                     "--upgrade", "--target", pkgs_dir],
                    check=True,
                )
            except Exception as e:
                debug_print(f"cmake install failed (natten's setup.py needs it to run): {e}")
        done += 1

        # natten==0.21.6 exactly — if a prior run already built and installed this
        # exact version, its dist-info is proof the CUDA-kernel compile already
        # succeeded, so skip re-running that multi-minute build every time.
        natten_wheel = find_local_wheel(wheels_dir, "natten")
        if dist_info_version(pkgs_dir, "natten") == "0.21.6":
            debug_print("natten==0.21.6 already installed — skipping")
            set_progress("install", value=done / total, label="natten already installed")
        elif natten_wheel:
            # A bundled wheel exists (either shipped with the addon, or saved by
            # a previous build below) — install directly, no compile needed.
            set_progress("install", value=done / total, label="Installing natten (bundled wheel)")
            debug_print(f"Using bundled prebuilt wheel for natten: {os.path.basename(natten_wheel)}")
            try:
                run_logged(
                    [py, "-m", "pip", "install", "--disable-pip-version-check", natten_wheel,
                     "--no-deps", "--upgrade", "--target", pkgs_dir],
                    check=True,
                )
            except Exception as e:
                debug_print(f"natten wheel install failed: {e}")
        elif remote_wheel_url("natten"):
            # No bundled/built wheel present — natten's ~131MB wheel is too large to
            # ship in wheels/, so download the matching prebuilt one from the
            # PozzettiAndrea/cuda-wheels release index (same mechanism as flash_attn).
            # This avoids the CUDA Toolkit + MSVC from-source build in the common case.
            natten_url = remote_wheel_url("natten")
            set_progress("install", value=done / total, label="Downloading natten wheel")
            debug_print(f"No bundled wheel for natten — downloading from {natten_url}")
            try:
                run_logged(
                    [py, "-m", "pip", "install", "--disable-pip-version-check", natten_url,
                     "--no-deps", "--upgrade", "--target", pkgs_dir],
                    check=True,
                )
            except Exception as e:
                debug_print(f"natten remote wheel install failed ({e}); "
                            "falling back to from-source build.")
                _install_natten_from_source(py, pkgs_dir, ext_dir, wheels_dir, done, total)
        else:
            try:
                _install_natten_from_source(py, pkgs_dir, ext_dir, wheels_dir, done, total)
            except Exception as e:
                debug_print(f"natten failed to install — Pixal3D's shape/texture stages need it "
                            f"and have no fallback (TRELLIS.2 backend is unaffected): {e}")
        done += 1

        # 4. Clone (if remote) and build each CUDA extension into addon_packages/.
        #    --no-build-isolation: compile against the CUDA torch installed above
        #    instead of a fresh isolated (possibly CPU) torch. --no-deps: their
        #    runtime deps are already in requirements.txt. A regular (non-editable)
        #    install copies the compiled .pyd into the target dir so it is
        #    importable via the sys.path.insert in the isolated script.
        #    Failures are collected (not raised) so one bad build doesn't abort
        #    the others, and are surfaced in the final status line.
        os.makedirs(ext_dir, exist_ok=True)
        ext_failures = []
        build_env = os.environ.copy()
        cuda_home = find_cuda_home()
        if cuda_home:
            build_env["CUDA_HOME"] = cuda_home
            build_env.setdefault("CUDA_PATH", cuda_home)
        else:
            debug_print("CUDA Toolkit (nvcc) not found — extension builds needing "
                        "CUDAExtension will fail until it is installed.")
        # setup.py does `import torch` to build a CUDAExtension. Without this,
        # that import falls through to Blender's own bundled (CPU-only) torch
        # in its system site-packages rather than the CUDA build we installed
        # under pkgs_dir — which makes torch.cuda._is_compiled() False and
        # forces cpp_extension.CUDA_HOME to None, masking the real CUDA_HOME
        # env var set above. Prepending pkgs_dir to PYTHONPATH makes the CUDA
        # torch resolve first.
        build_env["PYTHONPATH"] = pkgs_dir + os.pathsep + build_env.get("PYTHONPATH", "")
        # nvcc refuses to compile with a host MSVC version it doesn't recognize
        # yet (fatal error C1189), which happens with very new Visual Studio
        # releases ahead of what a given CUDA Toolkit ships support tables for.
        # NVCC_APPEND_FLAGS is nvcc's own env-var hook for injecting extra
        # flags into every invocation without touching setup.py.
        build_env["NVCC_APPEND_FLAGS"] = (
            build_env.get("NVCC_APPEND_FLAGS", "") + " -allow-unsupported-compiler"
        ).strip()
        for label, source in ext_specs:
            dist_name = EXT_WHEEL_PROJECT.get(label, label)
            try:
                wheel_path = find_local_wheel(wheels_dir, label)
                if wheel_path:
                    target_ver = _wheel_version(wheel_path)
                    if target_ver and dist_info_version(pkgs_dir, dist_name) == target_ver:
                        debug_print(f"{label} {target_ver} already installed — skipping")
                        done += 1
                        continue
                    set_progress("install", value=done / total, label=f"Building {label}")
                    debug_print(f"Using bundled prebuilt wheel for {label}: {os.path.basename(wheel_path)}")
                    run_logged(
                        [py, "-m", "pip", "install", "--disable-pip-version-check", wheel_path,
                         "--no-deps", "--upgrade", "--target", pkgs_dir],
                        check=True,
                    )
                    done += 1
                    continue
                if source is None:
                    url = remote_wheel_url(label)
                    if url is None:
                        raise RuntimeError(f"no local or remote wheel configured for '{label}'")
                    target_ver = _wheel_version(url)
                    if target_ver and dist_info_version(pkgs_dir, dist_name) == target_ver:
                        debug_print(f"{label} {target_ver} already installed — skipping")
                        done += 1
                        continue
                    set_progress("install", value=done / total, label=f"Building {label}")
                    debug_print(f"No bundled wheel for {label} — downloading from {url}")
                    run_logged(
                        [py, "-m", "pip", "install", "--disable-pip-version-check", url,
                         "--no-deps", "--upgrade", "--target", pkgs_dir],
                        check=True,
                    )
                    done += 1
                    continue
                set_progress("install", value=done / total, label=f"Building {label}")
                if isinstance(source, tuple):
                    url, branch = source
                    src_path = os.path.join(ext_dir, label)
                    if not os.path.exists(src_path):
                        cmd = ["git", "clone", "--recursive", url, src_path]
                        if branch:
                            cmd[2:2] = ["-b", branch]
                        run_logged(cmd, check=True)
                    else:
                        run_logged(["git", "submodule", "update", "--init", "--recursive"],
                                   cwd=src_path, check=False)
                else:
                    src_path = source
                if not os.path.exists(src_path):
                    raise RuntimeError(f"source not found at {src_path}")
                if label == "o-voxel":
                    patch_ovoxel_windows_build(src_path)
                debug_print(f"Building extension: {label}")
                run_logged(
                    [py, "-m", "pip", "install", "--disable-pip-version-check", src_path,
                     "--no-deps", "--no-build-isolation", "--upgrade", "--target", pkgs_dir],
                    check=True, env=build_env,
                )
            except Exception as e:
                debug_print(f"Extension '{label}' failed to build: {e}")
                ext_failures.append(label)
            done += 1

        # 6. Verify CUDA is actually visible. torch was already installed in
        #    step 2 and nothing above should replace it (pip leaves the satisfied
        #    torch untouched), so we only reinstall if verification fails —
        #    avoiding a second ~2.5 GB torch install in the normal case.
        set_progress("install", value=done / total, label="Verifying CUDA torch")
        if not verify_cuda_torch(py, pkgs_dir):
            debug_print("CUDA missing after install — re-asserting CUDA torch once.")
            set_progress("install", value=done / total, label="Repairing CUDA torch")
            install_cuda_torch(py, pkgs_dir)
        cuda_ok = verify_cuda_torch(py, pkgs_dir)
        if not cuda_ok:
            status = "Done — WARNING: CUDA not available (see console)"
        elif ext_failures:
            status = f"Done — CUDA OK, but these extensions failed: {', '.join(ext_failures)} (see console)"
        else:
            status = "Done — CUDA OK"
        set_progress("install", value=1.0, label=status)
        debug_print("Full dependency install completed.")
        if ext_failures:
            debug_print(f"Extensions that FAILED to build: {', '.join(ext_failures)}. "
                        "3D conversion needs cumesh, flexgemm and o-voxel. "
                        "Building them requires the CUDA Toolkit (nvcc) and MSVC Build Tools.")
    except Exception as e:
        debug_print(f"Dependency install error: {e}")
        set_progress("install", label=f"Error: {e}")
    finally:
        set_progress("install", active=False)

# --- UTILS: NAMING & TEXT ---

# A persistent reference to the items is required: Blender does not keep a
# reference to the strings returned by a dynamic EnumProperty callback, so
# building a fresh list each call can lead to garbage-collection crashes /
# corrupted UI. Cache it in a module-level global.
_texts_enum_cache = []

def texts_callback(self, context):
    _texts_enum_cache.clear()
    _texts_enum_cache.extend((t.name, t.name, "") for t in bpy.data.texts)
    return _texts_enum_cache

class Import_Text_Props(PropertyGroup):
    def update_text_list(self, context):
        if self.scene_texts in bpy.data.texts: self.script = bpy.data.texts[self.scene_texts].name
    input_type: EnumProperty(name="Input Type", items=[("PROMPT", "Prompt", ""), ("TEXT_BLOCK", "Text-Block", ""), ("FILE", "File", "")], default="TEXT_BLOCK")
    script: StringProperty(default="")
    scene_texts: EnumProperty(name="Text-Blocks", items=texts_callback, update=update_text_list)
    # FILE input: an existing image supplied by the user instead of a prompt.
    # It skips the Z-Image 2D generation step but is otherwise processed
    # identically (BiRefNet background removal, per-object crop, plane import).
    # No subtype='FILE_PATH' — that would render a second, built-in browse
    # button next to our explicit FILE_FOLDER one. bpy.path.abspath() in
    # execute() still resolves any // or ~ prefix at run time.
    input_image: StringProperty(name="Image", default="")

def get_unique_name(self, context):
    base = context.scene.asset_name or "Asset"
    existing = {o.name for o in bpy.data.objects}
    if base in existing:
        m = re.search(r"\((\d+)\)$", base)
        c = int(m.group(1)) + 1 if m else 1
        regex_pat = r' \(\d+\)$'
        clean = re.sub(regex_pat, '', base)
        new_name = f"{clean} ({c})"
        while new_name in existing:
            c += 1
            new_name = f"{clean} ({c})"
        context.scene.asset_name = new_name

_SMART_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "―": "-",
    "…": "...", " ": " ",
}

def sanitize_text(s: str) -> str:
    # Subprocess stdout is decoded as UTF-8 by the reader threads, but the
    # Windows console can write it in the system codepage (e.g. cp1252).
    # Any byte from a non-ASCII character survives that mismatch and breaks
    # the decode, so normalize common smart punctuation and drop the rest.
    if not s:
        return s
    for src, dst in _SMART_CHAR_MAP.items():
        s = s.replace(src, dst)
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii")

def get_unique_file_name(base_path):
    if not os.path.exists(base_path): return base_path
    base, ext = os.path.splitext(base_path)
    c = 1
    p = f"{base}_{c}{ext}"
    while os.path.exists(p):
        c += 1
        p = f"{base}_{c}{ext}"
    return p

# --- SUBPROCESS LOG → PROGRESS PARSERS ---
# The inference subprocesses print structured "[2d]"/"[3d]" lines on stdout.
# A reader thread drains the pipe (so it never blocks), echoes each line to the
# console, and maps known markers to a step-based progress fraction.

def _read_2d_progress(proc, total):
    # Generation runs first (Phase 1), then segmentation (Phase 2); each phase
    # gets roughly half the bar so it never jumps backwards when the segmenter
    # loads after all images are generated. On a first-ever run each phase's
    # checkpoint downloads from HuggingFace before that phase's work starts
    # (Z-Image-Turbo in Phase 1, BiRefNet in Phase 2); each download is given a
    # visible sub-slice of its phase, driven off HF's tqdm output. On a cached
    # run no download is seen and each phase uses its full range exactly as
    # before, filling smoothly from the phase's start.
    stage_re = re.compile(r"^(.*?):\s*(\d+)%\|")
    # HF's outer "Fetching N files:" bar — the authoritative overall download %.
    fetch_re = re.compile(r"Fetching\s+\d+\s+files:\s*(\d+)%")
    # tqdm's trailing transfer rate, e.g. ", 60.0MB/s]" (byte units only, so it
    # never matches a diffusion denoising bar's "it/s" rate).
    rate_re = re.compile(r",\s*([\d.]+\s*[kKMGT]?i?B/s)")
    # Per-phase (download-band start, download-band end / work-band start,
    # work-band end). Generation work spans P1_MID..P1_END when a download was
    # seen, else 0.05..P1_END; likewise for segmentation.
    P1_DL0, P1_MID, P1_END = 0.02, 0.22, 0.50
    P2_DL0, P2_MID, P2_END = 0.50, 0.56, 0.95
    last_frac = 0.0
    gen = seg = 0
    phase = 1                        # 1 = generation side, 2 = segmentation side
    work_started = False             # this phase's real work has begun (past dl)
    fetch_seen = False               # outer "Fetching N files" bar is driving %
    dl_seen = {1: False, 2: False}   # a download happened in each phase
    dl_desc = ""

    def _emit(frac, label):
        nonlocal last_frac
        last_frac = max(last_frac, frac)
        set_progress("infer_2d", value=last_frac, label=label)

    def _dl_label():
        return "Downloading model weights" + (f": {dl_desc}" if dl_desc else "")

    try:
        for raw in proc.stdout:
            line = raw.rstrip()
            if line:
                print(line)
            if "Loading pipeline" in line:
                phase, work_started, fetch_seen, dl_desc = 1, False, False, ""
                _emit(0.02, "Loading image model")
                continue
            if "Loading BiRefNet" in line:
                phase, work_started, fetch_seen, dl_desc = 2, False, False, ""
                _emit(0.50, "Loading segmenter")
                continue
            if "Generating:" in line:
                work_started = True
                gen += 1
                base = P1_MID if dl_seen[1] else 0.05
                _emit(base + (P1_END - base) * (gen / max(total, 1)), f"Generating {gen}/{total}")
                continue
            if "Segmenting:" in line:
                work_started = True
                seg += 1
                base = P2_MID if dl_seen[2] else 0.50
                _emit(base + (P2_END - base) * (seg / max(total, 1)), f"Segmenting {seg}/{total}")
                continue
            if "Done." in line:
                _emit(0.99, "Finalizing")
                continue
            # --- HuggingFace checkpoint download bars (only before the phase's
            # own work begins; a denoising/sampling tqdm bar during generation is
            # never a download). ---
            if work_started:
                continue
            dl0, dl1 = (P1_DL0, P1_MID) if phase == 1 else (P2_DL0, P2_MID)
            fm = fetch_re.search(line)
            if fm:
                dl_seen[phase] = fetch_seen = True
                lbl = _dl_label()
                rm = rate_re.search(line)
                if rm:
                    lbl += f" ({rm.group(1).replace(' ', '')})"
                _emit(dl0 + (dl1 - dl0) * (int(fm.group(1)) / 100.0), lbl)
                continue
            sm = stage_re.match(line)
            if sm:
                dl_seen[phase] = True
                dl_desc, pct = sm.group(1).strip(), int(sm.group(2))
                lbl = _dl_label() + f" ({pct}%)"
                rm = rate_re.search(line)
                if rm:
                    lbl += f" - {rm.group(1).replace(' ', '')}"
                # Until the outer "Fetching N files" bar takes over, drive the
                # download band off this file's own %, capped just under the band
                # end so one finished file can't claim the whole phase before the
                # remaining files are known. _emit's max() keeps the bar from
                # moving backward as each successive file resets to 0%.
                if not fetch_seen:
                    _emit(dl0 + (dl1 - dl0) * 0.95 * (pct / 100.0), lbl)
                else:
                    set_progress("infer_2d", value=last_frac, label=lbl)
    except Exception as e:
        debug_print(f"2D progress reader stopped: {e}")

# (stage name substring, base fraction, span) within one task's build, in the
# order o_voxel's tqdm bars print them.
_3D_STAGE_WEIGHTS = [
    ("Sampling sparse structure", 0.00, 0.10),
    ("Sampling shape SLat",       0.10, 0.25),
    ("Sampling texture SLat",     0.35, 0.15),
    ("Building BVH",              0.50, 0.05),
    ("Cleaning mesh",             0.55, 0.05),
    ("Parameterizing new mesh",   0.60, 0.15),
    ("Sampling attributes",       0.75, 0.05),
    ("Finalizing mesh",           0.80, 0.05),
]
_3D_STAGE_LABELS = {
    "Sampling sparse structure": "Sampling sparse structure",
    "Sampling shape SLat": "Sampling shape",
    "Sampling texture SLat": "Sampling texture",
    "Building BVH": "Building mesh",
    "Cleaning mesh": "Cleaning mesh",
    "Parameterizing new mesh": "Unwrapping UVs",
    "Sampling attributes": "Baking texture",
    "Finalizing mesh": "Finalizing mesh",
}

def _read_3d_progress(proc):
    # Progress model: on a first-ever run the multi-GB checkpoint download
    # (several files from microsoft/TRELLIS.2-4B or TencentARC/Pixal3D) dominates
    # wall-clock time, so once a download is detected it is given a real, visible
    # slice of the bar (0.0-DL_SLICE) driven off huggingface_hub's tqdm output,
    # and the generation stages are remapped into DL_SLICE-0.9. On a cached run
    # no download is seen, so generation keeps the full 0.0-0.9 range and the bar
    # fills smoothly from zero. The subprocess phase is capped at 0.9 so the
    # Blender-side per-asset import/cleanup pass (see
    # TRELLIS_OT_ConvertSelected.modal) visibly owns the remaining 0.9-1.0.
    DL_SLICE = 0.30
    stage_re = re.compile(r"^(.*?):\s*(\d+)%\|")
    proc_re = re.compile(r"Processing\s+(\d+)\s*/\s*(\d+)")
    # huggingface_hub's outer "Fetching N files:" tqdm bar — the authoritative
    # overall download percentage across every file in the repo snapshot.
    fetch_re = re.compile(r"Fetching\s+\d+\s+files:\s*(\d+)%")
    # tqdm's trailing transfer rate, e.g. ", 60.0MB/s]". Byte units only, so it
    # never matches a generation sampling bar's "it/s" rate.
    rate_re = re.compile(r",\s*([\d.]+\s*[kKMGT]?i?B/s)")
    last_frac = 0.0
    cur_task, total_tasks = 1, 1
    download_seen = False   # a checkpoint download has started this run
    fetch_seen = False      # the outer "Fetching N files" bar is driving the %
    dl_desc = ""            # current file being fetched (for the label)

    def _dl_label():
        return "Downloading model weights" + (f": {dl_desc}" if dl_desc else "")

    try:
        for raw in proc.stdout:
            line = raw.rstrip()
            if line:
                print(line)
            if "Loading pipeline" in line:
                last_frac = max(last_frac, 0.01)
                set_progress("infer_3d", value=last_frac, label="Loading pipeline")
                continue
            # Backend-selection banners print once at import time (before any
            # checkpoint download starts) — surfacing them keeps the bar moving
            # instead of sitting static on "Loading pipeline" for a while.
            if "[SPARSE] Conv backend" in line:
                last_frac = max(last_frac, 0.02)
                set_progress("infer_3d", value=last_frac, label=line.strip())
                continue
            if "[ATTENTION] Using backend" in line:
                last_frac = max(last_frac, 0.03)
                set_progress("infer_3d", value=last_frac, label=line.strip())
                continue
            if "Downloading" in line and "checkpoint" in line:
                download_seen = True
                last_frac = max(last_frac, 0.04)
                set_progress("infer_3d", value=last_frac, label="Downloading model weights...")
                continue
            # Outer HF snapshot bar: the real overall download percentage.
            fm = fetch_re.search(line)
            if fm:
                download_seen = fetch_seen = True
                last_frac = max(last_frac, DL_SLICE * (int(fm.group(1)) / 100.0))
                lbl = _dl_label()
                rm = rate_re.search(line)
                if rm:
                    lbl += f" ({rm.group(1).replace(' ', '')})"
                set_progress("infer_3d", value=last_frac, label=lbl)
                continue
            m = proc_re.search(line)
            if m:
                cur_task, total_tasks = int(m.group(1)), max(int(m.group(2)), 1)
                continue
            sm = stage_re.match(line)
            if sm:
                stage_name, pct = sm.group(1).strip(), int(sm.group(2))
                matched = False
                for name, base, span in _3D_STAGE_WEIGHTS:
                    if name not in stage_name:
                        continue
                    local = base + span * (pct / 100.0)
                    # Reserve 0.0-DL_SLICE for the first-run download when one was
                    # seen; otherwise generation owns the whole 0.0-0.9 range.
                    gen_base = DL_SLICE if download_seen else 0.0
                    frac = gen_base + (0.9 - gen_base) * ((cur_task - 1 + local) / total_tasks)
                    label = _3D_STAGE_LABELS.get(name, name)
                    last_frac = max(last_frac, frac)
                    set_progress("infer_3d", value=last_frac, label=f"{label} (asset {cur_task}/{total_tasks})")
                    matched = True
                    break
                if not matched:
                    # A huggingface_hub per-file download tqdm bar (e.g.
                    # "model.safetensors: 45%|..."). Use it for the descriptive
                    # label + transfer rate, and — until the outer "Fetching N
                    # files" bar takes over — to drive the download slice off this
                    # file's own %. Capped just under DL_SLICE so a single finished
                    # file never claims the whole phase before the remaining files
                    # are known; last_frac's max() guard keeps the bar from moving
                    # backward as each successive file's bar resets to 0%.
                    download_seen = True
                    dl_desc = stage_name
                    lbl = _dl_label() + f" ({pct}%)"
                    rm = rate_re.search(line)
                    if rm:
                        lbl += f" - {rm.group(1).replace(' ', '')}"
                    if not fetch_seen:
                        last_frac = max(last_frac, DL_SLICE * 0.95 * (pct / 100.0))
                    set_progress("infer_3d", value=last_frac, label=lbl)
                continue
            if "Done." in line:
                last_frac = max(last_frac, 0.9)
                set_progress("infer_3d", value=last_frac, label="Generation complete")
    except Exception as e:
        debug_print(f"3D progress reader stopped: {e}")

# --- ASYNC OPERATOR: 2D GENERATOR ---

class ZIMAGE_OT_GenerateAsset(Operator):
    bl_idname = "object.generate_asset"
    bl_label = "Generate 2D Asset"
    bl_options = {"REGISTER"}

    _process = None
    _timer = None
    _reader = None
    _result_file = ""

    def modal(self, context, event):
        if event.type == 'ESC':
            return {'CANCELLED'}
        if event.type == 'TIMER':
            tag_redraw_all(context)  # animate the progress bar
            if self._process.poll() is not None:
                self.process_finish(context)
                return {'FINISHED'}
        return {'PASS_THROUGH'}

    def cancel(self, context):
        # Called on Esc (see modal) and when Blender tears the operator down
        # for other reasons (e.g. closing the file) — either way the child
        # process must be killed or its VRAM/RAM stays held indefinitely.
        _terminate_subprocess(self._process)
        _untrack_subprocess(self._process)
        context.window_manager.event_timer_remove(self._timer)
        set_progress("infer_2d", active=False)
        flush()
        tag_redraw_all(context)

    def process_finish(self, context):
        _untrack_subprocess(self._process)
        context.window_manager.event_timer_remove(self._timer)
        set_progress("infer_2d", active=False)
        tag_redraw_all(context)

        # The subprocess finished — but did it succeed? A non-zero exit code
        # means generation failed (missing CUDA, model download error, etc.).
        # Without this check we would silently re-import a stale results.json
        # from a previous run.
        if self._process.returncode != 0:
            self.report({'ERROR'}, "2D generation failed. See the system console for details.")
            flush()
            return

        if not os.path.exists(self._result_file):
            self.report({'ERROR'}, "2D generation produced no output. See the system console.")
            flush()
            return

        try:
            with open(self._result_file, 'r') as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self.report({'ERROR'}, f"Could not read generation results: {e}")
            flush()
            return

        from PIL import Image
        for item in results:
            bpy.ops.mesh.primitive_plane_add(size=1, location=(context.scene.cursor.location.x + item["offset"], context.scene.cursor.location.y, context.scene.cursor.location.z), rotation=(math.radians(90), 0, 0))
            obj = context.object
            # Bake the 90° stand-up rotation into the mesh data instead of
            # leaving it on the object transform. The Asset Browser's drag-drop
            # placement resets object rotation to world-aligned (identity), which
            # would otherwise flatten these planes back onto the ground.
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
            obj.name = item["name"]
            img_pil = Image.open(item["path"])
            obj.scale = (img_pil.size[0] / img_pil.size[1], 1, 1)
            mat = bpy.data.materials.new(name=f"Mat_{obj.name}")
            mat.use_nodes = True
            # blend_method was removed from materials in EEVEE-Next (Blender 4.2+).
            if hasattr(mat, "blend_method"):
                mat.blend_method = 'HASHED'
            # Look up the BSDF by type, not by name: the node label is localized
            # in non-English Blender builds, so nodes.get("Principled BSDF")
            # can return None and crash.
            bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
            tex.image = bpy.data.images.load(item["path"])
            if bsdf is not None:
                mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
                mat.node_tree.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
            obj.data.materials.append(mat)
            obj.asset_mark()
            context.view_layer.update()
            with context.temp_override(id=obj):
                bpy.ops.ed.lib_id_load_custom_preview(filepath=item["path"])
        flush()

    def execute(self, context):
        activate_virtualenv()
        py = python_exec()
        props = context.scene.import_text
        # FILE mode supplies an existing image and skips 2D generation entirely;
        # input_image_path is injected into the isolated script (empty => generate).
        input_image_path = ""
        if props.input_type == "FILE":
            input_image_path = bpy.path.abspath(props.input_image) if props.input_image else ""
            if not input_image_path or not os.path.isfile(input_image_path):
                self.report({'ERROR'}, "Select a valid input image file.")
                return {'CANCELLED'}
            lines = []
        elif props.input_type == "PROMPT":
            lines = [context.scene.asset_prompt]
        else:
            text = bpy.data.texts.get(props.scene_texts)
            if text is None:
                self.report({'ERROR'}, "No text-block selected.")
                return {'CANCELLED'}
            lines = [l.body for l in text.lines if l.body.strip()]
        lines = [sanitize_text(l).strip() for l in lines]
        lines = [l for l in lines if l]
        if not lines and not input_image_path:
            self.report({'ERROR'}, "No prompt text to generate from.")
            return {'CANCELLED'}

        # name_base names the imported plane object(s). For FILE input with no
        # explicit asset name, fall back to the image's filename stem.
        name_base = context.scene.asset_name or "Asset"
        if input_image_path and not context.scene.asset_name:
            name_base = os.path.splitext(os.path.basename(input_image_path))[0] or "Asset"

        data_dir = os.path.join(bpy.utils.user_resource("DATAFILES"), "2D_Async_Queue")
        os.makedirs(data_dir, exist_ok=True)
        self._result_file = os.path.join(data_dir, "results.json")
        # Remove any stale results from a previous run so a failed subprocess
        # cannot leave us importing old assets.
        if os.path.exists(self._result_file):
            os.remove(self._result_file)
        sp_folder = packages_path()

        isolated_script = f"""
import sys, os

# PATH SETUP — must happen before any non-stdlib import so addon_packages
# takes priority over Blender's own site-packages for every subsequent import.
sp = r"{sp_folder}"
sys.path.insert(0, sp)
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(sp)
    _tl = os.path.join(sp, "torch", "lib")
    if os.path.exists(_tl):
        os.add_dll_directory(_tl)
    for _r, _d, _f in os.walk(sp):
        if any(n.endswith(('.pyd', '.dll')) for n in _f):
            try:
                os.add_dll_directory(_r)
            except Exception:
                pass

import json
import gc
import numpy as np
import torch
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Re-run 'Install 2D + 3D Dependencies' to install "
        "the CUDA-enabled torch build, then restart Blender."
    )

from PIL import Image
from scipy.ndimage import label, find_objects
from diffusers import ZImagePipeline
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

device = "cuda"
lines = {json.dumps(lines)}
name_base = {json.dumps(name_base)}
# Non-empty => FILE mode: use this image directly and skip 2D generation.
input_image_path = {json.dumps(input_image_path)}

def free_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# Two-phase design to keep VRAM low: the diffusion pipeline and the BiRefNet
# segmenter are NEVER resident at the same time. Phase 1 generates every image
# with only the pipeline loaded (and CPU-offloaded), then the pipeline is fully
# released; Phase 2 loads the segmenter and processes the images saved to disk.

# --- PHASE 1: obtain the raw image(s) (only the diffusion pipeline in VRAM) ----
raw_paths = []  # (index, path-to-raw-RGB-png)
if input_image_path:
    # FILE mode: use the user's image directly — no generation, no pipeline load.
    # Copy it into the queue dir as the "raw" image so Phase 2 processes (and
    # later deletes) the copy, never the user's original file.
    print("[2d] Using input image (skipping generation)...")
    _src0 = Image.open(input_image_path).convert("RGB")
    raw_p = os.path.join(r"{data_dir}", f"{{name_base}}_0_raw.png")
    _src0.save(raw_p)
    raw_paths.append((0, raw_p))
else:
    print("[2d] Loading pipeline...")
    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
    # enable_model_cpu_offload keeps only the actively-running submodule on the GPU
    # and parks the rest in CPU RAM — a big VRAM saving. It manages device
    # placement itself, so we must NOT also call .to(device).
    try:
        pipe.enable_model_cpu_offload()
    except Exception as _e:
        print(f"[2d] CPU offload unavailable ({{_e}}); falling back to full GPU load")
        pipe.to(device)

    for i, line in enumerate(lines):
        print(f"[2d] Generating: {{line}}")
        # 2048 instead of Z-Image's native 1024: the segmented per-object crops
        # are what feed TRELLIS later, and crops from a 1024 sheet sit far below
        # TRELLIS's 1024px input ceiling. Doubling the sheet doubles each crop.
        img = pipe(prompt="neutral background, " + line, height=2048, width=2048,
                   num_inference_steps=9, guidance_scale=0.0).images[0]
        raw_p = os.path.join(r"{data_dir}", f"{{name_base}}_{{i}}_raw.png")
        img.convert("RGB").save(raw_p)
        raw_paths.append((i, raw_p))
        torch.cuda.empty_cache()

    # Release the diffusion pipeline BEFORE loading the segmenter.
    del pipe
    free_vram()

# --- PHASE 2: segment + crop (only BiRefNet in VRAM) --------------------------
print("[2d] Loading BiRefNet...")
birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_HR", trust_remote_code=True).to(device).eval()
# Newer transformers loads the checkpoint in its saved dtype (fp16 here), so
# match the input tensor to the model's dtype to avoid a float/Half mismatch.
biref_dtype = next(birefnet.parameters()).dtype
trans = transforms.Compose([
    transforms.Resize((2048, 2048)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
output_data = []
offset_accum = 0.0
for i, raw_p in raw_paths:
    print(f"[2d] Segmenting: image {{i + 1}}")
    src = Image.open(raw_p).convert("RGB")
    with torch.no_grad():
        inp = trans(src).unsqueeze(0).to(device=device, dtype=biref_dtype)
        m_t = birefnet(inp)[-1].sigmoid().float().cpu()
    mask = transforms.ToPILImage()(m_t[0].squeeze()).resize(src.size)
    src.putalpha(mask)
    mask_arr = np.array(src)[:, :, 3] > 0
    labeled, n = label(mask_arr)
    for j, bbox in enumerate(find_objects(labeled), 1):
        if bbox:
            crop = src.crop((bbox[1].start, bbox[0].start, bbox[1].stop, bbox[0].stop))
            f_p = os.path.join(r"{data_dir}", f"{{name_base}}_{{i}}_{{j}}.png")
            crop.save(f_p)
            output_data.append({{"path": f_p, "offset": offset_accum, "name": f"{{name_base}}_{{j}}"}})
            offset_accum += (crop.size[0] / crop.size[1]) + 0.2
    # Intermediate raw image no longer needed.
    try:
        os.remove(raw_p)
    except OSError:
        pass
    torch.cuda.empty_cache()

del birefnet
free_vram()

with open(r"{self._result_file}", "w") as f:
    json.dump(output_data, f)
print("[2d] Done.")
"""
        script_path = os.path.join(data_dir, "run_2d.py")
        with open(script_path, "w") as f: f.write(isolated_script)
        set_progress("infer_2d", value=0.0, label="Starting...", active=True)
        # -u: unbuffered child stdout so progress markers arrive in real time
        # instead of being held in a pipe buffer until the process exits.
        self._process = subprocess.Popen(
            [py, "-u", script_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            encoding="utf-8", errors="replace",
        )
        _track_subprocess(self._process)
        self._reader = threading.Thread(target=_read_2d_progress, args=(self._process, max(len(lines), 1)), daemon=True)
        self._reader.start()
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

# --- ASYNC OPERATOR: 3D CONVERTER (FIXED FOR BINARY LOADING) ---

def _import_trellis_mesh(context, glb_path, location):
    """Import a TRELLIS-exported GLB and return its actual mesh object.

    trimesh always exports the scene graph with a root node literally named
    "world" (its default base_frame), with the real mesh nested as a child.
    context.active_object can land on that empty wrapper instead of the mesh,
    so find the mesh explicitly and bake the wrapper's transform into it.
    """
    bpy.ops.import_scene.gltf(filepath=glb_path)
    imported = context.selected_objects
    mesh_obj = next((o for o in imported if o.type == 'MESH'), None)
    if mesh_obj is None:
        return None
    root = mesh_obj
    while root.parent is not None:
        root = root.parent
    root.location = location
    context.view_layer.update()
    wrappers = []
    if mesh_obj.parent is not None:
        node = mesh_obj.parent
        with context.temp_override(selected_editable_objects=[mesh_obj], active_object=mesh_obj):
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        while node is not None:
            wrappers.append(node)
            node = node.parent
    for empty in wrappers:
        bpy.data.objects.remove(empty, do_unlink=True)
    return mesh_obj


class TRELLIS_OT_ConvertSelected(Operator):
    bl_idname = "object.trellis_convert"
    bl_label = "Convert selected to 3D"
    
    _process = None
    _timer = None
    _reader = None
    _tasks = []
    _post_index = -1  # -1 while the subprocess is still running; >=0 once
                       # per-task Blender-side post-processing has started.

    def modal(self, context, event):
        if event.type == 'ESC':
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        tag_redraw_all(context)  # animate the progress bar

        if self._post_index < 0:
            if self._process.poll() is None:
                return {'PASS_THROUGH'}
            _untrack_subprocess(self._process)
            if self._process.returncode != 0:
                context.window_manager.event_timer_remove(self._timer)
                set_progress("infer_3d", active=False)
                self.report({'ERROR'}, "3D conversion failed. See the system console for details.")
                flush()
                tag_redraw_all(context)
                return {'FINISHED'}
            # Subprocess finished — hand off to the per-task Blender-side pass
            # below. Handled one task per tick (not one big loop) so Blender's
            # UI actually gets to repaint the progress bar between tasks; a
            # single blocking loop would only ever show its very last state,
            # since bake calls block the interpreter and nothing repaints
            # until control returns to Blender's event loop.
            self._post_index = 0
            return {'PASS_THROUGH'}

        n = len(self._tasks)
        if self._post_index >= n:
            context.window_manager.event_timer_remove(self._timer)
            set_progress("infer_3d", active=False)
            flush()
            tag_redraw_all(context)
            return {'FINISHED'}

        i = self._post_index
        set_progress("infer_3d", value=0.9 + 0.1 * (i / n), label=f"Building asset {i + 1}/{n}")
        self._process_one_task(context, self._tasks[i])
        self._post_index += 1
        return {'PASS_THROUGH'}

    def cancel(self, context):
        # Called on Esc (see modal) and when Blender tears the operator down
        # for other reasons (e.g. closing the file). If the subprocess is
        # still running its VRAM/RAM would otherwise stay held indefinitely
        # since nothing else would ever terminate it.
        _terminate_subprocess(self._process)
        _untrack_subprocess(self._process)
        context.window_manager.event_timer_remove(self._timer)
        set_progress("infer_3d", active=False)
        flush()
        tag_redraw_all(context)

    def _process_one_task(self, context, t):
        glb_p = t["path"].replace(".png", "_3d.glb")
        if not os.path.exists(glb_p):
            return
        loc = t["loc"]
        target_loc = (loc.x, loc.y + 2.5, loc.z)
        mesh_obj = _import_trellis_mesh(context, glb_p, target_loc)
        if mesh_obj is None:
            return

        context.view_layer.objects.active = mesh_obj
        mesh_obj.select_set(True)

        # The glTF importer keeps the file's per-corner normals as custom
        # split normals, and TRELLIS exports can carry broken ones (convex
        # shadowing artifacts). Drop them so Blender recomputes clean normals
        # from the geometry instead.
        if mesh_obj.data.has_custom_normals:
            try:
                bpy.ops.mesh.customdata_custom_splitnormals_clear()
            except Exception as e:
                debug_print(f"Could not clear custom normals on {mesh_obj.name}: {e}")

        # Cheap insurance against voxel-decode debris (loose
        # floaters, inconsistent winding). o_voxel's own
        # remesh rarely leaves either, so this is normally a
        # no-op, not an expected fix. These meshes can be dense (six-figure+
        # vert counts), so each step is timed — a future slowdown shows up as
        # a number in the console instead of a silent-looking freeze.
        _t0 = time.perf_counter()
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete_loose()
        debug_print(f"[cleanup] delete_loose: {time.perf_counter() - _t0:.2f}s")

        # Auto-flatten extreme spikes: voxel-decode/remesh debris
        # occasionally leaves a lone vertex pulled far from its
        # neighbors (a visible poke/needle). Detect verts whose distance
        # from their neighbor average badly exceeds the local edge
        # length and pull them back toward the surface. This only moves
        # vertex positions — UVs are per-loop, not per-vertex, so
        # texturing is unaffected. Run this before Merge by Distance so
        # any coincident verts it creates get welded next.
        _t0 = time.perf_counter()
        SPIKE_FACTOR = 4.0   # how many local edge-lengths counts as "extreme"
        SPIKE_PULL = 0.9     # how far to pull a spike toward its neighbor average (1.0 = fully flatten)
        bm = bmesh.from_edit_mesh(mesh_obj.data)
        bm.verts.ensure_lookup_table()
        spikes = []
        for v in bm.verts:
            if len(v.link_edges) < 3:
                continue
            neighbors = [e.other_vert(v) for e in v.link_edges]
            avg_edge_len = sum((v.co - n.co).length for n in neighbors) / len(neighbors)
            if avg_edge_len < 1e-8:
                continue
            center = sum((n.co for n in neighbors), Vector()) / len(neighbors)
            if (v.co - center).length > SPIKE_FACTOR * avg_edge_len:
                spikes.append((v, center))
        for v, center in spikes:
            v.co = v.co.lerp(center, SPIKE_PULL)
        bmesh.update_edit_mesh(mesh_obj.data)
        debug_print(f"[cleanup] spike-flatten ({len(spikes)} vert(s)): {time.perf_counter() - _t0:.2f}s")

        # Merge by Distance is position-only — UVs live per face-corner
        # (loop), not per vertex, so welding two coincident vertices doesn't
        # touch either corner's UV; it only removes the redundant vertex.
        # A tight threshold (well under the mesh's own voxel resolution)
        # only catches true duplicates — the kind glTF export creates at UV
        # seams themselves, or float32 rounding noise — and leaves anything
        # that's actually meant to stay separate untouched.
        _t0 = time.perf_counter()
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        debug_print(f"[cleanup] remove_doubles: {time.perf_counter() - _t0:.2f}s")

        # Removed: Blender's fill_holes() has to trace and classify every
        # boundary loop in the mesh before it can even check which ones are
        # small enough to fill, and that trace itself hung repeatedly on
        # these dense remeshed characters even with `sides` capped — the
        # cost is in finding the holes, not filling them. o_voxel's own
        # to_glb() already runs mesh.fill_holes(max_hole_perimeter=3e-2) via
        # cumesh before this mesh ever reaches Blender, so this was
        # redundant insurance, not a fix for an observed defect.

        _t0 = time.perf_counter()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        debug_print(f"[cleanup] normals_make_consistent: {time.perf_counter() - _t0:.2f}s")
        bpy.ops.object.mode_set(mode='OBJECT')

        # Cubic interpolation reduces perceived blur on the
        # baked texture vs. Blender's default linear sampling.
        for mat in mesh_obj.data.materials:
            if mat is None or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE':
                    node.interpolation = 'Cubic'

        mesh_obj.asset_mark()
        context.view_layer.update()
        with context.temp_override(id=mesh_obj):
            bpy.ops.ed.lib_id_load_custom_preview(filepath=t["path"])

    def execute(self, context):
        selected = [o for o in context.selected_objects if o.type == 'MESH']
        if not selected: return {'CANCELLED'}
        activate_virtualenv()
        
        self._tasks = []
        for o in selected:
            for s in o.material_slots:
                if s.material and s.material.use_nodes:
                    for n in s.material.node_tree.nodes:
                        if n.type == 'TEX_IMAGE' and n.image:
                            self._tasks.append({"path": bpy.path.abspath(n.image.filepath), "loc": o.location.copy()})
        
        if not self._tasks: return {'CANCELLED'}
        self._post_index = -1
        # Remove stale .glb outputs so a failed subprocess cannot leave us
        # re-importing results from a previous run.
        for t in self._tasks:
            for suffix in ("_3d.glb", "_3d_hipoly.glb"):
                stale = t["path"].replace(".png", suffix)
                if os.path.exists(stale):
                    os.remove(stale)
        py = python_exec()
        data_dir = os.path.join(bpy.utils.user_resource("DATAFILES"), "3D_Async_Queue")
        os.makedirs(data_dir, exist_ok=True)
        task_json = os.path.join(data_dir, "tasks.json")
        with open(task_json, 'w') as f: json.dump([{"path": t["path"]} for t in self._tasks], f)

        sp = packages_path()
        bin_dir = os.path.join(sp, "torch", "lib")  # CUDA runtime DLLs live here
        repo = os.path.normpath(os.path.join(addon_script_path(), "TRELLIS_REPO"))
        pixal3d_repo = os.path.normpath(os.path.join(addon_script_path(), "PIXAL3D_REPO"))
        scn = context.scene

        if scn.trellis_model_backend == "PIXAL3D":
            isolated_3d = self._build_pixal3d_script(sp, bin_dir, pixal3d_repo, task_json, scn)
        else:
            isolated_3d = self._build_trellis2_script(sp, bin_dir, repo, task_json, scn)

        script_path = os.path.join(data_dir, "run_3d.py")
        with open(script_path, "w") as f: f.write(isolated_3d)
        set_progress("infer_3d", value=0.0, label="Starting...", active=True)
        # -u: unbuffered child stdout so progress markers arrive in real time.
        self._process = subprocess.Popen(
            [py, "-u", script_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            encoding="utf-8", errors="replace",
        )
        _track_subprocess(self._process)
        self._reader = threading.Thread(target=_read_3d_progress, args=(self._process,), daemon=True)
        self._reader.start()
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _build_trellis2_script(self, sp, bin_dir, repo, task_json, scn):
        return f"""
import sys, os

# PATH SETUP — must happen before any non-stdlib import so addon_packages
# takes priority over Blender's own site-packages for every subsequent import.
sp, bin_dir, repo = r"{sp}", r"{bin_dir}", r"{repo}"
sys.path.insert(0, repo)
sys.path.insert(0, sp)  # sp at 0 so addon_packages wins over repo and Blender

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(sp)
    if os.path.exists(bin_dir):           # torch/lib/ — CUDA runtime DLLs
        os.add_dll_directory(bin_dir)
    for _r, _d, _f in os.walk(sp):
        if any(n.endswith(('.pyd', '.dll')) for n in _f):
            try:
                os.add_dll_directory(_r)
            except Exception:
                pass

import json
import gc

# CRITICAL: Set env vars BEFORE any trellis2 import (modules read them at import time)
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Enable flex_gemm sparse conv backend if the extension was built
try:
    import flex_gemm  # noqa: F401
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
except ImportError:
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

# MANDATORY: Import Torch FIRST to load CUDA runtime for cumesh/flexgemm
import torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

# o_voxel.postprocess.to_glb() Telea-inpaints the ENTIRE unused texture atlas
# (not just a padding border around each UV island) to avoid black seams at
# chart edges. When the UV charts only cover a small fraction of the canvas
# (common with cumesh's chart packing), Telea hallucinates a repeating
# wood-grain-like pattern across all that empty space — cosmetically ugly,
# and it can bleed into visible mip levels at chart boundaries. Bound the
# inpaint to a small ring around real texels instead; texels farther out are
# never sampled by the mesh anyway, so nothing is lost by leaving them as-is.
import numpy as np
import cv2
_orig_cv2_inpaint = cv2.inpaint
def _bounded_cv2_inpaint(src, inpaintMask, inpaintRadius, flags):
    valid = (inpaintMask == 0).astype(np.uint8)
    pad = max(int(inpaintRadius) * 4, 16)
    kernel = np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8)
    dilated = cv2.dilate(valid, kernel)
    band = ((dilated > 0) & (valid == 0)).astype(np.uint8)
    filled = _orig_cv2_inpaint(src, band, inpaintRadius, flags)
    # Real cv2.inpaint silently drops a trailing size-1 channel dim on
    # single-channel input (metallic/roughness/alpha are (H, W, 1) -> (H, W)),
    # and o_voxel's own code compensates with a manual `[..., None]` after the
    # call. Mirror that exact squeeze here so this wrapper's return shape
    # matches what the real function would have produced — matching src's
    # shape instead would make that `[..., None]` add a spurious extra dim.
    src2d = src if src.ndim == filled.ndim else src.squeeze(-1)
    band_bool = band.astype(bool)
    out = src2d.copy()
    out[band_bool] = filled[band_bool]
    return out
cv2.inpaint = _bounded_cv2_inpaint

device = "cuda" if torch.cuda.is_available() else "cpu"

def free_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

with open(r"{task_json}", "r") as f:
    tasks = json.load(f)

# pipeline.json pins the rembg backend to briaai/RMBG-2.0, a gated HF repo
# that most users won't have been granted access to (403 GatedRepoError).
# Force the public, ungated BiRefNet_HR checkpoint instead, regardless of
# what the downloaded config says.
from trellis2.pipelines import rembg as _rembg
_orig_birefnet_init = _rembg.BiRefNet.__init__
def _patched_birefnet_init(self, model_name="ZhengPeng7/BiRefNet_HR", **kwargs):
    _orig_birefnet_init(self, model_name="ZhengPeng7/BiRefNet_HR")
_rembg.BiRefNet.__init__ = _patched_birefnet_init

# Our pinned transformers version nests DINOv3ViTModel's transformer layers
# under `.model.layer` (DINOv3ViTModel.model is the DINOv3ViTEncoder), but
# upstream trellis2 code was written against a transformers version where
# they lived directly at `.layer` on the top-level model. Patch the accessor
# rather than the property path so both layouts are supported going forward.
from trellis2.modules import image_feature_extractor as _dinov3_mod
def _patched_extract_features(self, image):
    image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
    hidden_states = self.model.embeddings(image, bool_masked_pos=None)
    position_embeddings = self.model.rope_embeddings(image)
    layers = getattr(self.model, "layer", None)
    if layers is None:
        layers = self.model.model.layer
    for layer_module in layers:
        hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
    import torch.nn.functional as _F
    return _F.layer_norm(hidden_states, hidden_states.shape[-1:])
_dinov3_mod.DinoV3FeatureExtractor.extract_features = _patched_extract_features

print("[3d] Loading pipeline...")
print("[3d] Downloading checkpoint if not already cached (microsoft/TRELLIS.2-4B, first run only)...")
pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipe.low_vram = True
pipe.to(device)

for idx, t in enumerate(tasks, 1):
    print(f"[3d] Processing {{idx}}/{{len(tasks)}}")
    # Load as RGBA — alpha mask from the 2D generation step lets the pipeline
    # skip background removal and go straight to crop+premultiply.
    img = Image.open(t["path"]).convert("RGBA")
    # return_latent=True gives us `res` (grid resolution) required by to_glb.
    # Sampler params come from the UI (TRELLIS_PT_SubPanel "Advanced" box);
    # see the scene.trellis_* properties in register().
    out_meshes, (_shape_slat, _tex_slat, res) = pipe.run(
        img,
        seed={scn.trellis_seed},
        preprocess_image=True,
        return_latent=True,
        pipeline_type="{scn.trellis_pipeline_type}",
        sparse_structure_sampler_params={{
            "steps": {scn.trellis_ss_steps},
            "guidance_strength": {scn.trellis_ss_guidance},
            "guidance_rescale": {scn.trellis_ss_guidance_rescale},
        }},
        shape_slat_sampler_params={{
            "steps": {scn.trellis_shape_steps},
            "guidance_strength": {scn.trellis_shape_guidance},
            "guidance_rescale": {scn.trellis_shape_guidance_rescale},
        }},
        # guidance_strength > 1 (stock: 1.0 = CFG off) trades a rescale
        # (color/saturation drift correction) for stronger adherence to the
        # input image.
        tex_slat_sampler_params={{
            "steps": {scn.trellis_tex_steps},
            "guidance_strength": {scn.trellis_tex_guidance},
            "guidance_rescale": {scn.trellis_tex_guidance_rescale},
        }},
        # Default 49152 makes the cascade silently step 1536 down (1408,
        # 1280, ...) until the voxel token count fits — which a full-body
        # character usually triggers.
        max_num_tokens={scn.trellis_max_num_tokens},
    )
    mesh = out_meshes[0]
    mesh.simplify(16777216)  # nvdiffrast face limit
    glb_p = t["path"].replace(".png", "_3d.glb")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipe.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        remesh=True,
        # 4096 (vs the library default of 2048): the voxel grid holds more
        # detail than a 2048 map can carry per-texel once the UV islands are
        # packed, and the larger map also reduces aliasing at UV seams.
        texture_size=4096,
        # o_voxel's own defaults (0 refine / 1 global iteration) are far
        # weaker than cumesh.compute_charts()'s own library defaults (100 /
        # 3) and leave the initial cone-angle clustering pass unrefined —
        # the result is heavily over-fragmented, jagged UV charts. Matching
        # cumesh's own defaults here lets it merge/smooth clusters properly
        # before xatlas packs them.
        mesh_cluster_refine_iterations=100,
        mesh_cluster_global_iterations=3,
        use_tqdm=True,
    )
    glb.export(glb_p, extension_webp=True)

    del img, out_meshes, _shape_slat, _tex_slat, res, mesh, glb
    free_vram()

del pipe
free_vram()
print("[3d] Done.")
"""

    def _build_pixal3d_script(self, sp, bin_dir, pixal3d_repo, task_json, scn):
        return f"""
import sys, os

# PATH SETUP — must happen before any non-stdlib import so addon_packages
# takes priority over Blender's own site-packages for every subsequent import.
sp, bin_dir, pixal3d_repo = r"{sp}", r"{bin_dir}", r"{pixal3d_repo}"
sys.path.insert(0, pixal3d_repo)
sys.path.insert(0, sp)  # sp at 0 so addon_packages wins over repo and Blender

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(sp)
    if os.path.exists(bin_dir):           # torch/lib/ — CUDA runtime DLLs
        os.add_dll_directory(bin_dir)
    for _r, _d, _f in os.walk(sp):
        if any(n.endswith(('.pyd', '.dll')) for n in _f):
            try:
                os.add_dll_directory(_r)
            except Exception:
                pass

import json
import gc
import math

# CRITICAL: Set env vars BEFORE any pixal3d import (modules read them at import time)
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Enable flex_gemm sparse conv backend if the extension was built
try:
    import flex_gemm  # noqa: F401
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
except ImportError:
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

# MANDATORY: Import Torch FIRST to load CUDA runtime for cumesh/flexgemm/natten
import torch
from PIL import Image
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
import o_voxel
import numpy as np

# Same bounded-inpaint fix as the TRELLIS.2 path: o_voxel.postprocess.to_glb()
# Telea-inpaints the ENTIRE unused texture atlas rather than just a border
# around each UV island, which hallucinates a wood-grain pattern over empty
# canvas. Bound it to a small ring around real texels instead.
import cv2
_orig_cv2_inpaint = cv2.inpaint
def _bounded_cv2_inpaint(src, inpaintMask, inpaintRadius, flags):
    valid = (inpaintMask == 0).astype(np.uint8)
    pad = max(int(inpaintRadius) * 4, 16)
    kernel = np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8)
    dilated = cv2.dilate(valid, kernel)
    band = ((dilated > 0) & (valid == 0)).astype(np.uint8)
    filled = _orig_cv2_inpaint(src, band, inpaintRadius, flags)
    src2d = src if src.ndim == filled.ndim else src.squeeze(-1)
    band_bool = band.astype(bool)
    out = src2d.copy()
    out[band_bool] = filled[band_bool]
    return out
cv2.inpaint = _bounded_cv2_inpaint

device = "cuda" if torch.cuda.is_available() else "cpu"

def free_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

with open(r"{task_json}", "r") as f:
    tasks = json.load(f)

# TencentARC/Pixal3D's pipeline.json pins the rembg backend to the gated
# briaai/RMBG-2.0 (same gate the TRELLIS.2 path already works around below) —
# and Pixal3DImageTo3DPipeline.from_pretrained() instantiates it eagerly
# regardless of whether our already-alpha-matted input ever calls it. Force
# the public, ungated BiRefNet_HR checkpoint instead.
from pixal3d.pipelines import rembg as _rembg
_orig_birefnet_init = _rembg.BiRefNet.__init__
def _patched_birefnet_init(self, model_name="ZhengPeng7/BiRefNet_HR", **kwargs):
    _orig_birefnet_init(self, model_name="ZhengPeng7/BiRefNet_HR")
_rembg.BiRefNet.__init__ = _patched_birefnet_init

print("[3d] Loading pipeline...")
print("[3d] Downloading checkpoint if not already cached (TencentARC/Pixal3D, ~24GB, first run only — this can take a while)...")
pipe = Pixal3DImageTo3DPipeline.from_pretrained("TencentARC/Pixal3D")

# "Proj mode" (pixel-aligned conditioning): from_pretrained() deliberately
# leaves the four image_cond_model_* attrs as None — they must be built
# externally. Configs match the reference TencentARC/Pixal3D inference.py
# exactly, including the camenduru mirror of DINOv3 (pipeline.json points at
# the gated facebook/dinov3-vitl16-pretrain-lvd1689m; this ungated mirror
# avoids a 403 for users without Meta's DINOv3 access grant).
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor

# Same bug (and same fix) as the stock TRELLIS.2 DinoV3FeatureExtractor patch
# above: DinoV3ProjFeatureExtractor.extract_features() hard-codes
# `self.model.layer`, but our pinned transformers version nests the
# transformer blocks one level deeper, under `self.model.model.layer`
# (DINOv3ViTModel.model is the DINOv3ViTEncoder). Unpatched, this raises
# AttributeError the first time the Pixal3D backend actually runs. Confirmed
# still present, unpatched, in TencentARC/Pixal3D upstream as of this writing.
import torch.nn.functional as _F
def _patched_proj_extract_features(self, image):
    image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
    hidden_states = self.model.embeddings(image, bool_masked_pos=None)
    position_embeddings = self.model.rope_embeddings(image)
    layers = getattr(self.model, "layer", None)
    if layers is None:
        layers = self.model.model.layer
    for layer_module in layers:
        hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
    return _F.layer_norm(hidden_states, hidden_states.shape[-1:])
DinoV3ProjFeatureExtractor.extract_features = _patched_proj_extract_features

_IMAGE_COND_CONFIGS = {{
    "ss": dict(model_name="camenduru/dinov3-vitl16-pretrain-lvd1689m", image_size=512, grid_resolution=16),
    "shape_512": dict(model_name="camenduru/dinov3-vitl16-pretrain-lvd1689m", image_size=512, grid_resolution=32,
                       use_naf_upsample=True, naf_target_size=512),
    "shape_1024": dict(model_name="camenduru/dinov3-vitl16-pretrain-lvd1689m", image_size=1024, grid_resolution=64,
                        use_naf_upsample=True, naf_target_size=512),
    "tex_1024": dict(model_name="camenduru/dinov3-vitl16-pretrain-lvd1689m", image_size=1024, grid_resolution=64,
                      use_naf_upsample=True, naf_target_size=1024),
}}

def _build_image_cond_model(cfg):
    m = DinoV3ProjFeatureExtractor(**cfg)
    m.eval()
    return m

pipe.image_cond_model_ss = _build_image_cond_model(_IMAGE_COND_CONFIGS["ss"])
pipe.image_cond_model_shape_512 = _build_image_cond_model(_IMAGE_COND_CONFIGS["shape_512"])
pipe.image_cond_model_shape_1024 = _build_image_cond_model(_IMAGE_COND_CONFIGS["shape_1024"])
pipe.image_cond_model_tex_1024 = _build_image_cond_model(_IMAGE_COND_CONFIGS["tex_1024"])

# Low-VRAM mode (default True, matching the TRELLIS.2 path): models stay on
# CPU and are moved to GPU on-demand per stage by the pipeline's own run()
# logic. Only the NAF upsampler weights need pre-loading here (reference
# TencentARC/Pixal3D inference.py does the same in its low_vram branch).
pipe.low_vram = True
for _attr in ("image_cond_model_ss", "image_cond_model_shape_512",
              "image_cond_model_shape_1024", "image_cond_model_tex_1024"):
    _m = getattr(pipe, _attr, None)
    if _m is not None and getattr(_m, "use_naf_upsample", False):
        _m._load_naf()
pipe._device = torch.device(device)

# Manual camera FOV/distance — Pixal3D's pixel back-projection requires
# camera_params (no default fallback), but its own camera estimation (MoGe-2)
# is trained on real photos, not the flat AI-generated illustrations this
# add-on feeds it. Reproduce the reference inference.py's manual-FOV math
# verbatim (its own recommended fallback when there's no real camera).
def _compute_f_pixels(camera_angle_x, resolution):
    focal_length = 16.0 / math.tan(camera_angle_x / 2.0)
    return focal_length * resolution / 32.0

def _distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    # Fixed axis-swap matching the reference implementation's world/grid convention.
    xw, yw, zw = grid_point[0], -grid_point[2], grid_point[1]
    xw, yw, zw = xw / mesh_scale / 2, yw / mesh_scale / 2, zw / mesh_scale / 2
    xt, yt = target_point[0], target_point[1]
    f_pixels = _compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    return f_pixels * xw / x_ndc - yw

_camera_angle_x = {scn.trellis_pixal_fov}
_mesh_scale = {scn.trellis_pixal_mesh_scale}
_image_resolution = 512
_distance = _distance_from_fov(_camera_angle_x, (-1.0, 0.0, 0.0), (0 - 0, _image_resolution - 1 + 0), _mesh_scale, _image_resolution)
camera_params = {{'camera_angle_x': _camera_angle_x, 'distance': _distance, 'mesh_scale': _mesh_scale}}
print(f"[3d] Manual camera: fov={{math.degrees(_camera_angle_x):.1f}} deg, distance={{_distance:.4f}}, mesh_scale={{_mesh_scale}}")

for idx, t in enumerate(tasks, 1):
    print(f"[3d] Processing {{idx}}/{{len(tasks)}}")
    # Load as RGBA — alpha mask from the 2D generation step lets the pipeline
    # skip background removal and go straight to crop+premultiply.
    img = Image.open(t["path"]).convert("RGBA")
    out_meshes, (_shape_slat, _tex_slat, res) = pipe.run(
        img,
        camera_params=camera_params,
        seed={scn.trellis_seed},
        preprocess_image=True,
        return_latent=True,
        pipeline_type="{scn.trellis_pipeline_type}",
        sparse_structure_sampler_params={{
            "steps": {scn.trellis_ss_steps},
            "guidance_strength": {scn.trellis_ss_guidance},
            "guidance_rescale": {scn.trellis_ss_guidance_rescale},
            "rescale_t": {scn.trellis_ss_rescale_t},
        }},
        shape_slat_sampler_params={{
            "steps": {scn.trellis_shape_steps},
            "guidance_strength": {scn.trellis_shape_guidance},
            "guidance_rescale": {scn.trellis_shape_guidance_rescale},
            "rescale_t": {scn.trellis_shape_rescale_t},
        }},
        tex_slat_sampler_params={{
            "steps": {scn.trellis_tex_steps},
            "guidance_strength": {scn.trellis_tex_guidance},
            "guidance_rescale": {scn.trellis_tex_guidance_rescale},
            "rescale_t": {scn.trellis_tex_rescale_t},
        }},
        max_num_tokens={scn.trellis_max_num_tokens},
    )
    mesh = out_meshes[0]
    mesh.simplify(16777216)  # nvdiffrast face limit
    glb_p = t["path"].replace(".png", "_3d.glb")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipe.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        texture_size=4096,
        mesh_cluster_refine_iterations=100,
        mesh_cluster_global_iterations=3,
        use_tqdm=True,
    )
    # Axis correction matching the reference TencentARC/Pixal3D inference.py —
    # Pixal3D's coordinate convention differs from stock TRELLIS.2's export.
    _rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(_rot)
    glb.export(glb_p, extension_webp=True)

    del img, out_meshes, _shape_slat, _tex_slat, res, mesh, glb
    free_vram()

del pipe
free_vram()
print("[3d] Done.")
"""

# --- BAKE TEXTURE TO VERTEX COLORS ---

# --- PRO ASYNC INSTALLER ---

class InstallOperator(Operator):
    bl_idname = "virtual_dependencies.install_all"
    bl_label = "Install 2D + 3D Dependencies"
    _thread = None
    _timer = None
    def modal(self, context, event):
        if event.type == 'TIMER':
            tag_redraw_all(context)
            if not self._thread.is_alive():
                context.window_manager.event_timer_remove(self._timer)
                self.report({'INFO'}, "Dependencies ready.")
                tag_redraw_all(context)
                return {'FINISHED'}
        return {'PASS_THROUGH'}
    def execute(self, context):
        self._thread = threading.Thread(target=install_all_dependencies, daemon=True)
        self._thread.start()
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class AssetGeneratorPreferences(AddonPreferences):
    bl_idname = __name__
    def draw(self, context):
        layout = self.layout
        layout.label(text="Pro Isolated Environment Management", icon='SYSTEM')
        row = layout.row()
        row.operator("virtual_dependencies.install_all")
        row.operator("virtual_dependencies.uninstall_all", text="Nuclear Wipe")
        # Live progress for the background install / uninstall jobs.
        draw_progress(layout, "install")
        draw_progress(layout, "uninstall")

def uninstall_all():
    set_progress("uninstall", value=0.0, label="Removing packages...", active=True)
    try:
        shutil.rmtree(packages_path(), ignore_errors=True)
        set_progress("uninstall", value=1.0, label="Done")
    except Exception as e:
        debug_print(f"Uninstall error: {e}")
        set_progress("uninstall", label=f"Error: {e}")
    finally:
        set_progress("uninstall", active=False)

class UninstallOperator(Operator):
    bl_idname = "virtual_dependencies.uninstall_all"
    bl_label = "Uninstall Everything"
    _thread = None
    _timer = None
    def modal(self, context, event):
        if event.type == 'TIMER':
            tag_redraw_all(context)
            if not self._thread.is_alive():
                context.window_manager.event_timer_remove(self._timer)
                self.report({'INFO'}, "Packages removed.")
                tag_redraw_all(context)
                return {'FINISHED'}
        return {'PASS_THROUGH'}
    def execute(self, context):
        self._thread = threading.Thread(target=uninstall_all, daemon=True)
        self._thread.start()
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

# --- UI PANELS ---

class ZIMAGE_OT_SelectInputImage(Operator):
    """Open a file browser to pick an image to convert into a 2D asset."""
    bl_idname = "object.select_input_image"
    bl_label = "Select Image"
    bl_options = {"INTERNAL"}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(
        default="*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp;*.exr",
        options={'HIDDEN'},
    )

    def execute(self, context):
        context.scene.import_text.input_image = self.filepath
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class ZIMAGE_PT_MainPanel(Panel):
    bl_label = "Asset Generator"
    bl_idname = "VIEW3D_PT_zimage"
    bl_space_type, bl_region_type, bl_category = "VIEW_3D", "UI", "Asset Gen"
    def draw(self, context):
        layout = self.layout
        scene, props = context.scene, context.scene.import_text
        box = layout.box()
        box.row().prop(props, "input_type", expand=True)
        if props.input_type == "TEXT_BLOCK":
            row = box.row(align=True)
            row.prop(props, "scene_texts", text="", icon="TEXT")
            row.prop(props, "script", text="")
        elif props.input_type == "FILE":
            row = box.row(align=True)
            row.prop(props, "input_image", text="")
            row.operator("object.select_input_image", text="", icon='FILE_FOLDER')
            box.prop(scene, "asset_name", text="Name")
        else:
            box.prop(scene, "asset_prompt", text="Prompt")
            box.prop(scene, "asset_name", text="Name")
        box.operator("object.generate_asset", text="Generate Assets")
        draw_progress(box, "infer_2d")

class TRELLIS_PT_SubPanel(Panel):
    bl_label = "3D Trellis Conversion"
    bl_idname = "VIEW3D_PT_trellis"
    bl_space_type, bl_region_type, bl_category = "VIEW_3D", "UI", "Asset Gen"
    def draw(self, context):
        scene = context.scene
        layout = self.layout
        layout.prop(scene, "trellis_model_backend")
        layout.operator("object.trellis_convert", text="Convert Selected to 3D", icon='MOD_MESHDEFORM')
        draw_progress(layout, "infer_3d")

        is_pixal3d = scene.trellis_model_backend == "PIXAL3D"
        if is_pixal3d:
            cam_box = layout.box()
            cam_box.label(text="Camera (manual — no real camera to estimate from)", icon='CAMERA_DATA')
            cam_box.prop(scene, "trellis_pixal_fov")
            cam_box.prop(scene, "trellis_pixal_mesh_scale")

        box = layout.box()
        row = box.row()
        row.prop(scene, "trellis_show_settings",
                 icon='TRIA_DOWN' if scene.trellis_show_settings else 'TRIA_RIGHT',
                 icon_only=True, emboss=False, text="Advanced Settings")
        if not scene.trellis_show_settings:
            return

        box.prop(scene, "trellis_pipeline_type")
        box.prop(scene, "trellis_seed")
        box.prop(scene, "trellis_max_num_tokens")

        col = box.column(align=True)
        col.label(text="Sparse Structure:")
        col.prop(scene, "trellis_ss_steps")
        col.prop(scene, "trellis_ss_guidance")
        col.prop(scene, "trellis_ss_guidance_rescale")
        if is_pixal3d:
            col.prop(scene, "trellis_ss_rescale_t")

        col = box.column(align=True)
        col.label(text="Shape:")
        col.prop(scene, "trellis_shape_steps")
        col.prop(scene, "trellis_shape_guidance")
        col.prop(scene, "trellis_shape_guidance_rescale")
        if is_pixal3d:
            col.prop(scene, "trellis_shape_rescale_t")

        col = box.column(align=True)
        col.label(text="Texture:")
        col.prop(scene, "trellis_tex_steps")
        col.prop(scene, "trellis_tex_guidance")
        col.prop(scene, "trellis_tex_guidance_rescale")
        if is_pixal3d:
            col.prop(scene, "trellis_tex_rescale_t")

# --- REGISTRATION ---

classes = (
    Import_Text_Props, AssetGeneratorPreferences,
    ZIMAGE_OT_GenerateAsset, ZIMAGE_OT_SelectInputImage, TRELLIS_OT_ConvertSelected,
    InstallOperator, UninstallOperator,
    ZIMAGE_PT_MainPanel, TRELLIS_PT_SubPanel
)

# scene.trellis_* property name -> (PropertyType, kwargs). Defaults match the
# values this addon used before these became user-editable (see the
# isolated_3d pipe.run() call in TRELLIS_OT_ConvertSelected.execute), except
# where noted as the pipeline's own stock config.
_TRELLIS_SCENE_PROPS = {
    "trellis_show_settings": (BoolProperty, dict(name="Advanced Settings", default=False)),
    "trellis_model_backend": (EnumProperty, dict(
        name="Model",
        description="3D generation backbone. Pixal3D adds pixel-aligned back-projection "
                    "conditioning (from TencentARC/Pixal3D) for closer fidelity to the input "
                    "image, at the cost of a large extra checkpoint download and a manual "
                    "camera FOV guess (see below) since it has no real camera to estimate from",
        default="TRELLIS2",
        items=[
            ("TRELLIS2", "TRELLIS.2", "Stock microsoft/TRELLIS.2-4B — proven, no camera params needed"),
            ("PIXAL3D", "Pixal3D", "TencentARC/Pixal3D — pixel-aligned conditioning, needs a manual FOV guess"),
        ],
    )),
    # Radians. 0.2 matches Pixal3D's own suggested fallback value ("Try 0.2 rad if you
    # notice distortion") for when there's no real camera to estimate FOV from — exactly
    # this addon's situation, since inputs are flat AI-generated illustrations, not photos.
    "trellis_pixal_fov": (FloatProperty, dict(
        name="Camera FOV", subtype='ANGLE',
        description="Manual horizontal FOV fed to Pixal3D's pixel back-projection, in place "
                    "of MoGe-2's photo-based camera estimation (not used here — these inputs "
                    "have no real camera to estimate from). Lower values approximate a more "
                    "distant/orthographic-like camera",
        default=0.2, min=0.01, max=3.0,
    )),
    "trellis_pixal_mesh_scale": (FloatProperty, dict(
        name="Mesh Scale", default=1.0, min=0.01, max=10.0,
    )),
    "trellis_pipeline_type": (EnumProperty, dict(
        name="Resolution", default="1536_cascade",
        items=[
            ("512", "512", "Low resolution, fast, less VRAM"),
            ("1024", "1024", "Medium resolution, no cascade"),
            ("1024_cascade", "1024 Cascade", "Medium resolution with cascade"),
            ("1536_cascade", "1536 Cascade", "High resolution with cascade, most VRAM"),
        ],
    )),
    "trellis_seed": (IntProperty, dict(name="Seed", default=42, min=0)),
    "trellis_max_num_tokens": (IntProperty, dict(
        name="Max Tokens",
        description="Max sparse-voxel tokens during cascade upsampling. Too low silently "
                    "reduces the cascade resolution below the selected pipeline type",
        default=65536, min=16384, max=131072, step=4096,
    )),
    "trellis_ss_steps": (IntProperty, dict(name="Steps", default=12, min=1, max=50)),
    "trellis_ss_guidance": (FloatProperty, dict(name="Guidance Strength", default=7.5, min=0.0, max=20.0)),
    "trellis_ss_guidance_rescale": (FloatProperty, dict(name="Guidance Rescale", default=0.7, min=0.0, max=1.0)),
    # rescale_t: Pixal3D-only sampler param (its reference inference.py default). Not
    # passed to the stock TRELLIS2 call, which has no such key.
    "trellis_ss_rescale_t": (FloatProperty, dict(name="Rescale T (Pixal3D)", default=5.0, min=0.0, max=20.0)),
    "trellis_shape_steps": (IntProperty, dict(name="Steps", default=28, min=1, max=50)),
    "trellis_shape_guidance": (FloatProperty, dict(name="Guidance Strength", default=7.5, min=0.0, max=20.0)),
    "trellis_shape_guidance_rescale": (FloatProperty, dict(name="Guidance Rescale", default=0.5, min=0.0, max=1.0)),
    "trellis_shape_rescale_t": (FloatProperty, dict(name="Rescale T (Pixal3D)", default=3.0, min=0.0, max=20.0)),
    "trellis_tex_steps": (IntProperty, dict(name="Steps", default=28, min=1, max=50)),
    # 2.0 (stock: 1.0 = CFG off) for stronger adherence to the input image.
    "trellis_tex_guidance": (FloatProperty, dict(name="Guidance Strength", default=2.0, min=0.0, max=20.0)),
    # 0.7 counters the color/saturation drift plain CFG introduces at
    # guidance_strength > 1 (stock tex default is 0.0, matching stock's
    # CFG-off guidance_strength of 1.0).
    "trellis_tex_guidance_rescale": (FloatProperty, dict(name="Guidance Rescale", default=0.7, min=0.0, max=1.0)),
    "trellis_tex_rescale_t": (FloatProperty, dict(name="Rescale T (Pixal3D)", default=3.0, min=0.0, max=20.0)),
}

def register():
    for cls in classes: bpy.utils.register_class(cls)
    bpy.types.Scene.import_text = PointerProperty(type=Import_Text_Props)
    bpy.types.Scene.asset_prompt = StringProperty(name="Prompt", default="Goofy monster character sheet, multiple poses, white background")
    bpy.types.Scene.asset_name = StringProperty(name="Asset Name", default="Asset", update=get_unique_name)
    for name, (prop_type, kwargs) in _TRELLIS_SCENE_PROPS.items():
        setattr(bpy.types.Scene, name, prop_type(**kwargs))

def unregister():
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
    del bpy.types.Scene.import_text
    del bpy.types.Scene.asset_prompt
    del bpy.types.Scene.asset_name
    for name in _TRELLIS_SCENE_PROPS:
        delattr(bpy.types.Scene, name)

if __name__ == "__main__": register()
