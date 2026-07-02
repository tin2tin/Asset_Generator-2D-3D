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

def debug_print(*args, **kwargs):
    """Console logging for the generation process."""
    if DEBUG:
        print("[2D/3D Asset Pro Log] ", *args, **kwargs)

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
    return subprocess.run(
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

EXT_WHEEL_PROJECT = {
    "o-voxel":    "o_voxel",
    "cumesh":     "cumesh",
    "flexgemm":   "flex_gemm",
    "nvdiffrast": "nvdiffrast",
    "flash_attn": "flash_attn",
}

# flash_attn's matching wheel is ~240MB — over GitHub's 100MB push limit, and
# far bigger than the other four wheels combined (~13MB), so unlike them it is
# NOT bundled in wheels/. Instead it is downloaded on demand straight from the
# same PozzettiAndrea/cuda-wheels release index at install time. Building it
# from source on Windows is not a practical fallback (hours-long compile).
REMOTE_WHEEL_VERSION = {
    "flash_attn": "2.8.3",
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

def install_all_dependencies():
    """Install the full 2D + 3D stack in one pass: CUDA torch, the 2D/3D runtime
    packages, the TRELLIS.2 repo, and its compiled CUDA extensions."""
    set_progress("install", value=0.0, label="Preparing...", active=True)
    try:
        activate_virtualenv()
        py = python_exec()
        pkgs_dir = packages_path()
        addon_dir = addon_script_path()
        repo_path = os.path.normpath(os.path.join(addon_dir, "TRELLIS_REPO"))
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
        # torch + each requirement + repo clone + ext builds + verify
        total = 1 + len(reqs) + 1 + len(ext_specs) + 1
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
        #    aborting the whole batch.
        for req in reqs:
            set_progress("install", value=done / total, label=f"Installing {_req_shortname(req)}")
            subprocess.run([py, "-m", "pip", "install", "--disable-pip-version-check", "--target", pkgs_dir, "--no-deps", "--upgrade", req])
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
            subprocess.run(["git", "clone", "--recurse-submodules", "-b", "fix-cascade-seqlen-cache",
                             "https://github.com/tin2tin/TRELLIS.2.git", repo_path], check=True)
        else:
            subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_path, check=True)
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
        wheels_dir = os.path.join(addon_dir, "wheels")
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
            set_progress("install", value=done / total, label=f"Building {label}")
            try:
                wheel_path = find_local_wheel(wheels_dir, label)
                if wheel_path:
                    debug_print(f"Using bundled prebuilt wheel for {label}: {os.path.basename(wheel_path)}")
                    subprocess.run(
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
                    debug_print(f"No bundled wheel for {label} — downloading from {url}")
                    subprocess.run(
                        [py, "-m", "pip", "install", "--disable-pip-version-check", url,
                         "--no-deps", "--upgrade", "--target", pkgs_dir],
                        check=True,
                    )
                    done += 1
                    continue
                if isinstance(source, tuple):
                    url, branch = source
                    src_path = os.path.join(ext_dir, label)
                    if not os.path.exists(src_path):
                        cmd = ["git", "clone", "--recursive", url, src_path]
                        if branch:
                            cmd[2:2] = ["-b", branch]
                        subprocess.run(cmd, check=True)
                    else:
                        subprocess.run(["git", "submodule", "update", "--init", "--recursive"],
                                       cwd=src_path, check=False)
                else:
                    src_path = source
                if not os.path.exists(src_path):
                    raise RuntimeError(f"source not found at {src_path}")
                if label == "o-voxel":
                    patch_ovoxel_windows_build(src_path)
                debug_print(f"Building extension: {label}")
                subprocess.run(
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
    input_type: EnumProperty(name="Input Type", items=[("PROMPT", "Prompt", ""), ("TEXT_BLOCK", "Text-Block", "")], default="TEXT_BLOCK")
    script: StringProperty(default="")
    scene_texts: EnumProperty(name="Text-Blocks", items=texts_callback, update=update_text_list)

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
    # loads after all images are generated.
    try:
        gen = 0
        seg = 0
        for raw in proc.stdout:
            line = raw.rstrip()
            if line:
                print(line)
            if "Loading pipeline" in line:
                set_progress("infer_2d", value=0.03, label="Loading image model")
            elif "Generating:" in line:
                gen += 1
                frac = 0.05 + 0.45 * (gen / max(total, 1))
                set_progress("infer_2d", value=frac, label=f"Generating {gen}/{total}")
            elif "Loading BiRefNet" in line:
                set_progress("infer_2d", value=0.50, label="Loading segmenter")
            elif "Segmenting:" in line:
                seg += 1
                frac = 0.50 + 0.45 * (seg / max(total, 1))
                set_progress("infer_2d", value=frac, label=f"Segmenting {seg}/{total}")
            elif "Done." in line:
                set_progress("infer_2d", value=0.99, label="Finalizing")
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
    # The subprocess phase is capped at 0.9 so the Blender-side per-asset
    # import/cleanup pass (see TRELLIS_OT_ConvertSelected.modal) visibly owns
    # the remaining 0.9-1.0 instead of the bar sitting frozen at ~99% while
    # that phase runs unseen.
    stage_re = re.compile(r"^(.*?):\s*(\d+)%\|")
    proc_re = re.compile(r"Processing\s+(\d+)\s*/\s*(\d+)")
    last_frac = 0.0
    cur_task, total_tasks = 1, 1
    try:
        for raw in proc.stdout:
            line = raw.rstrip()
            if line:
                print(line)
            if "Loading pipeline" in line:
                last_frac = max(last_frac, 0.02)
                set_progress("infer_3d", value=last_frac, label="Loading TRELLIS pipeline")
                continue
            m = proc_re.search(line)
            if m:
                cur_task, total_tasks = int(m.group(1)), max(int(m.group(2)), 1)
                continue
            sm = stage_re.match(line)
            if sm:
                stage_name, pct = sm.group(1).strip(), int(sm.group(2))
                for name, base, span in _3D_STAGE_WEIGHTS:
                    if name not in stage_name:
                        continue
                    local = base + span * (pct / 100.0)
                    frac = 0.9 * ((cur_task - 1 + local) / total_tasks)
                    label = _3D_STAGE_LABELS.get(name, name)
                    last_frac = max(last_frac, frac)
                    set_progress("infer_3d", value=last_frac, label=f"{label} (asset {cur_task}/{total_tasks})")
                    break
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
        if props.input_type == "PROMPT":
            lines = [context.scene.asset_prompt]
        else:
            text = bpy.data.texts.get(props.scene_texts)
            if text is None:
                self.report({'ERROR'}, "No text-block selected.")
                return {'CANCELLED'}
            lines = [l.body for l in text.lines if l.body.strip()]
        lines = [sanitize_text(l).strip() for l in lines]
        lines = [l for l in lines if l]
        if not lines:
            self.report({'ERROR'}, "No prompt text to generate from.")
            return {'CANCELLED'}

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
name_base = "{context.scene.asset_name or 'Asset'}"

def free_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# Two-phase design to keep VRAM low: the diffusion pipeline and the BiRefNet
# segmenter are NEVER resident at the same time. Phase 1 generates every image
# with only the pipeline loaded (and CPU-offloaded), then the pipeline is fully
# released; Phase 2 loads the segmenter and processes the images saved to disk.

# --- PHASE 1: generate all images (only the diffusion pipeline in VRAM) -------
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

raw_paths = []  # (index, path-to-raw-RGB-png)
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
        )
        _track_subprocess(self._process)
        self._reader = threading.Thread(target=_read_2d_progress, args=(self._process, len(lines)), daemon=True)
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
        scn = context.scene

        isolated_3d = f"""
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
        script_path = os.path.join(data_dir, "run_3d.py")
        with open(script_path, "w") as f: f.write(isolated_3d)
        set_progress("infer_3d", value=0.0, label="Starting...", active=True)
        # -u: unbuffered child stdout so progress markers arrive in real time.
        self._process = subprocess.Popen(
            [py, "-u", script_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        _track_subprocess(self._process)
        self._reader = threading.Thread(target=_read_3d_progress, args=(self._process,), daemon=True)
        self._reader.start()
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

# --- BAKE TEXTURE TO VERTEX COLORS ---

class TRELLIS_OT_BakeVertexColors(Operator):
    bl_idname = "object.bake_texture_to_vertex_colors"
    bl_label = "Bake Texture to Vertex Colors"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = [o for o in context.selected_objects if o.type == 'MESH']
        if not selected:
            self.report({'ERROR'}, "No mesh selected.")
            return {'CANCELLED'}

        scene = context.scene
        prev_engine = scene.render.engine
        scene.render.engine = 'CYCLES'
        try:
            for obj in selected:
                mesh = obj.data
                color_attr = mesh.color_attributes.get("Col")
                if color_attr is None:
                    color_attr = mesh.color_attributes.new(name="Col", type='BYTE_COLOR', domain='CORNER')
                mesh.color_attributes.active_color = color_attr
                context.view_layer.objects.active = obj
                obj.select_set(True)
                # DIFFUSE + COLOR-only pass filter bakes the flat albedo (the
                # image texture's color), ignoring scene lighting entirely —
                # target='VERTEX_COLORS' writes straight into the color
                # attribute above instead of a separate image.
                with context.temp_override(selected_editable_objects=[obj], active_object=obj, object=obj):
                    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'}, target='VERTEX_COLORS')
        except RuntimeError as e:
            self.report({'ERROR'}, f"Bake failed: {e}")
            return {'CANCELLED'}
        finally:
            scene.render.engine = prev_engine

        self.report({'INFO'}, f"Baked {len(selected)} mesh(es) to vertex colors.")
        return {'FINISHED'}

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
        layout.operator("object.trellis_convert", text="Convert Selected to 3D", icon='MOD_MESHDEFORM')
        draw_progress(layout, "infer_3d")
        layout.operator("object.bake_texture_to_vertex_colors", text="Bake Texture to Vertex Colors")

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

        col = box.column(align=True)
        col.label(text="Shape:")
        col.prop(scene, "trellis_shape_steps")
        col.prop(scene, "trellis_shape_guidance")
        col.prop(scene, "trellis_shape_guidance_rescale")

        col = box.column(align=True)
        col.label(text="Texture:")
        col.prop(scene, "trellis_tex_steps")
        col.prop(scene, "trellis_tex_guidance")
        col.prop(scene, "trellis_tex_guidance_rescale")

# --- REGISTRATION ---

classes = (
    Import_Text_Props, AssetGeneratorPreferences,
    ZIMAGE_OT_GenerateAsset, TRELLIS_OT_ConvertSelected, TRELLIS_OT_BakeVertexColors,
    InstallOperator, UninstallOperator,
    ZIMAGE_PT_MainPanel, TRELLIS_PT_SubPanel
)

# scene.trellis_* property name -> (PropertyType, kwargs). Defaults match the
# values this addon used before these became user-editable (see the
# isolated_3d pipe.run() call in TRELLIS_OT_ConvertSelected.execute), except
# where noted as the pipeline's own stock config.
_TRELLIS_SCENE_PROPS = {
    "trellis_show_settings": (BoolProperty, dict(name="Advanced Settings", default=False)),
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
    "trellis_shape_steps": (IntProperty, dict(name="Steps", default=28, min=1, max=50)),
    "trellis_shape_guidance": (FloatProperty, dict(name="Guidance Strength", default=7.5, min=0.0, max=20.0)),
    "trellis_shape_guidance_rescale": (FloatProperty, dict(name="Guidance Rescale", default=0.5, min=0.0, max=1.0)),
    "trellis_tex_steps": (IntProperty, dict(name="Steps", default=28, min=1, max=50)),
    # 2.0 (stock: 1.0 = CFG off) for stronger adherence to the input image.
    "trellis_tex_guidance": (FloatProperty, dict(name="Guidance Strength", default=2.0, min=0.0, max=20.0)),
    # 0.7 counters the color/saturation drift plain CFG introduces at
    # guidance_strength > 1 (stock tex default is 0.0, matching stock's
    # CFG-off guidance_strength of 1.0).
    "trellis_tex_guidance_rescale": (FloatProperty, dict(name="Guidance Rescale", default=0.7, min=0.0, max=1.0)),
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
