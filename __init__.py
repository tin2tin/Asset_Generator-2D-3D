bl_info = {
    "name": "2D Asset Generator & Trellis 3D (Async Pro)",
    "author": "tintwotin",
    "version": (5, 4),
    "blender": (3, 0, 0),
    "category": "3D View",
    "location": "3D Editor > Sidebar > 2D Asset",
    "description": "Async Z-Image 2D and TRELLIS.2 3D with Fixed CUDA Extension Loading.",
}

import bpy
import os
import re
import subprocess
import sys
import math
import venv
import importlib
import platform
import gc
import shutil
import json
import threading
import queue
import stat
from os.path import join
from mathutils import Vector
from typing import Optional
from bpy.types import Operator, PropertyGroup, Panel, AddonPreferences
from bpy.props import StringProperty, EnumProperty, PointerProperty, BoolProperty

# --- CRITICAL ENVIRONMENT PROTECTION ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

DEBUG = True

def debug_print(*args, **kwargs):
    """Console logging for the generation process."""
    if DEBUG:
        print("[2D/3D Asset Pro Log] ", *args, **kwargs)

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

def venv_path() -> str:
    return os.path.normpath(os.path.join(addon_script_path(), "virtual_dependencies"))

def python_exec() -> str:
    ext = ".exe" if os.name == 'nt' else ""
    bin_dir = 'Scripts' if os.name == 'nt' else 'bin'
    py = os.path.join(venv_path(), bin_dir, f'python{ext}')
    return py if os.path.exists(py) else sys.executable

def create_venv():
    path = venv_path()
    if not os.path.exists(path):
        debug_print(f"Creating virtual environment: {path}")
        venv.create(path, with_pip=True)
        subprocess.run([python_exec(), '-m', 'pip', 'install', '--upgrade', 'pip'], capture_output=True)

def activate_virtualenv():
    create_venv()
    path = venv_path()
    if platform.system() == 'Windows':
        sp = os.path.join(path, 'Lib', 'site-packages') 
    else:
        sp = os.path.join(path, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
    
    if os.path.exists(sp):
        if sp in sys.path: sys.path.remove(sp)
        sys.path.insert(0, sp)
    
    repo_path = os.path.normpath(os.path.join(addon_script_path(), "TRELLIS_REPO"))
    if os.path.exists(repo_path):
        if repo_path in sys.path: sys.path.remove(repo_path)
        sys.path.insert(0, repo_path)
    
    importlib.invalidate_caches()
    return True

# --- PRO INSTALLATION LOGIC (FIXED FOR CUMESH) ---

def install_pro_2d_core():
    activate_virtualenv()
    py = python_exec()
    debug_print("Installing 2D Pro core with SIMD support...")
    
    # 1. Handle Pillow-SIMD
    subprocess.run([py, "-m", "pip", "uninstall", "-y", "Pillow"], capture_output=True)
    subprocess.run([py, "-m", "pip", "install", "pillow-simd"], capture_output=True)
    
    # 2. Base 2D requirements (Pinned for stability)
    pkgs = [
        "numpy==1.26.4", 
        "git+https://github.com/huggingface/diffusers.git", 
        "transformers>=4.45.0", "accelerate>=1.0.0", 
        "scipy", "tqdm", "sympy==1.13.1", "einops", "kornia", "timm", "easydict"
    ]
    for pkg in pkgs:
        subprocess.run([py, "-m", "pip", "install", pkg, "--upgrade"])
    
    if gfx_device() == "cuda":
        subprocess.run([py, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"])

def build_trellis_pro_system():
    activate_virtualenv()
    py = python_exec()
    addon_dir = addon_script_path()
    repo_path = os.path.normpath(os.path.join(addon_dir, "TRELLIS_REPO"))

    debug_print("Verifying TRELLIS.2 Submodules...")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.9;9.0"
    os.environ["FORCE_CUDA"] = "1"
    
    # Check if submodules exist
    if os.path.exists(repo_path):
        sub_check = os.path.join(repo_path, "extensions", "cumesh", "setup.py")
        if not os.path.exists(sub_check):
            debug_print("Cumesh source missing. Clearing repo for fresh clone...")
            shutil.rmtree(repo_path, onerror=remove_readonly)

    if not os.path.exists(repo_path):
        subprocess.run(["git", "clone", "--recurse-submodules", "https://github.com/microsoft/TRELLIS.2.git", repo_path], check=True)
    else:
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_path, check=True)
    
    nvpath = os.path.normpath(os.path.join(addon_dir, "nvdiffrec"))
    if not os.path.exists(nvpath):
        subprocess.run(["git", "clone", "https://github.com/NVlabs/nvdiffrec.git", nvpath], check=True)

    # 1. Base requirements
    pro_deps = ["numpy==1.26.4", "ninja", "setuptools", "wheel", "trimesh", "imageio", "imageio-ffmpeg", "opencv-python", "easydict", "einops", "onnxruntime-gpu", "scikit-image"]
    for dep in pro_deps:
        subprocess.run([py, "-m", "pip", "install", dep, "--upgrade"])
    if os.name == 'nt': subprocess.run([py, "-m", "pip", "install", "triton-windows"])

    # 2. Build Extensions (Using Direct Install mode for Windows linking)
    exts = [
        os.path.join(repo_path, "extensions", "o-voxel"),
        os.path.join(repo_path, "extensions", "flexgemm"),
        os.path.join(repo_path, "extensions", "cumesh"),
        os.path.join(repo_path, "external", "nvdiffrast"),
        nvpath
    ]

    for ext in exts:
        if os.path.exists(ext):
            if any(os.path.exists(os.path.join(ext, f)) for f in ["setup.py", "pyproject.toml"]):
                debug_print(f"Building/Linking: {os.path.basename(ext)}")
                # -e (editable) ensures Windows links the binary correctly in the venv
                subprocess.run([py, "-m", "pip", "install", "-e", "."], cwd=ext, check=True)
    
    debug_print("Pro build completed.")

# --- UTILS: NAMING & TEXT ---

def texts_callback(self, context): return [(t.name, t.name, "") for t in bpy.data.texts]

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

def get_unique_file_name(base_path):
    if not os.path.exists(base_path): return base_path
    base, ext = os.path.splitext(base_path)
    c = 1
    p = f"{base}_{c}{ext}"
    while os.path.exists(p):
        c += 1
        p = f"{base}_{c}{ext}"
    return p

# --- ASYNC OPERATOR: 2D GENERATOR ---

class ZIMAGE_OT_GenerateAsset(Operator):
    bl_idname = "object.generate_asset"
    bl_label = "Generate 2D Asset"
    bl_options = {"REGISTER"}

    _process = None
    _timer = None
    _result_file = ""

    def modal(self, context, event):
        if event.type == 'TIMER' and self._process.poll() is not None:
            self.process_finish(context)
            return {'FINISHED'}
        return {'PASS_THROUGH'}

    def process_finish(self, context):
        context.window_manager.event_timer_remove(self._timer)
        if os.path.exists(self._result_file):
            with open(self._result_file, 'r') as f: results = json.load(f)
            for item in results:
                bpy.ops.mesh.primitive_plane_add(size=1, location=(context.scene.cursor.location.x + item["offset"], context.scene.cursor.location.y, context.scene.cursor.location.z))
                obj = context.object
                obj.name = item["name"]
                from PIL import Image
                img_pil = Image.open(item["path"])
                obj.scale = (img_pil.size[0] / img_pil.size[1], 1, 1)
                mat = bpy.data.materials.new(name=f"Mat_{obj.name}")
                mat.use_nodes = True
                mat.blend_method = 'HASHED'
                bsdf, tex = mat.node_tree.nodes.get("Principled BSDF"), mat.node_tree.nodes.new("ShaderNodeTexImage")
                tex.image = bpy.data.images.load(item["path"])
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
        lines = [context.scene.asset_prompt] if props.input_type == "PROMPT" else \
                [l.body for l in bpy.data.texts[props.scene_texts].lines if l.body.strip()]
        
        data_dir = os.path.join(bpy.utils.user_resource("DATAFILES"), "2D_Async_Queue")
        os.makedirs(data_dir, exist_ok=True)
        self._result_file = os.path.join(data_dir, "results.json")
        sp_folder = os.path.join(venv_path(), 'Lib' if os.name == 'nt' else 'lib', 'site-packages')

        isolated_script = f"""
import os, sys, torch, json, numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects
sp = r"{sp_folder}"
sys.path.insert(0, sp)
if hasattr(os, "add_dll_directory"): os.add_dll_directory(sp)
from diffusers import ZImagePipeline
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
lines = {json.dumps(lines)}
name_base = "{context.scene.asset_name or 'Asset'}"
pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16).to("cuda")
birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to("cuda")
trans = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
output_data = []
offset_accum = 0.0
for i, line in enumerate(lines):
    img = pipe(prompt="neutral background, "+line, height=1024, width=1024, num_inference_steps=9, guidance_scale=0.0).images[0]
    src = img.convert("RGB")
    with torch.no_grad():
        m_t = birefnet(trans(src).unsqueeze(0).to("cuda"))[-1].sigmoid().cpu()
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
with open(r"{self._result_file}", "w") as f: json.dump(output_data, f)
"""
        script_path = os.path.join(data_dir, "run_2d.py")
        with open(script_path, "w") as f: f.write(isolated_script)
        self._process = subprocess.Popen([py, script_path])
        self._timer = context.window_manager.event_timer_add(1.0, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

# --- ASYNC OPERATOR: 3D CONVERTER (FIXED FOR BINARY LOADING) ---

class TRELLIS_OT_ConvertSelected(Operator):
    bl_idname = "object.trellis_convert"
    bl_label = "Convert selected to 3D"
    
    _process = None
    _timer = None
    _tasks = []

    def modal(self, context, event):
        if event.type == 'TIMER' and self._process.poll() is not None:
            context.window_manager.event_timer_remove(self._timer)
            for t in self._tasks:
                glb_p = t["path"].replace(".png", "_3d.glb")
                if os.path.exists(glb_p):
                    bpy.ops.import_scene.gltf(filepath=glb_p)
                    context.active_object.location, context.active_object.location.y = t["loc"], t["loc"].y + 2.5
            return {'FINISHED'}
        return {'PASS_THROUGH'}

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
        py = python_exec()
        data_dir = os.path.join(bpy.utils.user_resource("DATAFILES"), "3D_Async_Queue")
        os.makedirs(data_dir, exist_ok=True)
        task_json = os.path.join(data_dir, "tasks.json")
        with open(task_json, 'w') as f: json.dump([{"path": t["path"]} for t in self._tasks], f)

        sp = os.path.join(venv_path(), 'Lib' if os.name == 'nt' else 'lib', 'site-packages')
        bin_dir = os.path.join(venv_path(), 'Scripts')
        repo = os.path.normpath(os.path.join(addon_script_path(), "TRELLIS_REPO"))

        isolated_3d = f"""
import os, sys, json
from PIL import Image

sp, bin_dir, repo = r"{sp}", r"{bin_dir}", r"{repo}"
sys.path.insert(0, sp)
sys.path.insert(0, repo)

# CRITICAL WINDOWS FIX: Register DLL directories for C++ Extensions
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(sp)
    os.add_dll_directory(bin_dir)
    # Recursively find any folder containing .pyd and add it to DLL search
    for r, d, f in os.walk(sp):
        if any(file.endswith('.pyd') for file in f):
            os.add_dll_directory(r)

# MANDATORY: Import Torch FIRST to load CUDA libraries for cumesh/flexgemm
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

with open(r"{task_json}", "r") as f: tasks = json.load(f)
pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B").cuda()

for t in tasks:
    img = Image.open(t["path"]).convert("RGB")
    with torch.no_grad():
        mesh = pipe.run(img)[0]
        mesh.simplify(16777216)
    glb_p = t["path"].replace(".png", "_3d.glb")
    glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=100000, texture_size=2048, remesh=True)
    glb.export(glb_p)
"""
        script_path = os.path.join(data_dir, "run_3d.py")
        with open(script_path, "w") as f: f.write(isolated_3d)
        self._process = subprocess.Popen([py, script_path])
        self._timer = context.window_manager.event_timer_add(1.0, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

# --- PRO ASYNC INSTALLERS ---

class Install2DOperator(Operator):
    bl_idname = "virtual_dependencies.install_pro_2d"
    bl_label = "Install Pro 2D Dependencies"
    _thread = None
    _timer = None
    def modal(self, context, event):
        if event.type == 'TIMER' and not self._thread.is_alive():
            context.window_manager.event_timer_remove(self._timer)
            self.report({'INFO'}, "2D Core ready.")
            return {'FINISHED'}
        return {'PASS_THROUGH'}
    def execute(self, context):
        self._thread = threading.Thread(target=install_pro_2d_core)
        self._thread.start()
        self._timer = context.window_manager.event_timer_add(1.0, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class Install3DOperator(Operator):
    bl_idname = "virtual_dependencies.install_pro_3d"
    bl_label = "Build Pro 3D CUDA Kernels"
    _thread = None
    _timer = None
    def modal(self, context, event):
        if event.type == 'TIMER' and not self._thread.is_alive():
            context.window_manager.event_timer_remove(self._timer)
            self.report({'INFO'}, "3D build finished.")
            return {'FINISHED'}
        return {'PASS_THROUGH'}
    def execute(self, context):
        self._thread = threading.Thread(target=build_trellis_pro_system)
        self._thread.start()
        self._timer = context.window_manager.event_timer_add(1.0, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class AssetGeneratorPreferences(AddonPreferences):
    bl_idname = __name__
    def draw(self, context):
        layout = self.layout
        layout.label(text="Pro Isolated Environment Management", icon='SYSTEM')
        row = layout.row()
        row.operator("virtual_dependencies.install_pro_2d")
        row.operator("virtual_dependencies.install_pro_3d")
        row.operator("virtual_dependencies.uninstall_all", text="Nuclear Wipe")

class UninstallOperator(Operator):
    bl_idname = "virtual_dependencies.uninstall_all"
    bl_label = "Uninstall Everything"
    def execute(self, context):
        shutil.rmtree(venv_path(), ignore_errors=True)
        return {'FINISHED'}

# --- UI PANELS ---

class ZIMAGE_PT_MainPanel(Panel):
    bl_label = "2D Asset Generator"
    bl_idname = "VIEW3D_PT_zimage"
    bl_space_type, bl_region_type, bl_category = "VIEW_3D", "UI", "2D Asset"
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
        box.operator("object.generate_asset", text="Generate 2D Assets")

class TRELLIS_PT_SubPanel(Panel):
    bl_label = "3D Trellis Conversion"
    bl_idname = "VIEW3D_PT_trellis"
    bl_space_type, bl_region_type, bl_category = "VIEW_3D", "UI", "2D Asset"
    def draw(self, context):
        self.layout.operator("object.trellis_convert", text="Convert Selected to 3D", icon='MOD_MESHDEFORM')

# --- REGISTRATION ---

classes = (
    Import_Text_Props, AssetGeneratorPreferences,
    ZIMAGE_OT_GenerateAsset, TRELLIS_OT_ConvertSelected,
    Install2DOperator, Install3DOperator, UninstallOperator,
    ZIMAGE_PT_MainPanel, TRELLIS_PT_SubPanel
)

def register():
    for cls in classes: bpy.utils.register_class(cls)
    bpy.types.Scene.import_text = PointerProperty(type=Import_Text_Props)
    bpy.types.Scene.asset_prompt = StringProperty(name="Prompt", default="Goofy monster character sheet, multiple poses, white background")
    bpy.types.Scene.asset_name = StringProperty(name="Asset Name", default="Asset", update=get_unique_name)

def unregister():
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
    del bpy.types.Scene.import_text
    del bpy.types.Scene.asset_prompt
    del bpy.types.Scene.asset_name

if __name__ == "__main__": register()
