# -*- mode: python ; coding: utf-8 -*-

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

# CUDA is opt-in to keep the default download small: set BABBLE_CUDA=1 (and build against a CUDA torch)
# to bundle the multi-GB NVIDIA runtime. Default Linux/macOS builds are slim CPU; Windows is DirectML.
WITH_CUDA = os.environ.get("BABBLE_CUDA", "0").lower() not in ("0", "", "false", "no")

# The trainer needs only torch / (torch_directml | CUDA) / timm / cv2 / numpy / onnx. The data-science
# stack (pandas, matplotlib, scipy, sklearn, pyarrow, pyzmq) gets pulled in as optional/transitive
# imports and each ships its OWN older msvcp140.dll on Windows — one of those stale MSVC runtimes
# shadows the system one and makes onnx's C extension (built against a newer runtime) fail with "DLL
# initialization routine failed". Excluding them fixes that AND massively slims the build. onnxscript/
# onnxruntime are only the dynamo exporter + export(check=True) path (we use the legacy exporter,
# check=False) and also crash PyInstaller's analysis, so they go too.
EXCLUDES = ['onnxscript', 'onnxruntime', 'pandas', 'matplotlib', 'scipy', 'sklearn',
            'pyarrow', 'zmq', 'IPython', 'triton']  # triton is torch.compile-only (unused), ~685MB

datas = []
binaries = []
hiddenimports = ['onnx', 'onnx.onnx_cpp2py_export']

if sys.platform == 'win32':
    # Windows: DirectML backend (any-vendor GPU). One-file build (the DirectML payload is small).
    binaries += collect_dynamic_libs('torch_directml')
    binaries += [("babble_data_win/opencv_world4100.dll", '.')]
    ONEFILE = True
else:
    # Linux/macOS: bundle torch + timm/torchvision (timm builds the backbone from a name string, so its
    # registry submodules must be collected explicitly). NVIDIA CUDA runtime libs are added only when
    # WITH_CUDA (collect_all('nvidia') is a no-op against a CPU torch anyway). One-DIR build: a multi-GB
    # CUDA payload as one-file would re-extract to a temp dir on every launch.
    pkgs = ['torch', 'torchvision', 'timm'] + (['nvidia'] if WITH_CUDA else [])
    for pkg in pkgs:
        try:
            d, b, h = collect_all(pkg)
            datas += d
            binaries += b
            hiddenimports += h
        except Exception as e:
            print(f"[spec] collect_all({pkg!r}) skipped: {e}")
    ONEFILE = not WITH_CUDA   # one-file for the slim CPU build; one-dir only for the multi-GB CUDA payload

if not WITH_CUDA:
    try:
        import torch
        if getattr(torch.version, "cuda", None):
            print("[spec] WARNING: BABBLE_CUDA is off but a CUDA torch is installed; build will be bloated "
                  "and may not run. Use a CPU torch (requirements.txt) or set BABBLE_CUDA=1.", flush=True)
    except Exception:
        pass

a = Analysis(
    ['train_frontend.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

if ONEFILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='BabbleTrainer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
else:
    # One-dir: thin launcher + a BabbleTrainer/ payload dir. UPX is disabled — it is not assumed
    # present and it corrupts the CUDA .so files.
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='BabbleTrainer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name='BabbleTrainer',
    )
