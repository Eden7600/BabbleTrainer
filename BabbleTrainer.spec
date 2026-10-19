# -*- mode: python ; coding: utf-8 -*-

Binaries = []

import sys

if sys.platform == 'win32':
    import importlib.util
    from PyInstaller.utils.hooks import collect_dynamic_libs
    # DirectML builds bundle its DLLs; CUDA builds get theirs from torch's own hooks.
    if importlib.util.find_spec('torch_directml'):
        Binaries = collect_dynamic_libs('torch_directml')
    # torchvision >= 0.29 ships its ops as *_stable.pyd, which PyInstaller's hook misses
    # ("operator torchvision::nms does not exist"); bundle its native files explicitly.
    from pathlib import Path
    vision = Path(importlib.util.find_spec('torchvision').origin).parent
    Binaries += [(str(f), 'torchvision') for f in vision.iterdir() if f.suffix in ('.pyd', '.dll')]
    print(Binaries)

a = Analysis(
    ['train_frontend.py'],
    pathex=[],
    binaries=Binaries,
    datas=[],
    hiddenimports=['onnx', 'onnx.onnx_cpp2py_export'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # The trainer needs only torch / torch_directml / timm / cv2 / numpy / onnx. The data-science stack
    # (pandas, matplotlib, scipy, sklearn, pyarrow, pyzmq) gets pulled in as optional/transitive imports
    # and each ships its OWN older msvcp140.dll — one of those stale MSVC runtimes shadows the system one
    # and makes onnx's C extension (built against a newer runtime) fail with "DLL initialization routine
    # failed". Excluding them fixes that AND massively slims the exe. onnxscript/onnxruntime are only the
    # dynamo exporter + export(check=True) path (we use the legacy exporter, check=False) and also crash
    # PyInstaller's analysis, so they go too.
    excludes=['onnxscript', 'onnxruntime', 'pandas', 'matplotlib', 'scipy', 'sklearn',
              'pyarrow', 'zmq', 'IPython'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# One folder (exe + _internal) rather than one file: CUDA torch is several GB, and a one-file exe
# would unpack all of it on every training run.
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
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='BabbleTrainer')
