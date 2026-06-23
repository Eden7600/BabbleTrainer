# -*- mode: python ; coding: utf-8 -*-

Binaries = []

import sys
import platform

if sys.platform == 'win32':
    import os
    from PyInstaller.utils.hooks import collect_dynamic_libs
    Binaries = collect_dynamic_libs('torch_directml') + [("babble_data_win/opencv_world4100.dll", '.'),]
    print(Binaries)
elif sys.platform.startswith('linux') and platform.machine() == 'x86_64':
    # toss in the pytorch_dlprim backend: pt_ocl.so and its libOpenCL.so.1 dep.
    from PyInstaller.utils.hooks import collect_dynamic_libs
    # default search_patterns is lib*.so, the backend file is pt_ocl.so
    Binaries = collect_dynamic_libs('pytorch_ocl', search_patterns=['*.so'])
    print(Binaries)

a = Analysis(
    ['train_frontend.py'],
    pathex=[],
    binaries=Binaries,
    datas=[],
    # pytorch_ocl is imported lazily (device.py), so PyInstaller's static scan misses it; force it in
    # so its __init__ (backend registration) and pt_ocl.so are bundled. Absent on non-x86_64-Linux ->
    # PyInstaller just warns and skips, the binary stays CPU/DirectML.
    hiddenimports=['onnx', 'onnx.onnx_cpp2py_export', 'pytorch_ocl', 'pytorch_ocl.pt_ocl'],
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
    # compressing the ocl backend causes failed kernel load at runtime for some reason 
    upx_exclude=['pt_ocl.so', 'libOpenCL.so.1'],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
