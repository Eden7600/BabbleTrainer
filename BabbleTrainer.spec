# -*- mode: python ; coding: utf-8 -*-

Binaries = []

import sys

if sys.platform == 'win32':
    import os
    from PyInstaller.utils.hooks import collect_dynamic_libs
    Binaries = collect_dynamic_libs('torch_directml') + [("babble_data_win/opencv_world4100.dll", '.'),]
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
