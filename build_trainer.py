"""Freeze and hash the matched trainer; never overwrite an existing package."""
import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from capture_manifest import sha256
from training_artifacts import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--assets", required=True, help="Existing trusted trainer checkpoint directory")
    args = parser.parse_args()
    root, output, assets = Path(__file__).resolve().parent, Path(args.output).resolve(), Path(args.assets).resolve()
    if output.exists():
        raise ValueError("Choose a new package output directory")
    pretrained = ["model_best.pt", "gaze_model_best.pt", "footage_unlabeled.npz"]
    for name in pretrained:
        if not (assets / name).is_file():
            raise FileNotFoundError(assets / name)
    if (assets / "presence_model.pt").is_file():  # optional headset-worn output
        pretrained.append("presence_model.pt")
    with tempfile.TemporaryDirectory(prefix="babble-freeze-") as temporary:
        tmp = Path(temporary)
        subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                        "--distpath", str(tmp / "dist"), "--workpath", str(tmp / "build"),
                        str(root / "BabbleTrainer.spec")], cwd=root, check=True)
        exe_name = "BabbleTrainer.exe" if sys.platform == "win32" else "BabbleTrainer"
        exe = tmp / "dist" / exe_name
        capabilities = json.loads(subprocess.check_output([str(exe), "--capabilities"], text=True))
        if (capabilities.get("protocol_version") != 1 or capabilities.get("capture_schema_version") != 1
                or capabilities.get("review_schema_version") != 1):
            raise ValueError("Frozen trainer capability mismatch")
        output.mkdir(parents=True)
        shutil.copy2(exe, output / exe_name)
        for name in pretrained:
            shutil.copy2(assets / name, output / name)
    git = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    revision = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(git + ["status", "--porcelain"], text=True).strip())
    source_files = {p.relative_to(root).as_posix(): sha256(p)
                    for p in sorted(root.glob("*.py"))}
    source_files["BabbleTrainer.spec"] = sha256(root / "BabbleTrainer.spec")
    dependencies = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    manifest = dict(schema_version=1, protocol_version=1, capture_schema_version=1, review_schema_version=1,
        source_revision=revision, source_dirty=dirty, source_files=source_files,
        files={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
        dependencies=dependencies, pretrained={name: sha256(output / name) for name in pretrained},
        capabilities=capabilities, python=sys.version,
        validation=["frozen --capabilities"], hardware_validation="not_performed")
    write_json(output / "trainer-manifest.json", manifest)
    print(f"Matched trainer package: {output}")


if __name__ == "__main__":
    main()
