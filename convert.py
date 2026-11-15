"""Convert one of NVIDIA's Parakeet speech models to the ONNX files leetsay uses.

Run by .github/workflows/convert.yml on GitHub's computers, in the open, so
anyone can see exactly how the published files were made. Steps:

1. Download NVIDIA's original model file (.nemo) from NVIDIA's own account on
   huggingface.co, at a fixed version (commit), and record its SHA-256.
2. Export it to ONNX with NVIDIA's own NeMo toolkit (model.export).
3. Write vocab.txt (the model's word pieces) and config.json.
4. Shrink the two networks to 8-bit weights with onnxruntime's standard
   dynamic quantization (QUInt8), as the files leetsay used before were made.
5. Write SHA256SUMS.txt and PROVENANCE.txt for everything published.

Usage: python convert.py <nvidia/repo> <commit> <SHA-256 of the .nemo> <output folder>
"""
import hashlib
import json
import os
import gc
import platform
import shutil
import subprocess
import sys
from importlib import metadata


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def report(step):
    """Free disk space, so a failed run shows whether the disk was the cause."""
    for place in sorted({os.path.abspath(os.environ.get("CONVERT_WORK", ".")), "/"}):
        try:
            print("[%s] free on %s: %.1f GB" % (step, place, shutil.disk_usage(place).free / 1e9),
                  flush=True)
        except OSError:
            pass


# Run in a fresh process for each part, so the memory the export used is
# fully given back before the large model is loaded again for shrinking.
_QUANTIZE = """
import sys
from onnxruntime.quantization import QuantType, quantize_dynamic
quantize_dynamic(sys.argv[1], sys.argv[2], weight_type=QuantType.QUInt8)
"""


def main():
    repo, commit, expected, out = sys.argv[1], sys.argv[2], sys.argv[3].lower(), sys.argv[4]
    if not repo.startswith("nvidia/"):
        sys.exit("Only NVIDIA's own originals are converted here.")
    if len(commit) != 40:
        sys.exit("Give the full 40-character commit, so the source is fixed.")
    os.makedirs(out, exist_ok=True)
    # The large working files go on the bigger disk when the workflow gives one.
    work = os.environ.get("CONVERT_WORK") or os.path.join(out, "_work")
    os.makedirs(work, exist_ok=True)

    # 1. NVIDIA's original, at exactly this commit.
    from huggingface_hub import hf_hub_download
    name = repo.split("/", 1)[1]
    source = hf_hub_download(repo, name + ".nemo", revision=commit)
    source_sha = sha256(source)
    print("Source:", repo, commit, name + ".nemo", source_sha, flush=True)
    if source_sha != expected:
        # Not the file NVIDIA published at that commit: stop, publish nothing.
        sys.exit("The original's SHA-256 is not the one NVIDIA published: %s" % source_sha)

    report("original downloaded")

    # 2. Export with NVIDIA's own toolkit.
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.restore_from(source, map_location="cpu")
    model.eval()
    # Loaded, and its SHA-256 is recorded: the file itself is no longer needed.
    os.remove(os.path.realpath(source))
    report("original loaded and removed")
    model.export(os.path.join(work, "model.onnx"))
    print("Exported:", sorted(f for f in os.listdir(work) if f.endswith(".onnx")), flush=True)
    report("exported")

    # 3. The word pieces and the settings onnx-asr reads.
    with open(os.path.join(out, "vocab.txt"), "w", encoding="utf-8", newline="\n") as f:
        for i, token in enumerate([*model.tokenizer.vocab, "<blk>"]):
            f.write(f"{token} {i}\n")
    config = {
        "model_type": "nemo-conformer-tdt",
        "features_size": int(model.cfg.preprocessor.features),
        "subsampling_factor": int(model.cfg.encoder.subsampling_factor),
    }
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(config, f, indent=4)
        f.write("\n")
    del model, nemo_asr
    gc.collect()

    # 4. 8-bit weights (onnxruntime's standard dynamic quantization).
    for part in ("decoder_joint-model", "encoder-model"):
        subprocess.run([sys.executable, "-c", _QUANTIZE, os.path.join(work, part + ".onnx"),
                        os.path.join(out, part + ".int8.onnx")], check=True)
        print("Quantized:", part, flush=True)
        report("quantized " + part)

    # 5. Fingerprints and the record of how it was made.
    published = ["config.json", "vocab.txt", "encoder-model.int8.onnx",
                 "decoder_joint-model.int8.onnx"]
    with open(os.path.join(out, "SHA256SUMS.txt"), "w", encoding="ascii", newline="\n") as f:
        for item in published:
            f.write("%s  %s\n" % (sha256(os.path.join(out, item)), item))
    packages = ["nemo_toolkit", "torch", "onnx", "onnxruntime", "huggingface_hub"]
    with open(os.path.join(out, "PROVENANCE.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("Converted from NVIDIA's original model, unchanged apart from the steps below.\n\n")
        f.write("Source:     https://huggingface.co/%s\n" % repo)
        f.write("Commit:     %s\n" % commit)
        f.write("File:       %s.nemo\n" % name)
        f.write("SHA-256:    %s\n" % source_sha)
        f.write("Licence:    CC-BY-4.0 (Creative Commons Attribution 4.0 International)\n\n")
        f.write("Changes made:\n")
        f.write("- Exported to ONNX with NVIDIA NeMo's model.export().\n")
        f.write("- Weights shrunk to 8 bits with onnxruntime quantize_dynamic (QUInt8).\n")
        f.write("- vocab.txt and config.json written from the model's own settings.\n\n")
        f.write("Made with: Python %s, %s\n" % (platform.python_version(), platform.platform()))
        for package in packages:
            try:
                f.write("  %s %s\n" % (package, metadata.version(package)))
            except metadata.PackageNotFoundError:
                pass
    print(open(os.path.join(out, "SHA256SUMS.txt")).read(), flush=True)


if __name__ == "__main__":
    main()
