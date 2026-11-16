# Notice

These files are a converted version of a speech recognition model made by
NVIDIA. They are published by the leetsay project (GitHub: sudo-csmith), not by
NVIDIA.

- **Original work:** Parakeet TDT 0.6B (v2 or v3, as named in the release),
  by NVIDIA Corporation.
- **Source:** https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 or
  https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 (the exact commit and the
  original file's SHA-256 are in PROVENANCE.txt).
- **Licence:** Creative Commons Attribution 4.0 International (CC-BY-4.0),
  https://creativecommons.org/licenses/by/4.0/ (full text in
  LICENSE-CC-BY-4.0.txt). These converted files are shared under the same
  licence.
- **Changes made:** exported to the ONNX format with NVIDIA's NeMo toolkit;
  weights reduced to 8 bits (dynamic quantization with onnxruntime); a word
  list (vocab.txt) and a settings file (config.json) written from the model's
  own settings. Nothing was retrained or fine-tuned. The steps are in
  convert.py and PROVENANCE.txt.
- **No endorsement:** NVIDIA has not reviewed, approved or endorsed this
  conversion, leetsay, or its author. NVIDIA and Parakeet are trademarks of
  NVIDIA Corporation, named here only to say where the model comes from.
- **No warranty:** as the licence says, the model is provided "as is", without
  warranties of any kind, by NVIDIA and by the leetsay project. See the
  disclaimer at https://github.com/sudo-csmith/leetsay-models#disclaimer.
