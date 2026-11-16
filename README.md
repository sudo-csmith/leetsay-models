# leetsay-models

The speech recognition files used by
[leetsay](https://github.com/sudo-csmith/leetsay), a free, local dictation app
for Windows. They are converted, in the open, from NVIDIA's original models.

leetsay downloads them only after you agree, and checks every file against a
SHA-256 fingerprint built into the app before using it. You do not need to
download anything from here yourself.

Please read the [Disclaimer](#disclaimer) before using these files.

## What is here

| Release | What it is | Licence |
|---|---|---|
| `parakeet-tdt-0.6b-v2-int8-r*` | NVIDIA Parakeet TDT 0.6B v2 (English), converted to ONNX, 8-bit | CC-BY-4.0 |
| `parakeet-tdt-0.6b-v3-int8-r*` | NVIDIA Parakeet TDT 0.6B v3 (25 European languages), converted to ONNX, 8-bit | CC-BY-4.0 |
| `silero-vad-v6.2.3` | Silero VAD, unchanged (detects pauses in speech) | MIT |

Each release lists its files' SHA-256 fingerprints (`SHA256SUMS.txt`) and how
it was made (`PROVENANCE.txt`).

## How the files are made

Nothing is made by hand. The workflow in
[`.github/workflows/convert.yml`](.github/workflows/convert.yml) runs on GitHub's
computers, where its log can be read:

1. It downloads NVIDIA's original model file from NVIDIA's own model page
   (linked under Credits), at a fixed version, and stops unless its SHA-256 is
   exactly the one NVIDIA published.
2. [`convert.py`](convert.py) exports it to ONNX with NVIDIA's own NeMo toolkit
   and reduces the weights to 8 bits with onnxruntime. The export steps follow
   the method documented by the [onnx-asr](https://github.com/istupakov/onnx-asr)
   project (MIT), the library leetsay uses to run these files.
3. The result goes into a draft release. It is published only after it has
   been checked against the earlier files (same text from the same
   recordings).

The pause-detection helper (Silero VAD) is not converted: it is the makers'
own file from their official release, checked against its known SHA-256.

## Credits and licences

- **Parakeet TDT 0.6B v2 and v3** by NVIDIA Corporation, licensed under
  [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Originals:
  [v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2),
  [v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3). The converted files
  here are shared under the same licence; the changes made are listed in
  [NOTICE.md](NOTICE.md). Full licence text: [LICENSE](LICENSE).
- **Silero VAD** by the Silero Team, under the MIT licence
  ([LICENSE-silero-vad.txt](LICENSE-silero-vad.txt)). Original:
  [snakers4/silero-vad](https://github.com/snakers4/silero-vad).
- The conversion scripts in this repository (`convert.py`, the workflow) are
  under the MIT licence, like leetsay
  ([LICENSE-scripts.txt](LICENSE-scripts.txt)).

Websites linked here are third-party websites that we do not run or control.

## Disclaimer

- **Not made, reviewed or endorsed by NVIDIA or the Silero Team.** This is an
  independent conversion. NVIDIA and Parakeet are trademarks of NVIDIA
  Corporation, named only to say where the originals come from.
- **No warranty.** The files are provided "as is", without warranty of any
  kind, express or implied, including fitness for a particular purpose,
  accuracy, and non-infringement. The original makers give no warranty either.
- **No liability.** To the fullest extent the law allows, the author of
  leetsay and of this repository is not liable for any claim, damages, loss or
  other liability arising from these files, the conversion, or their use. You
  use them entirely at your own risk.
- **The text comes from the model, not from us.** Speech recognition can
  mishear, leave out or add words, including words that were never said. The
  author does not control, and is not responsible for, the text the model
  produces. How well it works depends on the speaker's voice, accent and
  language, the microphone and background noise; NVIDIA's model cards
  describe the models' known limits. Do not rely on it where a wrong, missing
  or misplaced word could cause harm or loss, such as medical, legal,
  financial, safety or emergency use. It is not a medical or certified
  assistive device.
- **The conversion may differ from the original.** Exporting and reducing to
  8 bits can change results slightly. No promise is made that these files
  behave exactly like NVIDIA's originals.
- **Training data and the original model** are NVIDIA's. Questions about how
  the model was trained, or about its original licence, are for NVIDIA.
- **Your responsibility.** You are responsible for how you use these files,
  including any laws about recording or processing speech where you are, and
  for following the licences above if you share them further.
- **Official files only.** Only releases published in this repository are
  official. Copies offered anywhere else may have been changed; leetsay checks
  every file's SHA-256 and refuses any that do not match.
- **No obligation.** There is no promise of support, fixes, updates, or that
  these files stay available. Releases may be withdrawn at any time, for
  example if a problem is found; leetsay will then refuse them.
- **Limits of this disclaimer.** Nothing here limits any liability that
  cannot be limited by law. If any part cannot be enforced, the rest still
  applies.

## Problems

If you find a security problem in these files, please report it privately,
as described in leetsay's [SECURITY.md](https://github.com/sudo-csmith/leetsay/blob/main/SECURITY.md).
