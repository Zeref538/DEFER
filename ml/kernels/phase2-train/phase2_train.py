"""DEFER phase 2 launcher: LoRA training, two seeds. This file should never change.

All the logic lives in the attached `defer-code` dataset. Update it with
`python ml/kernels/publish.py --push`, then Save & Run All here.

The pip line is the one thing this stub does that the phase 1 stub does not.
Kaggle's image ships transformers and torch but not always a peft/bitsandbytes
pair new enough to load a 4-bit Llama, and a version mismatch surfaces as a
confusing shape error deep inside the model rather than as a missing import.
Pinning them here keeps that failure at second 30 instead of minute 9.
"""
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "peft>=0.11", "bitsandbytes>=0.43", "accelerate>=0.30"],
               check=False)

code_dir = None
for root, _dirs, files in os.walk("/kaggle/input"):
    if "phase2.py" in files and "train.py" in files:
        code_dir = root
        break

if code_dir is None:
    print("/kaggle/input actually contains:")
    for root, _dirs, files in os.walk("/kaggle/input"):
        if root.count(os.sep) - "/kaggle/input".count(os.sep) > 3:
            continue
        print(f"  {root}  ->  {files[:8]}")
    sys.exit(
        "Could not find phase2.py. Attach johnandreimartinez/defer-code to this "
        "notebook, or publish a new dataset version if the code just changed."
    )

sys.path.insert(0, code_dir)
import phase2  # noqa: E402

phase2.main()
