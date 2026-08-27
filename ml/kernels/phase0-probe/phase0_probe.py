"""DEFER phase 0 launcher. This file should almost never change.

All the logic lives in the attached `defer-code` dataset, not here. The reason
is a Kaggle constraint: the accelerator choice (T4 vs P100) cannot be set
through the API, and every `kaggle kernels push` resets the notebook to the
default P100 -- which cannot run this PyTorch at all, so the run dies. Setting
it back is a manual click in the browser.

So: push this stub once, set the GPU once by hand, and from then on ship changes
with `kaggle datasets version` instead. That updates the code without touching
the notebook or its settings. Re-running is one Save & Run All.
"""
import os
import sys

code_dir = None
for root, _dirs, files in os.walk("/kaggle/input"):
    if "phase0.py" in files and "metrics.py" in files:
        code_dir = root
        break

if code_dir is None:
    # Attached datasets do not mount at a predictable flat path, so print the
    # real tree. That one listing is the only signal Kaggle gives you.
    print("/kaggle/input actually contains:")
    for root, _dirs, files in os.walk("/kaggle/input"):
        if root.count(os.sep) - "/kaggle/input".count(os.sep) > 3:
            continue
        print(f"  {root}  ->  {files[:6]}")
    sys.exit(
        "Could not find phase0.py. Attach johnandreimartinez/defer-code to this "
        "notebook, or publish a new dataset version if the code just changed."
    )

sys.path.insert(0, code_dir)
import phase0  # noqa: E402

phase0.main()
