"""DEFER phase 0.5 launcher: the free baselines. This file should never change.

All the logic lives in the attached `defer-code` dataset, not here. A
`kaggle kernels push` replaces the notebook and starts a fresh version, while a
`kaggle datasets version` swaps the code under a notebook that stays put -- so
shipping logic through the dataset leaves the notebook, its settings and its run
history undisturbed.

Update the code with:  python ml/kernels/publish.py --push
Then Save & Run All here. Nothing about this file needs touching.
"""
import os
import sys

code_dir = None
for root, _dirs, files in os.walk("/kaggle/input"):
    if "phase1.py" in files and "metrics.py" in files:
        code_dir = root
        break

if code_dir is None:
    # Attached datasets do not mount at a predictable flat path, so print the
    # real tree. That one listing is the only signal Kaggle gives you.
    print("/kaggle/input actually contains:")
    for root, _dirs, files in os.walk("/kaggle/input"):
        if root.count(os.sep) - "/kaggle/input".count(os.sep) > 3:
            continue
        print(f"  {root}  ->  {files[:8]}")
    sys.exit(
        "Could not find phase1.py. Attach johnandreimartinez/defer-code to this "
        "notebook, or publish a new dataset version if the code just changed."
    )

sys.path.insert(0, code_dir)
import phase1  # noqa: E402

phase1.main()
