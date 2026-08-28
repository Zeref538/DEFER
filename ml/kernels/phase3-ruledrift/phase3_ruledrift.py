"""DEFER Gate D launcher: rule drift over ten turns. This file should never change.

All the logic lives in the attached `defer-code` dataset. Update it with
`python ml/kernels/publish.py --push`, then Save & Run All here.

No pip install and no training libraries: this gate is evaluation only, which is
the whole reason ADR 0003 put it in front of Arm B rather than after it.
"""
import os
import sys

code_dir = None
for root, _dirs, files in os.walk("/kaggle/input"):
    if "phase3.py" in files and "rules.py" in files:
        code_dir = root
        break

if code_dir is None:
    print("/kaggle/input actually contains:")
    for root, _dirs, files in os.walk("/kaggle/input"):
        if root.count(os.sep) - "/kaggle/input".count(os.sep) > 3:
            continue
        print(f"  {root}  ->  {files[:8]}")
    sys.exit(
        "Could not find phase3.py. Attach johnandreimartinez/defer-code to this "
        "notebook, or publish a new dataset version if the code just changed."
    )

sys.path.insert(0, code_dir)
import phase3  # noqa: E402

phase3.main()
