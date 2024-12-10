import os
import sys


def append_sys_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tube = os.path.join(root, "build", "rela")
    if tube not in sys.path:
        sys.path.append(tube)