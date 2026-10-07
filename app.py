#!/usr/bin/env python3
"""Root entry point to launch the Pearl-Chat local web application."""

import os
import sys
from pathlib import Path

# Automatically ensure the project virtual environment is used if available
try:
    import jax
except ImportError:
    venv_py = Path(__file__).parent / ".venv" / "bin" / "python"
    if venv_py.exists() and sys.executable != str(venv_py.resolve()):
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)
    else:
        raise

from ui.chat_app import main

if __name__ == "__main__":
    main()
