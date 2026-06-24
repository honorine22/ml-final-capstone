"""Compatibility entrypoint for the MaizeGuard model API.

The current capstone model is a PyTorch MobileNetV3 checkpoint loaded by
`pytorch_main.py`. This file intentionally re-exports that FastAPI app so older
commands such as `uvicorn main:app --reload --port 8000` keep working.
"""

from pytorch_main import app

