import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE / "pong" / "env"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
