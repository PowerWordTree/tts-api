import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


if __name__ == "__main__":
    from pwt.tts_api.api import main
    main()
