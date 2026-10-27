"""Write the checked-in JSON Schema for the runtime contracts."""

import json
from pathlib import Path

from .artifact_schema import build_json_schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    destination = PROJECT_ROOT / "protocols" / "artifacts.schema.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(build_json_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
