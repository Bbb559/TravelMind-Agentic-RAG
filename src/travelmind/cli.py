"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travelmind.data import inventory
from travelmind.graphs import AgenticRAGWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(prog="travelmind")
    parser.add_argument("--query", default="")
    parser.add_argument("--workflow", action="store_true")
    parser.add_argument("--retrieve", action="store_true")
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()

    if args.inventory:
        print(json.dumps(inventory(), ensure_ascii=False, indent=2))
        return
    workflow = AgenticRAGWorkflow()
    if args.workflow or args.retrieve:
        print(json.dumps(workflow.run(args.query).to_dict(), ensure_ascii=False, indent=2))
        return
    print(json.dumps(workflow.route(args.query).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
