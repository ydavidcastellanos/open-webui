#!/usr/bin/env python3
"""Import a versioned Open WebUI Function into a SQLite webui.db database."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def bool_to_int(value) -> int:
    return 1 if bool(value) else 0


def upsert_function(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_script()
    db_path = Path(args.db) if args.db else repo_root / "backend" / "data" / "webui.db"
    function_path = Path(args.function)
    metadata_path = Path(args.metadata)

    if not function_path.is_absolute():
        function_path = repo_root / function_path
    if not metadata_path.is_absolute():
        metadata_path = repo_root / metadata_path

    content = function_path.read_text(encoding="utf-8")
    compile(content, str(function_path), "exec")

    metadata = load_metadata(metadata_path)
    function_id = args.function_id or metadata["id"]
    name = args.name or metadata.get("name") or function_id
    function_type = metadata.get("type") or "pipe"
    meta = metadata.get("meta") or {}
    valves = metadata.get("valves")
    is_active = metadata.get("is_active", True) if args.active is None else args.active
    is_global = metadata.get("is_global", False) if args.global_function is None else args.global_function

    now = int(time.time())

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        existing = conn.execute(
            "select user_id, created_at from function where id = ?",
            (function_id,),
        ).fetchone()

        user_id = args.user_id
        if user_id is None and existing:
            user_id = existing["user_id"]

        created_at = existing["created_at"] if existing and existing["created_at"] else now
        row = {
            "id": function_id,
            "user_id": user_id,
            "name": name,
            "type": function_type,
            "content": content,
            "meta": json.dumps(meta, ensure_ascii=False),
            "valves": None if valves is None else json.dumps(valves, ensure_ascii=False),
            "is_active": bool_to_int(is_active),
            "is_global": bool_to_int(is_global),
            "updated_at": now,
            "created_at": created_at,
        }

        if args.dry_run:
            action = "update" if existing else "insert"
            print(f"dry-run: would {action} function {function_id!r} in {db_path}")
            return

        conn.execute(
            """
            insert into function (
                id, user_id, name, type, content, meta, valves,
                is_active, is_global, updated_at, created_at
            )
            values (
                :id, :user_id, :name, :type, :content, :meta, :valves,
                :is_active, :is_global, :updated_at, :created_at
            )
            on conflict(id) do update set
                user_id = excluded.user_id,
                name = excluded.name,
                type = excluded.type,
                content = excluded.content,
                meta = excluded.meta,
                valves = excluded.valves,
                is_active = excluded.is_active,
                is_global = excluded.is_global,
                updated_at = excluded.updated_at
            """,
            row,
        )
        conn.commit()

    print(f"Imported {function_id!r} into {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="Path to Open WebUI webui.db. Defaults to backend/data/webui.db.")
    parser.add_argument(
        "--function",
        default="custom/functions/multi_model_conversations_v2.py",
        help="Path to the versioned function Python file.",
    )
    parser.add_argument(
        "--metadata",
        default="custom/functions/multi_model_conversations_v2.meta.json",
        help="Path to the function metadata JSON.",
    )
    parser.add_argument("--function-id", help="Override function id from metadata.")
    parser.add_argument("--name", help="Override display name from metadata.")
    parser.add_argument("--user-id", help="Owner user id for a first-time import.")
    parser.add_argument("--active", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--global-function", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")
    upsert_function(parser.parse_args())


if __name__ == "__main__":
    main()
