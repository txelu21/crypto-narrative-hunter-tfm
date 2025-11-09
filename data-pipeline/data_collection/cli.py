import argparse
import logging
import os
from datetime import date

from .common.config import settings
from .common.logging_setup import setup_logging
from .common.db import get_cursor
from .common import checkpoints


def cmd_init_db(_: argparse.Namespace) -> None:
    setup_logging()
    logging.info("initializing database schema")
    schema_path = os.path.join(os.path.dirname(__file__), "..", "sql", "schema.sql")
    schema_path = os.path.abspath(schema_path)

    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()

    with get_cursor() as cur:
        cur.execute(sql)

    logging.info("schema applied")


def cmd_health(_: argparse.Namespace) -> None:
    setup_logging()
    logging.info("testing database connectivity")
    with get_cursor(readonly=True) as cur:
        cur.execute("SELECT 1")
        val = cur.fetchone()[0]
    logging.info({"db": "ok", "result": val})


def cmd_ensure_checkpoints(_: argparse.Namespace) -> None:
    setup_logging()
    checkpoints.ensure_table()
    logging.info("checkpoint table ensured")


def cmd_checkpoint_show(args: argparse.Namespace) -> None:
    setup_logging()
    cp = checkpoints.get_checkpoint(args.type)
    logging.info({"checkpoint": cp})


def cmd_checkpoint_update(args: argparse.Namespace) -> None:
    setup_logging()
    checkpoints.upsert_checkpoint(
        collection_type=args.type,
        last_processed_block=args.block,
        last_processed_date=args.date,
        records_collected=args.records,
        status=args.status,
    )
    logging.info("checkpoint updated")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("data-collection CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db").set_defaults(func=cmd_init_db)
    sub.add_parser("health").set_defaults(func=cmd_health)
    sub.add_parser("ensure-checkpoints").set_defaults(func=cmd_ensure_checkpoints)

    p_show = sub.add_parser("checkpoint-show")
    p_show.add_argument("--type", required=True)
    p_show.set_defaults(func=cmd_checkpoint_show)

    p_upd = sub.add_parser("checkpoint-update")
    p_upd.add_argument("--type", required=True)
    p_upd.add_argument("--block", type=int)
    p_upd.add_argument("--date")
    p_upd.add_argument("--records", type=int)
    p_upd.add_argument("--status")
    p_upd.set_defaults(func=cmd_checkpoint_update)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
