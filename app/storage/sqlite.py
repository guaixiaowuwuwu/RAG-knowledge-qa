import sqlite3
from pathlib import Path


def connect_sqlite(path: str | Path) -> sqlite3.Connection:
    sqlite_path = Path(path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    return connection
