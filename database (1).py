"""
database.py
===========
Production-ready, data-isolated SQLite database layer for BizAgent.
Includes secure SHA-256 salted password hashing engine with zero dummy configurations.
"""

from __future__ import annotations

import csv
import logging
import queue
import sqlite3
import threading
import hashlib
import hmac
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

DB_FILENAME = "supermart_ops.db"
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / DB_FILENAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("supermart.database")


class SecurityEngine:
    @staticmethod
    def hash_password(plaintext_password: str) -> str:
        salt = os.urandom(16)
        password_hash = hashlib.sha256(salt + plaintext_password.encode('utf-8')).hexdigest()
        return f"{salt.hex()}:{password_hash}"

    @staticmethod
    def verify_password(plaintext_password: str, stored_credential_string: str) -> bool:
        try:
            salt_hex, original_hash = stored_credential_string.split(":")
            salt = bytes.fromhex(salt_hex)
            current_hash = hashlib.sha256(salt + plaintext_password.encode('utf-8')).hexdigest()
            return hmac.compare_digest(current_hash, original_hash)
        except Exception:
            return False


class ConnectionPool:
    def __init__(self, db_path: Path, pool_size: int = 5, timeout: float = 30.0):
        self._db_path = db_path
        self._pool_size = pool_size
        self._timeout = timeout
        self._pool: "queue.Queue[sqlite3.Connection]" = queue.Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._initialized = False
        self._create_pool()

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path,
            timeout=self._timeout,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    def _create_pool(self) -> None:
        with self._lock:
            if self._initialized:
                return
            for _ in range(self._pool_size):
                self._pool.put(self._create_connection())
            self._initialized = True
            logger.info("Connection pool initialized with %d connections.", self._pool_size)

    @contextmanager
    def acquire(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._pool.get(timeout=self._timeout)
        try:
            yield conn
        finally:
            self._pool.put(conn)

    def close_all(self) -> None:
        with self._lock:
            while not self._pool.empty():
                conn = self._pool.get_nowait()
                conn.close()
            self._initialized = False
            logger.info("All pooled connections closed.")


@dataclass
class DatabaseManager:
    db_path: Path = DB_PATH
    pool_size: int = 5
    _pool: Optional[ConnectionPool] = None

    def __post_init__(self) -> None:
        self._pool = ConnectionPool(self.db_path, pool_size=self.pool_size)

    @contextmanager
    def get_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        conn = None
        cursor = None
        try:
            with self._pool.acquire() as conn:
                cursor = conn.cursor()
                conn.execute("BEGIN;")
                try:
                    yield cursor
                    conn.execute("COMMIT;")
                except Exception:
                    conn.execute("ROLLBACK;")
                    logger.exception("Transaction rolled back due to an error.")
                    raise
        except sqlite3.Error as db_err:
            logger.error("Database error: %s", db_err)
            raise
        finally:
            if cursor is not None:
                cursor.close()

    def initialize(self) -> None:
        with self.get_cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    username      TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at    TEXT DEFAULT (datetime('now'))
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    sku                     TEXT PRIMARY KEY,
                    barcode                 TEXT UNIQUE NOT NULL,
                    name                    TEXT NOT NULL,
                    category                TEXT NOT NULL,
                    stock_level             INTEGER NOT NULL DEFAULT 0 CHECK (stock_level >= 0),
                    minimum_required_stock  INTEGER NOT NULL DEFAULT 0 CHECK (minimum_required_stock >= 0),
                    wholesale_price         REAL NOT NULL CHECK (wholesale_price >= 0),
                    retail_price            REAL NOT NULL CHECK (retail_price >= 0),
                    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at              TEXT NOT NULL DEFAULT (datetime('now'))
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS vendor_contracts (
                    vendor_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    vendor_name         TEXT NOT NULL,
                    product_category    TEXT NOT NULL,
                    lead_time_days      INTEGER NOT NULL CHECK (lead_time_days >= 0),
                    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS sales_transactions (
                    transaction_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    receipt_id          TEXT NOT NULL,
                    sku                 TEXT NOT NULL,
                    quantity            INTEGER NOT NULL CHECK (quantity > 0),
                    timestamp           TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (sku) REFERENCES products (sku) ON DELETE RESTRICT
                );
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sales_timestamp ON sales_transactions(timestamp);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sales_sku ON sales_transactions(sku);")
            
            logger.info("All secure enterprise tables and analytical database indexes initialized successfully.")

    def register_user(self, username: str, plaintext_password: str) -> bool:
        hashed_value = SecurityEngine.hash_password(plaintext_password)
        try:
            with self.get_cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?);",
                    (username.strip().lower(), hashed_value)
                )
            logger.info("Security Clearance: Operator '%s' registered permanently.", username)
            return True
        except sqlite3.IntegrityError:
            logger.warning("Registration conflict: Username '%s' already exists.", username)
            return False

    def verify_user_credentials(self, username: str, plaintext_password: str) -> bool:
        try:
            with self.get_cursor() as cur:
                cur.execute(
                    "SELECT password_hash FROM users WHERE username = ?;",
                    (username.strip().lower(),)
                )
                row = cur.fetchone()
            
            if row is None:
                return False
                
            return SecurityEngine.verify_password(plaintext_password, row["password_hash"])
        except Exception as e:
            logger.error("Authentication layer exception encountered: %s", str(e))
            return False

    def seed_from_csv(self, csv_filepath: Path) -> None:
        """
        Bulk-load/refresh products from an uploaded CSV.

        Real-world CSVs (especially ones exported from Excel) are messy
        in ways that would otherwise crash a naive `row['sku']` lookup:
        - Excel's "CSV UTF-8" export prepends a BOM to the first header
          cell, turning "sku" into "\\ufeffsku" -- so it never matches.
        - Headers often use different case or spacing than expected,
          e.g. "SKU", "Stock Level" instead of "sku", "stock_level".

        This method normalizes headers (BOM-stripped, lowercased,
        whitespace-trimmed, spaces -> underscores) before matching them
        against the required column set, so "SKU", " Sku ", and "sku"
        are all treated the same. If required columns are still missing
        after normalization, it raises one clear error listing exactly
        what was expected vs. what was found in the file -- instead of
        an opaque KeyError partway through ingestion.
        """
        required_columns = {
            "sku", "barcode", "name", "category",
            "stock_level", "minimum_required_stock",
            "wholesale_price", "retail_price",
        }

        if not csv_filepath.exists():
            logger.warning("Ingestion Failed: Target CSV resource '%s' does not exist.", csv_filepath.name)
            return

        # utf-8-sig transparently strips a leading BOM if present, and is
        # a no-op (identical to plain utf-8) if there is no BOM.
        with open(csv_filepath, mode='r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                raise ValueError(f"CSV file '{csv_filepath.name}' has no header row.")

            def _normalize(col: str) -> str:
                return col.strip().lower().replace(" ", "_").replace("-", "_")

            # Map normalized name -> original header, so we can pull each
            # row's value by whatever the file actually calls that column.
            header_map = {_normalize(col): col for col in reader.fieldnames}

            missing = required_columns - header_map.keys()
            if missing:
                raise ValueError(
                    f"CSV file '{csv_filepath.name}' is missing required column(s): "
                    f"{sorted(missing)}. Columns found in file: {reader.fieldnames}. "
                    f"Expected (case/spacing-insensitive): {sorted(required_columns)}."
                )

            with self.get_cursor() as cur:
                row_count = 0
                for raw_row in reader:
                    row = {norm: raw_row[original] for norm, original in header_map.items()}
                    try:
                        cur.execute("""
                            INSERT OR REPLACE INTO products (
                                sku, barcode, name, category, stock_level,
                                minimum_required_stock, wholesale_price, retail_price
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                        """, (
                            row['sku'].strip(), row['barcode'].strip(), row['name'].strip(), row['category'].strip(),
                            int(row['stock_level']), int(row['minimum_required_stock']),
                            float(row['wholesale_price']), float(row['retail_price'])
                        ))
                        row_count += 1
                    except (ValueError, KeyError) as exc:
                        raise ValueError(
                            f"Could not import row {row_count + 1} of '{csv_filepath.name}': {exc}. "
                            f"Row contents: {row}"
                        ) from exc

                cur.execute("SELECT COUNT(*) as total FROM products;")
                res = cur.fetchone()
                count = res["total"] if res else 0
                logger.info(
                    "CSV Pipeline complete. Imported %d row(s) this run. Live verifiable row registry count: %d",
                    row_count, count,
                )


if __name__ == "__main__":
    print("Initializing Clean Database Layers...")
    db = DatabaseManager()
    db.initialize()
    print("Database built successfully inside 'supermart_ops.db'.")

