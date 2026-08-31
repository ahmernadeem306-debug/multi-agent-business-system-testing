 PY
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
        if not csv_filepath.exists():
            logger.warning("Ingestion Failed: Target CSV resource '%s' does not exist.", csv_filepath.name)
            return
 
        with open(csv_filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            with self.get_cursor() as cur:
                for row in reader:
                    cur.execute("""
                        INSERT OR REPLACE INTO products (
                            sku, barcode, name, category, stock_level, 
                            minimum_required_stock, wholesale_price, retail_price
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        row['sku'], row['barcode'], row['name'], row['category'],
                        int(row['stock_level']), int(row['minimum_required_stock']),
                        float(row['wholesale_price']), float(row['retail_price'])
                    ))
                cur.execute("SELECT COUNT(*) as total FROM products;")
                res = cur.fetchone()
                count = res["total"] if res else 0
                logger.info("CSV Pipeline complete. Live verifiable row registry count: %d", count)
 
 
if __name__ == "__main__":
    print("Initializing Clean Database Layers...")
    db = DatabaseManager()
    db.initialize()
    print("Database built successfully inside 'supermart_ops.db'.")

