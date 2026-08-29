"""
database.py
============
Production-ready SQLite data-access layer for a retail supermart
operational system.

Design goals
------------
1. Thread-safe connection pooling (SQLite connections are cheap but not
   free; pooling avoids repeated open/close overhead in a busy POS
   environment while respecting SQLite's threading model).
2. Every query is parameterized -- no string interpolation into SQL,
   ever. This is the primary SQL-injection defense.
3. All database access goes through context managers so connections
   and cursors are always released/committed/rolled-back deterministically,
   even on exceptions (try/except/finally pattern).
4. `seed_database()` is fully isolated from schema creation so you can
   re-seed, extend, or swap in fixture data without touching DDL.
5. WAL journal mode + foreign keys ON for better concurrent read/write
   behavior in a multi-terminal POS setup.

Usage
-----
    python database.py            # creates + seeds supermart_ops.db
    from database import DatabaseManager
    db = DatabaseManager()
    db.initialize()               # creates schema if not present
    db.seed_database()            # populates with sample data
    with db.get_cursor() as cur:
        cur.execute("SELECT * FROM products WHERE category = ?", ("Produce",))
        rows = cur.fetchall()
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, Optional

# --------------------------------------------------------------------------- 
# Configuration
# --------------------------------------------------------------------------- 

DB_FILENAME = "supermart_ops.db"
DB_PATH = Path(__file__).resolve().parent / DB_FILENAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("supermart.database")


# --------------------------------------------------------------------------- 
# Connection Pool
# --------------------------------------------------------------------------- 

class ConnectionPool:
    """
    A minimal, thread-safe connection pool for SQLite.

    SQLite connections are lightweight but each carries per-connection
    PRAGMA state and caching, so reusing a small fixed pool avoids
    the churn of opening a fresh file handle for every request in a
    multi-terminal (multiple checkout lanes) deployment.

    Connections are created with `check_same_thread=False` because they
    are handed out from a shared queue across worker threads; safety is
    instead guaranteed by the fact that only one thread holds a given
    connection at a time (enforced by the queue itself).
    """

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
            isolation_level=None,  # we manage transactions explicitly (BEGIN/COMMIT)
        )
        conn.row_factory = sqlite3.Row
        # Sane, production-friendly PRAGMAs
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
        """
        Borrow a connection from the pool. Guaranteed to return it,
        even if the caller's code raises.
        """
        conn = self._pool.get(timeout=self._timeout)
        try:
            yield conn
        finally:
            self._pool.put(conn)

    def close_all(self) -> None:
        """Drain and close every pooled connection (e.g. on app shutdown)."""
        with self._lock:
            while not self._pool.empty():
                conn = self._pool.get_nowait()
                conn.close()
            self._initialized = False
            logger.info("All pooled connections closed.")


# --------------------------------------------------------------------------- 
# Database Manager
# --------------------------------------------------------------------------- 

@dataclass
class DatabaseManager:
    """
    High-level facade over the connection pool: schema management,
    seeding, and a safe cursor context manager for application code.
    """

    db_path: Path = DB_PATH
    pool_size: int = 5
    _pool: Optional[ConnectionPool] = None

    def __post_init__(self) -> None:
        self._pool = ConnectionPool(self.db_path, pool_size=self.pool_size)

    # ---- Cursor context manager -------------------------------------------------

    @contextmanager
    def get_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """
        Yield a cursor bound to a pooled connection inside an explicit
        transaction. Commits on success, rolls back on any exception,
        and always releases the connection back to the pool.

        This is the single entry point application code should use for
        every read or write -- never touch sqlite3.connect() directly.
        """
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

    # ---- Schema (DDL) -------------------------------------------------

    def initialize(self) -> None:
        """Create all tables and indexes if they do not already exist."""
        schema_statements = [
            """
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
            """,
            """
            CREATE TABLE IF NOT EXISTS vendor_contracts (
                vendor_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor_name         TEXT NOT NULL,
                product_category    TEXT NOT NULL,
                lead_time_days      INTEGER NOT NULL CHECK (lead_time_days >= 0),
                created_at          TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS sales_transactions (
                receipt_id      TEXT NOT NULL,
                sku             TEXT NOT NULL,
                quantity        INTEGER NOT NULL CHECK (quantity > 0),
                timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (receipt_id, sku),
                FOREIGN KEY (sku) REFERENCES products (sku)
                    ON UPDATE CASCADE
                    ON DELETE RESTRICT
            );
            """,
            # Indexes to support the most common operational queries:
            # low-stock lookups, category filtering, and sales reporting.
            "CREATE INDEX IF NOT EXISTS idx_products_category ON products (category);",
            "CREATE INDEX IF NOT EXISTS idx_products_low_stock ON products (stock_level, minimum_required_stock);",
            "CREATE INDEX IF NOT EXISTS idx_sales_sku ON sales_transactions (sku);",
            "CREATE INDEX IF NOT EXISTS idx_sales_timestamp ON sales_transactions (timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_vendor_category ON vendor_contracts (product_category);",
        ]

        with self.get_cursor() as cur:
            for statement in schema_statements:
                cur.execute(statement)
        logger.info("Schema initialized (tables + indexes) at %s", self.db_path)

    # ---- Seeding -------------------------------------------------

    def seed_database(self, reset: bool = False) -> None:
        """
        Populate the schema with realistic sample data.

        This function is intentionally isolated from `initialize()` so
        you can edit or extend the fixture data below without touching
        DDL, and re-run seeding independently during local development.

        Parameters
        ----------
        reset : bool
            If True, existing rows in all three tables are wiped before
            inserting fresh sample data. Defaults to False (idempotent
            "INSERT OR IGNORE" seeding) so re-running seed_database()
            on a live dev DB won't duplicate rows.
        """
        products = [
            # sku,        barcode,          name,                     category,      stock, min_stock, wholesale, retail
            ("SKU-1001", "8901030875021", "Basmati Rice 5kg",         "Grains",       120, 20,  6.50, 9.99),
            ("SKU-1002", "8901030875038", "Whole Wheat Flour 2kg",    "Grains",        85, 15,  1.80, 3.49),
            ("SKU-1003", "8901030875045", "Sunflower Cooking Oil 1L", "Pantry",        60, 25,  2.10, 3.99),
            ("SKU-1004", "8901030875052", "Granulated Sugar 1kg",     "Pantry",       200, 30,  0.65, 1.29),
            ("SKU-1005", "8901030875069", "Iodized Salt 1kg",         "Pantry",       150, 25,  0.30, 0.79),
            ("SKU-2001", "8901030876011", "Fresh Tomatoes (kg)",      "Produce",       40, 20,  0.75, 1.49),
            ("SKU-2002", "8901030876028", "Red Onions (kg)",          "Produce",       55, 20,  0.60, 1.19),
            ("SKU-2003", "8901030876035", "Bananas (dozen)",          "Produce",       30, 15,  1.20, 2.29),
            ("SKU-2004", "8901030876042", "Green Apples (kg)",        "Produce",       25, 10,  1.80, 3.49),
            ("SKU-3001", "8901030877018", "Whole Milk 1L",            "Dairy",         70, 25,  0.85, 1.59),
            ("SKU-3002", "8901030877025", "Cheddar Cheese 200g",      "Dairy",         45, 15,  2.40, 4.29),
            ("SKU-3003", "8901030877032", "Salted Butter 250g",       "Dairy",         38, 12,  2.00, 3.79),
            ("SKU-4001", "8901030878015", "Frozen Chicken Breast 1kg","Meat & Poultry", 33, 12,  4.50, 7.99),
            ("SKU-4002", "8901030878022", "Ground Beef 500g",         "Meat & Poultry", 28, 10,  3.60, 6.49),
            ("SKU-5001", "8901030879012", "Dish Soap 750ml",          "Household",      90, 20,  1.10, 2.29),
            ("SKU-5002", "8901030879029", "Paper Towels (6-pack)",    "Household",      65, 15,  3.20, 5.99),
            ("SKU-6001", "8901030880018", "Bottled Water 1.5L (6pk)", "Beverages",     110, 30,  2.00, 3.99),
            ("SKU-6002", "8901030880025", "Orange Juice 1L",          "Beverages",      50, 20,  1.40, 2.79),
            ("SKU-7001", "8901030881015", "Milk Chocolate Bar 100g",  "Snacks",         10,  5,  0.55, 1.19),  # low stock example
            ("SKU-7002", "8901030881022", "Potato Chips 150g",        "Snacks",          8, 10,  0.90, 1.79),  # below minimum, example
        ]

        vendor_contracts = [
            # vendor_name,                  product_category,   lead_time_days
            ("Punjab Grains Co-op",          "Grains",           4),
            ("Metro Pantry Distributors",    "Pantry",           3),
            ("Green Valley Farms",           "Produce",          1),
            ("Fresh Dairy Collective",       "Dairy",            2),
            ("Prime Meat Supply Chain",      "Meat & Poultry",   3),
            ("CleanHome Wholesale",          "Household",        7),
            ("Sparkle Beverages Ltd.",       "Beverages",        5),
            ("SnackWorld Traders",           "Snacks",           4),
        ]

        # Generate a small, realistic batch of sales transactions
        # spread over the last 7 days, referencing only real SKUs.
        base_time = datetime.now() - timedelta(days=7)
        sales_transactions = []
        receipt_counter = 1
        sample_baskets = [
            ["SKU-1001", "SKU-3001", "SKU-2001"],
            ["SKU-6001", "SKU-7001"],
            ["SKU-4001", "SKU-2002", "SKU-1003"],
            ["SKU-3002", "SKU-3003", "SKU-1002"],
            ["SKU-2003", "SKU-2004", "SKU-6002"],
            ["SKU-5001", "SKU-5002"],
            ["SKU-7002", "SKU-6001", "SKU-1004"],
        ]
        for day_offset, basket in enumerate(sample_baskets):
            receipt_id = f"RCT-{1000 + receipt_counter}"
            timestamp = (base_time + timedelta(days=day_offset, hours=2 * day_offset)).isoformat(
                sep=" ", timespec="seconds"
            )
            for sku in basket:
                sales_transactions.append((receipt_id, sku, 1 + (receipt_counter % 3), timestamp))
            receipt_counter += 1

        with self.get_cursor() as cur:
            if reset:
                # Children first to respect the foreign key constraint.
                cur.execute("DELETE FROM sales_transactions;")
                cur.execute("DELETE FROM products;")
                cur.execute("DELETE FROM vendor_contracts;")
                logger.info("Existing data cleared before reseeding (reset=True).")

            cur.executemany(
                """
                INSERT OR IGNORE INTO products
                    (sku, barcode, name, category, stock_level,
                     minimum_required_stock, wholesale_price, retail_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                products,
            )

            cur.executemany(
                """
                INSERT OR IGNORE INTO vendor_contracts
                    (vendor_name, product_category, lead_time_days)
                VALUES (?, ?, ?)
                """,
                vendor_contracts,
            )

            cur.executemany(
                """
                INSERT OR IGNORE INTO sales_transactions
                    (receipt_id, sku, quantity, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                sales_transactions,
            )

        logger.info(
            "Seed complete: %d products, %d vendor contracts, %d sales line-items.",
            len(products), len(vendor_contracts), len(sales_transactions),
        )

    # ---- Convenience read helpers (examples of safe parameterized queries) ----

    def get_low_stock_products(self) -> list[sqlite3.Row]:
        """Return every product at or below its minimum required stock level."""
        with self.get_cursor() as cur:
            cur.execute(
                """
                SELECT sku, name, category, stock_level, minimum_required_stock
                FROM products
                WHERE stock_level <= minimum_required_stock
                ORDER BY category, name;
                """
            )
            return cur.fetchall()

    def get_product_by_barcode(self, barcode: str) -> Optional[sqlite3.Row]:
        """Point-of-sale lookup: fetch a single product by scanned barcode."""
        with self.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM products WHERE barcode = ?;",
                (barcode,),  # parameterized -- never f-string this
            )
            return cur.fetchone()

    def record_sale(self, receipt_id: str, sku: str, quantity: int) -> None:
        """
        Record a sale line-item and decrement stock atomically.
        Both statements run inside the same transaction via get_cursor(),
        so a failure in either one rolls back both.
        """
        with self.get_cursor() as cur:
            cur.execute(
                "INSERT INTO sales_transactions (receipt_id, sku, quantity) VALUES (?, ?, ?);",
                (receipt_id, sku, quantity),
            )
            cur.execute(
                """
                UPDATE products
                SET stock_level = stock_level - ?,
                    updated_at = datetime('now')
                WHERE sku = ? AND stock_level >= ?;
                """,
                (quantity, sku, quantity),
            )
            if cur.rowcount == 0:
                raise ValueError(f"Insufficient stock or unknown SKU '{sku}' for this sale.")

    def get_all_products(self, category: Optional[str] = None) -> list[sqlite3.Row]:
        """
        Return all products, optionally filtered by category.
        Used by the inventory view of downstream UI modules (e.g. BizAgent).
        """
        with self.get_cursor() as cur:
            if category:
                cur.execute(
                    "SELECT * FROM products WHERE category = ? ORDER BY name;",
                    (category,),
                )
            else:
                cur.execute("SELECT * FROM products ORDER BY category, name;")
            return cur.fetchall()

    def get_categories(self) -> list[str]:
        """Return the distinct set of product categories currently in stock."""
        with self.get_cursor() as cur:
            cur.execute("SELECT DISTINCT category FROM products ORDER BY category;")
            return [row["category"] for row in cur.fetchall()]

    def get_inventory_summary(self) -> sqlite3.Row:
        """
        Return aggregate inventory KPIs: total SKUs, total units on hand,
        total inventory value at retail price, and count of low-stock items.
        """
        with self.get_cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*)                                            AS total_skus,
                    COALESCE(SUM(stock_level), 0)                       AS total_units,
                    COALESCE(SUM(stock_level * retail_price), 0.0)      AS inventory_value,
                    COALESCE(SUM(CASE WHEN stock_level <= minimum_required_stock
                                       THEN 1 ELSE 0 END), 0)           AS low_stock_count
                FROM products;
                """
            )
            return cur.fetchone()

    def get_sales_summary(self, days: int = 7) -> sqlite3.Row:
        """
        Return aggregate sales KPIs over the trailing `days` window:
        number of receipts, total units sold, and total revenue.
        """
        with self.get_cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(DISTINCT st.receipt_id)                       AS receipt_count,
                    COALESCE(SUM(st.quantity), 0)                       AS units_sold,
                    COALESCE(SUM(st.quantity * p.retail_price), 0.0)    AS revenue
                FROM sales_transactions st
                JOIN products p ON p.sku = st.sku
                WHERE st.timestamp >= datetime('now', ?);
                """,
                (f"-{int(days)} days",),
            )
            return cur.fetchone()

    def get_recent_transactions(self, limit: int = 25) -> list[sqlite3.Row]:
        """Return the most recent sales line-items, newest first, joined to product names."""
        with self.get_cursor() as cur:
            cur.execute(
                """
                SELECT st.receipt_id, st.sku, p.name, st.quantity,
                       (st.quantity * p.retail_price) AS line_total, st.timestamp
                FROM sales_transactions st
                JOIN products p ON p.sku = st.sku
                ORDER BY st.timestamp DESC
                LIMIT ?;
                """,
                (int(limit),),
            )
            return cur.fetchall()

    def get_top_selling_products(self, days: int = 30, limit: int = 5) -> list[sqlite3.Row]:
        """Return the best-selling products by units sold in the trailing `days` window."""
        with self.get_cursor() as cur:
            cur.execute(
                """
                SELECT p.sku, p.name, p.category,
                       SUM(st.quantity)                        AS units_sold,
                       SUM(st.quantity * p.retail_price)        AS revenue
                FROM sales_transactions st
                JOIN products p ON p.sku = st.sku
                WHERE st.timestamp >= datetime('now', ?)
                GROUP BY p.sku
                ORDER BY units_sold DESC
                LIMIT ?;
                """,
                (f"-{int(days)} days", int(limit)),
            )
            return cur.fetchall()

    def get_vendor_contracts(self, category: Optional[str] = None) -> list[sqlite3.Row]:
        """Return vendor contracts, optionally filtered by product category."""
        with self.get_cursor() as cur:
            if category:
                cur.execute(
                    "SELECT * FROM vendor_contracts WHERE product_category = ? ORDER BY lead_time_days;",
                    (category,),
                )
            else:
                cur.execute("SELECT * FROM vendor_contracts ORDER BY product_category, lead_time_days;")
            return cur.fetchall()

    def get_reorder_recommendations(self) -> list[sqlite3.Row]:
        """
        Join low-stock products against vendor contracts for their category,
        surfacing which vendor to contact and the expected lead time.
        """
        with self.get_cursor() as cur:
            cur.execute(
                """
                SELECT p.sku, p.name, p.category, p.stock_level, p.minimum_required_stock,
                       v.vendor_name, v.lead_time_days
                FROM products p
                LEFT JOIN vendor_contracts v ON v.product_category = p.category
                WHERE p.stock_level <= p.minimum_required_stock
                ORDER BY p.stock_level ASC, v.lead_time_days ASC;
                """
            )
            return cur.fetchall()

    def search_products(self, term: str) -> list[sqlite3.Row]:
        """Case-insensitive search of products by name or SKU fragment."""
        with self.get_cursor() as cur:
            like_term = f"%{term}%"
            cur.execute(
                """
                SELECT * FROM products
                WHERE name LIKE ? ESCAPE '\\' OR sku LIKE ? ESCAPE '\\'
                ORDER BY name;
                """,
                (like_term, like_term),
            )
            return cur.fetchall()

    def row_counts(self) -> dict:
        """Return a quick table -> row-count map, used to detect an empty/unseeded DB."""
        counts = {}
        with self.get_cursor() as cur:
            for table in ("products", "sales_transactions", "vendor_contracts"):
                cur.execute(f"SELECT COUNT(*) AS c FROM {table};")  # table name is fixed/internal, not user input
                counts[table] = cur.fetchone()["c"]
        return counts

    def shutdown(self) -> None:
        """Release all pooled connections. Call this on application exit."""
        if self._pool is not None:
            self._pool.close_all()


# --------------------------------------------------------------------------- 
# Script entry point: build + seed the local database file
# --------------------------------------------------------------------------- 

def main() -> None:
    logger.info("Building local database at: %s", DB_PATH)
    db = DatabaseManager(db_path=DB_PATH)
    try:
        db.initialize()
        db.seed_database()

        low_stock = db.get_low_stock_products()
        if low_stock:
            logger.info("Products currently at/below minimum stock:")
            for row in low_stock:
                logger.info(
                    "  - %s | %s | stock=%d | minimum=%d",
                    row["sku"], row["name"], row["stock_level"], row["minimum_required_stock"],
                )
    finally:
        db.shutdown()

    logger.info("Done. Database file ready at '%s'.", DB_FILENAME)


if __name__ == "__main__":
    main()
