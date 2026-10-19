import os
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("price_comparison.database")
logging.basicConfig(level=logging.INFO)

DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_SQLITE_PATH = DEFAULT_DB_DIR / "laptop_comparison.db"


class DatabaseManager:
    """
    Manages persistent storage for Sri Lankan laptop sellers, products, offers,
    and historical price records. Supports SQLite by default and PostgreSQL
    if DATABASE_URL is configured.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_url = os.environ.get("DATABASE_URL")
        self.is_postgres = bool(self.db_url and self.db_url.startswith("postgres"))
        
        if not self.is_postgres:
            self.sqlite_path = db_path or DEFAULT_SQLITE_PATH
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Using SQLite database at %s", self.sqlite_path)
        else:
            logger.info("Configured for PostgreSQL: %s", self.db_url)
            
        self.init_db()

    def get_connection(self):
        """Returns a database connection."""
        if self.is_postgres:
            try:
                import psycopg2
                return psycopg2.connect(self.db_url)
            except Exception as e:
                logger.warning("PostgreSQL connection failed (%s); falling back to SQLite.", e)
                self.is_postgres = False
                self.sqlite_path = DEFAULT_SQLITE_PATH
                self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
                return sqlite3.connect(str(self.sqlite_path))
        else:
            conn = sqlite3.connect(str(self.sqlite_path))
            conn.row_factory = sqlite3.Row
            return conn

    def init_db(self):
        """Creates the relational schema if it does not already exist."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # 1. Sellers table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sellers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                code TEXT NOT NULL UNIQUE,
                base_url TEXT NOT NULL,
                feed_type TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 2. Products table (canonical normalized laptop models)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL UNIQUE,
                brand TEXT NOT NULL,
                model_series TEXT,
                processor TEXT,
                ram_gb INTEGER,
                storage_gb INTEGER,
                gpu TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 3. Offers table (current listed prices per seller)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS offers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                seller_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                product_url TEXT NOT NULL UNIQUE,
                current_price_lkr REAL NOT NULL,
                regular_price_lkr REAL,
                is_discounted INTEGER DEFAULT 0,
                stock_status TEXT NOT NULL,
                sku TEXT,
                source_type TEXT NOT NULL,
                last_checked_at TIMESTAMP NOT NULL,
                FOREIGN KEY (seller_id) REFERENCES sellers(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            );
            """)

            # 4. Price History table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                offer_id INTEGER NOT NULL,
                product_id INTEGER,
                seller_id INTEGER NOT NULL,
                price_lkr REAL NOT NULL,
                stock_status TEXT NOT NULL,
                recorded_at TIMESTAMP NOT NULL,
                FOREIGN KEY (offer_id) REFERENCES offers(id),
                FOREIGN KEY (seller_id) REFERENCES sellers(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            );
            """)

            # Pre-seed known sellers
            default_sellers = [
                ("Laptop.lk", "laptop_lk", "https://laptop.lk", "Shopify Public Feed"),
                ("LuxuryX.lk", "luxuryx_lk", "https://luxuryx.lk", "Schema.org Structured Feed"),
                ("PC.lk", "pc_lk", "https://www.pc.lk", "WooCommerce REST Feed"),
                ("Custom / CSV Import", "custom_csv", "local_feed", "Manual Seller Feed"),
            ]
            for name, code, url, feed_type in default_sellers:
                cursor.execute("""
                INSERT OR IGNORE INTO sellers (name, code, base_url, feed_type)
                VALUES (?, ?, ?, ?);
                """, (name, code, url, feed_type))

            conn.commit()
            logger.info("Database schema initialized successfully.")
        finally:
            conn.close()

    def get_or_create_seller(self, name: str, code: str, base_url: str, feed_type: str) -> int:
        """Retrieves or registers a retailer."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM sellers WHERE code = ?", (code,))
            row = cursor.fetchone()
            if row:
                return row[0]
            cursor.execute("""
            INSERT INTO sellers (name, code, base_url, feed_type) VALUES (?, ?, ?, ?)
            """, (name, code, base_url, feed_type))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_or_create_product(self, canonical_name: str, brand: str, model_series: Optional[str] = None,
                              processor: Optional[str] = None, ram_gb: Optional[int] = None,
                              storage_gb: Optional[int] = None, gpu: Optional[str] = None) -> int:
        """Retrieves or creates a canonical laptop entry for cross-store grouping."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM products WHERE canonical_name = ?", (canonical_name,))
            row = cursor.fetchone()
            if row:
                return row[0]
            cursor.execute("""
            INSERT INTO products (canonical_name, brand, model_series, processor, ram_gb, storage_gb, gpu)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (canonical_name, brand, model_series, processor, ram_gb, storage_gb, gpu))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def upsert_offer(self, offer_data: Dict[str, Any]) -> int:
        """
        Inserts or updates a seller offer and logs a price history record if price changed.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            seller_id = self.get_or_create_seller(
                name=offer_data["seller_name"],
                code=offer_data["seller_code"],
                base_url=offer_data.get("seller_url", "https://example.lk"),
                feed_type=offer_data.get("source_type", "Automated Feed")
            )
            
            product_id = offer_data.get("product_id")
            if not product_id and offer_data.get("canonical_name"):
                product_id = self.get_or_create_product(
                    canonical_name=offer_data["canonical_name"],
                    brand=offer_data.get("brand", "Unknown"),
                    model_series=offer_data.get("model_series"),
                    processor=offer_data.get("processor"),
                    ram_gb=offer_data.get("ram_gb"),
                    storage_gb=offer_data.get("storage_gb"),
                    gpu=offer_data.get("gpu")
                )

            # Check if offer exists by product_url
            cursor.execute("""
            SELECT id, current_price_lkr, stock_status FROM offers WHERE product_url = ?
            """, (offer_data["product_url"],))
            existing = cursor.fetchone()

            timestamp = offer_data.get("last_checked_at") or datetime.now().isoformat()
            current_price = float(offer_data["price_lkr"])
            regular_price = float(offer_data.get("regular_price_lkr") or current_price)
            is_discounted = 1 if (regular_price > current_price) else 0

            if existing:
                offer_id = existing[0]
                old_price = existing[1]
                old_stock = existing[2]

                cursor.execute("""
                UPDATE offers
                SET product_id = ?, title = ?, current_price_lkr = ?, regular_price_lkr = ?,
                    is_discounted = ?, stock_status = ?, source_type = ?, last_checked_at = ?
                WHERE id = ?
                """, (
                    product_id,
                    offer_data["title"],
                    current_price,
                    regular_price,
                    is_discounted,
                    offer_data["stock_status"],
                    offer_data.get("source_type", "Live Feed"),
                    timestamp,
                    offer_id
                ))

                # If price or stock changed, record price history
                if abs(old_price - current_price) > 0.01 or old_stock != offer_data["stock_status"]:
                    cursor.execute("""
                    INSERT INTO price_history (offer_id, product_id, seller_id, price_lkr, stock_status, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (offer_id, product_id, seller_id, current_price, offer_data["stock_status"], timestamp))
            else:
                cursor.execute("""
                INSERT INTO offers (
                    product_id, seller_id, title, product_url, current_price_lkr,
                    regular_price_lkr, is_discounted, stock_status, sku, source_type, last_checked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    product_id,
                    seller_id,
                    offer_data["title"],
                    offer_data["product_url"],
                    current_price,
                    regular_price,
                    is_discounted,
                    offer_data["stock_status"],
                    offer_data.get("sku"),
                    offer_data.get("source_type", "Live Feed"),
                    timestamp
                ))
                offer_id = cursor.lastrowid

                # Initial price history entry
                cursor.execute("""
                INSERT INTO price_history (offer_id, product_id, seller_id, price_lkr, stock_status, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (offer_id, product_id, seller_id, current_price, offer_data["stock_status"], timestamp))

            conn.commit()
            return offer_id
        finally:
            conn.close()

    def get_offers(self, search: Optional[str] = None, brand: Optional[str] = None,
                   processor: Optional[str] = None, ram_gb: Optional[int] = None,
                   min_price: Optional[float] = None, max_price: Optional[float] = None,
                   stock_status: Optional[str] = None, seller_code: Optional[str] = None,
                   sort_by: str = "price_asc") -> List[Dict[str, Any]]:
        """Queries offers with rich multi-parameter filters and sorting."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            query = """
            SELECT 
                o.id, o.title, o.product_url, o.current_price_lkr, o.regular_price_lkr,
                o.is_discounted, o.stock_status, o.source_type, o.last_checked_at,
                s.name as seller_name, s.code as seller_code, s.base_url as seller_url,
                p.id as product_id, p.canonical_name, p.brand, p.processor, p.ram_gb, p.storage_gb, p.gpu
            FROM offers o
            JOIN sellers s ON o.seller_id = s.id
            LEFT JOIN products p ON o.product_id = p.id
            WHERE 1=1
            """
            params = []

            if search:
                query += " AND (o.title LIKE ? OR p.canonical_name LIKE ? OR p.processor LIKE ? OR p.gpu LIKE ?)"
                kw = f"%{search}%"
                params.extend([kw, kw, kw, kw])

            if brand and brand != "All":
                query += " AND p.brand = ?"
                params.append(brand)

            if processor and processor != "All":
                query += " AND p.processor LIKE ?"
                params.append(f"%{processor}%")

            if ram_gb and ram_gb > 0:
                query += " AND p.ram_gb = ?"
                params.append(ram_gb)

            if min_price is not None:
                query += " AND o.current_price_lkr >= ?"
                params.append(min_price)

            if max_price is not None:
                query += " AND o.current_price_lkr <= ?"
                params.append(max_price)

            if stock_status and stock_status != "All":
                if stock_status == "In Stock Only":
                    query += " AND (o.stock_status = 'In Stock' OR o.stock_status = 'Available')"
                elif stock_status == "Discounted Only":
                    query += " AND o.is_discounted = 1"
                else:
                    query += " AND o.stock_status = ?"
                    params.append(stock_status)

            if seller_code and seller_code != "All":
                query += " AND s.code = ?"
                params.append(seller_code)

            # Sorting
            if sort_by == "price_asc":
                query += " ORDER BY o.current_price_lkr ASC"
            elif sort_by == "price_desc":
                query += " ORDER BY o.current_price_lkr DESC"
            elif sort_by == "discount_desc":
                query += " ORDER BY (o.regular_price_lkr - o.current_price_lkr) DESC"
            elif sort_by == "recent":
                query += " ORDER BY o.last_checked_at DESC"
            else:
                query += " ORDER BY o.current_price_lkr ASC"

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_price_history(self, offer_id: Optional[int] = None, product_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves price history records for trend charting."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            query = """
            SELECT 
                ph.id, ph.offer_id, ph.price_lkr, ph.stock_status, ph.recorded_at,
                s.name as seller_name, o.title as offer_title, p.canonical_name
            FROM price_history ph
            JOIN sellers s ON ph.seller_id = s.id
            JOIN offers o ON ph.offer_id = o.id
            LEFT JOIN products p ON ph.product_id = p.id
            WHERE 1=1
            """
            params = []
            if offer_id:
                query += " AND ph.offer_id = ?"
                params.append(offer_id)
            if product_id:
                query += " AND ph.product_id = ?"
                params.append(product_id)

            query += " ORDER BY ph.recorded_at ASC"
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Returns aggregate platform statistics."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), MIN(current_price_lkr), MAX(current_price_lkr), AVG(current_price_lkr) FROM offers")
            off_row = cursor.fetchone()
            total_offers = off_row[0] or 0
            min_price = off_row[1] or 0
            max_price = off_row[2] or 0
            avg_price = off_row[3] or 0

            cursor.execute("SELECT COUNT(DISTINCT seller_id) FROM offers")
            active_sellers = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(DISTINCT canonical_name) FROM products")
            total_products = cursor.fetchone()[0] or 0

            cursor.execute("SELECT MAX(last_checked_at) FROM offers")
            last_sync = cursor.fetchone()[0]

            return {
                "total_offers": total_offers,
                "total_products": total_products,
                "active_sellers": active_sellers,
                "min_price": min_price,
                "max_price": max_price,
                "avg_price": avg_price,
                "last_sync": last_sync
            }
        finally:
            conn.close()
