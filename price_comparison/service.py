import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from .database import DatabaseManager
from .fetchers import fetch_laptop_lk, fetch_luxuryx_lk, fetch_pc_lk, parse_custom_csv

logger = logging.getLogger("price_comparison.service")


class PriceComparisonService:
    """
    High-level service coordinating real-time data sync, database queries,
    cross-seller grouping, price history tracking, and ML comparison.
    """

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self._last_sync_time: Optional[datetime] = None

    def sync_live_sources(self, force: bool = False, min_interval_minutes: int = 30) -> Dict[str, Any]:
        """
        Retrieves real-time product feeds from permitted Sri Lankan retailers
        and saves them into the persistent database.
        """
        now = datetime.now()
        stats = self.db.get_summary_stats()
        last_sync_str = stats.get("last_sync")
        
        if not force and last_sync_str:
            try:
                last_dt = datetime.fromisoformat(last_sync_str)
                if now - last_dt < timedelta(minutes=min_interval_minutes):
                    logger.info("Using cached prices from database (last checked %s)", last_sync_str)
                    return {
                        "status": "cached",
                        "synced_count": stats["total_offers"],
                        "message": f"Prices current as of {last_sync_str}"
                    }
            except Exception:
                pass

        logger.info("Starting live price synchronization across Sri Lankan retailers...")
        total_synced = 0
        source_counts = {}

        # 1. Fetch Laptop.lk (Shopify public JSON feed)
        try:
            items_laptop = fetch_laptop_lk()
            for it in items_laptop:
                self.db.upsert_offer(it)
            total_synced += len(items_laptop)
            source_counts["Laptop.lk"] = len(items_laptop)
        except Exception as e:
            logger.error("Error syncing Laptop.lk: %s", e)
            source_counts["Laptop.lk"] = 0

        # 2. Fetch LuxuryX.lk (Schema.org standard structured JSON-LD feed)
        try:
            items_luxuryx = fetch_luxuryx_lk()
            for it in items_luxuryx:
                self.db.upsert_offer(it)
            total_synced += len(items_luxuryx)
            source_counts["LuxuryX.lk"] = len(items_luxuryx)
        except Exception as e:
            logger.error("Error syncing LuxuryX.lk: %s", e)
            source_counts["LuxuryX.lk"] = 0

        # 3. Fetch PC.lk (WooCommerce store feed)
        try:
            items_pc = fetch_pc_lk()
            for it in items_pc:
                self.db.upsert_offer(it)
            total_synced += len(items_pc)
            source_counts["PC.lk"] = len(items_pc)
        except Exception as e:
            logger.error("Error syncing PC.lk: %s", e)
            source_counts["PC.lk"] = 0

        self._last_sync_time = now
        return {
            "status": "success",
            "total_synced": total_synced,
            "source_counts": source_counts,
            "synced_at": now.isoformat()
        }

    def import_seller_csv(self, csv_content: str, seller_name: str) -> Tuple[int, List[str]]:
        """Imports seller data from a CSV file into the database."""
        items, errors = parse_custom_csv(csv_content, default_seller_name=seller_name)
        inserted = 0
        for it in items:
            try:
                self.db.upsert_offer(it)
                inserted += 1
            except Exception as e:
                errors.append(f"Failed to save '{it.get('title')}': {e}")
        return inserted, errors

    def get_offers(self, search: Optional[str] = None, brand: Optional[str] = None,
                   processor: Optional[str] = None, ram_gb: Optional[int] = None,
                   min_price: Optional[float] = None, max_price: Optional[float] = None,
                   stock_status: Optional[str] = None, seller_code: Optional[str] = None,
                   sort_by: str = "price_asc") -> List[Dict[str, Any]]:
        """Queries offers with user filters."""
        return self.db.get_offers(
            search=search,
            brand=brand,
            processor=processor,
            ram_gb=ram_gb,
            min_price=min_price,
            max_price=max_price,
            stock_status=stock_status,
            seller_code=seller_code,
            sort_by=sort_by
        )

    def get_cross_seller_comparison(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Groups all offers by canonical laptop name to find laptops sold across
        multiple retailers for direct price comparison.
        """
        all_offers = self.db.get_offers(sort_by="price_asc")
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for off in all_offers:
            cname = off.get("canonical_name") or off["title"]
            if cname not in grouped:
                grouped[cname] = []
            grouped[cname].append(off)
        return grouped

    def get_price_history(self, offer_id: Optional[int] = None, product_id: Optional[int] = None):
        """Returns price history records."""
        return self.db.get_price_history(offer_id=offer_id, product_id=product_id)

    def get_stats(self) -> Dict[str, Any]:
        """Returns aggregate platform metrics."""
        return self.db.get_summary_stats()
