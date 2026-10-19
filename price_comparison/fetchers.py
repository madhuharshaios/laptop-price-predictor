import io
import re
import csv
import json
import logging
import urllib.request
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("price_comparison.fetchers")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def extract_specs(title: str, fallback_brand: str = "Unknown") -> Dict[str, Any]:
    """
    Extracts brand, processor, RAM, storage, GPU, and produces a canonical
    grouped name for cross-retailer comparison.
    """
    clean_title = title.strip()
    
    # 1. Brand detection
    known_brands = ["ASUS", "Lenovo", "Dell", "HP", "Apple", "MSI", "Acer", "Huawei", "Microsoft", "Razer", "Gigabyte"]
    brand = fallback_brand
    for b in known_brands:
        if re.search(rf"\b{b}\b", clean_title, re.IGNORECASE):
            brand = b
            break

    # 2. Processor detection
    processor = "Unknown"
    cpu_patterns = [
        (r"\b(Ultra\s*[579](?:\s*[\w\d]+)?)\b", "Intel Core Ultra"),
        (r"\b(Core\s*i[3579](?:-[\w\d]+)?)\b", "Intel Core"),
        (r"\b(i[3579]-[\w\d]+)\b", "Intel Core"),
        (r"\b(Ryzen\s*[3579](?:\s*[\w\d]+)?)\b", "AMD Ryzen"),
        (r"\b(Apple\s*M[1234](?:\s*(?:Pro|Max|Ultra))?)\b", "Apple Silicon"),
        (r"\b(M[1234](?:\s*(?:Pro|Max|Ultra))?)\b", "Apple Silicon"),
    ]
    for pattern, prefix in cpu_patterns:
        m = re.search(pattern, clean_title, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            processor = f"{prefix} {val}" if not val.lower().startswith(prefix.lower()) else val
            break

    # 3. RAM detection
    ram_gb = None
    ram_match = re.search(r"\b(4|8|12|16|24|32|64)\s*GB(?:\s*(?:DDR[45]|RAM|LPDDR[45]))?\b", clean_title, re.IGNORECASE)
    if ram_match:
        try:
            ram_gb = int(ram_match.group(1))
        except ValueError:
            pass

    # 4. Storage detection
    storage_gb = None
    storage_match = re.search(r"\b(128|256|512)\s*GB\b", clean_title, re.IGNORECASE)
    tb_match = re.search(r"\b([12])\s*TB\b", clean_title, re.IGNORECASE)
    if tb_match:
        try:
            storage_gb = int(tb_match.group(1)) * 1024
        except ValueError:
            pass
    elif storage_match:
        try:
            storage_gb = int(storage_match.group(1))
        except ValueError:
            pass

    # 5. GPU detection
    gpu = "Integrated / Standard"
    gpu_match = re.search(r"\b(RTX\s*\d{4}(?:\s*Ti)?|GTX\s*\d{4}(?:\s*Ti)?|Radeon|Iris\s*Xe|Intel\s*Graphics)\b", clean_title, re.IGNORECASE)
    if gpu_match:
        gpu = gpu_match.group(1).strip()

    # 6. Canonical name for clustering identical models
    model_series = clean_title
    for s in ["Gaming", "Laptop", "Notebook", "Best Price In Sri Lanka", "Ultra", "Slim"]:
        model_series = re.sub(rf"\b{s}\b", "", model_series, flags=re.IGNORECASE)
    # Extract main series keyword e.g. "Lenovo Legion 5", "ASUS ROG Strix G16", "Dell XPS 15"
    series_match = re.search(r"((?:ASUS|Lenovo|Dell|HP|Apple|MSI|Acer)\s+[\w\d\-]+(?:\s+[\w\d\-]+)?)", clean_title, re.IGNORECASE)
    if series_match:
        canonical_name = series_match.group(1).strip()
    else:
        canonical_name = f"{brand} {clean_title[:30]}".strip()

    return {
        "brand": brand,
        "processor": processor,
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "gpu": gpu,
        "canonical_name": canonical_name,
        "model_series": canonical_name
    }


def fetch_laptop_lk() -> List[Dict[str, Any]]:
    """
    Retrieves official Shopify public product feed from Laptop.lk.
    Permitted under robots.txt (only /admin, /cart, /checkout are restricted).
    """
    url = "https://laptop.lk/products.json?limit=150"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    items = []
    
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.error("Failed to query laptop.lk product feed: %s", e)
        return []

    for p in data.get("products", []):
        ptype = (p.get("product_type") or "").strip().lower()
        title = (p.get("title") or "").strip()
        tags = [t.lower() for t in p.get("tags", [])]
        
        # Only process laptops
        if "laptop" not in ptype and "laptop" not in title.lower() and not any("laptop" in t for t in tags):
            continue

        handle = p.get("handle", "")
        vendor = p.get("vendor", "Unknown")
        product_url = f"https://laptop.lk/products/{handle}" if handle else "https://laptop.lk"
        specs = extract_specs(title, fallback_brand=vendor)

        for v in p.get("variants", []):
            try:
                price = float(v.get("price") or 0)
            except (ValueError, TypeError):
                continue
            if price < 50000:
                continue

            comp_price = None
            if v.get("compare_at_price"):
                try:
                    comp_price = float(v.get("compare_at_price"))
                except (ValueError, TypeError):
                    comp_price = None

            avail = bool(v.get("available"))
            stock_status = "In Stock" if avail else "Out of Stock"
            regular_price = comp_price or price

            items.append({
                "seller_name": "Laptop.lk",
                "seller_code": "laptop_lk",
                "seller_url": "https://laptop.lk",
                "title": title,
                "brand": specs["brand"],
                "price_lkr": price,
                "regular_price_lkr": regular_price,
                "is_discounted": bool(regular_price > price),
                "stock_status": stock_status,
                "product_url": product_url,
                "sku": v.get("sku"),
                "source_type": "Shopify Public Feed",
                "last_checked_at": datetime.now().isoformat(),
                **specs
            })
            break  # First primary variant
            
    return items


def fetch_luxuryx_lk() -> List[Dict[str, Any]]:
    """
    Extracts permitted Schema.org standard JSON-LD ProductGroup structured data
    from LuxuryX.lk laptop catalog page.
    """
    url = "https://luxuryx.lk/laptop-price-in-sri-lanka"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    items = []

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        logger.error("Failed to query luxuryx.lk catalog: %s", e)
        return []

    matches = re.findall(r'<script type=["\']application/ld\+json["\']>(.*?)</script>', html, re.DOTALL)
    for m in matches:
        try:
            d = json.loads(m.strip())
            if not isinstance(d, dict) or d.get("@type") != "ProductGroup":
                continue

            title = d.get("name", "").strip()
            brand_obj = d.get("brand") or {}
            brand = brand_obj.get("name", "Unknown") if isinstance(brand_obj, dict) else str(brand_obj)
            prod_url = d.get("url") or url
            specs = extract_specs(title, fallback_brand=brand)

            # Price from offers
            offers = d.get("offers") or {}
            price = None
            if isinstance(offers, dict):
                p_val = offers.get("price") or offers.get("lowPrice")
                if p_val:
                    try:
                        price = float(p_val)
                    except (ValueError, TypeError):
                        pass
                avail_url = str(offers.get("availability") or "")
                if "InStock" in avail_url:
                    stock_status = "In Stock"
                elif "LimitedAvailability" in avail_url:
                    stock_status = "Limited Stock"
                elif "OutOfStock" in avail_url:
                    stock_status = "Out of Stock"
                else:
                    stock_status = "In Stock"
            else:
                stock_status = "In Stock"

            if price and price > 0:
                items.append({
                    "seller_name": "LuxuryX.lk",
                    "seller_code": "luxuryx_lk",
                    "seller_url": "https://luxuryx.lk",
                    "title": title,
                    "brand": specs["brand"],
                    "price_lkr": price,
                    "regular_price_lkr": price,
                    "is_discounted": False,
                    "stock_status": stock_status,
                    "product_url": prod_url,
                    "sku": None,
                    "source_type": "Schema.org Structured Feed",
                    "last_checked_at": datetime.now().isoformat(),
                    **specs
                })
        except Exception:
            continue

    return items


def fetch_pc_lk() -> List[Dict[str, Any]]:
    """
    Extracts publicly accessible WooCommerce REST store products from PC.lk.
    """
    url = "https://www.pc.lk/wp-json/wc/store/v1/products?per_page=40"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    items = []

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.error("Failed to query pc.lk WooCommerce feed: %s", e)
        return []

    for p in data:
        name = p.get("name", "").strip()
        # Filter for laptop products
        if not re.search(r"\b(laptop|legion|ideapad|tuf|zenbook|thinkpad|macbook|vivobook)\b", name, re.IGNORECASE):
            continue

        prices = p.get("prices", {})
        try:
            raw_p = prices.get("price") or prices.get("regular_price") or 0
            price = float(raw_p)
        except (ValueError, TypeError):
            price = 0.0

        is_stock = p.get("is_in_stock", True)
        stock_status = "In Stock" if is_stock else "Out of Stock"
        if price <= 0:
            stock_status = "Call for Price"

        permalink = p.get("permalink") or "https://www.pc.lk"
        specs = extract_specs(name)

        if price > 0:
            items.append({
                "seller_name": "PC.lk",
                "seller_code": "pc_lk",
                "seller_url": "https://www.pc.lk",
                "title": name,
                "brand": specs["brand"],
                "price_lkr": price,
                "regular_price_lkr": price,
                "is_discounted": False,
                "stock_status": stock_status,
                "product_url": permalink,
                "sku": p.get("sku"),
                "source_type": "WooCommerce REST Feed",
                "last_checked_at": datetime.now().isoformat(),
                **specs
            })

    return items


def parse_custom_csv(csv_content: str, default_seller_name: str = "Imported Seller") -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Validates and imports seller laptop prices from CSV data.
    Expected columns: title, brand, price_lkr, [regular_price_lkr, stock_status, product_url, seller_name]
    """
    items = []
    errors = []
    
    f = io.StringIO(csv_content.strip())
    reader = csv.DictReader(f)
    
    if not reader.fieldnames:
        return [], ["CSV file is empty or missing headers."]
        
    lower_map = {col.lower().strip().replace(" ", "_"): col for col in reader.fieldnames}
    
    required = ["title", "price_lkr"]
    for req in required:
        if req not in lower_map:
            errors.append(f"Missing mandatory column: '{req}'.")
            
    if errors:
        return [], errors

    row_num = 1
    for row in reader:
        row_num += 1
        try:
            title = str(row[lower_map["title"]]).strip()
            if not title:
                continue

            price_str = str(row[lower_map["price_lkr"]]).replace(",", "").replace("Rs.", "").strip()
            price = float(price_str)
            if price <= 0:
                continue

            brand_col = lower_map.get("brand")
            brand = str(row[brand_col]).strip() if brand_col and row.get(brand_col) else "Unknown"

            reg_col = lower_map.get("regular_price_lkr")
            if reg_col and row.get(reg_col):
                try:
                    regular_price = float(str(row[reg_col]).replace(",", "").replace("Rs.", "").strip())
                except ValueError:
                    regular_price = price
            else:
                regular_price = price

            stock_col = lower_map.get("stock_status")
            stock_status = str(row[stock_col]).strip() if stock_col and row.get(stock_col) else "In Stock"

            url_col = lower_map.get("product_url")
            product_url = str(row[url_col]).strip() if url_col and row.get(url_col) else f"https://imported-seller.lk/p/{row_num}"

            seller_col = lower_map.get("seller_name")
            seller_name = str(row[seller_col]).strip() if seller_col and row.get(seller_col) else default_seller_name
            seller_code = re.sub(r"[^\w]", "_", seller_name.lower())

            specs = extract_specs(title, fallback_brand=brand)

            items.append({
                "seller_name": seller_name,
                "seller_code": seller_code,
                "seller_url": "https://imported-seller.lk",
                "title": title,
                "brand": specs["brand"],
                "price_lkr": price,
                "regular_price_lkr": regular_price,
                "is_discounted": bool(regular_price > price),
                "stock_status": stock_status,
                "product_url": product_url,
                "sku": f"IMP-{row_num}",
                "source_type": "CSV Seller Import",
                "last_checked_at": datetime.now().isoformat(),
                **specs
            })
        except Exception as err:
            errors.append(f"Row {row_num}: {err}")

    return items, errors
