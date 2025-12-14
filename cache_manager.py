"""
Cache Manager for Semantic Coverage Analyzer
============================================
Handles caching of LLM classification results using SQLite.
"""

import sqlite3
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

class CacheManager:
    """Manages a SQLite cache for function classifications."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "analysis_cache.db"
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS function_classifications (
                hash TEXT PRIMARY KEY,
                model_name TEXT,
                classification_json TEXT,
                timestamp REAL
            )
        """)
        conn.commit()
        conn.close()

    def _compute_hash(self, func_code: str, model_name: str) -> str:
        """Compute a unique hash for the function code and model."""
        content = f"{model_name}:{func_code}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get_classification(self, func_code: str, model_name: str) -> Optional[Dict[str, List[str]]]:
        """Retrieve cached classification if it exists."""
        func_hash = self._compute_hash(func_code, model_name)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT classification_json FROM function_classifications WHERE hash = ?", 
                (func_hash,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return json.loads(row[0])
        except Exception as e:
            print(f"  [WARN] Cache read error: {e}")
        
        return None

    def save_classification(self, func_code: str, model_name: str, classification: Dict[str, List[str]]):
        """Save classification result to cache."""
        func_hash = self._compute_hash(func_code, model_name)
        json_str = json.dumps(classification)
        timestamp = time.time()

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO function_classifications 
                (hash, model_name, classification_json, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (func_hash, model_name, json_str, timestamp)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"  [WARN] Cache write error: {e}")

    def clear(self):
        """Clear the cache."""
        try:
            if self.db_path.exists():
                self.db_path.unlink()
            self._init_db()
        except Exception as e:
            print(f"  [WARN] Cache clear error: {e}")
