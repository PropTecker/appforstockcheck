# database.py - Database models and operations for BNG backend data
"""
This module replaces the Excel backend workbook with an SQLite database.
It provides models for Banks, Pricing, HabitatCatalog, Stock, DistinctivenessLevels, SRM, and TradingRules.
"""

import sqlite3
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from io import BytesIO


class BNGDatabase:
    """Database handler for BNG backend data"""
    
    def __init__(self, db_path: str = "data/bng_backend.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = None
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Create database directory if it doesn't exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def connect(self):
        """Connect to the database"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def init_schema(self):
        """Initialize database schema"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Banks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS banks (
                bank_id TEXT PRIMARY KEY,
                bank_name TEXT NOT NULL,
                postcode TEXT,
                lpa_name TEXT,
                nca_name TEXT,
                lat REAL,
                lon REAL
            )
        """)
        
        # Pricing table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pricing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bank_id TEXT NOT NULL,
                habitat_name TEXT NOT NULL,
                tier TEXT NOT NULL,
                contract_size TEXT NOT NULL,
                price REAL NOT NULL,
                broader_type TEXT,
                distinctiveness_name TEXT,
                FOREIGN KEY (bank_id) REFERENCES banks(bank_id)
            )
        """)
        
        # HabitatCatalog table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS habitat_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                habitat_name TEXT UNIQUE NOT NULL,
                broader_type TEXT,
                distinctiveness_name TEXT
            )
        """)
        
        # Stock table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bank_id TEXT NOT NULL,
                habitat_name TEXT NOT NULL,
                quantity_available REAL NOT NULL DEFAULT 0,
                available_excl_quotes REAL,
                quoted REAL,
                FOREIGN KEY (bank_id) REFERENCES banks(bank_id)
            )
        """)
        
        # DistinctivenessLevels table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS distinctiveness_levels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                distinctiveness_name TEXT UNIQUE NOT NULL,
                level_value INTEGER NOT NULL
            )
        """)
        
        # SRM (Spatial Risk Multiplier) table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS srm (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_distinctiveness TEXT NOT NULL,
                to_distinctiveness TEXT NOT NULL,
                tier TEXT NOT NULL,
                is_legal INTEGER NOT NULL DEFAULT 1
            )
        """)
        
        # TradingRules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_type TEXT NOT NULL,
                rule_value TEXT NOT NULL,
                description TEXT
            )
        """)
        
        conn.commit()
    
    def load_from_excel(self, excel_bytes: bytes):
        """Load data from Excel workbook into database"""
        x = pd.ExcelFile(BytesIO(excel_bytes))
        conn = self.connect()
        
        # Load each sheet
        if "Banks" in x.sheet_names:
            df = pd.read_excel(x, "Banks")
            df.to_sql("banks", conn, if_exists="replace", index=False)
        
        if "Pricing" in x.sheet_names:
            df = pd.read_excel(x, "Pricing")
            df.to_sql("pricing", conn, if_exists="replace", index=False)
        
        if "HabitatCatalog" in x.sheet_names:
            df = pd.read_excel(x, "HabitatCatalog")
            df.to_sql("habitat_catalog", conn, if_exists="replace", index=False)
        
        if "Stock" in x.sheet_names:
            df = pd.read_excel(x, "Stock")
            df.to_sql("stock", conn, if_exists="replace", index=False)
        
        if "DistinctivenessLevels" in x.sheet_names:
            df = pd.read_excel(x, "DistinctivenessLevels")
            df.to_sql("distinctiveness_levels", conn, if_exists="replace", index=False)
        
        if "SRM" in x.sheet_names:
            df = pd.read_excel(x, "SRM")
            df.to_sql("srm", conn, if_exists="replace", index=False)
        
        if "TradingRules" in x.sheet_names:
            df = pd.read_excel(x, "TradingRules")
            df.to_sql("trading_rules", conn, if_exists="replace", index=False)
        
        conn.commit()
    
    def get_backend_dict(self) -> Dict[str, pd.DataFrame]:
        """Get backend data as dictionary of DataFrames (compatible with existing code)"""
        conn = self.connect()
        
        backend = {
            "Banks": pd.read_sql("SELECT * FROM banks", conn),
            "Pricing": pd.read_sql("SELECT * FROM pricing", conn),
            "HabitatCatalog": pd.read_sql("SELECT * FROM habitat_catalog", conn),
            "Stock": pd.read_sql("SELECT * FROM stock", conn),
            "DistinctivenessLevels": pd.read_sql("SELECT * FROM distinctiveness_levels", conn),
            "SRM": pd.read_sql("SELECT * FROM srm", conn),
        }
        
        # TradingRules might not exist
        try:
            backend["TradingRules"] = pd.read_sql("SELECT * FROM trading_rules", conn)
        except Exception:
            backend["TradingRules"] = pd.DataFrame()
        
        return backend
    
    def update_stock(self, bank_id: str, habitat_name: str, quantity: float):
        """Update stock quantity for a bank/habitat combination"""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE stock 
            SET quantity_available = ?
            WHERE bank_id = ? AND habitat_name = ?
        """, (quantity, bank_id, habitat_name))
        conn.commit()
