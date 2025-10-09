"""
Database module for BNG application
Replaces Excel backend with SQLite database
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class BNGDatabase:
    """SQLite database handler for BNG stock, pricing, and bank data"""
    
    def __init__(self, db_path: str = "bng_backend.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Create database and tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS banks (
                bank_id TEXT PRIMARY KEY,
                bank_name TEXT NOT NULL,
                lpa_name TEXT,
                nca_name TEXT,
                postcode TEXT,
                address TEXT,
                lat REAL,
                lon REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS habitat_catalog (
                habitat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                habitat_name TEXT UNIQUE NOT NULL,
                broader_type TEXT,
                distinctiveness_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS stock (
                stock_id TEXT PRIMARY KEY,
                bank_id TEXT NOT NULL,
                habitat_name TEXT NOT NULL,
                quantity_available REAL NOT NULL DEFAULT 0,
                available_excl_quotes REAL DEFAULT 0,
                quoted REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (bank_id) REFERENCES banks(bank_id),
                FOREIGN KEY (habitat_name) REFERENCES habitat_catalog(habitat_name)
            );
            
            CREATE TABLE IF NOT EXISTS pricing (
                pricing_id INTEGER PRIMARY KEY AUTOINCREMENT,
                bank_id TEXT NOT NULL,
                habitat_name TEXT NOT NULL,
                contract_size TEXT NOT NULL,
                tier TEXT NOT NULL,
                price REAL NOT NULL,
                broader_type TEXT,
                distinctiveness_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (bank_id) REFERENCES banks(bank_id),
                FOREIGN KEY (habitat_name) REFERENCES habitat_catalog(habitat_name),
                UNIQUE(bank_id, habitat_name, contract_size, tier)
            );
            
            CREATE TABLE IF NOT EXISTS distinctiveness_levels (
                distinctiveness_name TEXT PRIMARY KEY,
                level_value REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS spatial_risk_multipliers (
                srm_id INTEGER PRIMARY KEY AUTOINCREMENT,
                risk_category TEXT NOT NULL,
                multiplier REAL NOT NULL,
                description TEXT
            );
            
            CREATE TABLE IF NOT EXISTS trading_rules (
                rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                demand_habitat TEXT NOT NULL,
                supply_habitat TEXT NOT NULL,
                is_allowed INTEGER NOT NULL DEFAULT 1,
                notes TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_stock_bank ON stock(bank_id);
            CREATE INDEX IF NOT EXISTS idx_stock_habitat ON stock(habitat_name);
            CREATE INDEX IF NOT EXISTS idx_pricing_bank ON pricing(bank_id);
            CREATE INDEX IF NOT EXISTS idx_pricing_habitat ON pricing(habitat_name);
        """)
        self.conn.commit()
    
    def load_from_excel(self, excel_path: str):
        """Load data from Excel backend file into database"""
        try:
            # Read Excel file
            xl = pd.ExcelFile(excel_path)
            
            # Load Banks
            if "Banks" in xl.sheet_names:
                banks_df = pd.read_excel(xl, "Banks")
                banks_df.to_sql("banks", self.conn, if_exists="replace", index=False)
            
            # Load HabitatCatalog
            if "HabitatCatalog" in xl.sheet_names:
                catalog_df = pd.read_excel(xl, "HabitatCatalog")
                catalog_df.to_sql("habitat_catalog", self.conn, if_exists="replace", index=False)
            
            # Load Stock
            if "Stock" in xl.sheet_names:
                stock_df = pd.read_excel(xl, "Stock")
                stock_df.to_sql("stock", self.conn, if_exists="replace", index=False)
            
            # Load Pricing
            if "Pricing" in xl.sheet_names:
                pricing_df = pd.read_excel(xl, "Pricing")
                pricing_df.to_sql("pricing", self.conn, if_exists="replace", index=False)
            
            # Load DistinctivenessLevels
            if "DistinctivenessLevels" in xl.sheet_names:
                dist_df = pd.read_excel(xl, "DistinctivenessLevels")
                dist_df.to_sql("distinctiveness_levels", self.conn, if_exists="replace", index=False)
            
            # Load SRM
            if "SRM" in xl.sheet_names:
                srm_df = pd.read_excel(xl, "SRM")
                srm_df.to_sql("spatial_risk_multipliers", self.conn, if_exists="replace", index=False)
            
            # Load TradingRules
            if "TradingRules" in xl.sheet_names:
                rules_df = pd.read_excel(xl, "TradingRules")
                rules_df.to_sql("trading_rules", self.conn, if_exists="replace", index=False)
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error loading Excel data: {e}")
            return False
    
    def get_banks(self) -> pd.DataFrame:
        """Get all banks as DataFrame"""
        return pd.read_sql("SELECT * FROM banks", self.conn)
    
    def get_stock(self) -> pd.DataFrame:
        """Get all stock as DataFrame"""
        return pd.read_sql("SELECT * FROM stock", self.conn)
    
    def get_pricing(self) -> pd.DataFrame:
        """Get all pricing as DataFrame"""
        return pd.read_sql("SELECT * FROM pricing", self.conn)
    
    def get_habitat_catalog(self) -> pd.DataFrame:
        """Get habitat catalog as DataFrame"""
        return pd.read_sql("SELECT * FROM habitat_catalog", self.conn)
    
    def get_distinctiveness_levels(self) -> pd.DataFrame:
        """Get distinctiveness levels as DataFrame"""
        return pd.read_sql("SELECT * FROM distinctiveness_levels", self.conn)
    
    def get_srm(self) -> pd.DataFrame:
        """Get spatial risk multipliers as DataFrame"""
        return pd.read_sql("SELECT * FROM spatial_risk_multipliers", self.conn)
    
    def get_trading_rules(self) -> pd.DataFrame:
        """Get trading rules as DataFrame"""
        query = "SELECT * FROM trading_rules"
        try:
            return pd.read_sql(query, self.conn)
        except:
            return pd.DataFrame()
    
    def update_stock(self, stock_id: str, quantity_available: float):
        """Update stock quantity"""
        self.conn.execute(
            "UPDATE stock SET quantity_available = ?, updated_at = CURRENT_TIMESTAMP WHERE stock_id = ?",
            (quantity_available, stock_id)
        )
        self.conn.commit()
    
    def add_bank(self, bank_data: Dict) -> bool:
        """Add a new bank"""
        try:
            self.conn.execute(
                """INSERT INTO banks (bank_id, bank_name, lpa_name, nca_name, postcode, address, lat, lon)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    bank_data.get("bank_id"),
                    bank_data.get("bank_name"),
                    bank_data.get("lpa_name"),
                    bank_data.get("nca_name"),
                    bank_data.get("postcode"),
                    bank_data.get("address"),
                    bank_data.get("lat"),
                    bank_data.get("lon"),
                )
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error adding bank: {e}")
            return False
    
    def get_backend_dict(self) -> Dict[str, pd.DataFrame]:
        """Get all data as dictionary matching original backend structure"""
        return {
            "Banks": self.get_banks(),
            "Stock": self.get_stock(),
            "Pricing": self.get_pricing(),
            "HabitatCatalog": self.get_habitat_catalog(),
            "DistinctivenessLevels": self.get_distinctiveness_levels(),
            "SRM": self.get_srm(),
            "TradingRules": self.get_trading_rules(),
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
