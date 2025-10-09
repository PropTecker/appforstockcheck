"""
Database Initialization Script
Initializes the BNG database from an Excel backend file
"""

import sys
from pathlib import Path
from database import BNGDatabase


def init_database(excel_path: str, db_path: str = "bng_backend.db"):
    """Initialize database from Excel file"""
    print(f"Initializing database from {excel_path}...")
    
    if not Path(excel_path).exists():
        print(f"Error: Excel file not found: {excel_path}")
        return False
    
    try:
        db = BNGDatabase(db_path)
        success = db.load_from_excel(excel_path)
        
        if success:
            # Verify data was loaded
            backend = db.get_backend_dict()
            print(f"\n✓ Database initialized successfully at: {db_path}")
            print(f"  - Banks: {len(backend['Banks'])} records")
            print(f"  - Stock: {len(backend['Stock'])} records")
            print(f"  - Pricing: {len(backend['Pricing'])} records")
            print(f"  - Habitat Catalog: {len(backend['HabitatCatalog'])} records")
            print(f"  - Distinctiveness Levels: {len(backend['DistinctivenessLevels'])} records")
        else:
            print("✗ Failed to initialize database")
        
        db.close()
        return success
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python init_database.py <excel_file> [db_path]")
        print("Example: python init_database.py data/HabitatBackend_WITH_STOCK.xlsx")
        sys.exit(1)
    
    excel_file = sys.argv[1]
    db_file = sys.argv[2] if len(sys.argv) > 2 else "bng_backend.db"
    
    success = init_database(excel_file, db_file)
    sys.exit(0 if success else 1)
