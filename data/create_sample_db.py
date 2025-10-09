"""Create sample database for testing and demonstration"""
import sys
sys.path.insert(0, '/home/runner/work/appforstockcheck/appforstockcheck')

import pandas as pd
from database import BNGDatabase

print("Creating sample database...")

# Create database
db = BNGDatabase('data/sample_bng_backend.db')

# Sample Banks
banks_data = pd.DataFrame([
    {
        'bank_id': 'BNK001',
        'bank_name': 'Meadowlands BNG Bank',
        'lpa_name': 'Test District',
        'nca_name': 'Thames Basin',
        'postcode': 'SW1A 1AA',
        'address': '',
        'lat': 51.5074,
        'lon': -0.1278
    },
    {
        'bank_id': 'BNK002',
        'bank_name': 'Woodland Conservation Bank',
        'lpa_name': 'Forest District',
        'nca_name': 'Chilterns',
        'postcode': 'HP4 1AA',
        'address': '',
        'lat': 51.7520,
        'lon': -0.4746
    },
    {
        'bank_id': 'BNK003',
        'bank_name': 'Wetland Restoration Bank',
        'lpa_name': 'Coastal District',
        'nca_name': 'Thames Estuary',
        'postcode': 'SS3 9AA',
        'address': '',
        'lat': 51.5357,
        'lon': 0.7178
    }
])

# Sample Habitat Catalog
habitats_data = pd.DataFrame([
    {'habitat_name': 'Grassland - Modified grassland', 'broader_type': 'Grassland', 'distinctiveness_name': 'Low'},
    {'habitat_name': 'Grassland - Other neutral grassland', 'broader_type': 'Grassland', 'distinctiveness_name': 'Medium'},
    {'habitat_name': 'Grassland - Lowland meadows', 'broader_type': 'Grassland', 'distinctiveness_name': 'High'},
    {'habitat_name': 'Woodland - Mixed plantation', 'broader_type': 'Woodland', 'distinctiveness_name': 'Low'},
    {'habitat_name': 'Woodland - Other woodland; broadleaved', 'broader_type': 'Woodland', 'distinctiveness_name': 'Medium'},
    {'habitat_name': 'Woodland - Lowland beech and yew woodland', 'broader_type': 'Woodland', 'distinctiveness_name': 'High'},
    {'habitat_name': 'Wetland - Ponds (non-priority)', 'broader_type': 'Wetland', 'distinctiveness_name': 'Medium'},
    {'habitat_name': 'Wetland - Reedbeds', 'broader_type': 'Wetland', 'distinctiveness_name': 'High'},
    {'habitat_name': 'Heathland - Lowland heathland', 'broader_type': 'Heathland', 'distinctiveness_name': 'High'},
    {'habitat_name': 'Urban - Introduced shrub', 'broader_type': 'Urban', 'distinctiveness_name': 'Low'},
])

# Sample Stock
stock_data = pd.DataFrame([
    {'stock_id': 'STK001', 'bank_id': 'BNK001', 'habitat_name': 'Grassland - Modified grassland', 
     'quantity_available': 150.0, 'available_excl_quotes': 150.0, 'quoted': 0.0},
    {'stock_id': 'STK002', 'bank_id': 'BNK001', 'habitat_name': 'Grassland - Other neutral grassland', 
     'quantity_available': 80.0, 'available_excl_quotes': 100.0, 'quoted': 20.0},
    {'stock_id': 'STK003', 'bank_id': 'BNK001', 'habitat_name': 'Grassland - Lowland meadows', 
     'quantity_available': 45.0, 'available_excl_quotes': 45.0, 'quoted': 0.0},
    {'stock_id': 'STK004', 'bank_id': 'BNK002', 'habitat_name': 'Woodland - Mixed plantation', 
     'quantity_available': 200.0, 'available_excl_quotes': 200.0, 'quoted': 0.0},
    {'stock_id': 'STK005', 'bank_id': 'BNK002', 'habitat_name': 'Woodland - Other woodland; broadleaved', 
     'quantity_available': 120.0, 'available_excl_quotes': 150.0, 'quoted': 30.0},
    {'stock_id': 'STK006', 'bank_id': 'BNK002', 'habitat_name': 'Woodland - Lowland beech and yew woodland', 
     'quantity_available': 60.0, 'available_excl_quotes': 60.0, 'quoted': 0.0},
    {'stock_id': 'STK007', 'bank_id': 'BNK003', 'habitat_name': 'Wetland - Ponds (non-priority)', 
     'quantity_available': 35.0, 'available_excl_quotes': 35.0, 'quoted': 0.0},
    {'stock_id': 'STK008', 'bank_id': 'BNK003', 'habitat_name': 'Wetland - Reedbeds', 
     'quantity_available': 25.0, 'available_excl_quotes': 25.0, 'quoted': 0.0},
    {'stock_id': 'STK009', 'bank_id': 'BNK001', 'habitat_name': 'Urban - Introduced shrub', 
     'quantity_available': 90.0, 'available_excl_quotes': 90.0, 'quoted': 0.0},
])

# Sample Pricing (different contract sizes and tiers)
pricing_data = []
for habitat in habitats_data['habitat_name']:
    broader_type = habitats_data[habitats_data['habitat_name'] == habitat]['broader_type'].iloc[0]
    dist_name = habitats_data[habitats_data['habitat_name'] == habitat]['distinctiveness_name'].iloc[0]
    
    # Base prices by distinctiveness
    base_prices = {'Low': 8000, 'Medium': 12000, 'High': 18000}
    base_price = base_prices.get(dist_name, 10000)
    
    for bank_id in banks_data['bank_id']:
        for size in ['small', 'medium', 'large']:
            for tier in ['a1', 'a2', 'b1', 'b2']:
                # Adjust price by tier (A1 most expensive, B2 least)
                tier_multipliers = {'a1': 1.0, 'a2': 0.95, 'b1': 0.90, 'b2': 0.85}
                # Adjust by contract size (larger = cheaper per unit)
                size_multipliers = {'small': 1.0, 'medium': 0.95, 'large': 0.90}
                
                price = base_price * tier_multipliers[tier] * size_multipliers[size]
                
                pricing_data.append({
                    'bank_id': bank_id,
                    'habitat_name': habitat,
                    'contract_size': size,
                    'tier': tier,
                    'price': price,
                    'broader_type': broader_type,
                    'distinctiveness_name': dist_name
                })

pricing_df = pd.DataFrame(pricing_data)

# Distinctiveness Levels
dist_levels = pd.DataFrame([
    {'distinctiveness_name': 'Very Low', 'level_value': 0.0},
    {'distinctiveness_name': 'Low', 'level_value': 2.0},
    {'distinctiveness_name': 'Medium', 'level_value': 4.0},
    {'distinctiveness_name': 'High', 'level_value': 6.0},
    {'distinctiveness_name': 'Very High', 'level_value': 8.0},
])

# SRM data
srm_data = pd.DataFrame([
    {'risk_category': 'Very Low', 'multiplier': 1.0, 'description': 'Same LPA, same NCA'},
    {'risk_category': 'Low', 'multiplier': 1.5, 'description': 'Neighbouring LPA, same NCA'},
    {'risk_category': 'Medium', 'multiplier': 2.0, 'description': 'Same LPA, neighbouring NCA'},
    {'risk_category': 'High', 'multiplier': 2.5, 'description': 'Neighbouring LPA, neighbouring NCA'},
])

# Write to database
banks_data.to_sql('banks', db.conn, if_exists='replace', index=False)
habitats_data.to_sql('habitat_catalog', db.conn, if_exists='replace', index=False)
stock_data.to_sql('stock', db.conn, if_exists='replace', index=False)
pricing_df.to_sql('pricing', db.conn, if_exists='replace', index=False)
dist_levels.to_sql('distinctiveness_levels', db.conn, if_exists='replace', index=False)
srm_data.to_sql('spatial_risk_multipliers', db.conn, if_exists='replace', index=False)

db.conn.commit()
db.close()

print("✓ Sample database created successfully!")
print(f"  Location: data/sample_bng_backend.db")
print(f"  Banks: {len(banks_data)}")
print(f"  Habitats: {len(habitats_data)}")
print(f"  Stock items: {len(stock_data)}")
print(f"  Pricing entries: {len(pricing_df)}")
print(f"  Total available units: {stock_data['quantity_available'].sum():.0f}")
