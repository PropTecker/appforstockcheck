# Manual Hedgerow and Watercourse Units Feature

## Overview
This feature allows users to manually add hedgerow and watercourse units to their BNG quote after running the main optimisation.

## Key Components

### 1. Session State Variables
- `manual_hedgerow_rows`: List of manually added hedgerow entries
- `manual_watercourse_rows`: List of manually added watercourse entries
- `_next_manual_hedgerow_id`: ID counter for hedgerow entries
- `_next_manual_watercourse_id`: ID counter for watercourse entries

### 2. Helper Functions
- `is_watercourse(name: str)`: Checks if a habitat name contains "watercourse" or "water"
- `get_hedgerow_habitats(catalog_df)`: Returns filtered list of hedgerow habitats from catalog
- `get_watercourse_habitats(catalog_df)`: Returns filtered list of watercourse habitats from catalog

### 3. User Interface
After optimisation completes, a new section appears with:
- **Manual Hedgerow Units** expandable section
  - Habitat selection (dropdown filtered to hedgerow types)
  - Units required (number input)
  - Price per unit (£) (number input)
  - Add/Remove row buttons
  
- **Manual Watercourse Units** expandable section
  - Habitat selection (dropdown filtered to watercourse types)
  - Units required (number input)
  - Price per unit (£) (number input)
  - Add/Remove row buttons

### 4. Report Generation
The `generate_client_report_table_fixed()` function now:
- Accepts manual entry lists as parameters
- Processes manual entries and calculates costs
- Categorizes entries into appropriate sections (Area/Hedgerow/Watercourse)
- Includes manual entry costs in total calculations
- Updates unit totals to include manual entries

### 5. Report Structure
The final client report displays:
```
Development Impact | Mitigation Supplied from Wild Capital
---------------------------------------------------------
Area Habitats
  [area habitat rows]

Hedgerow Habitats
  [hedgerow habitat rows + manual hedgerow entries]

Watercourse Habitats
  [watercourse habitat rows + manual watercourse entries]

Spatial Risk Multiplier
  [empty placeholder sections]

Planning Discharge Pack: £500
Total: [total units] | [total cost]
```

## Usage Flow
1. User runs main optimisation with area habitats
2. Optimisation completes successfully
3. User expands "Manual Hedgerow Units" section
4. User adds hedgerow entries with habitat type, units, and price
5. User expands "Manual Watercourse Units" section
6. User adds watercourse entries with habitat type, units, and price
7. User generates client report
8. Report includes all optimised habitats + manual entries
9. All costs and totals are calculated correctly

## Cost Calculation
- Manual entry cost = units × price_per_unit
- Total hedgerow cost = sum of all manual hedgerow costs
- Total watercourse cost = sum of all manual watercourse costs
- Grand total = optimisation cost + manual hedgerow cost + manual watercourse cost + admin fee (£500)

## Validation
- Only hedgerow habitats can be selected in hedgerow section
- Only watercourse habitats can be selected in watercourse section
- Empty entries (0 units) are ignored in calculations
- All numeric fields validated (non-negative values)
