# Changes Summary: Ledger Separation Fix

## Issue Addressed
- Fixed "Optimiser error: 'type'" when using hedgerow/watercourse units
- Fixed illegal cross-trading between different habitat ledgers (e.g., Ditches being traded for Scrub/Orchard)
- Implemented proper three-ledger system: Area, Hedgerow, and Watercourse

## Root Causes
1. **Blanket ban on hedgerow/watercourse**: The original code completely filtered out hedgerow and watercourse habitats from Stock and Pricing, preventing them from being optimised at all.
2. **No ledger separation enforcement**: There was no mechanism to prevent cross-trading between the three separate ledgers (Area, Hedgerow, Watercourse).
3. **Name-based checks only**: The `is_hedgerow()` function only checked for "hedgerow" in the habitat name, not the `broader_type` field from the catalog.
4. **No watercourse detection**: There was no function to identify watercourse habitats.

## Changes Made

### 1. New Helper Functions (lines 102-130)
- **`is_watercourse(name: str)`**: Identifies watercourse habitats by name (ditch, river, stream, water, watercourse)
- **`get_habitat_ledger(habitat_name: str, catalog_df: pd.DataFrame)`**: Determines which ledger a habitat belongs to by checking the `broader_type` field in the catalog, with fallback to name-based checks

### 2. Moved NET_GAIN_LABEL Constant (line 31)
- Moved `NET_GAIN_LABEL = "Net Gain (Low-equivalent)"` to the constants section so it can be referenced by helper functions

### 3. Demand Validation (lines 881-888)
**Before**: Blocked all hedgerow units by name
```python
banned = [h for h in demand_df["habitat_name"] if is_hedgerow(h)]
if banned:
    st.error("Hedgerow units cannot be traded...")
    st.stop()
```

**After**: Validates that all demand habitats are from the same ledger
```python
ledgers = [get_habitat_ledger(h, backend["HabitatCatalog"]) for h in demand_df["habitat_name"]]
unique_ledgers = set(ledgers)
if len(unique_ledgers) > 1:
    st.error(f"Cannot mix habitats from different ledgers...")
    st.stop()
```

### 4. Removed Hedgerow Filtering (lines 477-496)
**Before**: `normalise_pricing()` and Stock processing filtered out all hedgerow/watercourse habitats
```python
df = df[~df["habitat_name"].map(is_hedgerow)].copy()
backend["Stock"] = backend["Stock"][~backend["Stock"]["habitat_name"].map(is_hedgerow)].copy()
```

**After**: Keeps all ledgers intact
```python
# No longer filter out hedgerow/watercourse - keep all ledgers
```

### 5. Ledger-Based Filtering in prepare_options (lines 936-945, 962-982)
**Added**: Determines active ledger from demand and filters Stock and Pricing to only include habitats from that ledger
```python
# Determine which ledger we're working with
demand_ledgers = set([get_habitat_ledger(h, Catalog) for h in demand_df["habitat_name"]])
if len(demand_ledgers) != 1:
    raise RuntimeError(f"All demand habitats must be from the same ledger...")
active_ledger = demand_ledgers.pop()

# Filter stock to only the active ledger
stock_full["_ledger"] = stock_full["habitat_name"].apply(lambda h: get_habitat_ledger(h, Catalog))
stock_full = stock_full[stock_full["_ledger"] == active_ledger].copy()

# Filter pricing to only the active ledger
pc_join["_ledger"] = pc_join["habitat_name"].apply(lambda h: get_habitat_ledger(h, Catalog))
pricing_enriched = pc_join[pc_join["_ledger"] == active_ledger].copy()
```

### 6. Trading Rules Ledger Check (lines 1066-1073)
**Before**: Filtered out hedgerow by name
```python
if is_hedgerow(sh):
    continue
```

**After**: Ensures supply habitat is from the same ledger
```python
if get_habitat_ledger(sh, Catalog) != active_ledger:
    continue
```

### 7. Paired Orchard+Scrub Restriction (line 1144)
**Added**: Restricted paired options to Area ledger only
```python
if active_ledger == "Area" and sstr(d_dist).lower() == "medium" and ORCHARD_NAME and SCRUB_NAME:
```

### 8. Defensive Coding for KeyError (lines 1298, 1395)
**Changed**: Use `.get()` with default value instead of direct dict access
```python
"allocation_type": opt.get("type", "normal"),  # Instead of opt["type"]
```

### 9. Improved Error Messages (lines 1205-1207, 1217-1221)
**Added**: Mention ledger constraints in error messages
```python
raise RuntimeError("No feasible options. Check prices/stock/rules or location tiers. "
                  "Ensure supply habitats exist in the same ledger (Area/Hedgerow/Watercourse) as demand.")
```

## How It Works Now

### Three Separate Ledgers
1. **Area Ledger**: Standard area habitats (Grassland, Woodland, Cropland, etc.)
2. **Hedgerow Ledger**: Habitats with broader_type containing "hedgerow"
3. **Watercourse Ledger**: Habitats with broader_type containing "watercourse" or "water"

### Trading Rules
- **Within-ledger only**: A habitat can only be traded for another habitat from the same ledger
- **No cross-trading**: Area ↔ Hedgerow, Area ↔ Watercourse, Hedgerow ↔ Watercourse are all blocked
- **Explicit validation**: Error messages inform users when they try to mix ledgers

### Example Scenarios

#### Valid: Area demand with Area supply
✅ Demand: Grassland → Supply: Woodland, Scrub, Cropland (all Area)

#### Valid: Watercourse demand with Watercourse supply  
✅ Demand: Ditches → Supply: Other watercourse units (same ledger)

#### Invalid: Watercourse demand with Area supply
❌ Demand: Ditches → Supply: Scrub/Orchard (different ledgers)
→ Error: "No legal options for: Ditches. Check that supply habitats exist in the same ledger."

#### Invalid: Mixed demand
❌ Demand: Grassland + Ditches (mixing Area + Watercourse)
→ Error: "Cannot mix habitats from different ledgers. Found: {'Area', 'Watercourse'}."

## Testing Recommendations
1. Test with pure Area habitats (existing use case)
2. Test with Hedgerow habitats only
3. Test with Watercourse habitats only (e.g., Ditches)
4. Test error handling when mixing ledgers
5. Verify Ditches cannot be traded for Scrub/Orchard anymore
6. Verify Trading Rules respect ledger boundaries

## Benefits
- ✅ Hedgerow and Watercourse units can now be optimised (not blocked entirely)
- ✅ Proper three-ledger separation enforced as per DEFRA requirements
- ✅ Better error messages guide users to correct configuration
- ✅ Defensive coding prevents KeyError exceptions
- ✅ Catalog-based ledger detection is more accurate than name-only checks
