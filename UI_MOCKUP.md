# UI Mockup - Manual Entry Sections

## Location
The manual entry sections appear **after** the optimization completes and download buttons, but **before** the client report generation section.

## UI Flow

```
┌─────────────────────────────────────────────────────────────┐
│  [Optimization Results]                                      │
│  - Allocation detail table                                   │
│  - Site/Habitat totals                                       │
│  - By bank / By habitat summaries                            │
│  - Order summary with totals                                 │
│  - Download buttons (CSV files)                              │
│  ✓ Map automatically updated message                         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  ────────────────────────────────────────────────────────   │
│  #### ➕ Manual Additions (Hedgerow & Watercourse)          │
│  ℹ️ Add additional hedgerow or watercourse units to         │
│     your quote. These will be included in the final          │
│     client report.                                           │
│                                                              │
│  ▶ 🌿 Manual Hedgerow Units                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ **Add hedgerow habitat entries:**                    │   │
│  │                                                       │   │
│  │ [Habitat ▼]         [Units]  [Price/Unit £]  [🗑️]   │   │
│  │ Native Hedgerow     5.00     100.00           [Del]  │   │
│  │                                                       │   │
│  │ [➕ Add Hedgerow Entry]  [🧹 Clear Hedgerow]         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ▶ 💧 Manual Watercourse Units                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ **Add watercourse habitat entries:**                 │   │
│  │                                                       │   │
│  │ [Habitat ▼]         [Units]  [Price/Unit £]  [🗑️]   │   │
│  │ Watercourse         2.00     200.00           [Del]  │   │
│  │                                                       │   │
│  │ [➕ Add Watercourse Entry]  [🧹 Clear Watercourse]   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  ────────────────────────────────────────────────────────   │
│  #### 📧 Client Report Generation                            │
│  [Form with client name, ref number, location]              │
│  [Generate button]                                           │
│  [Client Report Table Display]                              │
│  [Email generation options]                                 │
└─────────────────────────────────────────────────────────────┘
```

## Field Details

### Hedgerow Section
- **Habitat Dropdown**: Filtered to show only habitats containing "hedgerow"
  - Example: "Native Hedgerow", "Hedgerow with trees", etc.
- **Units**: Number input (min: 0.0, step: 0.01)
- **Price/Unit (£)**: Number input (min: 0.0, step: 1.0)
- **🗑️ Button**: Removes the entry row

### Watercourse Section
- **Habitat Dropdown**: Filtered to show only habitats containing "watercourse" or "water"
  - Example: "Watercourse", "Water body", etc.
- **Units**: Number input (min: 0.0, step: 0.01)
- **Price/Unit (£)**: Number input (min: 0.0, step: 1.0)
- **🗑️ Button**: Removes the entry row

## Report Integration

Manual entries are automatically included in the client report:

```
═══════════════════════════════════════════════════════════
Development Impact | Mitigation Supplied from Wild Capital
═══════════════════════════════════════════════════════════

Area Habitats
─────────────────────────────────────────────────────────
[Optimised area habitat rows]

Hedgerow Habitats
─────────────────────────────────────────────────────────
[Optimised hedgerow rows - if any]
[Manual hedgerow entries ★]  ← Added here

Watercourse Habitats
─────────────────────────────────────────────────────────
[Optimised watercourse rows - if any]
[Manual watercourse entries ★]  ← Added here

Spatial Risk Multiplier
─────────────────────────────────────────────────────────
[Empty sections]

Planning Discharge Pack:                            £500
─────────────────────────────────────────────────────────
Total:  [units] | [units] |                    £[TOTAL]
        (includes manual entries)
```

## Example Scenario

1. User completes optimization with area habitats only
2. Optimization shows: Subtotal £5,000, Admin £500, Total £5,500
3. User adds manual hedgerow entry: 5 units × £100 = £500
4. User adds manual watercourse entry: 2 units × £200 = £400
5. Updated report shows:
   - Area Habitats section (from optimization)
   - Hedgerow Habitats section (manual: £500)
   - Watercourse Habitats section (manual: £400)
   - Planning Discharge Pack: £500
   - **New Total: £6,400** (£5,000 + £500 + £400 + £500)
