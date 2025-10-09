# BNG Optimiser + DEFRA Metric Reader

Combined application for Biodiversity Net Gain (BNG) optimization and DEFRA BNG Metric reading.

## Features

### Core Functionality
- **BNG Optimization**: Allocate biodiversity units from banks to development sites based on trading rules, proximity, and cost
- **DEFRA Metric Reading**: Parse DEFRA BNG Metric spreadsheets to automatically extract habitat requirements
- **Database Backend**: SQLite database for managing banks, pricing, and stock data (replacing Excel workbooks)
- **Flow Visualization**: Interactive Sankey diagrams showing trading flows from demand to supply
- **Map Visualization**: Interactive maps showing bank locations and catchment areas
- **Quote Generation**: Generate client-facing quotes with email templates

### Input Methods

#### 1. Manual Demand Entry
- Add habitats one by one from the catalog
- Specify units required for each habitat
- Support for Net Gain (Low-equivalent) entries

#### 2. DEFRA Metric Upload
- Upload completed DEFRA BNG Metric spreadsheets (.xlsx)
- Automatically parse:
  - Area habitats (from A-1, A-2 sheets)
  - Hedgerow habitats (from B-1, B-2 sheets)
  - Watercourse habitats (from C-1, C-2 sheets)
- View summary by habitat type

### Backend Data Management

#### Excel Workbook (Legacy)
Upload Excel workbook with sheets:
- Banks: Bank locations, postcodes, LPA/NCA
- Pricing: Unit prices by habitat, bank, tier, contract size
- HabitatCatalog: Available habitats and classifications
- Stock: Available units by bank and habitat
- DistinctivenessLevels: Distinctiveness hierarchy
- SRM: Spatial Risk Multiplier rules
- TradingRules: Custom trading rules (optional)

#### Database (New)
- Initialize from Excel workbook
- Persistent SQLite database in `data/bng_backend.db`
- Editable and queryable
- Better performance for large datasets

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Architecture

### New Modules

#### `database.py`
- `BNGDatabase`: SQLite database handler
- Methods for initialization, loading from Excel, and querying
- Returns data in same format as Excel backend for compatibility

#### `metric_parser.py`
- `DEFRAMetricParser`: Parse DEFRA BNG Metric spreadsheets
- Extracts habitat requirements from area, hedgerow, and watercourse sheets
- Handles various metric format variations

#### `flow_viz.py`
- `create_flow_diagram()`: Generate Sankey flow diagrams using Plotly
- `create_simple_flow_table()`: Fallback table view of trading flows

### Integration Points

1. **Backend Loading** (app.py lines ~295-395)
   - Radio button to choose Excel or Database mode
   - Database initialization from Excel
   - Caching for both modes

2. **Demand Input** (app.py lines ~830-970)
   - Tab interface for Manual Entry vs Metric Upload
   - Metric parsing on file upload
   - Unified demand DataFrame for optimization

3. **Results Visualization** (app.py lines ~1880-1900)
   - Flow diagram after allocation results
   - Integrated with existing map and table views

## Trading Rules

The optimizer follows BNG trading rules:
- **Low distinctiveness**: Can be offset by any habitat
- **Medium distinctiveness**: Must be same broader type OR higher distinctiveness
- **High/Very High distinctiveness**: Like-for-like only
- **Proximity tiers**: Local (1x), Adjacent (1.33x), Far (2x) multipliers

## Optimization Strategy

1. **Stage A**: Minimize total cost
2. **Stage B**: Given minimum cost, minimize number of banks used
3. Uses PuLP linear programming solver (falls back to greedy if unavailable)

## Configuration

### Constants (app.py)
- `ADMIN_FEE_GBP = 500.0`: Admin fee added to quotes
- `SINGLE_BANK_SOFT_PCT = 0.01`: Preference for single-bank solutions
- `MAP_CATCHMENT_ALPHA = 0.03`: Map overlay transparency

## Future Enhancements

Potential improvements:
- Database web UI for editing banks/pricing
- Support for additional metric versions
- Real-time stock updates
- Multi-user access control
- API for external integrations
- Historical quote tracking
