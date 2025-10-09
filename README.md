# BNG Optimiser with DEFRA Metric Reader

Combined application integrating BNG Optimiser (Standalone) and DEFRA BNG Metric Reader functionality.

## Features

### Core Functionality
- **BNG Optimiser**: Optimize habitat bank allocations based on biodiversity requirements
- **DEFRA BNG Metric Reader**: Parse DEFRA BNG Metric spreadsheets to auto-populate habitat requirements
- **Database Backend**: SQL database (SQLite) replaces Excel backend for persistent storage
- **Excel Support**: Still supports Excel upload for backward compatibility

### Key Capabilities
- Upload DEFRA BNG Metric files to automatically extract habitat requirements
- Support for area habitats, hedgerow, and watercourse units
- Map visualization of banks, sites, and catchment areas
- Optimization engine using linear programming (PuLP)
- Client quote generation with email templates
- Trading rules enforcement based on habitat distinctiveness

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

### Quick Demo with Sample Data

1. Start the app: `streamlit run app.py`
2. In the sidebar:
   - Select "Database (SQL)" as backend source
   - Enter path: `data/sample_bng_backend.db`
   - Upload metric file: `data/sample_defra_metric.xlsx`
3. Click "Import from Metric" to auto-populate habitat requirements
4. Enter a postcode (e.g., "SW1A 1AA") and click "Locate"
5. Click "Optimise now" to see bank allocations and map visualization

## Database Setup

### Option 1: Initialize from Excel (Recommended)

1. In the app sidebar, select "Database (SQL)" as backend source
2. Click "Initialize/Reset DB from Excel"
3. Upload your Excel backend file
4. The database will be created at `bng_backend.db`

### Option 2: Command-Line Initialization

```bash
python init_database.py path/to/backend.xlsx [output_db_path]
```

Example:
```bash
python init_database.py data/HabitatBackend_WITH_STOCK.xlsx bng_backend.db
```

## Using DEFRA BNG Metric Files

1. Upload your backend (Excel or Database)
2. In the sidebar, upload a DEFRA BNG Metric spreadsheet (.xlsx)
3. Click "Import from Metric" to automatically populate habitat requirements
4. Review and adjust the imported habitats as needed
5. Proceed with location and optimization

The metric reader supports:
- **Area habitats**: From sheets like "A-1", "Baseline", or "Site Habitat Baseline"
- **Hedgerow**: From sheets like "B-1", "Hedgerow", or "Hedgerow Baseline"
- **Watercourse**: From sheets like "C-1", "Watercourse", or "River Baseline"

## Architecture

### Modules

- **app.py**: Main Streamlit application (original BNG Optimiser with enhancements)
- **database.py**: SQLite database handler for stock, pricing, and bank data
- **metric_reader.py**: DEFRA BNG Metric spreadsheet parser
- **init_database.py**: Database initialization utility

### Database Schema

The SQLite database includes tables for:
- `banks`: Bank information with location and LPA/NCA data
- `habitat_catalog`: Master list of habitats
- `stock`: Available habitat units by bank
- `pricing`: Pricing rules by bank, habitat, contract size, and tier
- `distinctiveness_levels`: Habitat distinctiveness values
- `spatial_risk_multipliers`: SRM data
- `trading_rules`: Explicit trading rules

### Data Flow

```
DEFRA Metric File → Metric Reader → Habitat Requirements
                                           ↓
Excel/Database Backend → Stock & Pricing Data
                                           ↓
                              Optimisation Engine
                                           ↓
                      Results (Allocations, Map, Reports)
```

## Key Changes from Original

1. **Added Database Support**: SQLite database option alongside Excel
2. **Metric File Parsing**: Automatic habitat extraction from DEFRA BNG Metric files
3. **Modular Architecture**: Separated database and metric parsing into distinct modules
4. **Backward Compatible**: Excel backend still works as before
5. **Enhanced UI**: Additional upload options and metric file preview

## Configuration

### Backend Mode Selection
- **Database (SQL)**: Uses SQLite database for persistent storage
- **Excel Upload**: Original behavior, loads from Excel file

### Quotes Policy
Controls how quoted units affect availability:
- **Ignore quotes**: All available units can be allocated
- **Quotes hold 100%**: Quoted units are fully reserved
- **Quotes hold 50%**: Half of quoted units are reserved

## Future Enhancements

- Support for additional metric formats
- Enhanced habitat name matching/fuzzy matching
- Database editing UI
- Multi-user support with user authentication
- API endpoints for programmatic access
- Advanced reporting and analytics

## Troubleshooting

### Database Errors
- Ensure database file path is accessible
- Check file permissions
- Try initializing a fresh database from Excel

### Metric Import Issues
- Verify the metric file follows DEFRA BNG format
- Check sheet names match expected patterns
- Review imported data and adjust manually if needed

### Optimization Failures
- Ensure sufficient stock is available
- Check trading rules allow the requested combinations
- Verify all required columns are present in backend data

## License

See repository license file.

## Contact

For issues and feature requests, please use the GitHub issue tracker.
