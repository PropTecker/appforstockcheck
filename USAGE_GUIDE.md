# BNG Optimiser with DEFRA Metric Reader - Usage Guide

## Overview

This application combines biodiversity net gain (BNG) optimization with DEFRA BNG Metric parsing to help you:
1. Parse habitat requirements from DEFRA BNG Metric spreadsheets
2. Optimize habitat bank allocations to meet those requirements
3. Visualize results on an interactive map
4. Generate client quotes and reports

## Step-by-Step Workflow

### 1. Backend Setup

Choose your backend data source in the sidebar:

#### Option A: Use Database (Recommended)
- Select "Database (SQL)" in the sidebar
- Enter database path (e.g., `bng_backend.db` or `data/sample_bng_backend.db`)
- Database contains: banks, stock, pricing, and habitat information

**First-time setup**: If starting fresh, use "Initialize/Reset DB from Excel" to create a database from an existing Excel backend file.

#### Option B: Use Excel Upload
- Select "Excel Upload" in the sidebar
- Upload your backend workbook (.xlsx)
- Backend should contain sheets: Banks, Stock, Pricing, HabitatCatalog, DistinctivenessLevels, SRM

### 2. Upload DEFRA BNG Metric (Optional)

If you have a DEFRA BNG Metric spreadsheet:

1. In the sidebar under "DEFRA BNG Metric", upload your .xlsx file
2. The app will detect sheets for:
   - Area habitats (A-1, Site Habitat Baseline)
   - Hedgerow (B-1, Hedgerow Baseline)
   - Watercourse (C-1, Watercourse Baseline)
3. Click "Import from Metric" to auto-populate habitat requirements
4. Review and adjust the imported habitats as needed

**Note**: You can also manually add habitats in Step 4 without a metric file.

### 3. Locate Target Site

Enter your development site location:

- **Postcode** (faster): e.g., "SW1A 1AA", "HP4 1AA"
- **Address** (if no postcode): e.g., "10 Downing Street, London"

Click "Locate" to:
- Find the Local Planning Authority (LPA)
- Find the National Character Area (NCA)
- Display site location on map
- Identify neighboring LPAs and NCAs for trading

The map will show:
- 🎯 Red border: Target LPA
- 🎯 Orange border: Target NCA
- Red marker: Your development site

### 4. Define Demand (Habitat Requirements)

Add the habitats and biodiversity units you need to offset:

#### Auto-import from Metric
If you uploaded a metric file, click "Import from Metric" to automatically populate this section.

#### Manual Entry
- Click "➕ Add habitat" to add a row
- Select habitat from dropdown (searchable)
- Enter biodiversity units required
- Click "🗑️" to remove a row
- Click "🧹 Clear all" to start over

**Special option**: "➕ Net Gain (Low-equivalent)" adds a flexible requirement that can be met by any habitat (trades like Low distinctiveness).

The total units required will be displayed below the entry form.

### 5. Run Optimization

Click "Optimise now" to:
- Find the best bank allocations
- Minimize cost while meeting all requirements
- Respect trading rules (distinctiveness, spatial risk)
- Prefer fewer banks when possible

**Results include**:
- Total cost breakdown (units + admin fee)
- Detailed allocation table showing which banks supply which habitats
- Site/habitat totals
- Bank totals

### 6. View Results

#### Interactive Map
The map updates automatically to show:
- 🏢 Green dotted borders: Bank catchment areas (LPA + NCA)
- Green markers: Selected banks with allocation details
- Lines connecting your site to each bank
- Click markers for detailed information

#### Tables
- **Allocation detail**: Full breakdown by habitat and bank
- **Site/Habitat totals**: Requirements vs. supply
- **Bank totals**: Units allocated per bank

#### Downloads
Export results as CSV:
- Site summary
- By habitat
- Order summary

### 7. Generate Client Report

Scroll to the bottom for email/report generation:

1. Enter client details:
   - Client name
   - Reference number
   - Site location
2. Review the client report table
3. Choose an option:
   - **Copy Email HTML**: Get formatted HTML to paste in email
   - **Download .eml**: Get email file to open in email client
   - **Open in default email**: Launch email with pre-filled content

The email includes:
- Professional formatting
- Habitat breakdown by type (area, hedgerow, watercourse)
- Pricing per unit
- Total cost with admin fee

## Tips and Best Practices

### Trading Rules
- **Low distinctiveness**: Can be supplied by any habitat
- **Medium**: Requires Medium or High supply
- **High**: Requires High supply only
- Same broader habitat type preferred

### Spatial Risk
- Best: Same LPA, same NCA (1x multiplier)
- Good: Neighboring LPA, same NCA (1.5x)
- OK: Same LPA, neighboring NCA (2x)
- Higher cost: Neighboring LPA, neighboring NCA (2.5x)

### Quotes Policy
Control how quoted/reserved units affect availability:
- **Ignore quotes**: All units available (default)
- **Quotes hold 100%**: Quoted units fully reserved
- **Quotes hold 50%**: Half of quoted units reserved

### Contract Sizes
The optimizer automatically selects the best contract size (small, medium, large) based on total units required.

## Troubleshooting

### "No legal options for: [habitat name]"
- Check if habitat exists in backend stock
- Verify trading rules allow the habitat combination
- Ensure sufficient distinctiveness level in supply

### "Upload your backend workbook or configure database to continue"
- Database might be empty or invalid
- Try initializing database from Excel
- Or switch to Excel upload mode

### Metric file not parsing correctly
- Verify file is DEFRA BNG Metric format
- Check sheet names include A-1, B-1, C-1 or similar
- Ensure "Biodiversity units" column exists
- Manually adjust imported habitats if needed

### Map not showing banks
- Ensure banks have location data (postcode, address, or lat/lon)
- Check internet connection for geocoding
- Some geocoding may be rate-limited

### Optimization taking too long
- Large problems (many habitats × many banks) can be slow
- Reduce number of habitats or simplify requirements
- Consider using smaller contract sizes

## Support

For issues, feature requests, or questions:
- Check the GitHub repository issues
- Review the main README.md for technical details
- Contact the development team

## Version History

- **v10.0**: Combined BNG Optimiser and DEFRA Metric Reader
  - Added database backend support
  - Added DEFRA BNG Metric parsing
  - Maintained all original visualization and reporting features
