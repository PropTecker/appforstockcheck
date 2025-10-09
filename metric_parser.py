# metric_parser.py - DEFRA BNG Metric spreadsheet parser
"""
This module parses DEFRA BNG Metric spreadsheets to extract biodiversity unit requirements.
It supports area, hedgerow, and watercourse habitats from the metric calculation sheets.
"""

import pandas as pd
from io import BytesIO
from typing import Dict, List, Tuple
import re


class DEFRAMetricParser:
    """Parser for DEFRA BNG Metric spreadsheets"""
    
    def __init__(self, excel_bytes: bytes):
        """Initialize parser with Excel file bytes"""
        self.excel_bytes = excel_bytes
        self.workbook = pd.ExcelFile(BytesIO(excel_bytes))
        self.results = None
    
    def parse(self) -> pd.DataFrame:
        """
        Parse the metric spreadsheet and return a DataFrame with habitat requirements.
        
        Returns:
            DataFrame with columns: habitat_name, habitat_type, units_required, broader_type
        """
        requirements = []
        
        # Try to find and parse relevant sheets
        # Common sheet names in DEFRA BNG Metric: 
        # - "A-1 Baseline" / "A-2 Post-intervention"
        # - "B-1 Baseline" / "B-2 Post-intervention" (Hedgerow)
        # - "C-1 Baseline" / "C-2 Post-intervention" (Watercourse)
        
        # Parse Area habitats
        area_reqs = self._parse_area_habitats()
        requirements.extend(area_reqs)
        
        # Parse Hedgerow habitats
        hedgerow_reqs = self._parse_hedgerow_habitats()
        requirements.extend(hedgerow_reqs)
        
        # Parse Watercourse habitats
        watercourse_reqs = self._parse_watercourse_habitats()
        requirements.extend(watercourse_reqs)
        
        # Convert to DataFrame
        if requirements:
            df = pd.DataFrame(requirements)
            # Aggregate by habitat_name to combine duplicates
            df = df.groupby(['habitat_name', 'habitat_type', 'broader_type'], as_index=False).agg({
                'units_required': 'sum'
            })
            self.results = df
            return df
        else:
            self.results = pd.DataFrame(columns=['habitat_name', 'habitat_type', 'units_required', 'broader_type'])
            return self.results
    
    def _parse_area_habitats(self) -> List[Dict]:
        """Parse area habitat requirements from metric spreadsheet"""
        requirements = []
        
        # Look for sheets with area habitat data
        area_sheets = [s for s in self.workbook.sheet_names if 
                      any(x in s.lower() for x in ['a-1', 'a-2', 'a1', 'a2', 'area', 'habitat'])]
        
        for sheet_name in area_sheets:
            try:
                df = pd.read_excel(self.workbook, sheet_name)
                
                # Look for columns that might contain habitat names and unit values
                # Common column patterns: "Broad habitat", "Habitat type", "Biodiversity units"
                habitat_cols = [c for c in df.columns if any(x in str(c).lower() for x in 
                               ['habitat', 'broad habitat', 'habitat type'])]
                unit_cols = [c for c in df.columns if any(x in str(c).lower() for x in 
                            ['biodiversity unit', 'unit', 'total unit'])]
                
                if habitat_cols and unit_cols:
                    habitat_col = habitat_cols[0]
                    unit_col = unit_cols[-1]  # Often the last unit column is the total
                    
                    for _, row in df.iterrows():
                        habitat = str(row.get(habitat_col, '')).strip()
                        units = row.get(unit_col, 0)
                        
                        # Skip empty or header rows
                        if habitat and habitat.lower() not in ['habitat', 'total', ''] and pd.notna(units):
                            try:
                                units_val = float(units)
                                if units_val > 0:
                                    requirements.append({
                                        'habitat_name': habitat,
                                        'habitat_type': 'area',
                                        'units_required': units_val,
                                        'broader_type': 'area'
                                    })
                            except (ValueError, TypeError):
                                continue
            except Exception:
                continue
        
        return requirements
    
    def _parse_hedgerow_habitats(self) -> List[Dict]:
        """Parse hedgerow habitat requirements from metric spreadsheet"""
        requirements = []
        
        # Look for hedgerow sheets
        hedgerow_sheets = [s for s in self.workbook.sheet_names if 
                          any(x in s.lower() for x in ['b-1', 'b-2', 'b1', 'b2', 'hedgerow'])]
        
        for sheet_name in hedgerow_sheets:
            try:
                df = pd.read_excel(self.workbook, sheet_name)
                
                # Look for hedgerow-specific columns
                habitat_cols = [c for c in df.columns if any(x in str(c).lower() for x in 
                               ['hedgerow', 'type'])]
                unit_cols = [c for c in df.columns if any(x in str(c).lower() for x in 
                            ['biodiversity unit', 'unit', 'total unit'])]
                
                if habitat_cols and unit_cols:
                    habitat_col = habitat_cols[0]
                    unit_col = unit_cols[-1]
                    
                    for _, row in df.iterrows():
                        habitat = str(row.get(habitat_col, '')).strip()
                        units = row.get(unit_col, 0)
                        
                        if habitat and habitat.lower() not in ['hedgerow', 'total', ''] and pd.notna(units):
                            try:
                                units_val = float(units)
                                if units_val > 0:
                                    requirements.append({
                                        'habitat_name': f"Hedgerow - {habitat}",
                                        'habitat_type': 'hedgerow',
                                        'units_required': units_val,
                                        'broader_type': 'hedgerow'
                                    })
                            except (ValueError, TypeError):
                                continue
            except Exception:
                continue
        
        return requirements
    
    def _parse_watercourse_habitats(self) -> List[Dict]:
        """Parse watercourse habitat requirements from metric spreadsheet"""
        requirements = []
        
        # Look for watercourse sheets
        watercourse_sheets = [s for s in self.workbook.sheet_names if 
                             any(x in s.lower() for x in ['c-1', 'c-2', 'c1', 'c2', 'watercourse', 'river'])]
        
        for sheet_name in watercourse_sheets:
            try:
                df = pd.read_excel(self.workbook, sheet_name)
                
                # Look for watercourse-specific columns
                habitat_cols = [c for c in df.columns if any(x in str(c).lower() for x in 
                               ['watercourse', 'type', 'habitat'])]
                unit_cols = [c for c in df.columns if any(x in str(c).lower() for x in 
                            ['biodiversity unit', 'unit', 'total unit'])]
                
                if habitat_cols and unit_cols:
                    habitat_col = habitat_cols[0]
                    unit_col = unit_cols[-1]
                    
                    for _, row in df.iterrows():
                        habitat = str(row.get(habitat_col, '')).strip()
                        units = row.get(unit_col, 0)
                        
                        if habitat and habitat.lower() not in ['watercourse', 'total', ''] and pd.notna(units):
                            try:
                                units_val = float(units)
                                if units_val > 0:
                                    requirements.append({
                                        'habitat_name': f"Watercourse - {habitat}",
                                        'habitat_type': 'watercourse',
                                        'units_required': units_val,
                                        'broader_type': 'watercourse'
                                    })
                            except (ValueError, TypeError):
                                continue
            except Exception:
                continue
        
        return requirements
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of units by habitat type"""
        if self.results is None:
            return {}
        
        summary = {}
        for habitat_type in ['area', 'hedgerow', 'watercourse']:
            type_df = self.results[self.results['habitat_type'] == habitat_type]
            summary[habitat_type] = type_df['units_required'].sum()
        
        return summary
