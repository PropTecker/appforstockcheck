"""
DEFRA BNG Metric Reader Module
Parses DEFRA BNG Metric spreadsheets to extract habitat requirements
"""

import pandas as pd
from io import BytesIO
from typing import Dict, List, Tuple, Optional
import re


class DEFRAMetricReader:
    """Parse DEFRA BNG Metric spreadsheets and extract habitat requirements"""
    
    def __init__(self):
        self.metric_data = None
        self.area_habitats = []
        self.hedgerow_habitats = []
        self.watercourse_habitats = []
    
    def load_metric_file(self, file_bytes: bytes) -> bool:
        """Load and parse DEFRA BNG Metric Excel file"""
        try:
            xl = pd.ExcelFile(BytesIO(file_bytes))
            self.metric_data = {}
            
            # Common sheet names in DEFRA BNG Metric files
            sheet_patterns = {
                'area': ['A-1', 'A1', 'Baseline', 'Site Habitat Baseline'],
                'hedgerow': ['B-1', 'B1', 'Hedgerow', 'Hedgerow Baseline'],
                'watercourse': ['C-1', 'C1', 'Watercourse', 'River Baseline'],
            }
            
            # Try to find and load relevant sheets
            for category, patterns in sheet_patterns.items():
                for sheet_name in xl.sheet_names:
                    if any(pattern.lower() in sheet_name.lower() for pattern in patterns):
                        try:
                            self.metric_data[category] = pd.read_excel(xl, sheet_name)
                        except Exception as e:
                            print(f"Warning: Could not read sheet {sheet_name}: {e}")
            
            return len(self.metric_data) > 0
        except Exception as e:
            print(f"Error loading metric file: {e}")
            return False
    
    def parse_area_requirements(self) -> List[Dict]:
        """Parse area habitat requirements from metric file"""
        requirements = []
        
        if 'area' not in self.metric_data:
            return requirements
        
        df = self.metric_data['area']
        
        # Look for columns containing habitat names and unit calculations
        # DEFRA metrics typically have columns like 'Habitat type', 'Units', 'Biodiversity units'
        habitat_cols = [col for col in df.columns if any(
            term in str(col).lower() for term in ['habitat', 'type']
        )]
        
        unit_cols = [col for col in df.columns if any(
            term in str(col).lower() for term in ['unit', 'biodiversity']
        )]
        
        if not habitat_cols or not unit_cols:
            return requirements
        
        habitat_col = habitat_cols[0]
        unit_col = unit_cols[0]
        
        # Extract requirements
        for _, row in df.iterrows():
            habitat = str(row.get(habitat_col, '')).strip()
            try:
                units = float(row.get(unit_col, 0))
            except (ValueError, TypeError):
                units = 0.0
            
            if habitat and units > 0 and habitat.lower() not in ['nan', 'none', '']:
                requirements.append({
                    'habitat_name': habitat,
                    'units': units,
                    'type': 'area'
                })
        
        self.area_habitats = requirements
        return requirements
    
    def parse_hedgerow_requirements(self) -> List[Dict]:
        """Parse hedgerow requirements from metric file"""
        requirements = []
        
        if 'hedgerow' not in self.metric_data:
            return requirements
        
        df = self.metric_data['hedgerow']
        
        # Look for hedgerow-specific columns
        habitat_cols = [col for col in df.columns if any(
            term in str(col).lower() for term in ['hedge', 'type', 'description']
        )]
        
        unit_cols = [col for col in df.columns if any(
            term in str(col).lower() for term in ['unit', 'biodiversity', 'length']
        )]
        
        if not habitat_cols or not unit_cols:
            return requirements
        
        habitat_col = habitat_cols[0]
        unit_col = unit_cols[0]
        
        for _, row in df.iterrows():
            habitat = str(row.get(habitat_col, '')).strip()
            try:
                units = float(row.get(unit_col, 0))
            except (ValueError, TypeError):
                units = 0.0
            
            if habitat and units > 0 and habitat.lower() not in ['nan', 'none', '']:
                # Ensure habitat name includes 'hedgerow' for consistency
                if 'hedgerow' not in habitat.lower():
                    habitat = f"{habitat} Hedgerow"
                
                requirements.append({
                    'habitat_name': habitat,
                    'units': units,
                    'type': 'hedgerow'
                })
        
        self.hedgerow_habitats = requirements
        return requirements
    
    def parse_watercourse_requirements(self) -> List[Dict]:
        """Parse watercourse requirements from metric file"""
        requirements = []
        
        if 'watercourse' not in self.metric_data:
            return requirements
        
        df = self.metric_data['watercourse']
        
        # Look for watercourse-specific columns
        habitat_cols = [col for col in df.columns if any(
            term in str(col).lower() for term in ['water', 'river', 'type', 'description']
        )]
        
        unit_cols = [col for col in df.columns if any(
            term in str(col).lower() for term in ['unit', 'biodiversity', 'length']
        )]
        
        if not habitat_cols or not unit_cols:
            return requirements
        
        habitat_col = habitat_cols[0]
        unit_col = unit_cols[0]
        
        for _, row in df.iterrows():
            habitat = str(row.get(habitat_col, '')).strip()
            try:
                units = float(row.get(unit_col, 0))
            except (ValueError, TypeError):
                units = 0.0
            
            if habitat and units > 0 and habitat.lower() not in ['nan', 'none', '']:
                requirements.append({
                    'habitat_name': habitat,
                    'units': units,
                    'type': 'watercourse'
                })
        
        self.watercourse_habitats = requirements
        return requirements
    
    def get_all_requirements(self) -> List[Dict]:
        """Get all parsed requirements (area, hedgerow, watercourse)"""
        self.parse_area_requirements()
        self.parse_hedgerow_requirements()
        self.parse_watercourse_requirements()
        
        return self.area_habitats + self.hedgerow_habitats + self.watercourse_habitats
    
    def to_demand_dataframe(self) -> pd.DataFrame:
        """Convert parsed requirements to demand DataFrame format"""
        all_reqs = self.get_all_requirements()
        
        if not all_reqs:
            return pd.DataFrame(columns=['habitat_name', 'units'])
        
        df = pd.DataFrame(all_reqs)
        
        # Group by habitat_name to aggregate duplicate entries
        if len(df) > 0:
            df = df.groupby('habitat_name', as_index=False).agg({
                'units': 'sum'
            })
        
        return df[['habitat_name', 'units']]
    
    def get_summary(self) -> Dict:
        """Get summary statistics of parsed requirements"""
        all_reqs = self.get_all_requirements()
        
        return {
            'total_requirements': len(all_reqs),
            'area_habitats': len(self.area_habitats),
            'hedgerow_habitats': len(self.hedgerow_habitats),
            'watercourse_habitats': len(self.watercourse_habitats),
            'total_units': sum(req['units'] for req in all_reqs),
            'area_units': sum(req['units'] for req in self.area_habitats),
            'hedgerow_units': sum(req['units'] for req in self.hedgerow_habitats),
            'watercourse_units': sum(req['units'] for req in self.watercourse_habitats),
        }
