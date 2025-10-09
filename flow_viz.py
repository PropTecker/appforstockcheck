# flow_viz.py - Flow diagram visualization for BNG trading
"""
This module provides Sankey diagram visualizations for biodiversity unit trading flows.
It shows the flow from demand habitats to supply habitats via banks.
"""

import pandas as pd
from typing import Optional
import streamlit as st


def create_flow_diagram(alloc_df: pd.DataFrame) -> Optional[str]:
    """
    Create a Sankey flow diagram showing habitat trading flows.
    
    Args:
        alloc_df: Allocation DataFrame with demand_habitat, supply_habitat, bank_name, units_supplied
    
    Returns:
        HTML string for the Plotly Sankey diagram, or None if no data
    """
    if alloc_df.empty:
        return None
    
    try:
        import plotly.graph_objects as go
        
        # Prepare data for Sankey
        # Nodes: demand habitats, banks, supply habitats
        demand_habitats = alloc_df['demand_habitat'].unique().tolist()
        banks = alloc_df['bank_name'].unique().tolist()
        supply_habitats = alloc_df['supply_habitat'].unique().tolist()
        
        # Create node list
        all_nodes = []
        node_colors = []
        
        # Add demand habitats (red/orange)
        for h in demand_habitats:
            all_nodes.append(f"Need: {h}")
            node_colors.append("rgba(255, 127, 80, 0.8)")
        
        # Add banks (blue)
        for b in banks:
            all_nodes.append(f"Bank: {b}")
            node_colors.append("rgba(70, 130, 180, 0.8)")
        
        # Add supply habitats (green)
        for h in supply_habitats:
            all_nodes.append(f"Supply: {h}")
            node_colors.append("rgba(60, 179, 113, 0.8)")
        
        # Create node index mapping
        node_idx = {node: i for i, node in enumerate(all_nodes)}
        
        # Create links
        sources = []
        targets = []
        values = []
        link_colors = []
        
        # Link 1: Demand -> Banks
        for _, row in alloc_df.iterrows():
            demand_node = f"Need: {row['demand_habitat']}"
            bank_node = f"Bank: {row['bank_name']}"
            
            sources.append(node_idx[demand_node])
            targets.append(node_idx[bank_node])
            values.append(float(row['units_supplied']))
            link_colors.append("rgba(70, 130, 180, 0.3)")
        
        # Link 2: Banks -> Supply
        for _, row in alloc_df.iterrows():
            bank_node = f"Bank: {row['bank_name']}"
            supply_node = f"Supply: {row['supply_habitat']}"
            
            sources.append(node_idx[bank_node])
            targets.append(node_idx[supply_node])
            values.append(float(row['units_supplied']))
            link_colors.append("rgba(60, 179, 113, 0.3)")
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color=node_colors
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            )
        )])
        
        fig.update_layout(
            title_text="Biodiversity Unit Trading Flow",
            font_size=12,
            height=600
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="flow_diagram")
        
    except ImportError:
        st.warning("Install plotly for flow diagram visualization: pip install plotly")
        return None
    except Exception as e:
        st.warning(f"Could not create flow diagram: {e}")
        return None


def create_simple_flow_table(alloc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple table view of trading flows (fallback if plotly not available).
    
    Args:
        alloc_df: Allocation DataFrame
    
    Returns:
        Aggregated flow DataFrame
    """
    if alloc_df.empty:
        return pd.DataFrame()
    
    # Aggregate flows
    flow_df = alloc_df.groupby(
        ['demand_habitat', 'bank_name', 'supply_habitat', 'tier'], 
        as_index=False
    ).agg({
        'units_supplied': 'sum',
        'cost': 'sum'
    })
    
    flow_df['unit_price'] = flow_df['cost'] / flow_df['units_supplied']
    flow_df = flow_df.rename(columns={
        'demand_habitat': 'Demand',
        'bank_name': 'Bank',
        'supply_habitat': 'Supply',
        'tier': 'Tier',
        'units_supplied': 'Units',
        'cost': 'Total Cost',
        'unit_price': 'Unit Price'
    })
    
    return flow_df
