"""Create a sample DEFRA BNG Metric file for testing"""
import pandas as pd

print("Creating sample DEFRA BNG Metric file...")

writer = pd.ExcelWriter('data/sample_defra_metric.xlsx', engine='openpyxl')

# Area habitats sheet (A-1 - Site Habitat Baseline)
area_data = pd.DataFrame({
    'Habitat type': [
        'Grassland - Modified grassland',
        'Grassland - Other neutral grassland',
        'Woodland - Other woodland; broadleaved',
        'Urban - Introduced shrub'
    ],
    'Distinctiveness': ['Low', 'Medium', 'Medium', 'Low'],
    'Condition': ['Poor', 'Moderate', 'Good', 'N/A'],
    'Strategic significance': ['Low', 'Medium', 'Medium', 'Low'],
    'Area (hectares)': [1.5, 0.8, 1.2, 0.5],
    'Biodiversity units': [6.2, 8.5, 15.3, 2.8],
    'Comments': ['Existing grassland to be lost', 'Adjacent to development', 'Ancient woodland edge', 'Landscaping']
})
area_data.to_excel(writer, sheet_name='A-1 Site Habitat Baseline', index=False)

# Hedgerow sheet (B-1 - Hedgerow Baseline)  
hedgerow_data = pd.DataFrame({
    'Hedgerow type': [
        'Native Hedgerow',
        'Native Hedgerow - associated with bank or ditch',
    ],
    'Distinctiveness': ['Medium', 'High'],
    'Condition': ['Moderate', 'Good'],
    'Length (km)': [0.35, 0.15],
    'Biodiversity units': [4.2, 3.6],
    'Comments': ['Field boundary', 'Historic boundary with ditch']
})
hedgerow_data.to_excel(writer, sheet_name='B-1 Hedgerow Baseline', index=False)

# Watercourse sheet (C-1 - Watercourse Baseline)
watercourse_data = pd.DataFrame({
    'Watercourse type': [
        'Ditch',
        'Stream (other rivers)',
    ],
    'Condition': ['Poor', 'Moderate'],
    'Length (km)': [0.12, 0.25],
    'Biodiversity units': [2.1, 5.8],
    'Comments': ['Farm drainage ditch', 'Small tributary']
})
watercourse_data.to_excel(writer, sheet_name='C-1 Watercourse Baseline', index=False)

# Summary sheet
summary_data = pd.DataFrame({
    'Metric Type': ['Area', 'Hedgerow', 'Watercourse', 'Total'],
    'Baseline Units': [32.8, 7.8, 7.9, 48.5],
    'Post-development Units': [0.0, 0.0, 0.0, 0.0],
    'Units Required': [32.8, 7.8, 7.9, 48.5],
    'Comments': [
        '4 area habitats lost',
        '2 hedgerows removed',
        '2 watercourses affected',
        'Full compensation required'
    ]
})
summary_data.to_excel(writer, sheet_name='Summary', index=False)

writer.close()

print("✓ Sample DEFRA BNG Metric file created!")
print("  Location: data/sample_defra_metric.xlsx")
print("  Contents:")
print("    - Area habitats: 4 types, 32.8 units")
print("    - Hedgerow: 2 types, 7.8 units")
print("    - Watercourse: 2 types, 7.9 units")
print("    - Total: 48.5 biodiversity units required")
