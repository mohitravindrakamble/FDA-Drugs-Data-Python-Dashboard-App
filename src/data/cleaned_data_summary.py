import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob

def load_cleaned_data():
    """Load all cleaned data files"""
    cleaned_path = "data/processed/"
    cleaned_files = glob.glob(os.path.join(cleaned_path, "*_no_nulls.csv"))
    
    tables = {}
    for file_path in cleaned_files:
        table_name = os.path.basename(file_path).replace('_no_nulls.csv', '')
        try:
            df = pd.read_csv(file_path)
            tables[table_name] = df
            print(f"Loaded {table_name}: {df.shape}")
        except Exception as e:
            print(f"Error loading {table_name}: {e}")
    
    return tables

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

def summarize_table(df, table_name):
    """Generate detailed summary for a single table"""
    print(f"\n{'-'*60}")
    print(f"TABLE: {table_name.upper()}")
    print(f"{'-'*60}")
    
    # Basic info
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for nulls (should be none or handled)
    null_count = df.isnull().sum().sum()
    print(f"Total Null Values: {null_count}")
    
    # Column info
    print("\nCOLUMN INFORMATION:")
    info_data = []
    for col in df.columns:
        info_data.append({
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null': df[col].notna().sum(),
            'Unique': df[col].nunique(),
            'Sample': str(df[col].iloc[0]) if len(df) > 0 else 'N/A'
        })
    
    info_df = pd.DataFrame(info_data)
    print(info_df.to_string(index=False))
    
    # For key columns, show value distributions
    print("\nKEY COLUMN DISTRIBUTIONS:")
    
    # Identify key columns
    key_columns = []
    for col in df.columns:
        # Include ID columns, status columns, type columns
        if any(keyword in col.lower() for keyword in ['id', 'status', 'type', 'code', 'flag', 'priority']):
            if df[col].nunique() < 50:  # Only if reasonable number of unique values
                key_columns.append(col)
    
    for col in key_columns[:5]:  # Limit to 5 key columns
        print(f"\n{col}:")
        value_counts = df[col].value_counts().head(5)
        for value, count in value_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {str(value):<30} : {count:>7,} ({pct:>5.1f}%)")

def analyze_data_quality(tables):
    """Analyze overall data quality after cleaning"""
    print_section("DATA QUALITY ANALYSIS")
    
    quality_summary = []
    
    for table_name, df in tables.items():
        # Calculate quality metrics
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        
        # Check for empty strings (our null replacement)
        empty_strings = 0
        for col in df.select_dtypes(include=['object']).columns:
            empty_strings += (df[col] == '').sum()
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        
        # Check for special flag columns we created
        flag_columns = [col for col in df.columns if 'flag' in col.lower() or 'missing' in col.lower()]
        
        quality_summary.append({
            'Table': table_name,
            'Rows': f"{len(df):,}",
            'Columns': df.shape[1],
            'Nulls': null_cells,
            'Empty_Strings': empty_strings,
            'Duplicates': duplicate_rows,
            'Flag_Columns': len(flag_columns)
        })
    
    quality_df = pd.DataFrame(quality_summary)
    print("\nQuality Summary:")
    print(quality_df.to_string(index=False))
    
    # Report on flag columns created
    print("\nFlag Columns Created During Null Handling:")
    for table_name, df in tables.items():
        flag_cols = [col for col in df.columns if any(word in col.lower() for word in ['flag', 'missing', 'has'])]
        if flag_cols:
            print(f"\n{table_name}:")
            for col in flag_cols:
                if col in df.columns:
                    true_count = (df[col] == 1).sum() if df[col].dtype in ['int64', 'float64'] else 0
                    print(f"  - {col}: {true_count:,} flagged records")

def analyze_relationships(tables):
    """Analyze relationships between tables"""
    print_section("TABLE RELATIONSHIPS ANALYSIS")
    
    # Check ApplNo consistency
    if 'applications' in tables:
        base_applnos = set(tables['applications']['ApplNo'].unique())
        print(f"\nBase Applications: {len(base_applnos):,} unique ApplNo values")
        
        for table_name, df in tables.items():
            if 'ApplNo' in df.columns and table_name != 'applications':
                table_applnos = set(df['ApplNo'].unique())
                matching = len(base_applnos.intersection(table_applnos))
                orphaned = len(table_applnos - base_applnos)
                
                if orphaned > 0:
                    print(f"\n{table_name}:")
                    print(f"  - Matching ApplNo: {matching:,}")
                    print(f"  - Orphaned ApplNo: {orphaned:,} (not in applications table)")

def create_summary_visualizations(tables):
    """Create summary visualizations"""
    print_section("CREATING VISUALIZATIONS")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('FDA Data Summary After Cleaning', fontsize=16)
    
    # 1. Table sizes
    ax1 = axes[0, 0]
    table_sizes = [(name, len(df)) for name, df in tables.items()]
    table_sizes.sort(key=lambda x: x[1], reverse=True)
    names, sizes = zip(*table_sizes[:10])  # Top 10 tables
    
    ax1.barh(names, sizes)
    ax1.set_xlabel('Number of Records')
    ax1.set_title('Top 10 Tables by Record Count')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(sizes):
        ax1.text(v + 1000, i, f'{v:,}', va='center')
    
    # 2. Data types distribution
    ax2 = axes[0, 1]
    all_dtypes = []
    for table_name, df in tables.items():
        for dtype in df.dtypes:
            all_dtypes.append(str(dtype))
    
    dtype_counts = pd.Series(all_dtypes).value_counts()
    ax2.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    ax2.set_title('Data Types Distribution Across All Columns')
    
    # 3. Empty strings by table (our null replacement)
    ax3 = axes[1, 0]
    empty_counts = []
    table_names = []
    
    for table_name, df in tables.items():
        empty_count = 0
        for col in df.select_dtypes(include=['object']).columns:
            empty_count += (df[col] == '').sum()
        if empty_count > 0:
            empty_counts.append(empty_count)
            table_names.append(table_name)
    
    if empty_counts:
        ax3.bar(table_names[:10], empty_counts[:10])
        ax3.set_xlabel('Table')
        ax3.set_ylabel('Count')
        ax3.set_title('Empty Strings by Table (Null Replacements)')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'No empty strings found', ha='center', va='center')
        ax3.set_title('Empty Strings by Table')
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary stats
    total_rows = sum(len(df) for df in tables.values())
    total_cols = sum(df.shape[1] for df in tables.values())
    total_cells = sum(df.shape[0] * df.shape[1] for df in tables.values())
    total_memory = sum(df.memory_usage(deep=True).sum() for df in tables.values()) / 1024**2
    
    summary_text = f"""
    OVERALL SUMMARY
    
    Total Tables: {len(tables)}
    Total Rows: {total_rows:,}
    Total Columns: {total_cols}
    Total Cells: {total_cells:,}
    Total Memory: {total_memory:.1f} MB
    
    Data Quality:
    - All nulls handled
    - Flag columns created for missing data
    - Text fields preserved
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = 'data/processed/data_summary_after_cleaning.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.show()

def main():
    """Main function to generate comprehensive data summary"""
    
    # Load cleaned data
    print("Loading cleaned data files...")
    tables = load_cleaned_data()
    
    if not tables:
        print("No cleaned data files found in data/processed/")
        print("Please run the null handling script first.")
        return
    
    print(f"\nLoaded {len(tables)} cleaned tables")
    
    # Overall summary
    print_section("OVERALL DATA SUMMARY AFTER NULL HANDLING")
    
    summary_data = []
    for table_name, df in tables.items():
        summary_data.append({
            'Table': table_name,
            'Rows': f"{len(df):,}",
            'Columns': df.shape[1],
            'Memory (MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}",
            'Nulls': df.isnull().sum().sum(),
            'Data Types': ', '.join(df.dtypes.value_counts().index.astype(str))
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))
    
    # Detailed table summaries
    print_section("DETAILED TABLE SUMMARIES")
    
    # Focus on main tables
    main_tables = ['applications', 'products', 'submissions', 'marketing_status']
    
    for table_name in main_tables:
        if table_name in tables:
            summarize_table(tables[table_name], table_name)
    
    # Data quality analysis
    analyze_data_quality(tables)
    
    # Relationship analysis
    analyze_relationships(tables)
    
    # Create visualizations
    create_summary_visualizations(tables)
    
    # Final recommendations
    print_section("NEXT STEPS")
    print("""
    1. Data Standardization:
       - Standardize text fields (uppercase, trim whitespace)
       - Parse and validate date columns
       - Standardize drug names and active ingredients
    
    2. Feature Engineering:
       - Create time-based features from dates
       - Create approval duration metrics
       - Aggregate features by sponsor, drug type, etc.
    
    3. Create Master Dataset:
       - Join key tables on ApplNo
       - Handle one-to-many relationships
       - Create analysis-ready dataset
    
    4. Statistical Analysis:
       - Approval rates by drug type
       - Time series analysis of submissions
       - Sponsor performance metrics
    """)

if __name__ == "__main__":
    main()