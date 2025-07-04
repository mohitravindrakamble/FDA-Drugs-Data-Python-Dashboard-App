import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import FDADataLoader
import pandas as pd
import numpy as np
from datetime import datetime

def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"{title.upper()}")
    print("=" * 80)

def explore_table(df, table_name):
    """Detailed exploration of a single table"""
    print(f"\n{'='*60}")
    print(f"TABLE: {table_name.upper()}")
    print(f"{'='*60}")
    
    # Basic info
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column information
    print("\nCOLUMN INFORMATION:")
    print("-" * 60)
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        print(f"{col:<30} | {str(dtype):<15} | "
              f"Nulls: {null_count:>7,} ({null_pct:>5.1f}%) | "
              f"Unique: {unique_count:>7,}")
    
    # Sample data
    print("\nFIRST 3 ROWS:")
    print("-" * 60)
    print(df.head(3).to_string())
    
    # Data quality issues
    print("\nDATA QUALITY ISSUES:")
    print("-" * 60)
    
    # Check for duplicates
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows: {dup_count:,}")
    
    # Check for completely null columns
    null_cols = [col for col in df.columns if df[col].isnull().all()]
    if null_cols:
        print(f"Completely null columns: {', '.join(null_cols)}")
    else:
        print("No completely null columns")
    
    # For specific columns, show value distributions
    print("\nVALUE DISTRIBUTIONS FOR KEY COLUMNS:")
    print("-" * 60)
    
    # Show distributions for categorical columns with < 20 unique values
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() < 20:
            print(f"\n{col}:")
            value_counts = df[col].value_counts().head(10)
            for value, count in value_counts.items():
                pct = (count / len(df)) * 100
                print(f"  {str(value):<30} : {count:>7,} ({pct:>5.1f}%)")

def explore_date_columns(df, table_name):
    """Explore date columns in the dataframe"""
    date_cols = []
    
    # Find potential date columns
    for col in df.columns:
        if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]':
            date_cols.append(col)
        elif df[col].dtype == 'object':
            # Try to parse as date
            try:
                sample = df[col].dropna().head(100)
                pd.to_datetime(sample, errors='coerce')
                if pd.to_datetime(sample, errors='coerce').notna().sum() > 50:
                    date_cols.append(col)
            except:
                pass
    
    if date_cols:
        print(f"\nDATE COLUMNS ANALYSIS FOR {table_name}:")
        print("-" * 60)
        for col in date_cols:
            try:
                if df[col].dtype != 'datetime64[ns]':
                    date_series = pd.to_datetime(df[col], errors='coerce')
                else:
                    date_series = df[col]
                
                valid_dates = date_series.dropna()
                if len(valid_dates) > 0:
                    print(f"\n{col}:")
                    print(f"  Date range: {valid_dates.min()} to {valid_dates.max()}")
                    print(f"  Valid dates: {len(valid_dates):,} / {len(df):,}")
            except:
                pass

def get_table_relationships(tables):
    """Analyze relationships between tables based on common columns"""
    print_section_header("TABLE RELATIONSHIPS")
    
    # Find common columns between tables
    relationships = []
    table_names = list(tables.keys())
    
    for i, table1 in enumerate(table_names):
        for table2 in table_names[i+1:]:
            cols1 = set(tables[table1].columns)
            cols2 = set(tables[table2].columns)
            common_cols = cols1.intersection(cols2)
            
            if common_cols:
                relationships.append({
                    'Table1': table1,
                    'Table2': table2,
                    'Common_Columns': ', '.join(common_cols)
                })
    
    if relationships:
        rel_df = pd.DataFrame(relationships)
        print("\nCommon columns between tables:")
        print(rel_df.to_string(index=False))
    
    # Check for primary key candidates
    print("\n\nPRIMARY KEY CANDIDATES:")
    print("-" * 60)
    
    for table_name, df in tables.items():
        print(f"\n{table_name}:")
        
        # Check single column uniqueness
        for col in df.columns:
            if df[col].nunique() == len(df) and df[col].notna().all():
                print(f"  - {col} (unique, no nulls)")
        
        # Check common ID columns
        id_cols = [col for col in df.columns if 'ID' in col.upper() or 'NO' in col.upper()]
        for col in id_cols:
            unique_ratio = df[col].nunique() / len(df)
            null_ratio = df[col].isnull().sum() / len(df)
            print(f"  - {col}: {unique_ratio:.1%} unique, {null_ratio:.1%} null")

def main():
    """Main data exploration function"""
    
    # Load data
    print("Loading FDA data...")
    loader = FDADataLoader()
    tables = loader.load_all_tables()
    
    if not tables:
        print("No data loaded!")
        return
    
    # Overall summary
    print_section_header("DATA SUMMARY")
    
    summary_data = []
    for table_name, df in tables.items():
        summary_data.append({
            'Table': table_name,
            'Rows': f"{len(df):,}",
            'Columns': len(df.columns),
            'Memory (MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}",
            'Nulls': f"{df.isnull().sum().sum():,}",
            'Duplicates': f"{df.duplicated().sum():,}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nOVERALL SUMMARY:")
    print(summary_df.to_string(index=False))
    
    total_rows = sum(len(df) for df in tables.values())
    total_memory = sum(df.memory_usage(deep=True).sum() for df in tables.values()) / 1024**2
    print(f"\nTotal rows across all tables: {total_rows:,}")
    print(f"Total memory usage: {total_memory:.1f} MB")
    
    # Analyze relationships
    get_table_relationships(tables)
    
    # Detailed exploration of each table
    print_section_header("DETAILED TABLE EXPLORATION")
    
    # Focus on main tables first
    main_tables = ['applications', 'products', 'submissions', 'marketing_status']
    
    for table_name in main_tables:
        if table_name in tables:
            explore_table(tables[table_name], table_name)
            explore_date_columns(tables[table_name], table_name)
    
    # Then lookup tables
    print_section_header("LOOKUP TABLES")
    
    lookup_tables = [name for name in tables.keys() if 'lookup' in name.lower()]
    for table_name in lookup_tables:
        if table_name in tables:
            print(f"\n{table_name.upper()}:")
            df = tables[table_name]
            print(f"Shape: {df.shape}")
            print("\nContent:")
            print(df.to_string() if len(df) < 20 else df.head(20).to_string())
    
    # Key insights
    print_section_header("KEY INSIGHTS FOR DATA CLEANING")
    
    print("\n1. DATA QUALITY ISSUES TO ADDRESS:")
    print("   - Missing values in key columns")
    print("   - Duplicate records")
    print("   - Inconsistent date formats")
    print("   - Text fields that need standardization")
    
    print("\n2. MAIN ENTITY RELATIONSHIPS:")
    print("   - Applications → Products (via ApplNo)")
    print("   - Applications → Submissions (via ApplNo)")
    print("   - Products → Marketing Status (via ApplNo, ProductNo)")
    
    print("\n3. RECOMMENDED CLEANING STEPS:")
    print("   - Standardize text fields (trim whitespace, consistent case)")
    print("   - Parse and validate date columns")
    print("   - Remove duplicates")
    print("   - Create derived features (approval status, time-based features)")
    print("   - Handle missing values appropriately")
    
    return tables

if __name__ == "__main__":
    tables = main()