import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# First install chardet if not already installed
try:
    import chardet
except ImportError:
    print("Installing chardet for encoding detection...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])
    import chardet

from data_loader import FDADataLoader
import pandas as pd

def fix_and_load_data():
    """Fix encoding issues and load all FDA data"""
    
    print("=" * 80)
    print("FIXING FDA DATA ENCODING ISSUES")
    print("=" * 80)
    
    # Initialize loader with the fixed version
    loader = FDADataLoader()
    
    # Load all tables
    print("\nLoading FDA tables with encoding detection...")
    tables = loader.load_all_tables()
    
    print("\n" + "=" * 80)
    print("LOADING RESULTS:")
    print("=" * 80)
    
    # Get summary
    if tables:
        summary = loader.get_loading_summary()
        print("\n", summary.to_string(index=False))
        
        # Specific check for problematic tables
        print("\n" + "-" * 80)
        print("DETAILED CHECK FOR PREVIOUSLY FAILED TABLES:")
        print("-" * 80)
        
        for table_name in ['submissions', 'application_docs']:
            if table_name in tables:
                df = tables[table_name]
                print(f"\n✅ {table_name.upper()} - SUCCESSFULLY LOADED")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                print(f"   First few rows:")
                print(df.head(2).to_string())
            else:
                print(f"\n❌ {table_name.upper()} - FAILED TO LOAD")
        
        # Save successfully loaded tables count
        print(f"\n" + "=" * 80)
        print(f"FINAL STATUS: Loaded {len(tables)} out of 12 tables")
        
        missing_tables = set(['applications', 'products', 'submissions', 'marketing_status', 
                            'te', 'application_docs', 'action_types_lookup', 
                            'application_docs_type_lookup', 'marketing_status_lookup', 
                            'submission_class_lookup', 'submission_property_type', 
                            'join_submission_action_type']) - set(tables.keys())
        
        if missing_tables:
            print(f"Missing tables: {', '.join(missing_tables)}")
        else:
            print("✅ ALL TABLES LOADED SUCCESSFULLY!")
        print("=" * 80)
        
    else:
        print("❌ No tables were loaded!")
    
    return tables

if __name__ == "__main__":
    # First, ensure chardet is in requirements.txt
    requirements_path = "requirements.txt"
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            content = f.read()
        
        if 'chardet' not in content:
            print("Adding chardet to requirements.txt...")
            with open(requirements_path, 'a') as f:
                f.write("\n# Encoding detection\nchardet==5.2.0\n")
    
    # Run the fix
    fix_and_load_data()