"""
FDA Drug Analytics Dashboard Launcher
Run this script to start the interactive dashboard
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'dash',
        'dash-bootstrap-components',
        'plotly',
        'pandas',
        'numpy',
        'scikit-learn',
        'networkx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        
        response = input("\nWould you like to install missing packages? (y/n): ")
        if response.lower() == 'y':
            for package in missing_packages:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            print("Please install the missing packages manually.")
            sys.exit(1)

def main():
    """Main function to run the dashboard"""
    print("="*60)
    print("FDA DRUG ANALYTICS DASHBOARD")
    print("="*60)
    
    # Check requirements
    print("\nChecking requirements...")
    check_requirements()
    
    # Check if data exists
    data_path = "data/processed"
    if not os.path.exists(data_path):
        print(f"\nError: Data directory '{data_path}' not found!")
        print("Please run the data processing scripts first.")
        sys.exit(1)
    
    # Check for required data files
    required_files = [
        "applications_no_nulls.csv",
        "products_no_nulls.csv",
        "submissions_no_nulls.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(data_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print("\nError: Missing required data files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run the data processing scripts first.")
        sys.exit(1)
    
    print("\nAll requirements satisfied!")
    print("\nStarting FDA Drug Analytics Dashboard...")
    print("-"*60)
    print("Dashboard will be available at: http://localhost:8050")
    print("Press Ctrl+C to stop the server")
    print("-"*60)
    
    # Import and run the dashboard
    try:
        from app.fda_dashboard import app
        app.run(debug=True, port=8050, host='0.0.0.0')
    except ImportError:
        print("\nError: Could not import dashboard app.")
        print("Make sure 'fda_dashboard.py' is in the 'app' directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()