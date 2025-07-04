import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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

def analyze_numeric_column(df, col, ax, table_name):
    """Analyze and plot numeric column distribution"""
    data = df[col].dropna()
    
    # Remove outliers for better visualization (keep 99% of data)
    q01 = data.quantile(0.005)
    q99 = data.quantile(0.995)
    data_filtered = data[(data >= q01) & (data <= q99)]
    
    # Plot histogram
    ax.hist(data_filtered, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(f'{table_name}.{col}\n(n={len(data):,})', fontsize=10)
    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    
    # Add statistics
    stats_text = f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nMin: {data.min():.2f}\nMax: {data.max():.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=7, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def analyze_categorical_column(df, col, ax, table_name, max_categories=20):
    """Analyze and plot categorical column distribution"""
    value_counts = df[col].value_counts()
    
    if len(value_counts) > max_categories:
        # Show top categories
        value_counts = value_counts.head(max_categories)
        title_suffix = f' (Top {max_categories})'
    else:
        title_suffix = ''
    
    # Plot bar chart
    if len(value_counts) > 10:
        # Horizontal bar for many categories
        value_counts.plot(kind='barh', ax=ax)
        ax.set_xlabel('Count', fontsize=8)
    else:
        # Vertical bar for few categories
        value_counts.plot(kind='bar', ax=ax)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
    
    ax.set_title(f'{table_name}.{col}{title_suffix}\n(n={len(df):,}, unique={df[col].nunique()})', fontsize=10)
    
    # Add percentage labels for top 5
    total = len(df)
    for i, (idx, val) in enumerate(value_counts.head(5).items()):
        pct = (val / total) * 100
        if len(value_counts) > 10:  # Horizontal
            ax.text(val + 0.01 * value_counts.max(), i, f'{pct:.1f}%', 
                   va='center', fontsize=7)
        else:  # Vertical
            ax.text(i, val + 0.01 * value_counts.max(), f'{pct:.1f}%', 
                   ha='center', fontsize=7)

def analyze_date_column(df, col, ax, table_name):
    """Analyze and plot date column distribution"""
    # Convert to datetime
    dates = pd.to_datetime(df[col], errors='coerce').dropna()
    
    if len(dates) == 0:
        ax.text(0.5, 0.5, 'No valid dates', ha='center', va='center')
        ax.set_title(f'{table_name}.{col}')
        return
    
    # Extract year
    years = dates.dt.year
    valid_years = years[(years >= 1980) & (years <= 2025)]
    
    if len(valid_years) > 0:
        # Plot year distribution
        year_counts = valid_years.value_counts().sort_index()
        year_counts.plot(kind='line', ax=ax, marker='o', markersize=4)
        ax.set_title(f'{table_name}.{col}\n(Date range: {dates.min()} to {dates.max()})', fontsize=10)
        ax.set_xlabel('Year', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Invalid date range', ha='center', va='center')
        ax.set_title(f'{table_name}.{col}')

def create_distribution_report(tables):
    """Create comprehensive distribution plots for all columns"""
    
    # Create output directory
    output_dir = 'data/processed/distributions'
    os.makedirs(output_dir, exist_ok=True)
    
    for table_name, df in tables.items():
        print(f"\nAnalyzing {table_name}...")
        
        # Determine number of columns to plot
        n_cols = len(df.columns)
        
        # Skip if too many columns
        if n_cols > 50:
            print(f"  Skipping {table_name} - too many columns ({n_cols})")
            continue
        
        # Calculate subplot layout
        n_rows = int(np.ceil(n_cols / 4))
        fig_height = max(12, n_rows * 3)
        
        # Create figure
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, fig_height))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = np.array([[axes]])
        
        fig.suptitle(f'Column Distributions - {table_name.upper()}', fontsize=16, y=0.995)
        
        # Plot each column
        for idx, col in enumerate(df.columns):
            row = idx // 4
            col_idx = idx % 4
            ax = axes[row, col_idx] if n_rows > 1 else axes[0, col_idx]
            
            # Determine column type and plot accordingly
            if df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (few unique values)
                if df[col].nunique() < 20:
                    analyze_categorical_column(df, col, ax, table_name)
                else:
                    analyze_numeric_column(df, col, ax, table_name)
            elif 'date' in col.lower():
                analyze_date_column(df, col, ax, table_name)
            else:
                analyze_categorical_column(df, col, ax, table_name)
        
        # Remove empty subplots
        for idx in range(n_cols, n_rows * 4):
            row = idx // 4
            col_idx = idx % 4
            ax = axes[row, col_idx] if n_rows > 1 else axes[0, col_idx]
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'{table_name}_distributions.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
        plt.close()

def create_key_insights_summary(tables):
    """Create a summary of key insights from distributions"""
    
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM DISTRIBUTIONS")
    print("="*80)
    
    insights = []
    
    for table_name, df in tables.items():
        table_insights = {
            'table': table_name,
            'numeric_cols': [],
            'categorical_cols': [],
            'date_cols': [],
            'high_cardinality': [],
            'low_cardinality': [],
            'skewed_cols': []
        }
        
        for col in df.columns:
            # Numeric columns
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 20:
                table_insights['numeric_cols'].append(col)
                
                # Check for skewness
                if len(df[col].dropna()) > 0:
                    skewness = df[col].skew()
                    if abs(skewness) > 1:
                        table_insights['skewed_cols'].append(f"{col} (skew={skewness:.2f})")
            
            # Categorical columns
            elif df[col].dtype == 'object' or df[col].nunique() < 20:
                unique_count = df[col].nunique()
                
                if unique_count > 100:
                    table_insights['high_cardinality'].append(f"{col} ({unique_count} unique)")
                elif unique_count <= 10:
                    table_insights['low_cardinality'].append(f"{col} ({unique_count} unique)")
                else:
                    table_insights['categorical_cols'].append(f"{col} ({unique_count} unique)")
            
            # Date columns
            if 'date' in col.lower():
                table_insights['date_cols'].append(col)
        
        insights.append(table_insights)
    
    # Print insights
    for insight in insights:
        print(f"\n{insight['table'].upper()}:")
        
        if insight['numeric_cols']:
            print(f"  Numeric columns ({len(insight['numeric_cols'])}): {', '.join(insight['numeric_cols'][:5])}")
        
        if insight['categorical_cols']:
            print(f"  Categorical columns ({len(insight['categorical_cols'])}): {', '.join(insight['categorical_cols'][:5])}")
        
        if insight['date_cols']:
            print(f"  Date columns: {', '.join(insight['date_cols'])}")
        
        if insight['high_cardinality']:
            print(f"  High cardinality: {', '.join(insight['high_cardinality'][:3])}")
        
        if insight['low_cardinality']:
            print(f"  Low cardinality (good for grouping): {', '.join(insight['low_cardinality'][:5])}")
        
        if insight['skewed_cols']:
            print(f"  Skewed distributions: {', '.join(insight['skewed_cols'][:3])}")

def create_summary_dashboard(tables):
    """Create a summary dashboard of key distributions"""
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('FDA Data - Key Distributions Summary', fontsize=20)
    
    # Select key columns to showcase
    showcases = [
        ('applications', 'ApplType', 'categorical'),
        ('products', 'Form', 'categorical'),
        ('submissions', 'SubmissionStatus', 'categorical'),
        ('submissions', 'ReviewPriority', 'categorical'),
        ('submissions', 'SubmissionYear', 'numeric'),
        ('marketing_status', 'MarketingStatusID', 'categorical')
    ]
    
    plot_idx = 1
    for table_name, col_name, col_type in showcases:
        if table_name in tables and col_name in tables[table_name].columns:
            ax = plt.subplot(3, 3, plot_idx)
            df = tables[table_name]
            
            if col_type == 'categorical':
                value_counts = df[col_name].value_counts().head(10)
                value_counts.plot(kind='bar', ax=ax)
                ax.set_title(f'{table_name}.{col_name}', fontsize=12)
                ax.set_xlabel('')
                ax.tick_params(axis='x', rotation=45)
                
                # Add count labels
                for i, v in enumerate(value_counts.values):
                    ax.text(i, v + 0.01 * value_counts.max(), f'{v:,}', 
                           ha='center', fontsize=8)
            
            elif col_type == 'numeric':
                data = df[col_name].dropna()
                if col_name == 'SubmissionYear':
                    # Year distribution
                    year_counts = data.value_counts().sort_index()
                    valid_years = year_counts[(year_counts.index >= 1980) & (year_counts.index <= 2025)]
                    valid_years.plot(kind='line', ax=ax, marker='o')
                    ax.set_title(f'Submissions by Year', fontsize=12)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
                    ax.set_title(f'{table_name}.{col_name}', fontsize=12)
            
            plot_idx += 1
    
    # Add overall statistics in remaining subplot
    ax = plt.subplot(3, 3, 9)
    ax.axis('off')
    
    # Calculate statistics
    total_applications = len(tables.get('applications', []))
    total_products = len(tables.get('products', []))
    total_submissions = len(tables.get('submissions', []))
    
    stats_text = f"""
    OVERALL STATISTICS
    
    Total Applications: {total_applications:,}
    Total Products: {total_products:,}
    Total Submissions: {total_submissions:,}
    
    Unique Sponsors: {tables.get('applications', pd.DataFrame())['SponsorName'].nunique():,}
    Unique Drug Names: {tables.get('products', pd.DataFrame())['DrugName'].nunique():,}
    
    Date Range:
    Earliest: 1939
    Latest: 2024
    """
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save dashboard
    output_path = 'data/processed/distributions/summary_dashboard.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSummary dashboard saved to: {output_path}")
    plt.show()

def main():
    """Main function to analyze all column distributions"""
    
    # Load cleaned data
    print("Loading cleaned data files...")
    tables = load_cleaned_data()
    
    if not tables:
        print("No cleaned data files found!")
        return
    
    print(f"\nLoaded {len(tables)} tables")
    
    # Create distribution plots for all tables
    print("\nCreating distribution plots for all columns...")
    create_distribution_report(tables)
    
    # Generate key insights
    create_key_insights_summary(tables)
    
    # Create summary dashboard
    print("\nCreating summary dashboard...")
    create_summary_dashboard(tables)
    
    print("\n" + "="*80)
    print("DISTRIBUTION ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutputs saved to: data/processed/distributions/")
    print("\nKey findings to consider for next steps:")
    print("- Highly skewed distributions may need transformation")
    print("- High cardinality columns may need grouping")
    print("- Date columns can be used for time-based features")
    print("- Low cardinality columns are good for grouping/aggregation")

if __name__ == "__main__":
    main()