import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load all cleaned datasets"""
    print("Loading cleaned FDA data for EDA...")
    
    tables = {}
    table_names = ['applications', 'products', 'submissions', 'marketing_status']
    
    for table in table_names:
        file_path = f'data/processed/{table}_no_nulls.csv'
        if os.path.exists(file_path):
            tables[table] = pd.read_csv(file_path)
            print(f"Loaded {table}: {tables[table].shape}")
    
    return tables

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

def safe_value_counts_plot(series, ax, plot_type='bar', title='', top_n=None, **kwargs):
    """Safely plot value counts with proper handling"""
    try:
        value_counts = series.value_counts()
        if top_n:
            value_counts = value_counts.head(top_n)
        
        if plot_type == 'bar':
            value_counts.plot(kind='bar', ax=ax, **kwargs)
            # Add count labels
            for i, v in enumerate(value_counts.values):
                ax.text(i, v + 0.01 * value_counts.max(), f'{v:,}', ha='center', fontsize=9)
        elif plot_type == 'barh':
            value_counts.plot(kind='barh', ax=ax, **kwargs)
        elif plot_type == 'pie':
            # For pie charts, create appropriate labels
            labels = []
            for idx in value_counts.index:
                if isinstance(idx, (int, float)):
                    if idx in [0, 1]:
                        labels.append('No' if idx == 0 else 'Yes')
                    else:
                        labels.append(f'Value: {idx}')
                else:
                    labels.append(str(idx)[:20])  # Truncate long labels
            
            value_counts.plot(kind='pie', ax=ax, labels=labels, **kwargs)
            
        ax.set_title(title, fontsize=14)
        return True
    except Exception as e:
        ax.text(0.5, 0.5, f'Error plotting:\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title + ' (Error)', fontsize=14)
        return False

def basic_data_overview(tables):
    """Provide basic overview of the datasets"""
    print_section("1. BASIC DATA OVERVIEW")
    
    # Overall statistics
    total_records = sum(len(df) for df in tables.values())
    print(f"\nTotal Records Across All Tables: {total_records:,}")
    
    for name, df in tables.items():
        print(f"\n{name.upper()} Table:")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"  Columns: {', '.join(df.columns)}")
        
        # Data types
        print(f"\n  Data Types:")
        for dtype, count in df.dtypes.value_counts().items():
            print(f"    {dtype}: {count} columns")

def univariate_analysis(tables):
    """Perform univariate analysis on key variables"""
    print_section("2. UNIVARIATE ANALYSIS")
    
    # Create figure for univariate plots
    fig = plt.figure(figsize=(20, 15))
    
    applications = tables['applications']
    submissions = tables['submissions']
    products = tables['products']
    
    # 1. Application Types Distribution
    ax1 = plt.subplot(3, 3, 1)
    safe_value_counts_plot(applications['ApplType'], ax1, 'bar', 
                          'Distribution of Application Types')
    ax1.set_xlabel('Application Type')
    ax1.set_ylabel('Count')
    
    # 2. Submission Status Distribution
    ax2 = plt.subplot(3, 3, 2)
    safe_value_counts_plot(submissions['SubmissionStatus'], ax2, 'bar', 
                          'Top 10 Submission Statuses', top_n=10, color='orange')
    ax2.set_xlabel('Status')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Review Priority Distribution
    ax3 = plt.subplot(3, 3, 3)
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    safe_value_counts_plot(submissions['ReviewPriority'], ax3, 'pie', 
                          'Review Priority Distribution', autopct='%1.1f%%', colors=colors[:3])
    ax3.set_ylabel('')
    
    # 4. Drug Forms Distribution
    ax4 = plt.subplot(3, 3, 4)
    safe_value_counts_plot(products['Form'], ax4, 'barh', 
                          'Top 15 Drug Forms', top_n=15, color='green')
    ax4.set_xlabel('Count')
    
    # 5. Reference Drug Distribution
    ax5 = plt.subplot(3, 3, 5)
    # First check what values are in ReferenceDrug
    ref_drug_unique = products['ReferenceDrug'].unique()
    print(f"\nReferenceDrug unique values: {ref_drug_unique}")
    
    safe_value_counts_plot(products['ReferenceDrug'], ax5, 'pie', 
                          'Reference Drug Distribution', autopct='%1.1f%%')
    ax5.set_ylabel('')
    
    # 6. Marketing Status Distribution
    ax6 = plt.subplot(3, 3, 6)
    if 'marketing_status' in tables:
        marketing = tables['marketing_status']
        safe_value_counts_plot(marketing['MarketingStatusID'], ax6, 'bar', 
                              'Marketing Status Distribution', color='purple')
        ax6.set_xlabel('Marketing Status ID')
        ax6.set_ylabel('Count')
    else:
        ax6.text(0.5, 0.5, 'Marketing Status\nData Not Available', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Marketing Status Distribution', fontsize=14)
    
    # 7. Submissions Over Time
    ax7 = plt.subplot(3, 3, 7)
    try:
        submissions['SubmissionStatusDate'] = pd.to_datetime(submissions['SubmissionStatusDate'], errors='coerce')
        submissions['Year'] = submissions['SubmissionStatusDate'].dt.year
        yearly_submissions = submissions['Year'].value_counts().sort_index()
        valid_years = yearly_submissions[(yearly_submissions.index >= 1990) & 
                                        (yearly_submissions.index <= 2024)]
        
        if len(valid_years) > 0:
            valid_years.plot(kind='line', ax=ax7, marker='o', markersize=4)
            ax7.set_title('Submissions Over Time', fontsize=14)
            ax7.set_xlabel('Year')
            ax7.set_ylabel('Number of Submissions')
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'No valid year data', 
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Submissions Over Time (No Data)', fontsize=14)
    except Exception as e:
        ax7.text(0.5, 0.5, f'Error: {str(e)}', 
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Submissions Over Time (Error)', fontsize=14)
    
    # 8. Top Sponsors
    ax8 = plt.subplot(3, 3, 8)
    safe_value_counts_plot(applications['SponsorName'], ax8, 'barh', 
                          'Top 10 Sponsors by Applications', top_n=10, color='brown')
    ax8.set_xlabel('Number of Applications')
    
    # 9. Summary Statistics Box
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    try:
        # Calculate statistics safely
        total_apps = len(applications)
        total_prods = len(products)
        total_subs = len(submissions)
        unique_sponsors = applications['SponsorName'].nunique()
        unique_forms = products['Form'].nunique()
        
        # Safe approval rate calculation
        if 'SubmissionStatus' in submissions.columns:
            approval_rate = (submissions['SubmissionStatus'] == 'AP').mean()
        else:
            approval_rate = 0
        
        # Safe priority rate calculation
        if 'ReviewPriority' in submissions.columns:
            priority_rate = (submissions['ReviewPriority'] == 'PRIORITY').mean()
        else:
            priority_rate = 0
        
        # Safe year range
        if 'Year' in submissions.columns:
            year_min = submissions['Year'].min()
            year_max = submissions['Year'].max()
        else:
            year_min = year_max = 'N/A'
        
        summary_text = f"""
        KEY STATISTICS
        
        Total Applications: {total_apps:,}
        Total Products: {total_prods:,}
        Total Submissions: {total_subs:,}
        
        Unique Sponsors: {unique_sponsors:,}
        Unique Drug Forms: {unique_forms:,}
        
        Approval Rate: {approval_rate:.1%}
        Priority Reviews: {priority_rate:.1%}
        
        Date Range: {year_min} - {year_max}
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except Exception as e:
        ax9.text(0.5, 0.5, f'Error calculating statistics:\n{str(e)}', 
                ha='center', va='center', transform=ax9.transAxes)
    
    plt.tight_layout()
    plt.savefig('data/processed/eda_univariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nUnivariate analysis completed and saved.")

def bivariate_analysis(tables):
    """Perform bivariate analysis"""
    print_section("3. BIVARIATE ANALYSIS")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Merge necessary data
    submissions = tables['submissions'].copy()
    applications = tables['applications'].copy()
    products = tables['products'].copy()
    
    # Ensure Year column exists
    if 'SubmissionStatusDate' in submissions.columns:
        submissions['SubmissionStatusDate'] = pd.to_datetime(submissions['SubmissionStatusDate'], errors='coerce')
        submissions['Year'] = submissions['SubmissionStatusDate'].dt.year
    
    # 1. Approval Rate by Application Type
    ax1 = plt.subplot(2, 3, 1)
    try:
        app_submissions = submissions.merge(applications[['ApplNo', 'ApplType']], on='ApplNo')
        approval_by_type = app_submissions.groupby('ApplType').agg({
            'SubmissionStatus': lambda x: (x == 'AP').mean()
        })['SubmissionStatus']
        
        approval_by_type.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Approval Rate by Application Type', fontsize=14)
        ax1.set_xlabel('Application Type')
        ax1.set_ylabel('Approval Rate')
        ax1.set_ylim(0, 1)
        
        # Add percentage labels
        for i, v in enumerate(approval_by_type.values):
            ax1.text(i, v + 0.01, f'{v:.1%}', ha='center')
    except Exception as e:
        ax1.text(0.5, 0.5, f'Error: {str(e)}', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Approval Rate by Application Type (Error)', fontsize=14)
    
    # 2. Approval Rate by Review Priority
    ax2 = plt.subplot(2, 3, 2)
    try:
        approval_by_priority = submissions.groupby('ReviewPriority').agg({
            'SubmissionStatus': lambda x: (x == 'AP').mean()
        })['SubmissionStatus']
        
        approval_by_priority.plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Approval Rate by Review Priority', fontsize=14)
        ax2.set_xlabel('Review Priority')
        ax2.set_ylabel('Approval Rate')
        ax2.set_ylim(0, 1)
        
        for i, v in enumerate(approval_by_priority.values):
            ax2.text(i, v + 0.01, f'{v:.1%}', ha='center')
    except Exception as e:
        ax2.text(0.5, 0.5, f'Error: {str(e)}', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Approval Rate by Review Priority (Error)', fontsize=14)
    
    # 3. Products per Application Distribution
    ax3 = plt.subplot(2, 3, 3)
    try:
        products_per_app = products.groupby('ApplNo').size()
        products_per_app[products_per_app <= 20].hist(bins=20, ax=ax3, edgecolor='black')
        ax3.set_title('Distribution of Products per Application', fontsize=14)
        ax3.set_xlabel('Number of Products')
        ax3.set_ylabel('Count')
    except Exception as e:
        ax3.text(0.5, 0.5, f'Error: {str(e)}', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Products per Application (Error)', fontsize=14)
    
    # 4. Submissions per Application
    ax4 = plt.subplot(2, 3, 4)
    try:
        subs_per_app = submissions.groupby('ApplNo').size()
        subs_per_app[subs_per_app <= 30].hist(bins=30, ax=ax4, edgecolor='black', color='green')
        ax4.set_title('Distribution of Submissions per Application', fontsize=14)
        ax4.set_xlabel('Number of Submissions')
        ax4.set_ylabel('Count')
    except Exception as e:
        ax4.text(0.5, 0.5, f'Error: {str(e)}', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Submissions per Application (Error)', fontsize=14)
    
    # 5. Approval Trends Over Time
    ax5 = plt.subplot(2, 3, 5)
    try:
        if 'Year' in submissions.columns:
            yearly_approval = submissions.groupby('Year').agg({
                'SubmissionStatus': [
                    ('Total', 'count'),
                    ('Approved', lambda x: (x == 'AP').sum())
                ]
            })
            yearly_approval.columns = ['Total', 'Approved']
            yearly_approval['ApprovalRate'] = yearly_approval['Approved'] / yearly_approval['Total']
            
            valid_years = yearly_approval[(yearly_approval.index >= 2000) & (yearly_approval.index <= 2023)]
            if len(valid_years) > 0:
                valid_years['ApprovalRate'].plot(kind='line', ax=ax5, marker='o', linewidth=2)
                ax5.set_title('Approval Rate Trend Over Time', fontsize=14)
                ax5.set_xlabel('Year')
                ax5.set_ylabel('Approval Rate')
                ax5.set_ylim(0, 1)
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'No valid year data', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Approval Trends (No Data)', fontsize=14)
        else:
            ax5.text(0.5, 0.5, 'Year column not found', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Approval Trends (No Year Data)', fontsize=14)
    except Exception as e:
        ax5.text(0.5, 0.5, f'Error: {str(e)}', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Approval Trends (Error)', fontsize=14)
    
    # 6. Top Drug-Sponsor Combinations
    ax6 = plt.subplot(2, 3, 6)
    try:
        drug_sponsor = products.merge(applications[['ApplNo', 'SponsorName']], on='ApplNo')
        
        # Handle potential missing DrugName column
        if 'DrugName' in drug_sponsor.columns:
            top_combinations = drug_sponsor.groupby(['SponsorName', 'DrugName']).size().nlargest(10)
            
            # Create labels for the combinations
            labels = []
            for sponsor, drug in top_combinations.index:
                sponsor_str = str(sponsor)[:20] + '...' if len(str(sponsor)) > 20 else str(sponsor)
                drug_str = str(drug)[:20] + '...' if len(str(drug)) > 20 else str(drug)
                labels.append(f"{sponsor_str} - {drug_str}")
            
            ax6.barh(range(len(top_combinations)), top_combinations.values)
            ax6.set_yticks(range(len(top_combinations)))
            ax6.set_yticklabels(labels, fontsize=9)
            ax6.set_title('Top 10 Sponsor-Drug Combinations', fontsize=14)
            ax6.set_xlabel('Count')
        else:
            # If no DrugName, show top sponsors instead
            top_sponsors = drug_sponsor['SponsorName'].value_counts().head(10)
            top_sponsors.plot(kind='barh', ax=ax6)
            ax6.set_title('Top 10 Sponsors by Products', fontsize=14)
            ax6.set_xlabel('Product Count')
    except Exception as e:
        ax6.text(0.5, 0.5, f'Error: {str(e)}', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Top Combinations (Error)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('data/processed/eda_bivariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nBivariate analysis completed and saved.")

def create_remaining_analyses(tables):
    """Create the remaining analysis plots in a simplified manner"""
    print_section("4. ADDITIONAL ANALYSES")
    
    # Create temporal analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    submissions = tables['submissions'].copy()
    
    # Ensure date columns
    if 'SubmissionStatusDate' in submissions.columns:
        submissions['SubmissionStatusDate'] = pd.to_datetime(submissions['SubmissionStatusDate'], errors='coerce')
        submissions['Year'] = submissions['SubmissionStatusDate'].dt.year
        submissions['Month'] = submissions['SubmissionStatusDate'].dt.month
        submissions['Quarter'] = submissions['SubmissionStatusDate'].dt.quarter
    
    # 1. Monthly patterns
    ax1 = axes[0, 0]
    if 'Month' in submissions.columns:
        monthly_pattern = submissions['Month'].value_counts().sort_index()
        monthly_pattern.plot(kind='bar', ax=ax1, color='teal')
        ax1.set_title('Submissions by Month')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Count')
    else:
        ax1.text(0.5, 0.5, 'No month data available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Monthly Patterns (No Data)')
    
    # 2. Quarterly patterns
    ax2 = axes[0, 1]
    if 'Quarter' in submissions.columns:
        quarterly_pattern = submissions['Quarter'].value_counts().sort_index()
        quarterly_pattern.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title('Submissions by Quarter')
        ax2.set_ylabel('')
    else:
        ax2.text(0.5, 0.5, 'No quarter data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Quarterly Patterns (No Data)')
    
    # 3. Distribution summary
    ax3 = axes[1, 0]
    products = tables['products']
    products_per_app = products.groupby('ApplNo').size()
    
    summary_stats = f"""
    Products per Application:
    Mean: {products_per_app.mean():.1f}
    Median: {products_per_app.median():.0f}
    Std Dev: {products_per_app.std():.1f}
    Min: {products_per_app.min()}
    Max: {products_per_app.max()}
    
    Total Applications: {len(products_per_app)}
    """
    
    ax3.text(0.1, 0.9, summary_stats, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax3.axis('off')
    ax3.set_title('Distribution Summary')
    
    # 4. Data quality summary
    ax4 = axes[1, 1]
    applications = tables['applications']
    
    quality_summary = f"""
    Data Quality Summary:
    
    Applications: {len(applications):,} records
    Products: {len(products):,} records
    Submissions: {len(submissions):,} records
    
    Unique Sponsors: {applications['SponsorName'].nunique():,}
    Unique Products: {products['ProductNo'].nunique():,}
    
    No missing values (cleaned)
    """
    
    ax4.text(0.1, 0.9, quality_summary, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Data Quality')
    
    plt.tight_layout()
    plt.savefig('data/processed/eda_additional_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nAdditional analyses completed and saved.")

def create_eda_summary_report(tables):
    """Create a comprehensive EDA summary report"""
    print_section("5. EDA SUMMARY REPORT")
    
    # Key findings
    applications = tables['applications']
    products = tables['products']
    submissions = tables['submissions']
    
    print("\nKEY FINDINGS FROM EDA:")
    
    print("\n1. Data Volume:")
    print(f"   - {len(applications):,} unique drug applications")
    print(f"   - {len(products):,} drug products")
    print(f"   - {len(submissions):,} submissions")
    
    print("\n2. Market Characteristics:")
    print(f"   - {applications['SponsorName'].nunique():,} pharmaceutical companies")
    print(f"   - {products['Form'].nunique():,} different drug forms")
    print(f"   - Average products per application: {products.groupby('ApplNo').size().mean():.1f}")
    
    print("\n3. Regulatory Patterns:")
    approval_rate = (submissions['SubmissionStatus'] == 'AP').mean()
    priority_rate = (submissions['ReviewPriority'] == 'PRIORITY').mean()
    print(f"   - Overall approval rate: {approval_rate:.1%}")
    print(f"   - Priority review rate: {priority_rate:.1%}")
    print(f"   - Most common submission status: {submissions['SubmissionStatus'].mode()[0]}")
    
    # Save summary statistics
    summary_stats = {
        'total_applications': len(applications),
        'total_products': len(products),
        'total_submissions': len(submissions),
        'unique_sponsors': applications['SponsorName'].nunique(),
        'unique_drug_forms': products['Form'].nunique(),
        'approval_rate': approval_rate,
        'priority_rate': priority_rate
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv('data/processed/eda_summary_statistics.csv', index=False)
    print("\n\nSummary statistics saved to: data/processed/eda_summary_statistics.csv")

def main():
    """Main EDA pipeline"""
    print("="*80)
    print("INITIAL EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    
    # Load data
    tables = load_data()
    
    if len(tables) < 3:  # At least need applications, products, and submissions
        print("Error: Could not load required tables")
        return
    
    # Perform EDA steps
    basic_data_overview(tables)
    univariate_analysis(tables)
    bivariate_analysis(tables)
    create_remaining_analyses(tables)
    create_eda_summary_report(tables)
    
    print("\n" + "="*80)
    print("EDA COMPLETE")
    print("="*80)
    print("\nVisualizations saved to:")
    print("  - data/processed/eda_univariate_analysis.png")
    print("  - data/processed/eda_bivariate_analysis.png")
    print("  - data/processed/eda_additional_analysis.png")
    print("  - data/processed/eda_summary_statistics.csv")

if __name__ == "__main__":
    main()