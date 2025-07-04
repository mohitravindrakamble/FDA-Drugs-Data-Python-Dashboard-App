import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import glob
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_data():
    """Load all cleaned data files"""
    cleaned_path = "data/processed/"
    tables = {}
    
    # Load specific tables we need
    key_tables = ['applications', 'products', 'submissions', 'marketing_status']
    
    for table_name in key_tables:
        file_path = os.path.join(cleaned_path, f"{table_name}_no_nulls.csv")
        if os.path.exists(file_path):
            tables[table_name] = pd.read_csv(file_path)
            print(f"Loaded {table_name}: {tables[table_name].shape}")
    
    return tables

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

def drug_approval_analysis(tables):
    """Analyze drug approval patterns and statistics"""
    print_section("1. DRUG APPROVAL ANALYSIS")
    
    # Merge submissions with applications
    submissions = tables['submissions'].copy()
    applications = tables['applications'].copy()
    
    # Convert date column
    submissions['SubmissionStatusDate'] = pd.to_datetime(submissions['SubmissionStatusDate'], errors='coerce')
    submissions['Year'] = submissions['SubmissionStatusDate'].dt.year
    
    # Approval rates by status
    print("\n1.1 Submission Status Distribution:")
    status_counts = submissions['SubmissionStatus'].value_counts()
    total_submissions = len(submissions)
    
    print(f"\nTotal Submissions: {total_submissions:,}")
    print("\nStatus Breakdown:")
    for status, count in status_counts.head(10).items():
        pct = (count / total_submissions) * 100
        print(f"  {status}: {count:,} ({pct:.1f}%)")
    
    # Approval rate calculation (AP = Approved)
    approved = submissions[submissions['SubmissionStatus'] == 'AP']
    approval_rate = (len(approved) / total_submissions) * 100
    print(f"\nOverall Approval Rate: {approval_rate:.1f}%")
    
    # Approval rates by year
    print("\n1.2 Approval Trends Over Time:")
    yearly_stats = submissions[submissions['Year'].between(2010, 2024)].groupby('Year').agg({
        'SubmissionStatus': 'count',
        'ApplNo': 'nunique'
    }).rename(columns={'SubmissionStatus': 'Total_Submissions', 'ApplNo': 'Unique_Applications'})
    
    # Calculate approval rate by year
    yearly_approvals = submissions[(submissions['Year'].between(2010, 2024)) & 
                                 (submissions['SubmissionStatus'] == 'AP')].groupby('Year').size()
    yearly_stats['Approvals'] = yearly_approvals
    yearly_stats['Approval_Rate'] = (yearly_stats['Approvals'] / yearly_stats['Total_Submissions'] * 100).round(1)
    
    print("\nYearly Statistics (2010-2024):")
    print(yearly_stats.to_string())
    
    # Statistical test for trend
    years = yearly_stats.index.values
    approval_rates = yearly_stats['Approval_Rate'].values
    correlation, p_value = stats.pearsonr(years, approval_rates)
    print(f"\nTrend Analysis:")
    print(f"  Correlation coefficient: {correlation:.3f}")
    print(f"  P-value: {p_value:.4f}")
    if p_value < 0.05:
        trend = "increasing" if correlation > 0 else "decreasing"
        print(f"  Significant {trend} trend in approval rates")
    else:
        print("  No significant trend in approval rates")

def review_priority_analysis(tables):
    """Analyze review priority patterns"""
    print_section("2. REVIEW PRIORITY ANALYSIS")
    
    submissions = tables['submissions'].copy()
    
    # Review priority distribution
    priority_counts = submissions['ReviewPriority'].value_counts()
    print("\nReview Priority Distribution:")
    for priority, count in priority_counts.items():
        pct = (count / len(submissions)) * 100
        print(f"  {priority}: {count:,} ({pct:.1f}%)")
    
    # Approval rates by review priority
    print("\nApproval Rates by Review Priority:")
    for priority in ['PRIORITY', 'STANDARD']:
        priority_subs = submissions[submissions['ReviewPriority'] == priority]
        if len(priority_subs) > 0:
            approved = priority_subs[priority_subs['SubmissionStatus'] == 'AP']
            approval_rate = (len(approved) / len(priority_subs)) * 100
            print(f"  {priority}: {approval_rate:.1f}% approval rate (n={len(priority_subs):,})")
    
    # Statistical test - Chi-square test for independence
    contingency_table = pd.crosstab(submissions['ReviewPriority'], 
                                   submissions['SubmissionStatus'] == 'AP')
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\nChi-square Test (Priority vs Approval):")
    print(f"  Chi-square statistic: {chi2:.2f}")
    print(f"  P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  Result: Review priority significantly affects approval probability")
    else:
        print("  Result: No significant relationship between priority and approval")

def sponsor_performance_analysis(tables):
    """Analyze sponsor performance metrics"""
    print_section("3. SPONSOR PERFORMANCE ANALYSIS")
    
    # Merge data
    applications = tables['applications'].copy()
    submissions = tables['submissions'].copy()
    products = tables['products'].copy()
    
    # Create sponsor statistics
    sponsor_stats = applications.groupby('SponsorName').agg({
        'ApplNo': 'count'
    }).rename(columns={'ApplNo': 'Total_Applications'})
    
    # Add product counts
    products_by_sponsor = products.merge(applications[['ApplNo', 'SponsorName']], on='ApplNo', how='left')
    sponsor_product_counts = products_by_sponsor.groupby('SponsorName')['ProductNo'].count()
    sponsor_stats['Total_Products'] = sponsor_product_counts
    
    # Add submission counts and approval rates
    subs_by_sponsor = submissions.merge(applications[['ApplNo', 'SponsorName']], on='ApplNo', how='left')
    sponsor_submission_stats = subs_by_sponsor.groupby('SponsorName').agg({
        'SubmissionNo': 'count',
        'SubmissionStatus': lambda x: (x == 'AP').sum()
    }).rename(columns={'SubmissionNo': 'Total_Submissions', 'SubmissionStatus': 'Approved_Submissions'})
    
    sponsor_stats = sponsor_stats.join(sponsor_submission_stats)
    sponsor_stats['Approval_Rate'] = (sponsor_stats['Approved_Submissions'] / 
                                     sponsor_stats['Total_Submissions'] * 100).round(1)
    
    # Filter top sponsors
    top_sponsors = sponsor_stats.nlargest(20, 'Total_Applications')
    
    print("\nTop 20 Sponsors by Application Count:")
    print(f"{'Sponsor':<40} {'Apps':>6} {'Prods':>6} {'Subs':>6} {'Appr%':>6}")
    print("-" * 70)
    
    for sponsor, row in top_sponsors.iterrows():
        if sponsor and sponsor != 'UNKNOWN':
            print(f"{sponsor[:40]:<40} {int(row['Total_Applications']):>6} "
                  f"{int(row['Total_Products']):>6} {int(row['Total_Submissions']):>6} "
                  f"{row['Approval_Rate']:>6.1f}")
    
    # Statistical analysis of sponsor performance
    print("\nSponsor Performance Statistics:")
    print(f"  Total unique sponsors: {len(sponsor_stats):,}")
    print(f"  Mean applications per sponsor: {sponsor_stats['Total_Applications'].mean():.1f}")
    print(f"  Median applications per sponsor: {sponsor_stats['Total_Applications'].median():.0f}")
    print(f"  Std dev: {sponsor_stats['Total_Applications'].std():.1f}")
    
    # Concentration analysis
    top_10_pct = sponsor_stats.nlargest(int(len(sponsor_stats) * 0.1), 'Total_Applications')
    concentration = (top_10_pct['Total_Applications'].sum() / 
                    sponsor_stats['Total_Applications'].sum() * 100)
    print(f"\nMarket Concentration:")
    print(f"  Top 10% of sponsors account for {concentration:.1f}% of applications")

def drug_form_analysis(tables):
    """Analyze drug forms and formulations"""
    print_section("4. DRUG FORM AND FORMULATION ANALYSIS")
    
    products = tables['products'].copy()
    
    # Top drug forms
    form_counts = products['Form'].value_counts()
    print("\nTop 15 Drug Forms:")
    print(f"{'Form':<40} {'Count':>8} {'Percent':>8}")
    print("-" * 60)
    
    total_products = len(products)
    for form, count in form_counts.head(15).items():
        if form:  # Skip empty strings
            pct = (count / total_products) * 100
            print(f"{form[:40]:<40} {count:>8,} {pct:>7.1f}%")
    
    # Drug form diversity analysis
    print(f"\nDrug Form Diversity:")
    print(f"  Total unique forms: {form_counts.count()}")
    print(f"  Forms with >100 products: {(form_counts > 100).sum()}")
    print(f"  Forms with >1000 products: {(form_counts > 1000).sum()}")
    
    # Reference drug analysis
    ref_drug_pct = (products['ReferenceDrug'] == 1).sum() / len(products) * 100
    print(f"\nReference Drug Statistics:")
    print(f"  Reference drugs: {(products['ReferenceDrug'] == 1).sum():,} ({ref_drug_pct:.1f}%)")
    print(f"  Non-reference drugs: {(products['ReferenceDrug'] == 0).sum():,}")

def time_to_approval_analysis(tables):
    """Analyze time to approval patterns"""
    print_section("5. TIME TO APPROVAL ANALYSIS")
    
    submissions = tables['submissions'].copy()
    
    # Focus on approved submissions with valid dates
    submissions['SubmissionStatusDate'] = pd.to_datetime(submissions['SubmissionStatusDate'], errors='coerce')
    approved = submissions[(submissions['SubmissionStatus'] == 'AP') & 
                         (submissions['SubmissionStatusDate'].notna())]
    
    # Group by application to find first submission and approval
    app_timelines = approved.groupby('ApplNo').agg({
        'SubmissionStatusDate': ['min', 'max'],
        'SubmissionNo': 'count'
    })
    app_timelines.columns = ['First_Submission', 'Final_Approval', 'Total_Submissions']
    
    # Calculate time to approval (in days)
    app_timelines['Days_to_Approval'] = (app_timelines['Final_Approval'] - 
                                        app_timelines['First_Submission']).dt.days
    
    # Filter reasonable values (0 to 10 years)
    reasonable_timelines = app_timelines[(app_timelines['Days_to_Approval'] >= 0) & 
                                       (app_timelines['Days_to_Approval'] <= 3650)]
    
    print(f"\nTime to Approval Statistics (n={len(reasonable_timelines):,} applications):")
    print(f"  Mean: {reasonable_timelines['Days_to_Approval'].mean():.1f} days")
    print(f"  Median: {reasonable_timelines['Days_to_Approval'].median():.1f} days")
    print(f"  Std Dev: {reasonable_timelines['Days_to_Approval'].std():.1f} days")
    print(f"  25th percentile: {reasonable_timelines['Days_to_Approval'].quantile(0.25):.1f} days")
    print(f"  75th percentile: {reasonable_timelines['Days_to_Approval'].quantile(0.75):.1f} days")
    
    # By review priority
    print("\nTime to Approval by Review Priority:")
    priority_times = {}
    
    for priority in ['PRIORITY', 'STANDARD']:
        priority_apps = approved[approved['ReviewPriority'] == priority]['ApplNo'].unique()
        priority_timelines = reasonable_timelines[reasonable_timelines.index.isin(priority_apps)]
        
        if len(priority_timelines) > 0:
            mean_days = priority_timelines['Days_to_Approval'].mean()
            median_days = priority_timelines['Days_to_Approval'].median()
            print(f"  {priority}: mean={mean_days:.1f} days, median={median_days:.1f} days (n={len(priority_timelines):,})")
            priority_times[priority] = priority_timelines['Days_to_Approval']
    
    # Statistical test for difference
    if len(priority_times) == 2:
        statistic, p_value = stats.mannwhitneyu(priority_times['PRIORITY'], 
                                               priority_times['STANDARD'], 
                                               alternative='less')
        print(f"\nMann-Whitney U Test (Priority vs Standard):")
        print(f"  P-value: {p_value:.4f}")
        if p_value < 0.05:
            print("  Priority reviews are significantly faster than standard reviews")

def active_ingredient_analysis(tables):
    """Analyze active ingredients"""
    print_section("6. ACTIVE INGREDIENT ANALYSIS")
    
    products = tables['products'].copy()
    
    # Clean and analyze active ingredients
    products['ActiveIngredient_Clean'] = products['ActiveIngredient'].str.upper().str.strip()
    
    # Single vs combination products
    products['Is_Combination'] = products['ActiveIngredient_Clean'].str.contains(';', na=False)
    
    combination_pct = products['Is_Combination'].sum() / len(products) * 100
    print(f"\nProduct Complexity:")
    print(f"  Single ingredient products: {(~products['Is_Combination']).sum():,} ({100-combination_pct:.1f}%)")
    print(f"  Combination products: {products['Is_Combination'].sum():,} ({combination_pct:.1f}%)")
    
    # Top active ingredients (excluding combinations for clarity)
    single_ingredients = products[~products['Is_Combination']]['ActiveIngredient_Clean'].value_counts()
    
    print("\nTop 20 Active Ingredients (Single-ingredient products):")
    print(f"{'Ingredient':<40} {'Products':>8} {'Percent':>8}")
    print("-" * 60)
    
    for ingredient, count in single_ingredients.head(20).items():
        if ingredient:
            pct = (count / len(products[~products['Is_Combination']])) * 100
            print(f"{ingredient[:40]:<40} {count:>8,} {pct:>7.1f}%")

def statistical_summary(tables):
    """Provide overall statistical summary"""
    print_section("7. STATISTICAL SUMMARY AND KEY FINDINGS")
    
    applications = tables['applications']
    products = tables['products']
    submissions = tables['submissions']
    
    print("\nDATASET OVERVIEW:")
    print(f"  Total Applications: {len(applications):,}")
    print(f"  Total Products: {len(products):,}")
    print(f"  Total Submissions: {len(submissions):,}")
    print(f"  Unique Sponsors: {applications['SponsorName'].nunique():,}")
    print(f"  Unique Drug Names: {products['DrugName'].nunique():,}")
    
    # Key statistical findings
    print("\nKEY STATISTICAL FINDINGS:")
    
    # 1. Application to product ratio
    products_per_app = products.groupby('ApplNo').size()
    print(f"\n1. Products per Application:")
    print(f"   Mean: {products_per_app.mean():.1f}")
    print(f"   Median: {products_per_app.median():.0f}")
    print(f"   Max: {products_per_app.max()}")
    
    # 2. Submission patterns
    subs_per_app = submissions.groupby('ApplNo').size()
    print(f"\n2. Submissions per Application:")
    print(f"   Mean: {subs_per_app.mean():.1f}")
    print(f"   Median: {subs_per_app.median():.0f}")
    print(f"   Max: {subs_per_app.max()}")
    
    # 3. Market concentration (Gini coefficient)
    sponsor_apps = applications['SponsorName'].value_counts().sort_values()
    cumsum = sponsor_apps.cumsum() / sponsor_apps.sum()
    n = len(sponsor_apps)
    gini = (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum))) / n
    
    print(f"\n3. Market Concentration:")
    print(f"   Gini coefficient: {gini:.3f}")
    print(f"   Interpretation: {'High' if gini > 0.6 else 'Moderate' if gini > 0.3 else 'Low'} concentration")
    
    # 4. Temporal patterns
    submissions['Year'] = pd.to_datetime(submissions['SubmissionStatusDate'], errors='coerce').dt.year
    year_range = submissions['Year'].dropna()
    print(f"\n4. Temporal Coverage:")
    print(f"   Date range: {int(year_range.min())} - {int(year_range.max())}")
    print(f"   Most active year: {int(year_range.mode()[0])} ({(year_range == year_range.mode()[0]).sum():,} submissions)")

def main():
    """Run complete statistical analysis"""
    
    # Load data
    print("Loading FDA drug data for statistical analysis...")
    tables = load_cleaned_data()
    
    if len(tables) < 4:
        print("Error: Could not load all required tables")
        return
    
    # Run analyses
    drug_approval_analysis(tables)
    review_priority_analysis(tables)
    sponsor_performance_analysis(tables)
    drug_form_analysis(tables)
    time_to_approval_analysis(tables)
    active_ingredient_analysis(tables)
    statistical_summary(tables)
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()