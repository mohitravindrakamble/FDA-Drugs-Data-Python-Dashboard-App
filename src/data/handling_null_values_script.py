import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import FDADataLoader
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NullValueHandler:
    """Handle null values in FDA drug data with appropriate strategies"""
    
    def __init__(self, tables):
        self.tables = tables
        self.null_report = {}
        
    def analyze_nulls(self, df, table_name):
        """Analyze null values in a dataframe"""
        null_counts = df.isnull().sum()
        null_percentages = (null_counts / len(df)) * 100
        
        null_info = pd.DataFrame({
            'Column': null_counts.index,
            'Null_Count': null_counts.values,
            'Null_Percentage': null_percentages.values
        })
        
        null_info = null_info[null_info['Null_Count'] > 0].sort_values('Null_Count', ascending=False)
        
        if len(null_info) > 0:
            logger.info(f"\n{table_name} - Null Values Analysis:")
            logger.info(f"{null_info.to_string(index=False)}")
            
        return null_info
    
    def handle_applications_nulls(self, df):
        """Handle nulls in Applications table"""
        logger.info("\nHandling Applications table nulls...")
        df = df.copy()
        
        # Analyze nulls
        null_info = self.analyze_nulls(df, 'Applications')
        
        # ApplPublicNotes - text field, fill with empty string
        if 'ApplPublicNotes' in df.columns:
            null_count = df['ApplPublicNotes'].isnull().sum()
            if null_count > 0:
                df['ApplPublicNotes'] = df['ApplPublicNotes'].fillna('')
                logger.info(f"  - ApplPublicNotes: Filled {null_count} nulls with empty string")
        
        self.null_report['applications'] = {
            'before': null_info,
            'after': df.isnull().sum().sum()
        }
        
        return df
    
    def handle_products_nulls(self, df):
        """Handle nulls in Products table"""
        logger.info("\nHandling Products table nulls...")
        df = df.copy()
        
        # Analyze nulls
        null_info = self.analyze_nulls(df, 'Products')
        
        # Strength - text field, fill with empty string
        if 'Strength' in df.columns:
            null_count = df['Strength'].isnull().sum()
            if null_count > 0:
                df['Strength'] = df['Strength'].fillna('')
                logger.info(f"  - Strength: Filled {null_count} nulls with empty string")
        
        # ReferenceDrug - check if it's actually binary or text
        if 'ReferenceDrug' in df.columns:
            null_count = df['ReferenceDrug'].isnull().sum()
            if null_count > 0:
                # Check if column contains numeric values
                non_null_values = df['ReferenceDrug'].dropna()
                if non_null_values.empty or pd.api.types.is_numeric_dtype(non_null_values):
                    try:
                        # Try to convert to numeric
                        df['ReferenceDrug'] = pd.to_numeric(df['ReferenceDrug'], errors='coerce')
                        df['ReferenceDrug'] = df['ReferenceDrug'].fillna(0).astype(int)
                        logger.info(f"  - ReferenceDrug: Filled {null_count} nulls with 0")
                    except:
                        # If conversion fails, create a flag
                        df['HasReferenceDrug'] = df['ReferenceDrug'].notna().astype(int)
                        df['ReferenceDrug'] = df['ReferenceDrug'].fillna('')
                        logger.info(f"  - ReferenceDrug: Contains text, filled nulls with empty string and created HasReferenceDrug flag")
                else:
                    # Text column - fill with empty string
                    df['ReferenceDrug'] = df['ReferenceDrug'].fillna('')
                    logger.info(f"  - ReferenceDrug: Text column, filled {null_count} nulls with empty string")
        
        # ReferenceStandard - appears to contain drug names, not binary values
        if 'ReferenceStandard' in df.columns:
            null_count = df['ReferenceStandard'].isnull().sum()
            if null_count > 0:
                # Check data type
                non_null_values = df['ReferenceStandard'].dropna()
                
                # If it contains text (drug names), treat as text field
                if not non_null_values.empty and df['ReferenceStandard'].dtype == 'object':
                    # Check if it contains text
                    sample = non_null_values.head(10).astype(str)
                    contains_text = any(any(c.isalpha() for c in str(v)) for v in sample)
                    
                    if contains_text:
                        # It's a text field with drug names
                        df['ReferenceStandard'] = df['ReferenceStandard'].fillna('')
                        # Also create a binary flag
                        df['HasReferenceStandard'] = (df['ReferenceStandard'] != '').astype(int)
                        logger.info(f"  - ReferenceStandard: Contains drug names, filled {null_count} nulls with empty string")
                        logger.info(f"  - Created HasReferenceStandard binary flag")
                    else:
                        # Try numeric conversion
                        try:
                            df['ReferenceStandard'] = pd.to_numeric(df['ReferenceStandard'], errors='coerce')
                            df['ReferenceStandard'] = df['ReferenceStandard'].fillna(0).astype(int)
                            logger.info(f"  - ReferenceStandard: Filled {null_count} nulls with 0")
                        except:
                            df['ReferenceStandard'] = df['ReferenceStandard'].fillna('')
                            logger.info(f"  - ReferenceStandard: Filled {null_count} nulls with empty string")
                else:
                    # Empty or numeric
                    try:
                        df['ReferenceStandard'] = df['ReferenceStandard'].fillna(0).astype(int)
                        logger.info(f"  - ReferenceStandard: Filled {null_count} nulls with 0")
                    except:
                        df['ReferenceStandard'] = df['ReferenceStandard'].fillna('')
                        logger.info(f"  - ReferenceStandard: Filled {null_count} nulls with empty string")
        
        self.null_report['products'] = {
            'before': null_info,
            'after': df.isnull().sum().sum()
        }
        
        return df
    
    def handle_submissions_nulls(self, df):
        """Handle nulls in Submissions table"""
        logger.info("\nHandling Submissions table nulls...")
        df = df.copy()
        
        # Analyze nulls
        null_info = self.analyze_nulls(df, 'Submissions')
        
        # SubmissionClassCodeID - categorical, create a separate category
        if 'SubmissionClassCodeID' in df.columns:
            null_count = df['SubmissionClassCodeID'].isnull().sum()
            if null_count > 0:
                # Use -1 to indicate unknown/missing
                df['SubmissionClassCodeID'] = df['SubmissionClassCodeID'].fillna(-1).astype(int)
                logger.info(f"  - SubmissionClassCodeID: Filled {null_count} nulls with -1 (unknown)")
        
        # SubmissionStatus - important field
        if 'SubmissionStatus' in df.columns:
            null_count = df['SubmissionStatus'].isnull().sum()
            if null_count > 0:
                df['SubmissionStatus'] = df['SubmissionStatus'].fillna('UNK')
                logger.info(f"  - SubmissionStatus: Filled {null_count} nulls with 'UNK'")
        
        # SubmissionStatusDate - date field
        if 'SubmissionStatusDate' in df.columns:
            # First convert to datetime if not already
            df['SubmissionStatusDate'] = pd.to_datetime(df['SubmissionStatusDate'], errors='coerce')
            null_count = df['SubmissionStatusDate'].isnull().sum()
            
            # For missing dates, we'll create a flag column
            df['SubmissionDateMissing'] = df['SubmissionStatusDate'].isnull().astype(int)
            logger.info(f"  - SubmissionStatusDate: Created flag for {null_count} missing dates")
        
        # SubmissionsPublicNotes - text field
        if 'SubmissionsPublicNotes' in df.columns:
            null_count = df['SubmissionsPublicNotes'].isnull().sum()
            if null_count > 0:
                df['SubmissionsPublicNotes'] = df['SubmissionsPublicNotes'].fillna('')
                logger.info(f"  - SubmissionsPublicNotes: Filled {null_count} nulls with empty string")
        
        # ReviewPriority - categorical
        if 'ReviewPriority' in df.columns:
            null_count = df['ReviewPriority'].isnull().sum()
            if null_count > 0:
                df['ReviewPriority'] = df['ReviewPriority'].fillna('STANDARD')
                logger.info(f"  - ReviewPriority: Filled {null_count} nulls with 'STANDARD'")
        
        self.null_report['submissions'] = {
            'before': null_info,
            'after': df.isnull().sum().sum()
        }
        
        return df
    
    def handle_te_nulls(self, df):
        """Handle nulls in TE table"""
        logger.info("\nHandling TE table nulls...")
        df = df.copy()
        
        # Analyze nulls
        null_info = self.analyze_nulls(df, 'TE')
        
        # TECode - important field
        if 'TECode' in df.columns:
            null_count = df['TECode'].isnull().sum()
            if null_count > 0:
                # Check if there's a pattern for missing TECode
                df['TECode'] = df['TECode'].fillna('NONE')
                logger.info(f"  - TECode: Filled {null_count} nulls with 'NONE'")
        
        self.null_report['te'] = {
            'before': null_info,
            'after': df.isnull().sum().sum()
        }
        
        return df
    
    def handle_lookup_table_nulls(self, df, table_name):
        """Handle nulls in lookup tables"""
        logger.info(f"\nHandling {table_name} table nulls...")
        df = df.copy()
        
        # Analyze nulls
        null_info = self.analyze_nulls(df, table_name)
        
        # Specific handling for each lookup table
        if table_name == 'action_types_lookup':
            if 'ActionTypes_LookupDescription' in df.columns:
                null_count = df['ActionTypes_LookupDescription'].isnull().sum()
                if null_count > 0:
                    df['ActionTypes_LookupDescription'] = df['ActionTypes_LookupDescription'].fillna('UNKNOWN')
                    logger.info(f"  - ActionTypes_LookupDescription: Filled {null_count} nulls with 'UNKNOWN'")
            
            if 'SupplCategoryLevel1Code' in df.columns:
                null_count = df['SupplCategoryLevel1Code'].isnull().sum()
                if null_count > 0:
                    df['SupplCategoryLevel1Code'] = df['SupplCategoryLevel1Code'].fillna('UNK')
                    logger.info(f"  - SupplCategoryLevel1Code: Filled {null_count} nulls with 'UNK'")
            
            if 'SupplCategoryLevel2Code' in df.columns:
                null_count = df['SupplCategoryLevel2Code'].isnull().sum()
                if null_count > 0:
                    df['SupplCategoryLevel2Code'] = df['SupplCategoryLevel2Code'].fillna('UNK')
                    logger.info(f"  - SupplCategoryLevel2Code: Filled {null_count} nulls with 'UNK'")
        
        elif table_name == 'submission_class_lookup':
            if 'SubmissionClassCode' in df.columns:
                null_count = df['SubmissionClassCode'].isnull().sum()
                if null_count > 0:
                    df['SubmissionClassCode'] = df['SubmissionClassCode'].fillna('UNK')
                    logger.info(f"  - SubmissionClassCode: Filled {null_count} nulls with 'UNK'")
            
            if 'SubmissionClassCodeDescription' in df.columns:
                null_count = df['SubmissionClassCodeDescription'].isnull().sum()
                if null_count > 0:
                    df['SubmissionClassCodeDescription'] = df['SubmissionClassCodeDescription'].fillna('NOT SPECIFIED')
                    logger.info(f"  - SubmissionClassCodeDescription: Filled {null_count} nulls with 'NOT SPECIFIED'")
        
        self.null_report[table_name] = {
            'before': null_info,
            'after': df.isnull().sum().sum()
        }
        
        return df
    
    def handle_join_table_nulls(self, df):
        """Handle nulls in join_submission_action_type table"""
        logger.info("\nHandling join_submission_action_type table nulls...")
        df = df.copy()
        
        # Analyze nulls
        null_info = self.analyze_nulls(df, 'join_submission_action_type')
        
        # Handle ActionTypes_LookupID
        if 'ActionTypes_LookupID' in df.columns:
            null_count = df['ActionTypes_LookupID'].isnull().sum()
            if null_count > 0:
                # Use -1 to indicate missing/unknown action type
                df['ActionTypes_LookupID'] = df['ActionTypes_LookupID'].fillna(-1).astype(int)
                logger.info(f"  - ActionTypes_LookupID: Filled {null_count} nulls with -1")
                
                # Also create a flag for records with missing action types
                df['ActionTypeMissing'] = (df['ActionTypes_LookupID'] == -1).astype(int)
                logger.info(f"  - Created ActionTypeMissing flag for {null_count} records")
        
        self.null_report['join_submission_action_type'] = {
            'before': null_info,
            'after': df.isnull().sum().sum()
        }
        
        return df
    
    def process_all_tables(self):
        """Process all tables to handle null values"""
        cleaned_tables = {}
        
        # Process each table with appropriate strategy
        if 'applications' in self.tables:
            cleaned_tables['applications'] = self.handle_applications_nulls(self.tables['applications'])
        
        if 'products' in self.tables:
            cleaned_tables['products'] = self.handle_products_nulls(self.tables['products'])
        
        if 'submissions' in self.tables:
            cleaned_tables['submissions'] = self.handle_submissions_nulls(self.tables['submissions'])
        
        if 'te' in self.tables:
            cleaned_tables['te'] = self.handle_te_nulls(self.tables['te'])
        
        # Handle lookup tables
        lookup_tables = ['action_types_lookup', 'submission_class_lookup']
        for table_name in lookup_tables:
            if table_name in self.tables:
                cleaned_tables[table_name] = self.handle_lookup_table_nulls(
                    self.tables[table_name], 
                    table_name
                )
        
        # Handle join table
        if 'join_submission_action_type' in self.tables:
            cleaned_tables['join_submission_action_type'] = self.handle_join_table_nulls(
                self.tables['join_submission_action_type']
            )
        
        # Copy over tables with no nulls
        no_null_tables = ['marketing_status', 'application_docs', 'application_docs_type_lookup',
                         'marketing_status_lookup', 'submission_property_type']
        
        for table_name in no_null_tables:
            if table_name in self.tables:
                cleaned_tables[table_name] = self.tables[table_name].copy()
                logger.info(f"\n{table_name}: No nulls to handle")
        
        return cleaned_tables
    
    def generate_null_handling_report(self):
        """Generate a report of null handling"""
        logger.info("\n" + "="*60)
        logger.info("NULL HANDLING SUMMARY REPORT")
        logger.info("="*60)
        
        for table_name, report in self.null_report.items():
            before_nulls = report['before']['Null_Count'].sum() if len(report['before']) > 0 else 0
            after_nulls = report['after']
            
            logger.info(f"\n{table_name}:")
            logger.info(f"  Before: {before_nulls:,} null values")
            logger.info(f"  After:  {after_nulls:,} null values")
            logger.info(f"  Handled: {before_nulls - after_nulls:,} null values")

def main():
    """Main function to handle null values"""
    
    # Load data
    logger.info("Loading FDA data...")
    loader = FDADataLoader()
    tables = loader.load_all_tables()
    
    if not tables:
        logger.error("No data loaded!")
        return
    
    # Initialize null handler
    handler = NullValueHandler(tables)
    
    # Process all tables
    logger.info("\nProcessing null values...")
    cleaned_tables = handler.process_all_tables()
    
    # Generate report
    handler.generate_null_handling_report()
    
    # Save cleaned data
    logger.info("\nSaving cleaned data...")
    os.makedirs('data/processed', exist_ok=True)
    
    for table_name, df in cleaned_tables.items():
        output_path = f'data/processed/{table_name}_no_nulls.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {table_name} to {output_path}")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("NULL HANDLING COMPLETE")
    logger.info("="*60)
    logger.info(f"Processed {len(cleaned_tables)} tables")
    logger.info("All cleaned data saved to data/processed/")
    
    return cleaned_tables

if __name__ == "__main__":
    cleaned_tables = main()