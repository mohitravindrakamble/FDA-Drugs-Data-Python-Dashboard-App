import pandas as pd
import os
from typing import Dict, Optional
import logging
import chardet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FDADataLoader:
    """Load FDA drug data from files with automatic encoding detection"""
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = data_path
        self.tables = {}
        
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        with open(file_path, 'rb') as f:
            # Read a sample of the file
            sample = f.read(100000)  # Read first 100KB
            result = chardet.detect(sample)
            encoding = result['encoding']
            confidence = result['confidence']
            logger.info(f"Detected encoding for {os.path.basename(file_path)}: {encoding} (confidence: {confidence:.2f})")
            return encoding
    
    def load_table_with_fallback(self, file_path: str, table_name: str) -> Optional[pd.DataFrame]:
        """Try to load a table with multiple encoding options"""
        
        # List of encodings to try in order
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        # First try auto-detection
        try:
            detected_encoding = self.detect_encoding(file_path)
            if detected_encoding and detected_encoding not in encodings:
                encodings.insert(0, detected_encoding)
        except Exception as e:
            logger.warning(f"Could not auto-detect encoding: {e}")
        
        # Try each encoding
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, sep='\t', encoding=encoding, low_memory=False)
                logger.info(f"Successfully loaded {table_name} with encoding: {encoding}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error with encoding {encoding}: {str(e)}")
                continue
        
        # If all encodings fail, try with error handling
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='latin-1', 
                           low_memory=False, on_bad_lines='skip')
            logger.warning(f"Loaded {table_name} with latin-1 encoding and skipped bad lines")
            return df
        except Exception as e:
            logger.error(f"Failed to load {table_name} with all attempted encodings: {str(e)}")
            return None
        
    def load_all_tables(self) -> Dict[str, pd.DataFrame]:
        """Load all FDA tables from CSV/Excel files"""
        
        table_files = {
            'applications': 'Applications.txt',
            'products': 'Products.txt',
            'submissions': 'Submissions.txt',
            'marketing_status': 'MarketingStatus.txt',
            'te': 'TE.txt',
            'application_docs': 'ApplicationDocs.txt',
            'action_types_lookup': 'ActionTypes_Lookup.txt',
            'application_docs_type_lookup': 'ApplicationsDocsType_Lookup.txt',
            'marketing_status_lookup': 'MarketingStatus_Lookup.txt',
            'submission_class_lookup': 'SubmissionClass_Lookup.txt',
            'submission_property_type': 'SubmissionPropertyType.txt',
            'join_submission_action_type': 'Join_Submission_ActionTypes_Lookup.txt'
        }
        
        for table_name, file_name in table_files.items():
            file_path = os.path.join(self.data_path, file_name)
            
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
            
            # Use the new loading method with encoding fallback
            df = self.load_table_with_fallback(file_path, table_name)
            
            if df is not None:
                self.tables[table_name] = df
                logger.info(f"Loaded {table_name}: {df.shape}")
                
                # Show sample of problematic characters if any
                if table_name in ['submissions', 'application_docs']:
                    self._check_for_special_characters(df, table_name)
            else:
                logger.error(f"Failed to load {table_name}")
                
        return self.tables
    
    def _check_for_special_characters(self, df: pd.DataFrame, table_name: str):
        """Check for special characters in text columns"""
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns[:3]:  # Check first 3 text columns
            if col in df.columns:
                # Find rows with non-ASCII characters
                try:
                    non_ascii_mask = df[col].astype(str).str.contains('[^\x00-\x7F]', na=False)
                    if non_ascii_mask.any():
                        logger.info(f"{table_name}.{col} contains {non_ascii_mask.sum()} rows with special characters")
                except Exception:
                    pass
    
    def get_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """Get a specific table"""
        return self.tables.get(table_name)
    
    def get_loading_summary(self) -> pd.DataFrame:
        """Get a summary of all loaded tables"""
        summary_data = []
        
        for table_name, df in self.tables.items():
            summary_data.append({
                'Table': table_name,
                'Rows': len(df),
                'Columns': len(df.columns),
                'Memory (MB)': df.memory_usage(deep=True).sum() / 1024**2,
                'Null Values': df.isnull().sum().sum(),
                'Duplicate Rows': df.duplicated().sum()
            })
        
        return pd.DataFrame(summary_data)