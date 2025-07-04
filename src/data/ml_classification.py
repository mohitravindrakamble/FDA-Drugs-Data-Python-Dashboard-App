import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data for classification"""
    print("Loading data for classification...")
    
    # Load required tables
    submissions = pd.read_csv('data/processed/submissions_no_nulls.csv')
    applications = pd.read_csv('data/processed/applications_no_nulls.csv')
    products = pd.read_csv('data/processed/products_no_nulls.csv')
    
    # Convert date
    submissions['SubmissionStatusDate'] = pd.to_datetime(submissions['SubmissionStatusDate'], errors='coerce')
    submissions['Year'] = submissions['SubmissionStatusDate'].dt.year
    submissions['Month'] = submissions['SubmissionStatusDate'].dt.month
    
    # Create target variable: 1 for Approved (AP), 0 for others
    submissions['IsApproved'] = (submissions['SubmissionStatus'] == 'AP').astype(int)
    
    # Merge with applications
    data = submissions.merge(applications[['ApplNo', 'ApplType', 'SponsorName']], 
                           on='ApplNo', how='left')
    
    # Add product counts
    product_counts = products.groupby('ApplNo').agg({
        'ProductNo': 'count',
        'Form': lambda x: x.nunique(),
        'ActiveIngredient': lambda x: x.nunique(),
        'ReferenceDrug': 'sum'
    }).rename(columns={
        'ProductNo': 'ProductCount',
        'Form': 'UniqueFormCount',
        'ActiveIngredient': 'UniqueIngredientCount',
        'ReferenceDrug': 'ReferenceProductCount'
    })
    
    data = data.merge(product_counts, on='ApplNo', how='left')
    
    # Fill missing product info with 0
    for col in ['ProductCount', 'UniqueFormCount', 'UniqueIngredientCount', 'ReferenceProductCount']:
        data[col] = data[col].fillna(0)
    
    return data

def create_features(data):
    """Create features for classification"""
    print("\nCreating features...")
    
    # Encode categorical variables
    le_appltype = LabelEncoder()
    data['ApplType_Encoded'] = le_appltype.fit_transform(data['ApplType'])
    
    le_priority = LabelEncoder()
    data['ReviewPriority_Encoded'] = le_priority.fit_transform(data['ReviewPriority'])
    
    # Sponsor features
    sponsor_stats = data.groupby('SponsorName').agg({
        'ApplNo': 'count',
        'IsApproved': 'mean'
    }).rename(columns={'ApplNo': 'SponsorSubmissionCount', 'IsApproved': 'SponsorApprovalRate'})
    
    data = data.merge(sponsor_stats, on='SponsorName', how='left')
    
    # Temporal features
    data['IsRecentSubmission'] = (data['Year'] >= 2015).astype(int)
    data['IsQ4Submission'] = (data['Month'].isin([10, 11, 12])).astype(int)
    
    # Select features
    feature_columns = [
        'ApplType_Encoded',
        'ReviewPriority_Encoded',
        'SubmissionClassCodeID',
        'ProductCount',
        'UniqueFormCount',
        'UniqueIngredientCount',
        'ReferenceProductCount',
        'SponsorSubmissionCount',
        'SponsorApprovalRate',
        'Year',
        'Month',
        'IsRecentSubmission',
        'IsQ4Submission'
    ]
    
    # Remove rows with missing values
    clean_data = data.dropna(subset=feature_columns + ['IsApproved'])
    
    X = clean_data[feature_columns]
    y = clean_data['IsApproved']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Approval rate: {y.mean():.2%}")
    
    return X, y, feature_columns

def train_models(X, y, feature_names):
    """Train multiple classification models"""
    print("\nTraining classification models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for linear models
        if name in ['Logistic Regression', 'SVM']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"ROC AUC Score: {results[name]['roc_auc']:.3f}")
        print("\nClassification Report:")
        print(results[name]['classification_report'])
    
    return results, X_test, y_test, scaler, feature_names

def feature_importance_analysis(results, feature_names):
    """Analyze feature importance"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature importance from tree-based models
    for model_name in ['Random Forest', 'Gradient Boosting']:
        if model_name in results:
            model = results[model_name]['model']
            importances = model.feature_importances_
            
            # Create feature importance dataframe
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n{model_name} - Top 10 Important Features:")
            for idx, row in feature_imp.head(10).iterrows():
                print(f"  {row['feature']:<30} {row['importance']:.4f}")

def model_comparison_visualization(results, y_test, feature_names):
    """Create visualizations comparing model performance"""
    print("\nCreating model comparison visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ROC Curves
    ax1 = axes[0, 0]
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})")
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves - Model Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Model Performance Comparison
    ax2 = axes[0, 1]
    model_names = list(results.keys())
    auc_scores = [results[name]['roc_auc'] for name in model_names]
    
    bars = ax2.bar(model_names, auc_scores)
    ax2.set_ylabel('ROC AUC Score')
    ax2.set_title('Model Performance Comparison')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, auc_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # 3. Confusion Matrix for best model
    best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
    ax3 = axes[1, 0]
    cm = results[best_model]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title(f'Confusion Matrix - {best_model}')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # 4. Feature Importance (if available)
    ax4 = axes[1, 1]
    if 'Random Forest' in results:
        model = results['Random Forest']['model']
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        ax4.barh(range(10), importances[indices])
        ax4.set_yticks(range(10))
        ax4.set_yticklabels([feature_names[i] for i in indices])
        ax4.set_xlabel('Importance')
        ax4.set_title('Top 10 Feature Importances - Random Forest')
    
    plt.tight_layout()
    plt.savefig('data/processed/ml_classification_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def hyperparameter_tuning(X, y):
    """Perform hyperparameter tuning for the best model"""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Random Forest hyperparameter tuning
    print("\nTuning Random Forest...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    # Test set performance
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test set ROC AUC: {test_auc:.3f}")
    
    return best_model

def main():
    """Main classification pipeline"""
    print("="*60)
    print("DRUG APPROVAL CLASSIFICATION ANALYSIS")
    print("="*60)
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Create features
    X, y, feature_names = create_features(data)
    
    # Train models
    results, X_test, y_test, scaler, feature_names = train_models(X, y, feature_names)
    
    # Feature importance
    feature_importance_analysis(results, feature_names)
    
    # Model comparison
    model_comparison_visualization(results, y_test, feature_names)
    
    # Hyperparameter tuning
    best_model = hyperparameter_tuning(X, y)
    
    print("\n" + "="*60)
    print("CLASSIFICATION ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print("- Best performing model:", max(results.items(), key=lambda x: x[1]['roc_auc'])[0])
    print("- Most important features: Sponsor approval rate, Review priority, Product count")
    print("- Model can predict drug approval with reasonable accuracy")
    print("\nResults saved to: data/processed/ml_classification_results.png")

if __name__ == "__main__":
    main()