import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_sponsor_data():
    """Load and prepare sponsor-level data for clustering"""
    print("Loading data for sponsor segmentation...")
    
    # Load tables
    applications = pd.read_csv('data/processed/applications_no_nulls.csv')
    products = pd.read_csv('data/processed/products_no_nulls.csv')
    submissions = pd.read_csv('data/processed/submissions_no_nulls.csv')
    
    # Convert dates
    submissions['SubmissionStatusDate'] = pd.to_datetime(submissions['SubmissionStatusDate'], errors='coerce')
    submissions['Year'] = submissions['SubmissionStatusDate'].dt.year
    
    # Create sponsor features
    sponsor_features = applications.groupby('SponsorName').agg({
        'ApplNo': 'count'
    }).rename(columns={'ApplNo': 'TotalApplications'})
    
    # Add product metrics
    products_with_sponsor = products.merge(applications[['ApplNo', 'SponsorName']], on='ApplNo')
    sponsor_product_stats = products_with_sponsor.groupby('SponsorName').agg({
        'ProductNo': 'count',
        'Form': 'nunique',
        'ActiveIngredient': 'nunique',
        'ReferenceDrug': 'sum'
    }).rename(columns={
        'ProductNo': 'TotalProducts',
        'Form': 'UniqueForms',
        'ActiveIngredient': 'UniqueIngredients',
        'ReferenceDrug': 'ReferenceProducts'
    })
    
    sponsor_features = sponsor_features.join(sponsor_product_stats)
    
    # Add submission metrics
    submissions_with_sponsor = submissions.merge(applications[['ApplNo', 'SponsorName']], on='ApplNo')
    sponsor_submission_stats = submissions_with_sponsor.groupby('SponsorName').agg({
        'SubmissionNo': 'count',
        'SubmissionStatus': lambda x: (x == 'AP').sum(),
        'ReviewPriority': lambda x: (x == 'PRIORITY').sum(),
        'Year': ['min', 'max']
    })
    
    sponsor_submission_stats.columns = ['TotalSubmissions', 'ApprovedSubmissions', 
                                       'PriorityReviews', 'FirstYear', 'LastYear']
    
    sponsor_features = sponsor_features.join(sponsor_submission_stats)
    
    # Calculate derived features
    sponsor_features['ApprovalRate'] = sponsor_features['ApprovedSubmissions'] / sponsor_features['TotalSubmissions']
    sponsor_features['ProductsPerApplication'] = sponsor_features['TotalProducts'] / sponsor_features['TotalApplications']
    sponsor_features['YearsActive'] = sponsor_features['LastYear'] - sponsor_features['FirstYear'] + 1
    sponsor_features['SubmissionsPerYear'] = sponsor_features['TotalSubmissions'] / sponsor_features['YearsActive']
    sponsor_features['PriorityRate'] = sponsor_features['PriorityReviews'] / sponsor_features['TotalSubmissions']
    
    # Remove sponsors with very few applications
    sponsor_features = sponsor_features[sponsor_features['TotalApplications'] >= 5]
    
    # Handle any remaining NaN values
    sponsor_features = sponsor_features.fillna(0)
    
    print(f"Sponsor feature matrix shape: {sponsor_features.shape}")
    
    return sponsor_features

def load_and_prepare_drug_data():
    """Load and prepare drug-level data for clustering"""
    print("\nLoading data for drug segmentation...")
    
    # Load tables
    products = pd.read_csv('data/processed/products_no_nulls.csv')
    applications = pd.read_csv('data/processed/applications_no_nulls.csv')
    submissions = pd.read_csv('data/processed/submissions_no_nulls.csv')
    
    # Create drug features
    drug_features = products.copy()
    
    # Add application info
    drug_features = drug_features.merge(applications[['ApplNo', 'ApplType']], on='ApplNo', how='left')
    
    # Add submission statistics
    submission_stats = submissions.groupby('ApplNo').agg({
        'SubmissionNo': 'count',
        'SubmissionStatus': lambda x: (x == 'AP').any(),
        'ReviewPriority': lambda x: (x == 'PRIORITY').any()
    }).rename(columns={
        'SubmissionNo': 'TotalSubmissions',
        'SubmissionStatus': 'IsApproved',
        'ReviewPriority': 'HadPriorityReview'
    })
    
    drug_features = drug_features.merge(submission_stats, on='ApplNo', how='left')
    
    # Create features for clustering
    # One-hot encode ApplType
    appltype_dummies = pd.get_dummies(drug_features['ApplType'], prefix='ApplType')
    
    # Binary features
    drug_features['IsReferenceDrug'] = drug_features['ReferenceDrug'].astype(int)
    drug_features['HasReferenceStandard'] = (drug_features['ReferenceStandard'] != '').astype(int)
    
    # Combine features
    clustering_features = pd.concat([
        drug_features[['TotalSubmissions', 'IsApproved', 'HadPriorityReview', 
                      'IsReferenceDrug', 'HasReferenceStandard']],
        appltype_dummies
    ], axis=1)
    
    # Remove rows with missing values
    clustering_features = clustering_features.dropna()
    
    print(f"Drug feature matrix shape: {clustering_features.shape}")
    
    return clustering_features, drug_features.loc[clustering_features.index]

def perform_kmeans_clustering(features, name, n_clusters_range=range(2, 11)):
    """Perform K-means clustering with elbow method"""
    print(f"\n{'='*60}")
    print(f"K-MEANS CLUSTERING - {name}")
    print(f"{'='*60}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Elbow method
    inertias = []
    silhouette_scores = []
    
    for k in n_clusters_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette = silhouette_score(features_scaled, labels)
        silhouette_scores.append(silhouette)
        
        print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette:.3f}")
    
    # Find optimal k using silhouette score
    optimal_k = n_clusters_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    # Fit final model with optimal k
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(features_scaled)
    
    # Analyze clusters
    cluster_analysis = pd.DataFrame(features)
    cluster_analysis['Cluster'] = final_labels
    
    print("\nCluster Sizes:")
    print(cluster_analysis['Cluster'].value_counts().sort_index())
    
    print("\nCluster Characteristics (Mean values):")
    cluster_means = cluster_analysis.groupby('Cluster').mean()
    print(cluster_means.to_string())
    
    return final_labels, features_scaled, optimal_k, inertias, silhouette_scores

def perform_hierarchical_clustering(features, name, n_clusters=5):
    """Perform hierarchical clustering"""
    print(f"\n{'='*60}")
    print(f"HIERARCHICAL CLUSTERING - {name}")
    print(f"{'='*60}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hierarchical.fit_predict(features_scaled)
    
    # Calculate metrics
    silhouette = silhouette_score(features_scaled, labels)
    davies_bouldin = davies_bouldin_score(features_scaled, labels)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
    
    # Analyze clusters
    cluster_analysis = pd.DataFrame(features)
    cluster_analysis['Cluster'] = labels
    
    print("\nCluster Sizes:")
    print(cluster_analysis['Cluster'].value_counts().sort_index())
    
    return labels, features_scaled

def visualize_clusters(features_scaled, labels, title, feature_names=None):
    """Visualize clusters using PCA and t-SNE"""
    print(f"\nCreating visualizations for {title}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. PCA visualization
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    ax1 = axes[0, 0]
    scatter = ax1.scatter(features_pca[:, 0], features_pca[:, 1], 
                         c=labels, cmap='viridis', alpha=0.6)
    ax1.set_title(f'PCA Visualization - {title}')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter, ax=ax1)
    
    # 2. t-SNE visualization
    ax2 = axes[0, 1]
    try:
        if len(features_scaled) < 1000:  # t-SNE is slow for large datasets
            tsne = TSNE(n_components=2, random_state=42)
            features_tsne = tsne.fit_transform(features_scaled)
            scatter = ax2.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                                c=labels, cmap='viridis', alpha=0.6)
            ax2.set_title(f't-SNE Visualization - {title}')
            plt.colorbar(scatter, ax=ax2)
        else:
            ax2.text(0.5, 0.5, 't-SNE skipped (too many samples)', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('t-SNE Visualization')
    except Exception as e:
        ax2.text(0.5, 0.5, f't-SNE failed:\n{str(e)[:50]}', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('t-SNE Visualization (Error)')
    
    # 3. Feature importance for PCA
    ax3 = axes[1, 0]
    if feature_names is not None:
        # Get top contributing features for PC1
        pc1_contributions = abs(pca.components_[0])
        n_features = min(10, len(pc1_contributions))
        top_features_idx = np.argsort(pc1_contributions)[-n_features:]
        
        ax3.barh(range(n_features), pc1_contributions[top_features_idx])
        ax3.set_yticks(range(n_features))
        ax3.set_yticklabels([feature_names[i] for i in top_features_idx])
        ax3.set_xlabel('Absolute Contribution')
        ax3.set_title(f'Top {n_features} Features Contributing to PC1')
    else:
        ax3.axis('off')
    
    # 4. Cluster summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    summary_text = f"{title} - Cluster Summary\n\n"
    summary_text += "Cluster Sizes:\n"
    for cluster, count in cluster_counts.items():
        summary_text += f"  Cluster {cluster}: {count} ({count/len(labels)*100:.1f}%)\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            verticalalignment='top', fontsize=12, fontfamily='monospace')
    
    plt.tight_layout()
    return fig

def sponsor_segmentation_analysis(sponsor_features):
    """Complete sponsor segmentation analysis"""
    print("\n" + "="*60)
    print("SPONSOR SEGMENTATION ANALYSIS")
    print("="*60)
    
    # Select features for clustering
    clustering_features = [
        'TotalApplications', 'TotalProducts', 'ApprovalRate',
        'ProductsPerApplication', 'YearsActive', 'SubmissionsPerYear',
        'PriorityRate', 'UniqueForms', 'UniqueIngredients'
    ]
    
    # Filter to only include features that exist
    available_features = [f for f in clustering_features if f in sponsor_features.columns]
    features = sponsor_features[available_features]
    
    print(f"Using {len(available_features)} features for clustering: {available_features}")
    
    # K-means clustering
    kmeans_labels, features_scaled, optimal_k, inertias, silhouette_scores = perform_kmeans_clustering(
        features, "Sponsors", n_clusters_range=range(2, 8)
    )
    
    # Visualize clusters
    fig = visualize_clusters(features_scaled, kmeans_labels, "Sponsor Clusters", available_features)
    plt.savefig('data/processed/ml_sponsor_segmentation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Interpret clusters
    sponsor_clusters = sponsor_features.copy()
    sponsor_clusters['Cluster'] = kmeans_labels
    
    print("\n" + "="*60)
    print("SPONSOR CLUSTER INTERPRETATION")
    print("="*60)
    
    for cluster in range(optimal_k):
        cluster_sponsors = sponsor_clusters[sponsor_clusters['Cluster'] == cluster]
        print(f"\nCluster {cluster} ({len(cluster_sponsors)} sponsors):")
        print(f"  Avg Applications: {cluster_sponsors['TotalApplications'].mean():.1f}")
        print(f"  Avg Approval Rate: {cluster_sponsors['ApprovalRate'].mean():.1%}")
        print(f"  Avg Years Active: {cluster_sponsors['YearsActive'].mean():.1f}")
        print(f"  Top sponsors: {', '.join(cluster_sponsors.nlargest(3, 'TotalApplications').index.tolist()[:3])}")
    
    return sponsor_clusters

def drug_segmentation_analysis(drug_features, drug_info):
    """Complete drug segmentation analysis"""
    print("\n" + "="*60)
    print("DRUG SEGMENTATION ANALYSIS")
    print("="*60)
    
    # K-means clustering
    kmeans_labels, features_scaled, optimal_k, inertias, silhouette_scores = perform_kmeans_clustering(
        drug_features, "Drugs", n_clusters_range=range(2, 8)
    )
    
    # Visualize clusters
    fig = visualize_clusters(features_scaled, kmeans_labels, "Drug Clusters", drug_features.columns.tolist())
    plt.savefig('data/processed/ml_drug_segmentation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Interpret clusters
    drug_clusters = drug_info.copy()
    drug_clusters['Cluster'] = kmeans_labels
    
    print("\n" + "="*60)
    print("DRUG CLUSTER INTERPRETATION")
    print("="*60)
    
    for cluster in range(optimal_k):
        cluster_drugs = drug_clusters[drug_clusters['Cluster'] == cluster]
        print(f"\nCluster {cluster} ({len(cluster_drugs)} drugs):")
        
        # Most common drug forms in cluster
        top_forms = cluster_drugs['Form'].value_counts().head(3)
        print(f"  Top forms: {', '.join(top_forms.index.tolist())}")
        
        # Application type distribution
        appltype_dist = cluster_drugs['ApplType'].value_counts()
        print(f"  Application types: {dict(appltype_dist)}")
        
        # Approval statistics
        approval_rate = cluster_drugs['IsApproved'].mean()
        priority_rate = cluster_drugs['HadPriorityReview'].mean()
        print(f"  Approval rate: {approval_rate:.1%}")
        print(f"  Priority review rate: {priority_rate:.1%}")
    
    return drug_clusters

def create_segmentation_summary(sponsor_clusters, drug_clusters):
    """Create summary visualization of segmentation results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('FDA Data Segmentation Summary', fontsize=16)
    
    # 1. Sponsor cluster sizes
    ax1 = axes[0, 0]
    sponsor_clusters['Cluster'].value_counts().sort_index().plot(kind='bar', ax=ax1)
    ax1.set_title('Sponsor Cluster Sizes')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Number of Sponsors')
    
    # 2. Sponsor cluster characteristics
    ax2 = axes[0, 1]
    cluster_means = sponsor_clusters.groupby('Cluster')[['ApprovalRate', 'TotalApplications']].mean()
    cluster_means.plot(kind='bar', ax=ax2)
    ax2.set_title('Sponsor Cluster Characteristics')
    ax2.set_xlabel('Cluster')
    ax2.legend(['Approval Rate', 'Avg Applications'])
    
    # 3. Drug cluster sizes
    ax3 = axes[1, 0]
    drug_clusters['Cluster'].value_counts().sort_index().plot(kind='bar', ax=ax3)
    ax3.set_title('Drug Cluster Sizes')
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Number of Drugs')
    
    # 4. Drug cluster approval rates
    ax4 = axes[1, 1]
    drug_approval_by_cluster = drug_clusters.groupby('Cluster')['IsApproved'].mean()
    drug_approval_by_cluster.plot(kind='bar', ax=ax4)
    ax4.set_title('Drug Cluster Approval Rates')
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Approval Rate')
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('data/processed/ml_segmentation_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main segmentation pipeline"""
    print("="*60)
    print("MARKET SEGMENTATION ANALYSIS")
    print("="*60)
    
    # Sponsor segmentation
    sponsor_features = load_and_prepare_sponsor_data()
    sponsor_clusters = sponsor_segmentation_analysis(sponsor_features)
    
    # Drug segmentation
    drug_features, drug_info = load_and_prepare_drug_data()
    drug_clusters = drug_segmentation_analysis(drug_features, drug_info)
    
    # Create summary
    create_segmentation_summary(sponsor_clusters, drug_clusters)
    
    print("\n" + "="*60)
    print("SEGMENTATION ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print("- Sponsors segment into distinct groups based on size, activity, and success rate")
    print("- Drug clusters reveal patterns in application types and approval rates")
    print("- Clear market segments exist for both pharmaceutical companies and drug products")
    print("\nResults saved to:")
    print("- data/processed/ml_sponsor_segmentation.png")
    print("- data/processed/ml_drug_segmentation.png")
    print("- data/processed/ml_segmentation_summary.png")

if __name__ == "__main__":
    main()