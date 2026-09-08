"""
Customer Churn Prediction in Telecommunications Using Machine Learning and Explainable AI
Complete Implementation - Iranian Telecommunication Customer Churn Dataset

Author: Jaya Krishna Alapati
"""

# ============================================================
# SECTION 1: LIBRARY IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Preprocessing and Model Selection
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from mcnemar_comparison import evaluate_mcnemar
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, brier_score_loss,
                             classification_report, confusion_matrix, roc_curve)

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Explainability
import shap

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("All libraries imported successfully.")


# ============================================================
# SECTION 2: DATA LOADING AND INITIAL INSPECTION
# ============================================================

def load_and_inspect_data(filepath):
    """Load the dataset from the given filepath and perform initial inspection."""

    df = pd.read_csv(filepath)
    
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"\nDataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nFirst 5 Records:")
    print(df.head())
    
    print(f"\nData Types:")
    print(df.dtypes)
    
    print(f"\nStatistical Summary:")
    print(df.describe())
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")
    
    print(f"\nDuplicate Records: {df.duplicated().sum()}")
    
    return df


# Load the dataset
dataset_path = r"C:\Users\m\Documents\My_Project\iranian+churn+dataset\Customer Churn.csv"

if Path(dataset_path).exists():
    df = load_and_inspect_data(dataset_path)
else:
    print(f"Dataset not found. Expected file at: {dataset_path}")
    print("Please provide the Iranian Telecom Customer Churn Dataset")
    raise SystemExit(1)


# ============================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

def perform_eda(df):
    """Conduct comprehensive exploratory data analysis."""
    
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # 3.1 Target Variable Distribution
    print("\n--- Target Variable Distribution ---")
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    print(f"Non-Churn (0): {churn_counts[0]} ({churn_pct[0]:.2f}%)")
    print(f"Churn (1): {churn_counts[1]} ({churn_pct[1]:.2f}%)")
    print(f"Imbalance Ratio: {churn_counts[0]/churn_counts[1]:.2f}:1")
    
    # Plot target distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    sns.countplot(x='Churn', data=df, ax=axes[0], palette=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Customer Churn Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Churn Status')
    axes[0].set_ylabel('Count')
    axes[0].set_xticklabels(['Retained (0)', 'Churned (1)'])
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=12)
    
    # Pie chart
    axes[1].pie(churn_counts.values, labels=['Retained', 'Churned'],
                autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
                explode=[0, 0.05], shadow=True, startangle=90)
    axes[1].set_title('Churn Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig_churn_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 3.2 Numerical Feature Distributions
    print("\n--- Numerical Feature Distributions ---")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in numerical_cols:
        numerical_cols.remove('Churn')
    print(f"Numerical Columns: {numerical_cols}")
    print(f"\nDescriptive Statistics:")
    print(df[numerical_cols].describe().round(2))

    # Histograms for numerical features
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        df[col].hist(bins=30, ax=axes[idx], color='#3498db', edgecolor='white', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}', fontsize=10, pad=10)
        axes[idx].set_xlabel(col, fontsize=9)
        axes[idx].set_ylabel('Frequency', fontsize=9)
        axes[idx].tick_params(axis='x', labelsize=8, rotation=30)
        axes[idx].tick_params(axis='y', labelsize=8)
    
    # Remove empty subplots
    for idx in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.95, bottom=0.08)
    plt.tight_layout(pad=3.0)
    plt.savefig('fig_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3.3 Box Plots by Churn Status
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        sns.boxplot(x='Churn', y=col, data=df, ax=axes[idx],
                   palette=['#2ecc71', '#e74c3c'], hue='Churn', legend=False)
        axes[idx].set_title(f'{col} by Churn Status', fontsize=10, pad=10)
        axes[idx].set_xticklabels(['Retained', 'Churned'], rotation=0, fontsize=9)
        axes[idx].tick_params(axis='y', labelsize=8)
    
    for idx in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.95, bottom=0.08)
    plt.tight_layout(pad=3.0)
    plt.savefig('fig_boxplots_by_churn.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3.4 Correlation Analysis
    print("\n--- Correlation Analysis ---")
    if 'Churn' not in numerical_cols:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Churn' in numerical_cols:
            numerical_cols.remove('Churn')

    correlation_matrix = df[numerical_cols + ['Churn']].corr()

    # Correlations with target variable
    churn_corr = correlation_matrix['Churn'].drop('Churn').sort_values(ascending=False)
    print("\nCorrelations with Churn (sorted):")
    print(churn_corr.round(4))

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.92, bottom=0.15)
    plt.tight_layout(pad=2.0)
    plt.savefig('fig_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3.5 Churn by Categorical Features
    print("\n--- Churn by Categorical Features ---")
    categorical_features = ['Complains', 'Tariff Plan', 'Status', 'Age Group']
    available_cats = [c for c in categorical_features if c in df.columns]

    if available_cats:
        fig, axes = plt.subplots(
            1,
            len(available_cats),
            figsize=(5 * len(available_cats), 5),
            squeeze=False
        )

        for idx, col in enumerate(available_cats):
            churn_rate = df.groupby(col)['Churn'].mean().mul(100)
            axes[0, idx].bar(churn_rate.index.astype(str), churn_rate.values,
                             color='#e74c3c', alpha=0.7)
            axes[0, idx].set_title(f'Churn Rate by {col}', fontsize=12, fontweight='bold')
            axes[0, idx].set_xlabel(col)
            axes[0, idx].set_ylabel('Churn Rate (%)')
            axes[0, idx].tick_params(axis='x', rotation=0)

        plt.tight_layout()
        plt.savefig('fig_churn_by_category.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        print("No supported categorical columns were found for churn-rate plotting.")
    
    return numerical_cols

numerical_cols = perform_eda(df)

# ============================================================
# SECTION 4: DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================

def preprocess_data(df):
    """Perform data preprocessing and prepare for model training."""

    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # 4.1 Handle Missing Values
    print("\n--- Handling Missing Values ---")
    missing_before = df.isnull().sum().sum()
    print(f"Total missing values: {missing_before}")

    if missing_before > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        print("Missing values handled with median for numeric features and mode for categorical features.")
    else:
        print("No missing values detected. No imputation required.")

    # 4.2 Remove Duplicates
    print("\n--- Removing Duplicates ---")
    duplicates_before = df.duplicated().sum()
    print(f"Duplicate rows before cleaning: {duplicates_before}")

    if duplicates_before > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {duplicates_before} duplicate records.")
    else:
        print("No duplicate records found.")

    duplicates_after = df.duplicated().sum()
    print(f"Duplicate rows after cleaning: {duplicates_after}")

    # 4.3 Feature-Target Separation
    print("\n--- Feature-Target Separation ---")
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"Target distribution:\n  Non-Churn: {(y==0).sum()} ({(y==0).mean()*100:.2f}%)")
    print(f"  Churn: {(y==1).sum()} ({(y==1).mean()*100:.2f}%)")

    # 4.4 Train-Test Split
    print("\n--- Train-Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Train churn rate: {y_train.mean()*100:.2f}%")
    print(f"Test churn rate: {y_test.mean()*100:.2f}%")

    # 4.5 Feature Scaling (for Logistic Regression)
    print("\n--- Feature Scaling ---")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    print("StandardScaler applied for algorithms requiring normalised features.")
    print(f"Scaled training set mean (sample): {X_train_scaled.iloc[:, 0].mean():.4f}")
    print(f"Scaled training set std (sample): {X_train_scaled.iloc[:, 0].std():.4f}")

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, X, y


X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, X, y = preprocess_data(df)

# ============================================================
# SECTION 5: MODEL DEVELOPMENT AND TRAINING
# ============================================================

def train_and_evaluate_models(X_train, X_test, y_train, y_test,
                              X_train_scaled, X_test_scaled):
    """Train all six ML models and evaluate their performance."""

    print("\n" + "=" * 60)
    print("MODEL DEVELOPMENT AND EVALUATION")
    print("=" * 60)

    results = {}
    trained_models = {}

    model_specs = [
        ("Logistic Regression", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='lbfgs'
        ), X_train_scaled, X_test_scaled),
        ("Decision Tree", DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        ), X_train, X_test),
        ("Random Forest", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ), X_train, X_test),
        ("XGBoost", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        ), X_train, X_test),
        ("LightGBM", LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=42,
            verbose=-1
        ), X_train, X_test),
        ("CatBoost", CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            auto_class_weights='Balanced',
            random_state=42,
            verbose=0
        ), X_train, X_test),
    ]

    for name, model, X_fit, X_pred in model_specs:
        print(f"\n--- Model: {name} ---")
        try:
            model.fit(X_fit, y_train)
            pred = model.predict(X_pred)
            proba = model.predict_proba(X_pred)[:, 1]

            results[name] = {
                'Accuracy': accuracy_score(y_test, pred),
                'Precision': precision_score(y_test, pred),
                'Recall': recall_score(y_test, pred),
                'F1-Score': f1_score(y_test, pred),
                'ROC-AUC': roc_auc_score(y_test, proba),
                'Brier-Score': brier_score_loss(y_test, proba)
            }
            trained_models[name] = (model, X_pred)
            print(f"  Status: SUCCESS")
            print(f"  Accuracy: {results[name]['Accuracy']:.4f}")
            print(f"  F1-Score: {results[name]['F1-Score']:.4f}")
            print(f"  ROC-AUC: {results[name]['ROC-AUC']:.4f}")
            print(f"  Brier-Score: {results[name]['Brier-Score']:.4f}")
        except Exception as exc:
            print(f"  Status: FAILED")
            print(f"  Error: {type(exc).__name__}: {exc}")

    return results, trained_models
results, trained_models = train_and_evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled
)

# ============================================================
# SECTION 6: RESULTS COMPARISON AND VISUALISATION
# ============================================================

def visualise_results(results, trained_models, X_test, y_test, X_test_scaled):
    """Generate comprehensive visualisations of model performance."""
    
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON AND VISUALISATION")
    print("=" * 60)
    
    # 6.1 Results Summary Table
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    results_df = results_df.sort_values('ROC-AUC', ascending=False)
    
    print("\n--- Model Performance Comparison ---")
    print(results_df.to_string())
    
    # Save results to CSV
    results_df.to_csv('model_comparison_results.csv')
    print("\nResults saved to 'model_comparison_results.csv'")
    
    # 6.2 Performance Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Brier-Score']
    x = np.arange(len(results_df.index))
    width = 0.12
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, results_df[metric], width, 
                     label=metric, color=colors[i], alpha=0.85)
    
    ax.set_xlabel('Machine Learning Algorithm', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparative Performance of Machine Learning Algorithms\nfor Customer Churn Prediction', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(results_df.index, rotation=30, ha='right')
    fig.set_size_inches(22, 7)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.30, 0.92), ncol=1, fontsize=10, frameon=True)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    fig.subplots_adjust(bottom=0.20, top=0.88, right=0.60, left=0.05)
    plt.savefig('fig_model_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()
    
    # 6.3 ROC Curves for All Models
    fig, ax = plt.subplots(figsize=(10, 8))
    
    model_predictions = {
        'Logistic Regression': (trained_models['Logistic Regression'][0].predict_proba(X_test_scaled)[:, 1]),
        'Decision Tree': (trained_models['Decision Tree'][0].predict_proba(X_test)[:, 1]),
        'Random Forest': (trained_models['Random Forest'][0].predict_proba(X_test)[:, 1]),
        'XGBoost': (trained_models['XGBoost'][0].predict_proba(X_test)[:, 1]),
        'LightGBM': (trained_models['LightGBM'][0].predict_proba(X_test)[:, 1]),
        'CatBoost': (trained_models['CatBoost'][0].predict_proba(X_test)[:, 1]),
    }
    
    colors_roc = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    
    for (name, proba), color in zip(model_predictions.items(), colors_roc):
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=color, linewidth=2, 
                label=f'{name} (AUC = {auc_val:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Baseline')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - All Models Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6.4 Confusion Matrices for All Models
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    model_preds = {
        'Logistic Regression': trained_models['Logistic Regression'][0].predict(X_test_scaled),
        'Decision Tree': trained_models['Decision Tree'][0].predict(X_test),
        'Random Forest': trained_models['Random Forest'][0].predict(X_test),
        'XGBoost': trained_models['XGBoost'][0].predict(X_test),
        'LightGBM': trained_models['LightGBM'][0].predict(X_test),
        'CatBoost': trained_models['CatBoost'][0].predict(X_test),
    }
    
    for idx, (name, preds) in enumerate(model_preds.items()):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Retained', 'Churned'],
                   yticklabels=['Retained', 'Churned'], cbar=False)
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold', pad=12)
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=0, labelsize=9)
        axes[idx].tick_params(axis='y', labelsize=9)
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.02)
    fig.subplots_adjust(top=0.92, hspace=0.35, wspace=0.3)
    plt.tight_layout(pad=2.5)
    plt.savefig('fig_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6.5 Cross-Validation Results
    print("\n--- Cross-Validation Results (5-Fold Stratified) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {}
    models_for_cv = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', use_label_encoder=False),
        'LightGBM': LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, is_unbalance=True, random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, auto_class_weights='Balanced', random_state=42, verbose=0)
    }
    
    cv_scores_all = {}
    for name, model in models_for_cv.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        cv_scores_all[name] = scores
        cv_results[name] = {'Mean AUC': scores.mean(), 'Std AUC': scores.std()}
        print(f"  {name}: Mean AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # Cross-validation boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.boxplot(list(cv_scores_all.values()), patch_artist=True)
    ax.set_ylabel('ROC-AUC Score', fontsize=12)
    ax.set_title('5-Fold Cross-Validation ROC-AUC Scores', fontsize=14, fontweight='bold')
    ax.set_xticklabels(list(cv_scores_all.keys()), rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3)
    fig.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig('fig_cross_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df


mcnemar_results = evaluate_mcnemar(trained_models, y_test)
results_df = visualise_results(results, trained_models, X_test, y_test, X_test_scaled)

# ============================================================
# SECTION 7: SHAP EXPLAINABILITY ANALYSIS
# ============================================================

def perform_shap_analysis(trained_models, X_train, X_test, results):
    """Apply SHAP to the best-performing model for explainability."""

    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    results_df = pd.DataFrame(results).T
    best_model_name = results_df['ROC-AUC'].idxmax()
    print(f"\nBest performing model (by ROC-AUC): {best_model_name}")
    print(f"ROC-AUC: {results_df.loc[best_model_name, 'ROC-AUC']:.4f}")

    best_model = trained_models[best_model_name][0]
    best_X_test = trained_models[best_model_name][1]

    print("\n--- Computing SHAP Values ---")

    try:
        if best_model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 'Decision Tree']:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(best_X_test)
        else:
            explainer = shap.LinearExplainer(best_model, X_train)
            shap_values = explainer.shap_values(best_X_test)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        print(f"SHAP values computed. Shape: {shap_values.shape}")
    except Exception as exc:
        print(f"SHAP computation failed: {type(exc).__name__}: {exc}")
        return None, best_model_name, None

    print("\n--- Global Feature Importance ---")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, best_X_test, plot_type="bar", show=False)
    plt.title(f'SHAP Global Feature Importance ({best_model_name})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n--- SHAP Summary Plot (Feature Impact Direction) ---")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, best_X_test, show=False)
    plt.title(f'SHAP Summary Plot ({best_model_name})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n--- SHAP Dependence Plots (Top 4 Features) ---")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': best_X_test.columns,
        'Mean |SHAP|': mean_abs_shap
    }).sort_values('Mean |SHAP|', ascending=False)

    print("\nFeature Importance Ranking (Mean |SHAP Value|):")
    print(feature_importance_df.to_string(index=False))

    top_features = feature_importance_df['Feature'].head(4).tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(top_features):
        shap.dependence_plot(feature, shap_values, best_X_test, ax=axes[idx], show=False)
        axes[idx].set_title(f'SHAP Dependence: {feature}', fontsize=11, fontweight='bold')

    plt.suptitle(f'SHAP Dependence Plots - Top 4 Features ({best_model_name})', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig_shap_dependence_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n--- Local Explanations (Individual Predictions) ---")
    churned_indices = y_test[y_test == 1].index
    if len(churned_indices) > 0:
        sample_idx = 0
        print(f"\nExplanation for Customer (Predicted to Churn):")
        print(f"Actual outcome: {'Churn' if y_test.iloc[sample_idx] == 1 else 'Retained'}")

        try:
            plt.figure(figsize=(16, 5))
            shap.force_plot(
                explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray)
                else explainer.expected_value[1],
                shap_values[sample_idx],
                best_X_test.iloc[sample_idx],
                matplotlib=True,
                show=False
            )
            plt.title('SHAP Force Plot - Individual Churn Prediction', fontsize=12, fontweight='bold', pad=16)
            plt.tight_layout(pad=3.0)
            plt.subplots_adjust(bottom=0.25)
            plt.savefig('fig_shap_force_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as exc:
            print(f"Force plot failed: {type(exc).__name__}: {exc}")

    print("\n--- SHAP Waterfall Plot (Single Prediction Breakdown) ---")
    try:
        plt.figure(figsize=(10, 8))
        shap_explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray)
            else explainer.expected_value[1],
            data=best_X_test.iloc[0].values,
            feature_names=best_X_test.columns.tolist()
        )
        shap.waterfall_plot(shap_explanation, show=False)
        plt.title('SHAP Waterfall Plot - Prediction Breakdown', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('fig_shap_waterfall.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as exc:
        print(f"Waterfall plot failed: {type(exc).__name__}: {exc}")

    return feature_importance_df, best_model_name, shap_values


feature_importance_df, best_model_name, shap_values = perform_shap_analysis(
    trained_models, X_train, X_test, results
)


# ============================================================
# SECTION 8: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE - FINAL SUMMARY")
print("=" * 60)

print(f"\nBest Performing Model: {best_model_name}")
print(f"\nFinal Performance Metrics:")
try:
    print(results_df.to_string())
except Exception as exc:
    print(f"Performance metrics unavailable: {exc}")

if feature_importance_df is not None and not feature_importance_df.empty:
    print(f"\nTop 5 Most Important Features (SHAP):")
    print(feature_importance_df.head().to_string(index=False))
else:
    print(f"\nTop 5 Most Important Features (SHAP): not available because SHAP analysis did not complete.")

print(f"\nFiles Generated:")
output_files = [
    "mcnemar_results.csv",
    "mcnemar_test_predictions.csv",
    "model_comparison_results.csv",
    "fig_churn_distribution.png",
    "fig_feature_distributions.png",
    "fig_boxplots_by_churn.png",
    "fig_correlation_heatmap.png",
    "fig_churn_by_category.png",
    "fig_model_comparison.png",
    "fig_roc_curves.png",
    "fig_confusion_matrices.png",
    "fig_cross_validation.png",
    "fig_shap_feature_importance.png",
    "fig_shap_summary_beeswarm.png",
    "fig_shap_dependence_plots.png",
    "fig_shap_force_plot.png",
    "fig_shap_waterfall.png",
]

for file_name in output_files:
    status = "[OK]" if Path(file_name).exists() else "[MISSING]"
    print(f"  {status} {file_name}")

print("\n" + "=" * 60)
print("END OF IMPLEMENTATION")
print("=" * 60)
