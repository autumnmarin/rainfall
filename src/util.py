import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score


# --------------------------- Data Loading --------------------------- #

def load_data(train_path, test_path, **kwargs):
    train_df = pd.read_csv(train_path, **kwargs)
    test_df = pd.read_csv(test_path, **kwargs)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


# --------------------------- Data Summary --------------------------- #

def summarize_dataframe(df):
    summary = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Count': df.nunique(),
        'Sample Unique': df.apply(lambda x: x.unique()[:5] if x.nunique() < 10 else x.unique()[:5]),
        'Min': df.min(numeric_only=True),
        'Max': df.max(numeric_only=True),
        'Mean': df.mean(numeric_only=True),
        'Std': df.std(numeric_only=True)
    }).reset_index().rename(columns={'index': 'Column'})
    return summary


# --------------------------- Visualization --------------------------- #

def plot_missing_data(df, figsize=(12, 6), color='skyblue'):
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
    if missing_percent.empty:
        print("No missing data found!")
        return
    plt.figure(figsize=figsize)
    missing_percent.plot(kind='bar', color=color)
    plt.title('Percentage of Missing Data by Column')
    plt.ylabel('Percent Missing')
    plt.xlabel('Columns')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, figsize=(12, 8), cmap='coolwarm', annot=False):
    plt.figure(figsize=figsize)
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap=cmap, annot=annot, fmt=".2f", square=True, linewidths=.5)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()


def plot_numeric_distributions(df, figsize=(16, 12), bins=30):
    numeric_cols = df.select_dtypes(include=['number']).columns
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    for idx, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


# --------------------------- Data Preprocessing --------------------------- #

def preprocess_train_test(train_df, test_df, numeric_strategy='median'):
    train_df, test_df = train_df.copy(), test_df.copy()
    numeric_cols = train_df.select_dtypes(include='number').columns
    for col in numeric_cols:
        fill_value = train_df[col].median() if numeric_strategy == 'median' else train_df[col].mean()
        train_df[col].fillna(fill_value, inplace=True)
        test_df[col].fillna(fill_value, inplace=True)
    return train_df, test_df


def remove_outliers_iqr(df, columns=None, multiplier=1.5, return_outliers=False):
    columns = columns or df.select_dtypes(include='number').columns
    df_clean, outliers_list = df.copy(), []
    for col in columns:
        Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR
        outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
        outliers_list.append(outliers)
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    if return_outliers:
        return df_clean, pd.concat(outliers_list).drop_duplicates()
    return df_clean


# --------------------------- Categorical Summary --------------------------- #

def summarize_categorical(df, top_n=5):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    return pd.DataFrame([
        {'Column': col, 'Unique Values': df[col].nunique(), 'Top Values': df[col].value_counts().head(top_n).to_dict()}
        for col in cat_cols
    ])


# --------------------------- Cross-Validation --------------------------- #

def cross_validate_rmse(model, X, y, n_splits=5, stratify=None, random_state=42):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state) if stratify else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = kf.split(X, pd.qcut(stratify, q=10, duplicates='drop')) if stratify else kf.split(X)
    rmse_list = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        rmse = mean_squared_error(y.iloc[val_idx], preds, squared=False)
        rmse_list.append(rmse)
        print(f"Fold {fold + 1} RMSE: {rmse:.4f}")
    print(f"\nAverage RMSE: {np.mean(rmse_list):.4f}")
    return np.mean(rmse_list), rmse_list


def cross_validate_classification(model, X, y, n_splits=5, stratify=True, random_state=42):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state) if stratify else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracy_list = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        acc = accuracy_score(y.iloc[val_idx], preds)
        accuracy_list.append(acc)
        print(f"Fold {fold + 1} Accuracy: {acc:.4f}")
    print(f"\nAverage Accuracy: {np.mean(accuracy_list):.4f}")
    return np.mean(accuracy_list), accuracy_list


# --------------------------- Test Prediction & Submission --------------------------- #

def predict_test(model, test_df, features):
    return model.predict(test_df[features])


def write_submission(test_df, predictions, id_column, output_path='submission.csv', prediction_column='Prediction'):
    submission = pd.DataFrame({id_column: test_df[id_column], prediction_column: predictions})
    submission.to_csv(output_path, index=False)
    print(f"Submission file written to: {output_path}")


def align_columns(train_df, test_df):
    missing_cols = set(train_df.columns) - set(test_df.columns)
    for col in missing_cols:
        test_df[col] = 0
    extra_cols = set(test_df.columns) - set(train_df.columns)
    test_df.drop(columns=extra_cols, inplace=True)
    return test_df[train_df.columns]

import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importance"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]  # Top N features

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Add this in util.py

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Add this in util.py

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_probs, model_name="Model"):
    """
    Plot ROC curve and AUC score.

    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1).
    y_probs : array-like
        Predicted probabilities for the positive class.
    model_name : str
        Label for the model in the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

# Add this to util.py

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", cmap='Blues'):
    """
    Plot a confusion matrix with true vs predicted labels.

    Parameters:
    -----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    labels : list or None
        List of class labels to display. If None, auto-detect from y_true.
    title : str
        Title of the plot.
    cmap : str
        Color map for the heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    plt.title(title)
    plt.show()

# Add this to util.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv=5, scoring='roc_auc', train_sizes=np.linspace(0.1, 1.0, 5), random_state=42):
    """
    Plots a learning curve showing model performance as training size increases.

    Parameters:
    -----------
    estimator : object
        The model to evaluate (must implement fit and predict).
    X : array-like
        Feature dataset.
    y : array-like
        Target labels.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric for evaluation (default: 'roc_auc').
    train_sizes : array-like
        List of training set sizes (fractions or counts).
    random_state : int
        Random seed for reproducibility.
    """
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, random_state=random_state, n_jobs=-1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, valid_scores_mean, 'o-', color='green', label='Cross-validation score')
    plt.title('Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel(scoring)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
