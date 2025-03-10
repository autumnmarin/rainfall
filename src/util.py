import pandas as pd

def load_data(train_path, test_path, **kwargs):
    """
    Load train and test datasets.

    Parameters:
    -----------
    train_path : str
        Path to the training CSV.
    test_path : str
        Path to the test CSV.
    kwargs : dict
        Additional keyword arguments for pd.read_csv.

    Returns:
    --------
    pd.DataFrame, pd.DataFrame
        Train and test DataFrames.
    """
    train_df = pd.read_csv(train_path, **kwargs)
    test_df = pd.read_csv(test_path, **kwargs)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


def summarize_dataframe(df):
    """
    Generate a robust summary table for a pandas DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to summarize.

    Returns:
    --------
    pd.DataFrame
        Summary table with column info, missing values, unique counts, dtypes, and stats.
    """
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
    })

    # Reset index to make 'Column' a column instead of index
    summary = summary.reset_index().rename(columns={'index': 'Column'})

    return summary

import matplotlib.pyplot as plt

def plot_missing_data(df, figsize=(12, 6), color='skyblue'):

    # Calculate percent missing
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

    if missing_percent.empty:
        print("No missing data found!")
        return

    # Plot
    plt.figure(figsize=figsize)
    missing_percent.plot(kind='bar', color=color)
    plt.title('Percentage of Missing Data by Column')
    plt.ylabel('Percent Missing')
    plt.xlabel('Columns')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df, figsize=(12, 8), cmap='coolwarm', annot=False):
    """
    Plot a correlation heatmap for numerical features in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze.
    figsize : tuple
        Size of the plot.
    cmap : str
        Color map for the heatmap.
    annot : bool
        Whether to annotate cells with correlation values.
    """
    plt.figure(figsize=figsize)
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap=cmap, annot=annot, fmt=".2f", square=True, linewidths=.5)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()

def plot_numeric_distributions(df, figsize=(16, 12), bins=30):
    """
    Plot histograms for all numeric columns in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze.
    figsize : tuple
        Size of the overall figure.
    bins : int
        Number of bins for histograms.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    n_cols = 3  # Number of plots per row
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols  # Auto-calculate rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
    
    # Turn off empty plots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def summarize_categorical(df, top_n=5):
    """
    Summarize categorical columns: number of unique values and top frequent values.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze.
    top_n : int
        Number of top frequent values to display.

    Returns:
    --------
    pd.DataFrame
        Summary of categorical variables.
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    summary = []

    for col in cat_cols:
        top_values = df[col].value_counts().head(top_n).to_dict()
        summary.append({
            'Column': col,
            'Unique Values': df[col].nunique(),
            'Top Values': top_values
        })

    return pd.DataFrame(summary)


def preprocess_train_test(train_df, test_df, numeric_strategy='median'):
    """
    Preprocess train and test datasets consistently:
    - Fill missing numeric data using median or mean of train.
    - Align columns (ensure test has same columns).

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset.
    test_df : pd.DataFrame
        Test dataset.
    numeric_strategy : str
        Strategy for numeric imputation: 'median' or 'mean'.

    Returns:
    --------
    pd.DataFrame, pd.DataFrame
        Preprocessed train and test DataFrames.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Handle numeric columns
    numeric_cols = train_df.select_dtypes(include='number').columns

    for col in numeric_cols:
        if numeric_strategy == 'median':
            fill_value = train_df[col].median()
        else:
            fill_value = train_df[col].mean()

        # Fill NaN in both datasets with train's value
        train_df[col] = train_df[col].fillna(fill_value)
        test_df[col] = test_df[col].fillna(fill_value)

    return train_df, test_df


def remove_outliers_iqr(df, columns=None, multiplier=1.5, return_outliers=False):
    """
    Remove outliers from specified numeric columns using the IQR method.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process.
    columns : list or None
        List of numeric columns to check for outliers. If None, all numeric columns are used.
    multiplier : float
        The IQR multiplier to define outlier thresholds (1.5 is standard).
    return_outliers : bool
        If True, also return a DataFrame of the outliers that were removed.

    Returns:
    --------
    pd.DataFrame
        DataFrame with outliers removed.
    pd.DataFrame (optional)
        DataFrame of outliers if return_outliers=True.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns

    df_clean = df.copy()
    outliers_list = []

    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Find outliers for this column
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        outliers_list.append(outliers)

        # Remove outliers for this column
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    if return_outliers:
        # Combine all detected outliers into one DataFrame
        all_outliers = pd.concat(outliers_list).drop_duplicates()
        return df_clean, all_outliers
    else:
        return df_clean

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

def cross_validate_rmse(model, X, y, n_splits=5, stratify=None, random_state=42):
    """
    Perform cross-validation and return average RMSE.

    Parameters:
    -----------
    model : sklearn-like estimator
        Model to be evaluated.
    X : pd.DataFrame or np.ndarray
        Features.
    y : pd.Series or np.ndarray
        Target.
    n_splits : int
        Number of folds.
    stratify : pd.Series or None
        If provided, will use StratifiedKFold.
    random_state : int
        Random seed for reproducibility.

    Returns:
    --------
    float
        Average RMSE across folds.
    list
        List of RMSE for each fold.
    """
    if stratify is not None:
        # Bin target for stratification (if it's regression)
        bins = pd.qcut(stratify, q=10, duplicates='drop')  # 10 quantiles as bins
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kf.split(X, bins)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kf.split(X)

    rmse_list = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)  # RMSE
        rmse_list.append(rmse)
        print(f"Fold {fold + 1} RMSE: {rmse:.4f}")

    avg_rmse = np.mean(rmse_list)
    print(f"\nAverage RMSE: {avg_rmse:.4f}")
    return avg_rmse, rmse_list

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

def cross_validate_classification(model, X, y, n_splits=5, stratify=True, random_state=42):
    """
    Perform cross-validation for classification tasks and return average accuracy.

    Parameters:
    -----------
    model : sklearn-like estimator
        Classification model to be evaluated.
    X : pd.DataFrame or np.ndarray
        Features.
    y : pd.Series or np.ndarray
        Target.
    n_splits : int
        Number of folds.
    stratify : bool
        Whether to use StratifiedKFold (recommended for classification).
    random_state : int
        Random seed for reproducibility.

    Returns:
    --------
    float
        Average accuracy across folds.
    list
        List of accuracy scores for each fold.
    """
    if stratify:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accuracy_list = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)
        accuracy_list.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

    avg_accuracy = np.mean(accuracy_list)
    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    return avg_accuracy, accuracy_list


def predict_test(model, test_df, features):
    """
    Make predictions on the test dataset using the trained model.

    Parameters:
    -----------
    model : sklearn-like estimator
        Trained model.
    test_df : pd.DataFrame
        Test dataset (without target column).
    features : list
        List of feature column names used in training.

    Returns:
    --------
    np.ndarray
        Predictions for the test dataset.
    """
    X_test = test_df[features]  # Ensure only feature columns are used
    preds = model.predict(X_test)
    return preds

def write_submission(test_df, predictions, id_column, output_path='submission.csv', prediction_column='Prediction'):
    """
    Create and write a submission CSV file with a custom prediction column name.

    Parameters:
    -----------
    test_df : pd.DataFrame
        Test dataset (must contain the ID column).
    predictions : array-like
        Model predictions for the test set.
    id_column : str
        Name of the ID column to include in submission.
    output_path : str
        Path to output CSV file.
    prediction_column : str
        Name of the prediction column in submission file.

    Returns:
    --------
    None
    """
    submission = pd.DataFrame({
        id_column: test_df[id_column],
        prediction_column: predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission file written to: {output_path}")

def align_columns(train_df, test_df):
    """
    Align the test DataFrame's columns to match the training DataFrame's columns.
    Adds missing columns in test_df filled with 0, and drops extra columns.

    Parameters:
    -----------
    train_df : pd.DataFrame
        The training DataFrame after feature engineering.
    test_df : pd.DataFrame
        The test DataFrame that needs to be aligned.

    Returns:
    --------
    pd.DataFrame
        Aligned test DataFrame with the same columns as train_df.
    """
    # Find missing columns in test
    missing_cols = set(train_df.columns) - set(test_df.columns)
    for col in missing_cols:
        test_df[col] = 0  # Add missing columns with 0

    # Drop any extra columns in test_df not in train_df
    extra_cols = set(test_df.columns) - set(train_df.columns)
    test_df = test_df.drop(columns=extra_cols)

    # Ensure same column order
    test_df = test_df[train_df.columns]

    return test_df
