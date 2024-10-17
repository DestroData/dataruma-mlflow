import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_correlations(df, target_col, save_path=None):
    """
    Plots the correlation of each variable in the dataframe with the target column.
    """

    correlations = df.corr()[target_col].drop(target_col).sort_values()

    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = correlations.map(colors)

    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )

    fig = plt.figure(figsize=(12, 8))
    plt.barh(correlations.index, correlations.values, color=color_mapped)

    plt.title("Correlation with Demand", fontsize=18)
    plt.xlabel("Correlation Coefficient", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    plt.close(fig)

    return fig


def plot_residuals(pred_y, test_y, save_path=None):
    """
    Plots the residuals of the model predictions against the true values.
    """

    residuals = test_y - pred_y

    sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

    fig = plt.figure(figsize=(12, 8))
    plt.scatter(test_y, residuals, color="blue", alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="-")

    plt.title("Residuals vs True Values", fontsize=18)
    plt.xlabel("True Values", fontsize=16)
    plt.ylabel("Residuals", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    plt.close(fig)

    return fig


def plot_feature_importance(model, X_train, save_path=None):
    """
    Plots feature importance for the GradientBoostingRegressor.
    """

    feature_importances = pd.DataFrame({
    'Caractéristique': X_train.columns,
    'Importance': model.feature_importances_
    })
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    fig = plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Caractéristique', data=feature_importances, palette='viridis', hue='Caractéristique',legend=False)
    plt.title('Importances des Caractéristiques - GradientBoostingRegressor', fontsize=16)
    plt.xlabel('Importance')
    plt.ylabel('Caractéristique')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    plt.close(fig)

    return fig


def plot_residuals_hist(pred_y, test_y, save_path=None):
    """
    Plots histogram of the absolut errors.
    """

    residuals = np.abs(test_y - pred_y)

    sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

    fig = plt.figure(figsize=(12, 8))
    plt.hist(residuals, bins=30, color="blue", alpha=0.7)

    plt.title("Histogram of Residuals", fontsize=18)
    plt.xlabel("Residuals", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    plt.close(fig)

    return fig
