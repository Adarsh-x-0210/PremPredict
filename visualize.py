"""
Visualization Module using Matplotlib
Generates visual evaluation charts for the Transfer Value Predictor:
1. Actual vs. Predicted Market Values (with ideal y=x reference line)
2. Learned Feature Coefficients (Economic impact of each stat)
3. Goals vs. Market Value distribution by Position
"""

import os
import matplotlib
# Use non-interactive Agg backend to save files cleanly without GUI popup issues
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_actual_vs_predicted(y_true, y_pred, r2: float, mae: float, player_names=None, filename="actual_vs_predicted.png"):
    """
    Scatter plot comparing actual transfer value against predicted transfer value.
    Points on the red diagonal line indicate perfect predictions.
    """
    plt.figure(figsize=(9, 7), dpi=150)
    
    # Scatter plot
    plt.scatter(y_true, y_pred, color="#1f77b4", alpha=0.75, edgecolors="k", s=70, label="Players (Test Set)")
    
    # Ideal 45-degree reference line (y = x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color="#d62728", linestyle="--", linewidth=2, label="Perfect Fit (y = x)")
    
    # Annotate select interesting players if names are provided
    if player_names is not None and len(player_names) == len(y_true):
        # Annotate players with high values or big deviations
        for name, actual, pred in zip(player_names, y_true, y_pred):
            if actual > 70 or abs(actual - pred) > 25:
                plt.annotate(
                    name,
                    (actual, pred),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                    weight="bold",
                    color="#2c3e50"
                )
    
    plt.title("Actual vs. Predicted Premier League Transfer Value", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Actual Market Value (€ Millions)", fontsize=11)
    plt.ylabel("Model Predicted Value (€ Millions)", fontsize=11)
    
    # Annotation box for metrics
    metrics_text = f"$R^2$ Score: {r2:.3f}\nMAE: €{mae:.2f}M"
    plt.text(
        0.05, 0.92, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#b0bec5", alpha=0.9)
    )
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"[matplotlib] Saved: {out_path}")
    return out_path


def plot_feature_coefficients(coef_df: pd.DataFrame, filename="feature_importance.png"):
    """
    Horizontal bar chart showing the magnitude and direction of learned weights.
    Explains which statistics increase or decrease a player's valuation.
    """
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Sort for bottom-to-top rendering
    df_sorted = coef_df.sort_values(by="coefficient", ascending=True)
    
    features = df_sorted["feature"]
    coefficients = df_sorted["coefficient"]
    
    # Color bars: Green for positive contribution, Crimson for negative
    colors = ["#2ca02c" if c >= 0 else "#d62728" for c in coefficients]
    
    bars = plt.barh(features, coefficients, color=colors, edgecolor="black", alpha=0.85, height=0.6)
    
    plt.axvline(0, color="black", linewidth=1.2, linestyle="-")
    plt.title("Linear Regression Coefficients (Feature Impact on Value)", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Valuation Change (€ Millions per unit)", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    
    # Add value labels next to bars
    for bar in bars:
        width = bar.get_width()
        ha = "left" if width >= 0 else "right"
        offset = 0.3 if width >= 0 else -0.3
        plt.text(
            width + offset, bar.get_y() + bar.get_height() / 2,
            f"{width:+.2f}M",
            va="center", ha=ha, fontsize=9, fontweight="semibold"
        )
        
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"[matplotlib] Saved: {out_path}")
    return out_path


def plot_goals_vs_value(df: pd.DataFrame, filename="goals_vs_value.png"):
    """
    Scatter plot showing the relationship between Goals scored and Transfer Market Value,
    colored by player position.
    """
    plt.figure(figsize=(9, 6), dpi=150)
    
    position_colors = {
        "Forward": "#e41a1c",
        "Midfielder": "#377eb8",
        "Defender": "#4daf4a",
        "Goalkeeper": "#984ea3"
    }
    
    for pos, color in position_colors.items():
        subset = df[df["position"] == pos]
        plt.scatter(
            subset["goals"], subset["market_value_eur_m"],
            c=color, label=pos, alpha=0.75, edgecolors="k", s=60
        )
        
    # Annotate top goalscorers
    top_scorers = df.nlargest(6, "goals")
    for _, row in top_scorers.iterrows():
        plt.annotate(
            row["player_name"],
            (row["goals"], row["market_value_eur_m"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            fontweight="bold"
        )
        
    plt.title("Goals Scored vs. Market Value (€M) by Position", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Goals Scored (Season)", fontsize=11)
    plt.ylabel("Market Value (€ Millions)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(title="Position", loc="upper left")
    plt.tight_layout()
    
    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"[matplotlib] Saved: {out_path}")
    return out_path


def generate_all_plots(predictor, df: pd.DataFrame):
    """
    Helper function to generate all project plots in one call.
    """
    if not predictor.is_trained:
        raise ValueError("Predictor must be trained before generating visualizations.")
        
    # 1. Actual vs Predicted on Test Set
    test_player_indices = predictor.X_test.index
    test_player_names = df.loc[test_player_indices, "player_name"].values
    plot_actual_vs_predicted(
        predictor.y_test,
        predictor.y_test_pred,
        r2=predictor.test_metrics["r2"],
        mae=predictor.test_metrics["mae"],
        player_names=test_player_names
    )
    
    # 2. Coefficients Bar Chart
    coef_df = predictor.get_coefficients()
    plot_feature_coefficients(coef_df)
    
    # 3. Goals vs Value Scatter Plot
    plot_goals_vs_value(df)
    
    print(f"[matplotlib] All visualizations generated in: {PLOTS_DIR}")
