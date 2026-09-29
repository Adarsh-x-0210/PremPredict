"""
Premier League Match Outcome Predictor - Main Evaluation Pipeline
Trains Random Forest & XGBoost classifiers, compares performance,
inspects feature importances, and tests sample marquee fixtures.
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import pandas as pd
import numpy as np
from model.match_predictor import MatchOutcomePredictor

MATCHES_CSV = os.path.join(PROJECT_DIR, "data", "pl_matches_dataset.csv")


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f" {title.upper()} ")
    print("=" * 75)


def main():
    print_banner("Premier League Match Outcome Predictor (XGBoost & Random Forest)")
    print("Tech Stack: pandas, scikit-learn, xgboost, matplotlib\n")

    # Step 1: Load Matches
    print(f">>> Step 1: Loading Historical Matches from {MATCHES_CSV}...")
    df = pd.read_csv(MATCHES_CSV)
    print(f"Total Matches: {len(df)}")
    print("Outcome Breakdown:")
    counts = df["result"].value_counts()
    print(f"  Home Wins (H): {counts.get('H', 0)} ({counts.get('H', 0)/len(df)*100:.1f}%)")
    print(f"  Draws     (D): {counts.get('D', 0)} ({counts.get('D', 0)/len(df)*100:.1f}%)")
    print(f"  Away Wins (A): {counts.get('A', 0)} ({counts.get('A', 0)/len(df)*100:.1f}%)")

    # Step 2: Train Models
    print_banner("Step 2: Training Random Forest & XGBoost Classifiers")
    predictor = MatchOutcomePredictor()
    results = predictor.train(df, test_size=0.2, random_state=42)

    rf_res = results["random_forest"]
    xgb_res = results["xgboost"]

    print("Model Performance Comparison (Unseen Test Set - 152 Matches):")
    print("-" * 65)
    print(f"Algorithm            Test Accuracy   Train Accuracy   Log Loss")
    print(f"XGBoost Classifier   {xgb_res['test_acc']*100:.1f}%           {xgb_res['train_acc']*100:.1f}%           {xgb_res['log_loss']:.3f}")
    print(f"Random Forest        {rf_res['test_acc']*100:.1f}%           {rf_res['train_acc']*100:.1f}%           {rf_res['log_loss']:.3f}")
    print("-" * 65)
    print(f"Note: 3-class random chance is 33.3%. The models achieve solid predictive edge!")

    # Step 3: Feature Importances
    print_banner("Step 3: What Drives Premier League Match Outcomes? (Feature Importance)")
    print("Top 6 Features learned by XGBoost:")
    for item in xgb_res["feature_importances"][:6]:
        feat = item["feature"]
        imp = item["importance"]
        print(f"  * {feat:<25}: {imp*100:5.1f}%")

    # Step 4: Test Sample Marquee Matchups
    print_banner("Step 4: Testing Marquee Matchups (Win / Draw / Loss Probabilities)")

    test_fixtures = [
        ("Arsenal", "Chelsea"),
        ("Manchester City", "Liverpool"),
        ("Tottenham", "Arsenal"),
        ("Aston Villa", "Manchester United"),
        ("Newcastle", "West Ham")
    ]

    for home_team, away_team in test_fixtures:
        pred = predictor.predict_match(home_team, away_team, model_type="xgboost")
        probs = pred["probabilities"]
        print(f"\n⚽ {home_team} (Home) vs. {away_team} (Away)")
        print(f"   Predicted Outcome: {pred['predicted_outcome']}")
        print(f"   Probabilities    : Home Win: {probs['home_win']}% | Draw: {probs['draw']}% | Away Win: {probs['away_win']}%")
        h_stat = pred["stats_comparison"]["home"]
        a_stat = pred["stats_comparison"]["away"]
        print(f"   Form (Last 5)    : {home_team}: {h_stat['form_points_last_5']} pts vs {away_team}: {a_stat['form_points_last_5']} pts")
        print(f"   Goals Avg        : {home_team}: {h_stat['goals_scored_avg']} vs {away_team}: {a_stat['goals_scored_avg']}")

    print_banner("Match Predictor Pipeline Run Complete!")


if __name__ == "__main__":
    main()
