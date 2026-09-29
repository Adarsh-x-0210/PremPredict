"""
Premier League Transfer Value Predictor - Position-Specific ML Pipeline
Trains dedicated Linear Regression models tailored to each position's exact metrics:
- Strikers (CF, SS): goals, assists
- Wingers (RW, LW, LMF, RMF): goals, assists, chances created, dribbles completed
- Attacking Midfielders (AMF): goals, assists, chances created
- Central Midfielders (CMF): balls recovered, line breaking passes, pass accuracy
- Defensive Midfielders (DMF): balls recovered, duels won, aerial duels won, line breaking passes
- Centre Backs (CB): duels won, successful tackles, aerial duels won
- Fullbacks (RB, LB): assists, duels won, successful tackles, aerial duels won
- Goalkeepers (GK): saves, penalties saved, pass accuracy
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from data.fetch_data import load_dataset
from model.predictor import TransferValuePredictor


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f" {title.upper()} ")
    print("=" * 75)


def main():
    print_banner("Premier League Transfer Predictor (Role-Specific ML Models)")
    print("Tech Stack: requests, pandas, scikit-learn, matplotlib\n")

    # Step 1: Load Data
    print(">>> Step 1: Loading Enriched Premier League Dataset...")
    df = load_dataset()
    print(f"Total Players: {len(df)} across 20 Premier League Clubs")
    print(f"Position breakdown:\n{df['position'].value_counts().to_string()}\n")

    # Step 2: Train Multi-Model Position Predictor
    print_banner("Step 2: Training Role-Specialized Regression Models")
    predictor = TransferValuePredictor()
    predictor.train(df)

    group_details = predictor.get_all_group_details()

    for key, info in group_details.items():
        print(f"\n[{info['name']}] - {info['sample_count']} players")
        print(f"  * Features evaluated: {', '.join(info['features'])}")
        print(f"  * Test R2: {info['metrics'].get('r2', 0):.3f} | Test MAE: EUR {info['metrics'].get('mae', 0):.2f}M")
        print(f"  * Base Intercept: EUR {info['intercept']:.2f}M")
        print("  * Learned Feature Weights:")
        for coef in info["coefficients"]:
            feat = coef["feature"]
            w = coef["weight"]
            sign = "+" if w >= 0 else ""
            print(f"      - {feat:<22}: {sign}EUR {w:6.2f}M")

    # Step 3: Evaluate Real Players by Role
    print_banner("Step 3: Sample Star Player Valuations Across Specific Roles")
    evaluated_df = predictor.evaluate_all(df)

    sample_stars = [
        "Erling Haaland",      # CF
        "Bukayo Saka",         # RW
        "Martin Odegaard",     # AMF
        "Alexis Mac Allister", # CMF
        "Declan Rice",         # DMF
        "William Saliba",      # CB
        "Trent Alexander-Arnold", # RB
        "David Raya"           # GK
    ]

    selected = evaluated_df[evaluated_df["player_name"].isin(sample_stars)][
        ["player_name", "team", "position", "age", "minutes_played", "market_value_eur_m", "predicted_value_eur_m", "valuation_diff_eur_m", "market_status"]
    ]
    print(selected.to_string(index=False))

    # Step 4: Role-Specific Custom Prediction Demonstration
    print_banner("Step 4: Custom Role-Specific Inference Demonstration")

    # 1. Custom Goalkeeper
    gk_pred = predictor.predict_player(
        position="GK",
        age=26,
        minutes_played=2800,
        saves=110,
        penalties_saved=2,
        pass_accuracy=84.5
    )
    print(f"1. Custom Goalkeeper (Age 26, 110 Saves, 2 Pen Saves, 84.5% PA): EUR {gk_pred:.2f}M")

    # 2. Custom Centre Back
    cb_pred = predictor.predict_player(
        position="CB",
        age=23,
        minutes_played=3000,
        duels_won=170,
        successful_tackles=65,
        aerial_duels_won=85
    )
    print(f"2. Custom Centre Back (Age 23, 170 Duels Won, 65 Tackles, 85 Aerials): EUR {cb_pred:.2f}M")

    # 3. Custom Central Midfielder
    cmf_pred = predictor.predict_player(
        position="CMF",
        age=24,
        minutes_played=2700,
        balls_recovered=180,
        line_breaking_passes=160,
        pass_accuracy=89.0
    )
    print(f"3. Custom Central Midfielder (Age 24, 180 Recoveries, 160 Line Breaks, 89% PA): EUR {cmf_pred:.2f}M")

    # 4. Custom Winger
    winger_pred = predictor.predict_player(
        position="RW",
        age=22,
        minutes_played=2600,
        goals=14,
        assists=9,
        chances_created=70,
        dribbles_completed=60
    )
    print(f"4. Custom Winger (Age 22, 14 Goals, 9 Assists, 70 Chances, 60 Dribbles): EUR {winger_pred:.2f}M")

    print_banner("Position-Specific Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
