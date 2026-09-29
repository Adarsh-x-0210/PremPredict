"""
Interactive CLI for Premier League Transfer Value Predictor
Supports Position-Specific Machine Learning Models and Granular Roles:
- CF, SS: Goals, Assists
- RW, LW, LMF, RMF: Goals, Assists, Chances Created, Dribbles Completed
- AMF: Goals, Assists, Chances Created
- CMF: Balls Recovered, Line-Breaking Passes, Pass Accuracy (%)
- DMF: Balls Recovered, Duels Won, Aerial Duels Won, Line-Breaking Passes
- CB: Duels Won, Successful Tackles, Aerial Duels Won
- RB, LB: Assists, Duels Won, Successful Tackles, Aerial Duels Won
- GK: Saves, Penalties Saved, Pass Accuracy (%)
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from data.fetch_data import load_dataset
from model.predictor import TransferValuePredictor


def clear_and_header():
    print("\n" + "=" * 70)
    print("   PREMIER LEAGUE TRANSFER VALUE PREDICTOR (ROLE-SPECIFIC CLI)    ")
    print("=" * 70)


def get_float_input(prompt: str, min_val: float, max_val: float, default: float) -> float:
    while True:
        val_str = input(f"{prompt} (default {default}): ").strip()
        if not val_str:
            return default
        try:
            val = float(val_str)
            if min_val <= val <= max_val:
                return val
            print(f"Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid number. Please try again.")


def get_int_input(prompt: str, min_val: int, max_val: int, default: int) -> int:
    return int(get_float_input(prompt, min_val, max_val, default))


def get_role_input() -> str:
    roles = {
        "1": ("CF", "Centre Forward / Striker"),
        "2": ("RW", "Right Winger"),
        "3": ("LW", "Left Winger"),
        "4": ("AMF", "Attacking Midfielder"),
        "5": ("CMF", "Central Midfielder"),
        "6": ("DMF", "Defensive Midfielder"),
        "7": ("CB", "Centre Back"),
        "8": ("RB", "Right Back / Wingback"),
        "9": ("LB", "Left Back / Wingback"),
        "10": ("GK", "Goalkeeper")
    }
    print("\nSelect Player Role:")
    for key, (role_code, name) in roles.items():
        print(f"  [{key:>2}] {role_code:<4} - {name}")
    while True:
        choice = input("\nEnter choice (1-10, default 1): ").strip() or "1"
        if choice in roles:
            return roles[choice][0]
        print("Invalid choice. Please enter a number between 1 and 10.")


def predict_custom_player(predictor: TransferValuePredictor):
    print("\n--- [Predict Value for a Custom Player] ---")
    role = get_role_input()
    age = get_float_input("Player Age (years)", min_val=16.0, max_val=42.0, default=24.0)
    minutes = get_float_input("Minutes Played this Season", min_val=0.0, max_val=3420.0, default=2400.0)

    stats = {"age": age, "minutes_played": minutes}

    if role in ["CF", "SS"]:
        stats["goals"] = get_int_input("Goals Scored", 0, 40, 16)
        stats["assists"] = get_int_input("Assists Provided", 0, 25, 5)
    elif role in ["RW", "LW", "RMF", "LMF"]:
        stats["goals"] = get_int_input("Goals Scored", 0, 35, 12)
        stats["assists"] = get_int_input("Assists Provided", 0, 25, 8)
        stats["chances_created"] = get_int_input("Chances Created", 0, 130, 65)
        stats["dribbles_completed"] = get_int_input("Dribbles Completed", 0, 120, 50)
    elif role == "AMF":
        stats["goals"] = get_int_input("Goals Scored", 0, 30, 8)
        stats["assists"] = get_int_input("Assists Provided", 0, 25, 9)
        stats["chances_created"] = get_int_input("Chances Created", 0, 140, 75)
    elif role == "CMF":
        stats["balls_recovered"] = get_int_input("Balls Recovered", 20, 280, 165)
        stats["line_breaking_passes"] = get_int_input("Line-Breaking Passes", 20, 300, 160)
        stats["pass_accuracy"] = get_float_input("Pass Accuracy (%)", 60.0, 98.0, 88.0)
    elif role == "DMF":
        stats["balls_recovered"] = get_int_input("Balls Recovered", 20, 300, 190)
        stats["duels_won"] = get_int_input("Duels Won", 20, 260, 175)
        stats["aerial_duels_won"] = get_int_input("Aerial Duels Won", 5, 100, 45)
        stats["line_breaking_passes"] = get_int_input("Line-Breaking Passes", 20, 300, 150)
    elif role == "CB":
        stats["duels_won"] = get_int_input("Duels Won", 20, 260, 165)
        stats["successful_tackles"] = get_int_input("Successful Tackles", 10, 110, 58)
        stats["aerial_duels_won"] = get_int_input("Aerial Duels Won", 10, 140, 80)
    elif role in ["RB", "LB"]:
        stats["assists"] = get_int_input("Assists Provided", 0, 20, 5)
        stats["duels_won"] = get_int_input("Duels Won", 20, 250, 145)
        stats["successful_tackles"] = get_int_input("Successful Tackles", 10, 110, 68)
        stats["aerial_duels_won"] = get_int_input("Aerial Duels Won", 5, 70, 32)
    elif role == "GK":
        stats["saves"] = get_int_input("Saves Made", 10, 180, 95)
        stats["penalties_saved"] = get_int_input("Penalties Saved", 0, 6, 1)
        stats["pass_accuracy"] = get_float_input("Passing Accuracy (%)", 50.0, 96.0, 82.0)

    predicted_val = predictor.predict_player(position=role, **stats)

    print("\n" + "*" * 55)
    print(f"  Player Role: {role} | Age: {int(age)} | {int(minutes)} mins")
    metric_str = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in stats.items() if k not in ['age', 'minutes_played'])
    print(f"  Metrics:     {metric_str}")
    print(f"  >> PREDICTED VALUE: EUR {predicted_val:.2f} Million")
    print("*" * 55)


def search_existing_player(predictor: TransferValuePredictor, df):
    print("\n--- [Search Player in Premier League Dataset] ---")
    query = input("Enter player name to search (e.g., Saka, Saliba, Palmer, Rice, Raya): ").strip().lower()
    matches = df[df["player_name"].str.lower().str.contains(query)]

    if matches.empty:
        print(f"No players found matching '{query}'.")
        return

    evaluated = predictor.evaluate_all(matches)
    for _, row in evaluated.iterrows():
        print("\n" + "-" * 60)
        print(f"Player:     {row['player_name']} ({row['team']})")
        print(f"Role:       {row['position']} | Age: {row['age']} | {row['minutes_played']} mins")
        pos = row["position"]
        if pos == "GK":
            print(f"GK Stats:   {row['saves']} Saves | {row['penalties_saved']} Pens Saved | {row['pass_accuracy']}% Pass")
        elif pos == "CB":
            print(f"CB Stats:   {row['duels_won']} Duels | {row['successful_tackles']} Tackles | {row['aerial_duels_won']} Aerials")
        elif pos in ["RB", "LB"]:
            print(f"FB Stats:   {row['assists']} Assists | {row['duels_won']} Duels | {row['successful_tackles']} Tackles")
        elif pos in ["DMF", "CMF"]:
            print(f"MF Stats:   {row['line_breaking_passes']} Line Breaks | {row['balls_recovered']} Recoveries | {row['pass_accuracy']}% Pass")
        elif pos in ["AMF", "RW", "LW"]:
            print(f"Att Stats:  {row['goals']}G / {row['assists']}A | {row['chances_created']} Chances | {row['dribbles_completed']} Dribbles")
        else:
            print(f"CF Stats:   {row['goals']} Goals | {row['assists']} Assists")

        print(f"Actual Value:     EUR {row['market_value_eur_m']:.1f}M")
        print(f"Model Prediction: EUR {row['predicted_value_eur_m']:.1f}M")
        diff = row["valuation_diff_eur_m"]
        sign = "+" if diff > 0 else ""
        print(f"Difference:       {sign}EUR {diff:.1f}M ({row['market_status']})")
        print("-" * 60)


def view_coefficients(predictor: TransferValuePredictor):
    print("\n--- [Role-Specific Model Coefficients] ---")
    details = predictor.get_all_group_details()
    for key, info in details.items():
        print(f"\n{info['name']} (Base Intercept: EUR {info['intercept']:.2f}M):")
        for coef in info["coefficients"]:
            feat = coef["feature"]
            w = coef["weight"]
            sign = "+" if w >= 0 else ""
            print(f"  * {feat:<22} : {sign}EUR {w:.2f}M")


def main():
    clear_and_header()
    print("Loading data and training role-specific models...")
    df = load_dataset()
    predictor = TransferValuePredictor()
    predictor.train(df)
    print(f"Models ready! (Trained on {len(df)} Premier League players across 20 clubs)\n")

    while True:
        print("\nMENU:")
        print("  1. Predict value for a custom player (with role-specific metrics)")
        print("  2. Look up an existing Premier League player (Actual vs Predicted)")
        print("  3. View Role-Specific Model Weights (Coefficients)")
        print("  4. Exit")

        choice = input("\nChoose an option (1-4): ").strip()
        if choice == "1":
            predict_custom_player(predictor)
        elif choice == "2":
            search_existing_player(predictor, df)
        elif choice == "3":
            view_coefficients(predictor)
        elif choice == "4":
            print("\nThank you for using the Premier League Transfer Value Predictor! Happy learning!")
            break
        else:
            print("Invalid selection. Please choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
