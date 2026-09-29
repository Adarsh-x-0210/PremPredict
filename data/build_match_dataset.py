"""
Realistic Premier League Match Dataset Builder
Incorporates:
1. Squad Quality & Market Valuations directly from pl_players_dataset.csv
2. Realistic Poisson goal expectancy based on Squad Strength & Home Advantage
3. Realistic team rolling stats (Form, Goals, Shots, SoT, Possession)
4. Realistic match outcomes:
   - Super teams (City, Arsenal, Liverpool) dominate bottom teams even AWAY
   - Home advantage provides ~0.35 goal boost, making equal teams favored at home
   - Balanced league-wide outcome distribution (~45% Home Win, ~25% Draw, ~30% Away Win)
"""

import pandas as pd
import numpy as np
import json
import os

PLAYERS_CSV = r"E:\premier_league_transfer_predictor\data\pl_players_dataset.csv"
MATCHES_CSV = r"E:\premier_league_transfer_predictor\data\pl_matches_dataset.csv"
TEAM_STATS_JSON = r"E:\premier_league_transfer_predictor\data\team_stats.json"

# Load squad values from player dataset
players_df = pd.read_csv(PLAYERS_CSV)
squad_values = players_df.groupby("team")["market_value_eur_m"].sum().to_dict()

# Base team profile anchored on real squad valuations
TEAM_PROFILES = {}
for team, val in squad_values.items():
    # Relative rating normalized between 60 and 95
    rating = 60.0 + (val - 200.0) / (1260.0 - 200.0) * 35.0
    
    # Expected goals per game baseline based on squad rating
    att_exp = 0.8 + (rating - 60.0) / 35.0 * 1.6   # 0.8 (bottom) to 2.4 (top)
    def_exp = 2.0 - (rating - 60.0) / 35.0 * 1.2   # 2.0 (bottom) to 0.8 (top)
    poss_exp = 40.0 + (rating - 60.0) / 35.0 * 25.0 # 40% to 65%
    shots_exp = 10.0 + (rating - 60.0) / 35.0 * 6.5 # 10 to 16.5
    sot_exp = 3.2 + (rating - 60.0) / 35.0 * 3.6   # 3.2 to 6.8
    
    TEAM_PROFILES[team] = {
        "squad_value_m": round(val, 1),
        "rating": round(rating, 1),
        "att_exp": round(att_exp, 2),
        "def_exp": round(def_exp, 2),
        "poss_exp": round(poss_exp, 1),
        "shots_exp": round(shots_exp, 1),
        "sot_exp": round(sot_exp, 1)
    }

teams = sorted(list(TEAM_PROFILES.keys()))
np.random.seed(42)

matches = []
match_id = 1

# Generate 2 full seasons (760 matches)
for season in ["2022-23", "2023-24"]:
    for h_team in teams:
        for a_team in teams:
            if h_team == a_team:
                continue

            h_prof = TEAM_PROFILES[h_team]
            a_prof = TEAM_PROFILES[a_team]

            # Strength differential
            rating_diff = h_prof["rating"] - a_prof["rating"]

            # Real football home advantage: ~+0.35 expected goals for home
            # But heavily overpowered by squad quality!
            # Example: Ipswich (rating 60) vs City (rating 95): rating_diff = -35
            # h_xg = 0.8 + (60 - 95)*0.03 + 0.35 = 0.8 - 1.05 + 0.35 = 0.25 xG (Ipswich rarely scores)
            # a_xg = 2.4 + (95 - 60)*0.03 - 0.20 = 2.4 + 1.05 - 0.20 = 3.25 xG (City creates high chances!)
            h_xg = max(0.2, (h_prof["att_exp"] * 0.55 + a_prof["def_exp"] * 0.45) + (rating_diff * 0.028) + 0.32 + np.random.normal(0, 0.25))
            a_xg = max(0.15, (a_prof["att_exp"] * 0.55 + h_prof["def_exp"] * 0.45) - (rating_diff * 0.028) - 0.18 + np.random.normal(0, 0.25))

            h_goals = int(np.random.poisson(max(0.1, h_xg)))
            a_goals = int(np.random.poisson(max(0.1, a_xg)))

            if h_goals > a_goals:
                res = "H"
            elif h_goals < a_goals:
                res = "A"
            else:
                res = "D"

            # Possession strongly dictated by squad quality
            poss_diff = (h_prof["poss_exp"] - a_prof["poss_exp"]) * 0.6
            h_poss = float(np.clip(round(50.0 + poss_diff + 2.5 + np.random.normal(0, 2.5), 1), 22.0, 78.0))
            a_poss = round(100.0 - h_poss, 1)

            h_shots = int(max(4, np.random.poisson(h_prof["shots_exp"] + rating_diff * 0.12 + 1.5)))
            a_shots = int(max(3, np.random.poisson(a_prof["shots_exp"] - rating_diff * 0.12 - 1.0)))
            h_sot = int(min(h_shots, max(1, np.random.poisson(h_prof["sot_exp"] + rating_diff * 0.06 + 0.8))))
            a_sot = int(min(a_shots, max(0, np.random.poisson(a_prof["sot_exp"] - rating_diff * 0.06 - 0.5))))

            matches.append({
                "match_id": match_id,
                "season": season,
                "home_team": h_team,
                "away_team": a_team,
                "home_squad_value_m": h_prof["squad_value_m"],
                "away_squad_value_m": a_prof["squad_value_m"],
                "squad_value_ratio": round(h_prof["squad_value_m"] / max(1.0, a_prof["squad_value_m"]), 2),
                "squad_value_diff": round(h_prof["squad_value_m"] - a_prof["squad_value_m"], 1),
                "home_goals": h_goals,
                "away_goals": a_goals,
                "result": res,
                "home_possession": h_poss,
                "away_possession": a_poss,
                "home_shots": h_shots,
                "away_shots": a_shots,
                "home_sot": h_sot,
                "away_sot": a_sot
            })
            match_id += 1

matches_df = pd.DataFrame(matches)

# Rolling history
team_history = {t: [] for t in teams}
enhanced = []

for _, row in matches_df.iterrows():
    h = row["home_team"]
    a = row["away_team"]

    h_hist = team_history[h][-5:]
    a_hist = team_history[a][-5:]

    def get_form(hist, default_pts):
        if not hist:
            return default_pts
        return float(sum(x["points"] for x in hist))

    def get_avg(hist, key, default):
        if not hist:
            return default
        return float(np.mean([x[key] for x in hist]))

    default_h_pts = 10.0 if TEAM_PROFILES[h]["rating"] > 85 else (7.0 if TEAM_PROFILES[h]["rating"] > 70 else 4.0)
    default_a_pts = 10.0 if TEAM_PROFILES[a]["rating"] > 85 else (7.0 if TEAM_PROFILES[a]["rating"] > 70 else 4.0)

    h_form = get_form(h_hist, default_h_pts)
    a_form = get_form(a_hist, default_a_pts)

    h_goals_scored = get_avg(h_hist, "goals_scored", TEAM_PROFILES[h]["att_exp"])
    a_goals_scored = get_avg(a_hist, "goals_scored", TEAM_PROFILES[a]["att_exp"])
    h_goals_conceded = get_avg(h_hist, "goals_conceded", TEAM_PROFILES[h]["def_exp"])
    a_goals_conceded = get_avg(a_hist, "goals_conceded", TEAM_PROFILES[a]["def_exp"])

    h_shots = get_avg(h_hist, "shots", TEAM_PROFILES[h]["shots_exp"])
    a_shots = get_avg(a_hist, "shots", TEAM_PROFILES[a]["shots_exp"])
    h_sot = get_avg(h_hist, "sot", TEAM_PROFILES[h]["sot_exp"])
    a_sot = get_avg(a_hist, "sot", TEAM_PROFILES[a]["sot_exp"])
    h_poss = get_avg(h_hist, "possession", TEAM_PROFILES[h]["poss_exp"])
    a_poss = get_avg(a_hist, "possession", TEAM_PROFILES[a]["poss_exp"])

    rdata = row.to_dict()
    rdata.update({
        "home_field_adv": 1.0,
        "home_form_points": round(h_form, 1),
        "away_form_points": round(a_form, 1),
        "form_diff": round(h_form - a_form, 1),
        "home_goals_scored_avg": round(h_goals_scored, 2),
        "away_goals_scored_avg": round(a_goals_scored, 2),
        "home_goals_conceded_avg": round(h_goals_conceded, 2),
        "away_goals_conceded_avg": round(a_goals_conceded, 2),
        "goal_diff_form": round((h_goals_scored - h_goals_conceded) - (a_goals_scored - a_goals_conceded), 2),
        "home_shots_avg": round(h_shots, 1),
        "away_shots_avg": round(a_shots, 1),
        "home_sot_avg": round(h_sot, 1),
        "away_sot_avg": round(a_sot, 1),
        "home_possession_avg": round(h_poss, 1),
        "away_possession_avg": round(a_poss, 1)
    })
    enhanced.append(rdata)

    h_pts = 3 if row["result"] == "H" else (1 if row["result"] == "D" else 0)
    a_pts = 3 if row["result"] == "A" else (1 if row["result"] == "D" else 0)

    team_history[h].append({
        "points": h_pts, "goals_scored": row["home_goals"], "goals_conceded": row["away_goals"],
        "shots": row["home_shots"], "sot": row["home_sot"], "possession": row["home_possession"]
    })
    team_history[a].append({
        "points": a_pts, "goals_scored": row["away_goals"], "goals_conceded": row["home_goals"],
        "shots": row["away_shots"], "sot": row["away_sot"], "possession": row["away_possession"]
    })

full_df = pd.DataFrame(enhanced)
full_df.to_csv(MATCHES_CSV, index=False)

print(f"Generated {len(full_df)} matches at: {MATCHES_CSV}")
print("Outcome breakdown:")
print(full_df["result"].value_counts(normalize=True))

# Export realistic current team profiles
latest_team_stats = {}
for team in teams:
    hist = team_history[team][-5:]
    pts = int(sum(x["points"] for x in hist))
    prof = TEAM_PROFILES[team]
    latest_team_stats[team] = {
        "team": team,
        "squad_value_m": prof["squad_value_m"],
        "rating": prof["rating"],
        "form_points_last_5": pts,
        "goals_scored_avg": round(float(np.mean([x["goals_scored"] for x in hist])), 2),
        "goals_conceded_avg": round(float(np.mean([x["goals_conceded"] for x in hist])), 2),
        "shots_avg": round(float(np.mean([x["shots"] for x in hist])), 1),
        "sot_avg": round(float(np.mean([x["sot"] for x in hist])), 1),
        "possession_avg": round(float(np.mean([x["possession"] for x in hist])), 1)
    }

with open(TEAM_STATS_JSON, "w", encoding="utf-8") as f:
    json.dump(latest_team_stats, f, indent=2)
print(f"Saved latest realistic team profiles to: {TEAM_STATS_JSON}")
