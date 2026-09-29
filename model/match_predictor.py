"""
Premier League Match Outcome Predictor using Machine Learning
Integrates:
- Squad Market Value & Quality Ratios (from 441 Premier League players)
- Rolling 5-match Form & Goal Difference
- Shots, Shots on Target, and Possession
- XGBoost & Random Forest Classifiers
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from xgboost import XGBClassifier


class MatchOutcomePredictor:
    """Predicts Premier League match outcomes (H, D, A) with realistic squad quality weights."""

    FEATURES = [
        "squad_value_diff",
        "squad_value_ratio",
        "home_squad_value_m",
        "away_squad_value_m",
        "form_diff",
        "goal_diff_form",
        "home_form_points",
        "away_form_points",
        "home_goals_scored_avg",
        "away_goals_scored_avg",
        "home_goals_conceded_avg",
        "away_goals_conceded_avg",
        "home_shots_avg",
        "away_shots_avg",
        "home_sot_avg",
        "away_sot_avg",
        "home_possession_avg",
        "away_possession_avg"
    ]

    LABEL_MAP = {"H": 0, "D": 1, "A": 2}
    INV_LABEL_MAP = {0: "H", 1: "D", 2: "A"}
    CLASS_NAMES = ["Home Win (H)", "Draw (D)", "Away Win (A)"]

    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=140, max_depth=6, random_state=42)
        self.xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.07,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            eval_metric="mlogloss"
        )
        self.is_trained = False
        self.evaluation_results = {}
        self.team_stats = {}

    def load_team_stats(self, json_path: str = None):
        if json_path is None:
            json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "team_stats.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                self.team_stats = json.load(f)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        for col in self.FEATURES:
            if col not in data.columns:
                data[col] = 0.0
        return data[self.FEATURES]

    def train(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """Trains both Random Forest and XGBoost classifiers."""
        self.load_team_stats()

        X = self.prepare_features(df)
        y = df["result"].map(self.LABEL_MAP).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train Random Forest
        self.rf_model.fit(X_train, y_train)
        rf_test_pred = self.rf_model.predict(X_test)
        rf_test_proba = self.rf_model.predict_proba(X_test)

        # Train XGBoost
        self.xgb_model.fit(X_train, y_train)
        xgb_test_pred = self.xgb_model.predict(X_test)
        xgb_test_proba = self.xgb_model.predict_proba(X_test)

        self.is_trained = True

        self.evaluation_results = {
            "random_forest": {
                "train_acc": float(accuracy_score(y_train, self.rf_model.predict(X_train))),
                "test_acc": float(accuracy_score(y_test, rf_test_pred)),
                "log_loss": float(log_loss(y_test, rf_test_proba)),
                "confusion_matrix": confusion_matrix(y_test, rf_test_pred).tolist(),
                "feature_importances": [
                    {"feature": f, "importance": round(float(imp), 4)}
                    for f, imp in sorted(zip(self.FEATURES, self.rf_model.feature_importances_), key=lambda x: x[1], reverse=True)
                ]
            },
            "xgboost": {
                "train_acc": float(accuracy_score(y_train, self.xgb_model.predict(X_train))),
                "test_acc": float(accuracy_score(y_test, xgb_test_pred)),
                "log_loss": float(log_loss(y_test, xgb_test_proba)),
                "confusion_matrix": confusion_matrix(y_test, xgb_test_pred).tolist(),
                "feature_importances": [
                    {"feature": f, "importance": round(float(imp), 4)}
                    for f, imp in sorted(zip(self.FEATURES, self.xgb_model.feature_importances_), key=lambda x: x[1], reverse=True)
                ]
            },
            "test_sample_count": len(y_test)
        }

        return self.evaluation_results

    def predict_match(self, home_team: str, away_team: str, model_type: str = "xgboost") -> dict:
        """
        Predicts match probabilities and outcome for Home Team vs Away Team.
        Accurately respects squad quality so elite teams win away against bottom teams.
        """
        if not self.is_trained:
            raise ValueError("Predictor must be trained first.")

        if not self.team_stats:
            self.load_team_stats()

        h_stat = self.team_stats.get(home_team, {
            "squad_value_m": 450.0, "form_points_last_5": 7, "goals_scored_avg": 1.4,
            "goals_conceded_avg": 1.4, "shots_avg": 12.5, "sot_avg": 4.5, "possession_avg": 50.0
        })
        a_stat = self.team_stats.get(away_team, {
            "squad_value_m": 450.0, "form_points_last_5": 7, "goals_scored_avg": 1.4,
            "goals_conceded_avg": 1.4, "shots_avg": 12.5, "sot_avg": 4.5, "possession_avg": 50.0
        })

        h_val = float(h_stat.get("squad_value_m", 450.0))
        a_val = float(a_stat.get("squad_value_m", 450.0))
        squad_diff = h_val - a_val
        squad_ratio = h_val / max(1.0, a_val)

        form_diff = float(h_stat["form_points_last_5"] - a_stat["form_points_last_5"])
        h_net = h_stat["goals_scored_avg"] - h_stat["goals_conceded_avg"]
        a_net = a_stat["goals_scored_avg"] - a_stat["goals_conceded_avg"]
        goal_diff_form = float(h_net - a_net)

        features_dict = {
            "squad_value_diff": [squad_diff],
            "squad_value_ratio": [squad_ratio],
            "home_squad_value_m": [h_val],
            "away_squad_value_m": [a_val],
            "form_diff": [form_diff],
            "goal_diff_form": [goal_diff_form],
            "home_form_points": [float(h_stat["form_points_last_5"])],
            "away_form_points": [float(a_stat["form_points_last_5"])],
            "home_goals_scored_avg": [float(h_stat["goals_scored_avg"])],
            "away_goals_scored_avg": [float(a_stat["goals_scored_avg"])],
            "home_goals_conceded_avg": [float(h_stat["goals_conceded_avg"])],
            "away_goals_conceded_avg": [float(a_stat["goals_conceded_avg"])],
            "home_shots_avg": [float(h_stat["shots_avg"])],
            "away_shots_avg": [float(a_stat["shots_avg"])],
            "home_sot_avg": [float(h_stat["sot_avg"])],
            "away_sot_avg": [float(a_stat["sot_avg"])],
            "home_possession_avg": [float(h_stat["possession_avg"])],
            "away_possession_avg": [float(a_stat["possession_avg"])]
        }

        X_in = pd.DataFrame(features_dict)[self.FEATURES]

        model = self.xgb_model if model_type.lower() == "xgboost" else self.rf_model
        proba = model.predict_proba(X_in)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self.INV_LABEL_MAP[pred_idx]

        return {
            "home_team": home_team,
            "away_team": away_team,
            "model_used": "XGBoost" if model_type.lower() == "xgboost" else "Random Forest",
            "predicted_result": pred_label,
            "predicted_outcome": self.CLASS_NAMES[pred_idx],
            "probabilities": {
                "home_win": round(float(proba[0]) * 100, 1),
                "draw": round(float(proba[1]) * 100, 1),
                "away_win": round(float(proba[2]) * 100, 1)
            },
            "stats_comparison": {
                "home": h_stat,
                "away": a_stat
            }
        }
