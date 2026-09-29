"""
Position-Specific Transfer Value Predictor using Linear Regression
Trains specialized Machine Learning models for distinct football roles:
- Strikers (CF, SS)
- Wingers & Wide Midfielders (RW, LW, LMF, RMF)
- Attacking Midfielders (AMF)
- Central Midfielders (CMF)
- Defensive Midfielders (DMF)
- Centre Backs (CB)
- Fullbacks & Wingbacks (RB, LB)
- Goalkeepers (GK)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class PositionGroupModel:
    """Represents a trained Linear Regression model for a specific position role."""
    def __init__(self, name: str, positions: list, features: list):
        self.name = name
        self.positions = positions
        self.features = features
        self.model = LinearRegression()
        self.train_metrics = {}
        self.test_metrics = {}
        self.sample_count = 0

    def fit(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        group_df = df[df["position"].isin(self.positions)].copy()
        self.sample_count = len(group_df)
        if self.sample_count < 5:
            return

        X = group_df[self.features].fillna(0)
        y = group_df["market_value_eur_m"].values

        if self.sample_count >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        self.model.fit(X_train, y_train)

        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        self.train_metrics = {
            "r2": float(r2_score(y_train, y_train_pred)),
            "mae": float(mean_absolute_error(y_train, y_train_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        }
        self.test_metrics = {
            "r2": float(r2_score(y_test, y_test_pred)),
            "mae": float(mean_absolute_error(y_test, y_test_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        }

    def predict(self, stats: dict) -> float:
        input_data = {feat: [stats.get(feat, 0)] for feat in self.features}
        X_in = pd.DataFrame(input_data)
        val = self.model.predict(X_in)[0]
        return max(0.5, float(val))

    def get_coefficients(self) -> list:
        coefs = []
        for feat, w in zip(self.features, self.model.coef_):
            coefs.append({"feature": feat, "weight": float(w)})
        return coefs


class TransferValuePredictor:
    """Multi-model predictor managing position-specialized regression pipelines."""

    METRIC_CONFIGS = {
        "CF_SS": {
            "name": "Strikers (CF, SS)",
            "positions": ["CF", "SS"],
            "features": ["age", "minutes_played", "goals", "assists"]
        },
        "WINGERS": {
            "name": "Wingers (RW, LW, LMF, RMF)",
            "positions": ["RW", "LW", "LMF", "RMF"],
            "features": ["age", "minutes_played", "goals", "assists", "chances_created", "dribbles_completed"]
        },
        "AMF": {
            "name": "Attacking Midfielders (AMF)",
            "positions": ["AMF"],
            "features": ["age", "minutes_played", "goals", "assists", "chances_created"]
        },
        "CMF": {
            "name": "Central Midfielders (CMF)",
            "positions": ["CMF"],
            "features": ["age", "minutes_played", "balls_recovered", "line_breaking_passes", "pass_accuracy"]
        },
        "DMF": {
            "name": "Defensive Midfielders (DMF)",
            "positions": ["DMF"],
            "features": ["age", "minutes_played", "balls_recovered", "duels_won", "aerial_duels_won", "line_breaking_passes"]
        },
        "CB": {
            "name": "Centre Backs (CB)",
            "positions": ["CB"],
            "features": ["age", "minutes_played", "duels_won", "successful_tackles", "aerial_duels_won"]
        },
        "FULLBACK": {
            "name": "Fullbacks (RB, LB)",
            "positions": ["RB", "LB"],
            "features": ["age", "minutes_played", "assists", "duels_won", "successful_tackles", "aerial_duels_won"]
        },
        "GK": {
            "name": "Goalkeepers (GK)",
            "positions": ["GK"],
            "features": ["age", "minutes_played", "saves", "penalties_saved", "pass_accuracy"]
        }
    }

    def __init__(self):
        self.models = {}
        for group_key, cfg in self.METRIC_CONFIGS.items():
            self.models[group_key] = PositionGroupModel(
                name=cfg["name"],
                positions=cfg["positions"],
                features=cfg["features"]
            )
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        """Trains all position-group models on their respective player cohorts."""
        for group_key, group_model in self.models.items():
            group_model.fit(df)
        self.is_trained = True

    def _get_group_for_position(self, position: str):
        pos_upper = position.upper().strip()
        for group_key, group_model in self.models.items():
            if pos_upper in group_model.positions:
                return group_model
        # Fallback to CF_SS
        return self.models["CF_SS"]

    def predict_player(self, position: str, **kwargs) -> float:
        """Predicts transfer valuation using the player's position-specific model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting.")
        group_model = self._get_group_for_position(position)
        return group_model.predict(kwargs)

    def evaluate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluates every player in the dataset using their position-specific model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first.")

        results = df.copy()
        preds = []
        for _, row in results.iterrows():
            pos = row["position"]
            group_model = self._get_group_for_position(pos)
            stats = row.to_dict()
            pred = group_model.predict(stats)
            preds.append(round(pred, 2))

        results["predicted_value_eur_m"] = preds
        results["valuation_diff_eur_m"] = np.round(
            results["market_value_eur_m"] - results["predicted_value_eur_m"], 2
        )
        results["market_status"] = np.where(
            results["valuation_diff_eur_m"] > 5.0,
            "Market Premium (Overvalued by stats)",
            np.where(
                results["valuation_diff_eur_m"] < -5.0,
                "Bargain (Undervalued by stats)",
                "Fairly Valued"
            )
        )
        return results

    def get_all_group_details(self) -> dict:
        """Returns parameters, weights, and metrics for all position models."""
        summary = {}
        for key, m in self.models.items():
            summary[key] = {
                "name": m.name,
                "positions": m.positions,
                "features": m.features,
                "sample_count": m.sample_count,
                "intercept": float(m.model.intercept_),
                "coefficients": m.get_coefficients(),
                "metrics": m.test_metrics
            }
        return summary
