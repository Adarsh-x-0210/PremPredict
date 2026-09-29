# PremPredict ⚽🤖

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-15B9A8?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

**An intelligent, dual-engine Premier League analytics platform powering role-specific transfer valuation and calibrated match outcome forecasting.**

[Features](#-key-features) • [Quickstart](#-quickstart) • [Machine Learning Architecture](#-machine-learning-architecture) • [Project Structure](#-project-structure) • [Web Application](#-web-application) • [License](#-license)

</div>

---

## 🌟 Key Features

PremPredict combines two distinct machine learning systems under a single unified, responsive interface:

### 1. 🎯 Position-Specific Player Transfer Value Predictor
- **441 Premier League Players**: Comprehensive dataset across all 20 Premier League clubs.
- **Granular Tactical Roles**: Custom sub-models tailored for **10 specific tactical roles**:
  - **CF / SS**: Goals, Assists, Minutes, Age.
  - **RW / LW / LMF / RMF**: Chances Created, Dribbles Completed, Goals, Assists.
  - **AMF**: Chances Created, Goals, Assists, Minutes, Age.
  - **CMF**: Line-Breaking Passes, Balls Recovered, Pass Accuracy, Goals, Assists.
  - **DMF**: Balls Recovered, Duels Won, Aerial Duels Won, Line-Breaking Passes.
  - **CB**: Duels Won, Successful Tackles, Aerial Duels Won, Minutes, Age.
  - **LB / RB**: Assists, Duels Won, Successful Tackles, Aerial Duels Won.
  - **GK**: Saves, Save % / Pass Accuracy, Penalties Saved, Minutes, Age.
- **Real-Time Dynamic Interactive Sliders**: Test hypothetical player profiles or adjust existing stars to see valuation swings in € Millions.
- **Value Efficiency Analytics**: Spot overvalued stars and undervalued bargains relative to market value.

### 2. 🔮 Match Outcome Predictor (XGBoost + Random Forest)
- **760 Historical Premier League Fixtures**: Rigorously compiled across 2 complete Premier League seasons.
- **Engineered Contextual Features**:
  - **Squad Market Valuation Difference & Ratio**: Captures financial gravity and elite player depth (eliminating naive home-advantage bias so top sides realistically win away against relegation contenders).
  - **Rolling Form**: Exponentially weighted points from last 5 fixtures.
  - **Attacking Threat**: Rolling shots on target per match.
  - **Game Control**: Rolling average possession percentage.
  - **Calibrated Home Ground Advantage**: Evaluated relative to opponent strength.
- **Three-Way Match Probabilities**: Calibrated Win %, Draw %, and Away Win % forecasts with head-to-head stat comparisons.

---

## 📁 Project Structure

```
PremPredict/
├── app.py                      # Lightweight Python HTTP server for the web interface
├── index.html                  # Full-featured single-page web app (Obsidian Black & Electric Blue)
├── main.py                     # Player valuation ML pipeline (training, evaluation, metrics)
├── main_matches.py             # Match outcome ML pipeline (XGBoost/Random Forest training)
├── interactive_cli.py          # Interactive terminal CLI for player valuation
├── visualize.py                # Visual diagnostics generator (Matplotlib)
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── pl_players_dataset.csv  # 441 Premier League players with position-specific stats
│   ├── players.json            # Structured player records for web application
│   ├── pl_matches_dataset.csv  # 760 historical match rows with rolling metrics & squad valuations
│   ├── team_stats.json         # Team profiles, squad market values, and rolling averages
│   └── match_predictions.json  # Precomputed fixture probabilities for all 380 matchups
│
├── model/
│   ├── __init__.py
│   ├── predictor.py            # Role-specific Linear Regression engine
│   └── match_predictor.py      # Match Outcome Classifier (XGBoost / RandomForest)
│
└── plots/                      # Diagnostic Matplotlib charts
    ├── actual_vs_predicted.png
    ├── feature_importance.png
    └── goals_vs_value.png
```

---

## 🚀 Quickstart

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/your-username/PremPredict.git
cd PremPredict
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Web Interface

```bash
python app.py
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser to explore the interactive dashboard.

### 3. Run Pipelines via Terminal

* **Train Player Valuation Models & View Evaluation**:
  ```bash
  python main.py
  ```
* **Train Match Outcome Models (XGBoost & Random Forest)**:
  ```bash
  python main_matches.py
  ```
* **Interactive Terminal CLI**:
  ```bash
  python interactive_cli.py
  ```

---

## 🧠 Machine Learning Architecture

### 1. Player Valuation Regression Engine
Because transfer values are continuous monetary values (€ Millions), we employ **Linear Regression** with role-tailored feature sets:

$$\hat{y}_{\text{value}} = \beta_0 + \sum_{i=1}^{k} \beta_i X_i$$

Where:
- $\beta_0$ represents the baseline market entry valuation.
- $\beta_i$ represents the learned marginal valuation coefficient per unit metric (e.g., value per goal for forwards, value per tackle and duel won for center-backs, value per penalty saved for goalkeepers).
- Age carries a distinct non-linear depreciation factor for older veterans vs youth prospects.

### 2. Match Outcome Classification Engine
Match forecasting is formulated as a 3-class probability classification task:
$$\text{Outcome} \in \{\text{Home Win}, \text{Draw}, \text{Away Win}\}$$

We benchmark **XGBoost (Extreme Gradient Boosting)** against **Random Forest Ensembles**. The models leverage:
- **Squad Market Value Ratio & Difference**: Prevents naive home-bias artifacts, ensuring elite away clubs (e.g. Manchester City at Kenilworth Road or Portman Road) are realistically favored based on sheer squad quality.
- **Rolling Form Factor**: Captures recent momentum and squad confidence over the preceding 5 matches.
- **Shot Dominance & Possession Indices**: Quantifies chance creation and pitch territory control.

---

## 🎨 Web Application Interface

The front-end is crafted in modern **Tailwind CSS** with an **Obsidian Black (`#0B0F17`) & Electric Blue (`#3B82F6`)** dark theme:
- **Unified Navigation Bar**: Switch seamlessly between Player Valuation and Match Prediction modes.
- **Filterable Rosters**: Filter players by club, position role, search query, or sorting (Highest Value, Most Undervalued, Most Overvalued).
- **Tactical Pitch Visualizer**: Dynamic stat cards featuring tactical radar metrics and real-time custom value recalculation.
- **Fixture Simulator**: Select any home and away pairing to preview forecasted win probabilities and team comparative radar stats.

---

## 📊 Evaluation & Metrics

| Pipeline | Model | Key Metrics |
| :--- | :--- | :--- |
| **Player Valuation** | Role-Specific OLS Regression | $R^2 \approx 0.76$, MAE $\approx €7.2\text{M}$ |
| **Match Outcome** | XGBoost & Random Forest | Multi-Class Log-Loss, Realistic Probability Calibration |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ⚽ for football analytics and machine learning enthusiasts.</sub>
</div>
