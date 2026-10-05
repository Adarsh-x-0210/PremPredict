# PremPredict ⚽🤖

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-15B9A8?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

**An intelligent, multi-engine Premier League analytics platform powering role-specific transfer valuation, calibrated match outcome forecasting, and AI player scouting style similarity.**

[Features](#-key-features) • [Quickstart](#-quickstart) • [Scouting Engine](#-player-scouting--similarity-engine) • [Machine Learning Architecture](#-machine-learning-architecture) • [Project Structure](#-project-structure) • [Web Application](#-web-application) • [License](#-license)

</div>

---

## 🌟 Key Features

PremPredict combines three interconnected machine learning systems:

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
  - **Squad Market Valuation Difference & Ratio**: Captures financial gravity and elite player depth (eliminating naive home-advantage bias so top sides realistically win away against bottom clubs).
  - **Rolling Form**: Exponentially weighted points from last 5 fixtures.
  - **Attacking Threat & Game Control**: Rolling shots on target and average possession percentage.
  - **Calibrated Home Ground Advantage**: Evaluated relative to opponent strength.
- **Three-Way Match Probabilities**: Calibrated Win %, Draw %, and Away Win % forecasts with head-to-head stat comparisons.

### 3. 🧭 AI Player Scouting & Style Similarity Engine (Streamlit + Scikit-Learn)
- **Multi-Vector Player Matching**: Uses **Cosine Similarity** across 12 standardized tactical metrics (goals, assists, chances created, dribbles, tackles, duels, recoveries, passing, saves).
- **Find Statistical Twins**: Select any Premier League star (e.g., *Bukayo Saka*) and discover their closest stylistic counterparts (e.g., *Matheus Cunha, Phil Foden, Mohamed Salah*).
- **K-Means Tactical Archetype Clustering**: Unsupervised clustering ($k=6$) automatically segments players into playing style archetypes (*Creative Winger / Attacking Spark*, *Goal Poacher / Lethal Finisher*, *Playmaker / Deep Midfield Engine*, *Defensive Stopper / Ball Winner*, *Goalkeeper / Shot Stopper*, *Box-to-Box / Balanced Operator*).
- **2D Tactical Style Space (PCA)**: Projects multidimensional player vectors onto orthogonal 2D principal components (Attacking Threat vs Defensive Solidity).
- **Interactive Radar Comparison Charts**: Built with Matplotlib for head-to-head tactical profile comparisons.

---

## 📁 Project Structure

```
PremPredict/
├── app.py                      # Lightweight Python HTTP server for unified web interface
├── streamlit_app.py            # Interactive Streamlit scouting & similarity dashboard
├── index.html                  # Full-featured 3-in-1 web app (Obsidian Black & Electric Blue)
├── generate_web_ui.py          # Unified UI compiler and artifact generator
├── main.py                     # Player valuation ML pipeline (training, evaluation, metrics)
├── main_matches.py             # Match outcome ML pipeline (XGBoost/Random Forest training)
├── interactive_cli.py          # Interactive terminal CLI for player valuation
├── visualize.py                # Visual diagnostics generator (Matplotlib)
├── requirements.txt            # Python dependencies (includes Streamlit & XGBoost)
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── pl_players_dataset.csv  # 441 Premier League players with position-specific stats
│   ├── players.json            # Structured player records for web application
│   ├── pl_matches_dataset.csv  # 760 historical match rows with rolling metrics & squad valuations
│   ├── team_stats.json         # Team profiles, squad market values, and rolling averages
│   ├── match_predictions.json  # Precomputed fixture probabilities for all 380 matchups
│   ├── scouting_similarity.json# Precomputed similarity matches for all 441 players
│   └── player_styles.json      # Tactical style cluster mapping for all players
│
├── model/
│   ├── __init__.py
│   ├── predictor.py            # Role-specific Linear Regression engine
│   ├── match_predictor.py      # Match Outcome Classifier (XGBoost / RandomForest)
│   └── scouting_engine.py      # Cosine similarity, K-Means clustering & PCA engine
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
git clone https://github.com/Adarsh-x-0210/PremPredict.git
cd PremPredict
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Streamlit Scouting Dashboard

```bash
streamlit run streamlit_app.py
```
Open **[http://localhost:8501](http://localhost:8501)** to explore interactive statistical twin scouting, K-Means style clusters, and 2D PCA tactical maps.

### 3. Launch the Unified Web App (Valuations + Matches + Scouting)

```bash
python app.py
```
Open **[http://localhost:8080](http://localhost:8080)** to use the unified single-page application with Top Navigation switching.

### 4. Run Pipelines via Terminal

* **Train Valuation Regression Models**:
  ```bash
  python main.py
  ```
* **Train Match Outcome Models (XGBoost & Random Forest)**:
  ```bash
  python main_matches.py
  ```
* **Test Scouting Similarity Engine in Terminal**:
  ```bash
  python model/scouting_engine.py
  ```

---

## 🧭 Player Scouting & Similarity Engine

### How it finds player matches:
1. **Feature Standardisation**: Player metrics are standardized with `StandardScaler` so that high-magnitude metrics (minutes, passes) do not drown out low-magnitude metrics (goals, assists, penalties saved).
2. **Cosine Vector Similarity**:
   $$\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
   Evaluates angular alignment between two players' multi-metric profiles, returning an intuitive percentage match (0% to 100%).
3. **K-Means Clustering**:
   Unsupervised clustering partitions the player population into tactical style clusters by minimizing intra-cluster variance (inertia):
   $$\arg\min_{S} \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2$$

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
