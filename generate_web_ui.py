"""
Unified Web UI Generator for PremPredict
Integrates:
1. Player Valuation Predictor (Role-specific Linear Regression for 441 players)
2. Match Outcome Predictor (XGBoost & Random Forest Win/Draw/Loss classifier)
3. Player Scouting & Similarity Recommender (Cosine Similarity & K-Means Style Clusters)
With a unified Top Navigation Bar in obsidian black & electric blue theme.
"""

import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from data.fetch_data import load_dataset
from model.predictor import TransferValuePredictor

PLAYERS_JSON_PATH = os.path.join(PROJECT_DIR, "data", "players.json")
MATCHES_JSON_PATH = os.path.join(PROJECT_DIR, "data", "match_predictions.json")
TEAM_STATS_JSON_PATH = os.path.join(PROJECT_DIR, "data", "team_stats.json")
SCOUTING_SIM_PATH = os.path.join(PROJECT_DIR, "data", "scouting_similarity.json")
PLAYER_STYLES_PATH = os.path.join(PROJECT_DIR, "data", "player_styles.json")

WEB_OUTPUT_PATH = os.path.join(PROJECT_DIR, "index.html")
WIDGET_OUTPUT_PATH = r"C:\Users\Admin\.gemini\antigravity\brain\8d4c148a-7a62-48eb-8644-4a4628d38d4b\transfer_predictor_widget.html"

with open(PLAYERS_JSON_PATH, "r", encoding="utf-8") as f:
    players_data = json.load(f)

with open(MATCHES_JSON_PATH, "r", encoding="utf-8") as f:
    match_predictions = json.load(f)

with open(TEAM_STATS_JSON_PATH, "r", encoding="utf-8") as f:
    team_stats = json.load(f)

with open(SCOUTING_SIM_PATH, "r", encoding="utf-8") as f:
    scouting_similarity = json.load(f)

with open(PLAYER_STYLES_PATH, "r", encoding="utf-8") as f:
    player_styles = json.load(f)

teams = sorted(list(team_stats.keys()))
all_player_names = sorted(list(scouting_similarity.keys()))

# Train player predictor to extract model weights
df = load_dataset()
predictor = TransferValuePredictor()
predictor.train(df)
player_models_summary = predictor.get_all_group_details()

html_content = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PremPredict | Player Valuation, Match Outcomes & Scouting AI</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    :root {{
      --bg-main: #0B0F17;
      --card-bg: #111827;
      --card-sub: #161F30;
      --border-color: #1F293D;
      --primary-blue: #2563EB;
      --electric-blue: #3B82F6;
    }}
    body {{
      background-color: #0B0F17 !important;
      color: #F8FAFC !important;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    input[type=range] {{
      accent-color: #3B82F6;
    }}
    ::-webkit-scrollbar {{
      width: 6px;
    }}
    ::-webkit-scrollbar-track {{
      background: #0B0F17;
    }}
    ::-webkit-scrollbar-thumb {{
      background: #1F293D;
      border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
      background: #3B82F6;
    }}
  </style>
</head>
<body class="min-h-screen p-3 sm:p-6 antialiased" style="background-color: #0B0F17; color: #F8FAFC;">
  <div class="max-w-4xl mx-auto space-y-6">
    
    <!-- TOP NAVIGATION BAR -->
    <header class="rounded-2xl p-4 sm:p-5 border flex flex-col sm:flex-row items-center justify-between gap-4 shadow-xl" style="background-color: #111827; border-color: #1F293D;">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl flex items-center justify-center text-xl shadow" style="background: linear-gradient(135deg, #2563EB, #1D4ED8);">
          ⚽
        </div>
        <div>
          <h1 class="text-base sm:text-lg font-black tracking-tight text-white flex items-center gap-2">
            PremPredict <span class="text-xs px-2 py-0.5 rounded font-bold uppercase tracking-wider text-blue-400 border border-blue-500/30" style="background-color: rgba(37, 99, 235, 0.1);">AI Hub</span>
          </h1>
          <p class="text-xs text-slate-400">Valuations • Match Forecasting • AI Player Scouting</p>
        </div>
      </div>

      <!-- Top Bar Functionality Tabs -->
      <nav class="flex p-1 rounded-xl border w-full sm:w-auto justify-center gap-1" style="background-color: #0B0F17; border-color: #1F293D;">
        <button id="nav-player-btn" onclick="switchNav('player')" class="flex-1 sm:flex-none px-3.5 py-2 text-xs sm:text-sm font-bold rounded-lg text-white transition-all shadow" style="background: linear-gradient(135deg, #2563EB, #1D4ED8);">
          ⚽ Valuation
        </button>
        <button id="nav-match-btn" onclick="switchNav('match')" class="flex-1 sm:flex-none px-3.5 py-2 text-xs sm:text-sm font-bold rounded-lg text-slate-400 hover:text-white transition-all">
          🏆 Matches
        </button>
        <button id="nav-scout-btn" onclick="switchNav('scout')" class="flex-1 sm:flex-none px-3.5 py-2 text-xs sm:text-sm font-bold rounded-lg text-slate-400 hover:text-white transition-all">
          🧭 Scouting
        </button>
      </nav>
    </header>

    <!-- ======================================================== -->
    <!-- SECTION 1: PLAYER VALUATION PREDICTOR -->
    <!-- ======================================================== -->
    <main id="section-player" class="rounded-2xl p-5 sm:p-7 shadow-2xl border space-y-6" style="background-color: #111827; border-color: #1F293D;">
      
      <!-- Sub-Header for Player Predictor -->
      <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b pb-4 gap-3" style="border-color: #1F293D;">
        <div>
          <h2 class="text-lg sm:text-xl font-bold text-white flex items-center gap-2">
            Role-Specific Transfer Value Predictor
            <span class="text-[10px] font-bold px-2 py-0.5 rounded-full bg-blue-600/20 text-blue-400 border border-blue-500/30">
              441 PLAYERS • 20 CLUBS
            </span>
          </h2>
          <p class="text-xs text-slate-400 mt-0.5">Linear Regression models with role-tailored stats (CBs by duels, GKs by saves, Wingers by dribbles)</p>
        </div>

        <div class="flex p-1 rounded-xl border self-start sm:self-auto" style="background-color: #0B0F17; border-color: #1F293D;">
          <button id="tab-predict-btn" onclick="switchPlayerTab('predict')" class="px-3.5 py-1.5 text-xs font-semibold rounded-lg text-white" style="background: linear-gradient(135deg, #2563EB, #1D4ED8);">
            Custom Predictor
          </button>
          <button id="tab-lookup-btn" onclick="switchPlayerTab('lookup')" class="px-3.5 py-1.5 text-xs font-semibold rounded-lg text-slate-400 hover:text-white">
            Player Browser ({len(players_data)})
          </button>
        </div>
      </div>

      <!-- Player Tab 1: Custom Role Predictor -->
      <div id="tab-player-predict" class="space-y-6">
        <!-- Position Pill Selector -->
        <div>
          <div class="flex justify-between items-center mb-2">
            <label class="text-xs font-bold uppercase tracking-wider text-slate-400">Select Player Role</label>
            <span id="pos-group-desc" class="text-xs font-medium text-blue-400">CF (Strikers)</span>
          </div>
          <div class="flex flex-wrap gap-2">
            <button type="button" onclick="selectRole('CF')" class="role-btn px-3 py-1.5 text-xs font-bold rounded-lg border text-white" style="background-color: #2563EB; border-color: #3B82F6;" data-role="CF">CF</button>
            <button type="button" onclick="selectRole('SS')" class="role-btn px-3 py-1.5 text-xs font-bold rounded-lg border text-slate-400 hover:text-white" style="background-color: #0B0F17; border-color: #1F293D;" data-role="SS">SS</button>
            <button type="button" onclick="selectRole('RW')" class="role-btn px-3 py-1.5 text-xs font-bold rounded-lg border text-slate-400 hover:text-white" style="background-color: #0B0F17; border-color: #1F293D;" data-role="RW">RW</button>
            <button type="button" onclick="selectRole('LW')" class="role-btn px-3 py-1.5 text-xs font-bold rounded-lg border text-slate-400 hover:text-white" style="background-color: #0B0F17; border-color: #1F293D;" data-role="LW">LW</button>
            <button type="button" onclick="selectRole('AMF')" class="role-btn px-3 py-1.5 text-xs font-bold rounded-lg border text-slate-400 hover:text-white" style="background-color: #0B0F17; border-color: #1F293D;" data-role="AMF">AMF</button>
            <button type="button" onclick="selectRole('CMF')" class="role-btn px-3 py-1.5 text-xs font-bold rounded-lg border text-slate-400 hover:text-white" style="background-color: #0B0F17; border-color: #1F293D;" data-role="CMF">CMF</button>
            <button type="button" onclick="selectRole('DMF')" class="role-btn px-3 py-1.5 text-xs font-bold rounded-lg border text-slate-400 hover:text-white" style="background-color: #0B0F17; border-color: #1F293D;" data-role="DMF">DMF</button>
            <button type="button" onclick="selectRole('CB')" class="role-btn px-3 py-1.5 text-xs font-bold rounded-lg border text-slate-400 hover:text-white" style="background-color: #0B0F17; border-color: #1F293D;" data-role="CB">CB</button>
            <button type="button" onclick="selectRole('RB')" class="role-btn px-3 py-1.5 text-xs font-bold rounded-lg border text-slate-400 hover:text-white" style="background-color: #0B0F17; border-color: #1F293D;" data-role="RB">RB</button>
            <button type="button" onclick="selectRole('LB')" class="role-btn px-3 py-1.5 text-xs font-bold rounded-lg border text-slate-400 hover:text-white" style="background-color: #0B0F17; border-color: #1F293D;" data-role="LB">LB</button>
            <button type="button" onclick="selectRole('GK')" class="role-btn px-3 py-1.5 text-xs font-bold rounded-lg border text-slate-400 hover:text-white" style="background-color: #0B0F17; border-color: #1F293D;" data-role="GK">GK</button>
          </div>
        </div>

        <!-- Universal Age & Minutes Sliders -->
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div class="p-4 rounded-xl border" style="background-color: #0B0F17; border-color: #1F293D;">
            <div class="flex justify-between items-center text-xs sm:text-sm mb-2">
              <span class="font-medium text-slate-400">Player Age</span>
              <span id="age-val" class="font-black text-white px-2 py-0.5 rounded" style="background-color: #161F30;">24 years</span>
            </div>
            <input id="age-slider" type="range" min="17" max="38" value="24" class="w-full cursor-pointer" oninput="calculatePlayer()">
          </div>

          <div class="p-4 rounded-xl border" style="background-color: #0B0F17; border-color: #1F293D;">
            <div class="flex justify-between items-center text-xs sm:text-sm mb-2">
              <span class="font-medium text-slate-400">Minutes Played</span>
              <span id="mins-val" class="font-black text-white px-2 py-0.5 rounded" style="background-color: #161F30;">2,500 mins</span>
            </div>
            <input id="mins-slider" type="range" min="100" max="3420" step="50" value="2500" class="w-full cursor-pointer" oninput="calculatePlayer()">
          </div>
        </div>

        <!-- Dynamic Role-Specific Sliders Container -->
        <div id="dynamic-sliders" class="grid grid-cols-1 sm:grid-cols-2 gap-4"></div>

        <!-- Valuation Card -->
        <div class="rounded-2xl p-6 border text-center relative overflow-hidden" style="background: linear-gradient(135deg, rgba(37,99,235,0.15), rgba(15,23,42,0.95), rgba(6,182,212,0.15)); border-color: #2563EB;">
          <span class="text-xs uppercase font-extrabold tracking-widest text-blue-400">Estimated Market Valuation</span>
          <div class="text-4xl sm:text-6xl font-black text-white my-2 tracking-tight">
            €<span id="predicted-val">85.8</span>M
          </div>
          <div id="breakdown-container" class="flex flex-wrap items-center justify-center gap-2 mt-3 text-xs font-semibold"></div>
        </div>
      </div>

      <!-- Player Tab 2: All Players Browser -->
      <div id="tab-player-lookup" class="space-y-4 hidden">
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div class="sm:col-span-2">
            <input 
              id="player-search" 
              type="text" 
              placeholder="Search player name (e.g. Saka, Haaland, Saliba, Odegaard)..." 
              class="w-full px-4 py-2.5 rounded-xl border text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder-slate-500"
              style="background-color: #0B0F17; border-color: #1F293D;"
              oninput="filterPlayers()"
            >
          </div>
          <div>
            <select 
              id="team-filter" 
              class="w-full px-3.5 py-2.5 rounded-xl border text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              style="background-color: #0B0F17; border-color: #1F293D;"
              onchange="filterPlayers()"
            >
              <option value="">All 20 Clubs ({len(players_data)} players)</option>
              {"".join(f'<option value="{t}">{t}</option>' for t in teams)}
            </select>
          </div>
        </div>

        <div class="flex flex-wrap items-center justify-between text-xs text-slate-400 gap-2 border-b pb-3" style="border-color: #1F293D;">
          <div class="flex flex-wrap gap-1.5">
            <button onclick="setListRoleFilter('')" class="list-role-btn px-2.5 py-1 rounded-lg font-bold bg-blue-600 text-white" data-lrole="">All</button>
            <button onclick="setListRoleFilter('CF')" class="list-role-btn px-2.5 py-1 rounded-lg font-medium text-slate-400" style="background-color: #0B0F17;" data-lrole="CF">CF</button>
            <button onclick="setListRoleFilter('RW')" class="list-role-btn px-2.5 py-1 rounded-lg font-medium text-slate-400" style="background-color: #0B0F17;" data-lrole="RW">RW</button>
            <button onclick="setListRoleFilter('LW')" class="list-role-btn px-2.5 py-1 rounded-lg font-medium text-slate-400" style="background-color: #0B0F17;" data-lrole="LW">LW</button>
            <button onclick="setListRoleFilter('AMF')" class="list-role-btn px-2.5 py-1 rounded-lg font-medium text-slate-400" style="background-color: #0B0F17;" data-lrole="AMF">AMF</button>
            <button onclick="setListRoleFilter('CMF')" class="list-role-btn px-2.5 py-1 rounded-lg font-medium text-slate-400" style="background-color: #0B0F17;" data-lrole="CMF">CMF</button>
            <button onclick="setListRoleFilter('DMF')" class="list-role-btn px-2.5 py-1 rounded-lg font-medium text-slate-400" style="background-color: #0B0F17;" data-lrole="DMF">DMF</button>
            <button onclick="setListRoleFilter('CB')" class="list-role-btn px-2.5 py-1 rounded-lg font-medium text-slate-400" style="background-color: #0B0F17;" data-lrole="CB">CB</button>
            <button onclick="setListRoleFilter('RB')" class="list-role-btn px-2.5 py-1 rounded-lg font-medium text-slate-400" style="background-color: #0B0F17;" data-lrole="RB">RB</button>
            <button onclick="setListRoleFilter('LB')" class="list-role-btn px-2.5 py-1 rounded-lg font-medium text-slate-400" style="background-color: #0B0F17;" data-lrole="LB">LB</button>
            <button onclick="setListRoleFilter('GK')" class="list-role-btn px-2.5 py-1 rounded-lg font-medium text-slate-400" style="background-color: #0B0F17;" data-lrole="GK">GK</button>
          </div>
          <div id="player-count" class="font-medium text-blue-400">
            Showing {len(players_data)} players
          </div>
        </div>

        <div id="players-list" class="space-y-2.5 max-h-[480px] overflow-y-auto pr-1.5"></div>
      </div>
    </main>


    <!-- ======================================================== -->
    <!-- SECTION 2: MATCH OUTCOME PREDICTOR -->
    <!-- ======================================================== -->
    <main id="section-match" class="rounded-2xl p-5 sm:p-7 shadow-2xl border space-y-6 hidden" style="background-color: #111827; border-color: #1F293D;">
      
      <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b pb-4 gap-3" style="border-color: #1F293D;">
        <div>
          <h2 class="text-lg sm:text-xl font-bold text-white flex items-center gap-2">
            Match Outcome Predictor
            <span class="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-600/20 text-emerald-400 border border-emerald-500/30">
              760 FIXTURES • 2 SEASONS
            </span>
          </h2>
          <p class="text-xs text-slate-400 mt-0.5">XGBoost & Random Forest multi-class classifiers predicting Win / Draw / Loss probabilities</p>
        </div>

        <div class="flex p-1 rounded-xl border self-start sm:self-auto" style="background-color: #0B0F17; border-color: #1F293D;">
          <button id="model-xgb-btn" onclick="setMatchModel('xgboost')" class="px-3.5 py-1.5 text-xs font-semibold rounded-lg text-white" style="background: linear-gradient(135deg, #2563EB, #1D4ED8);">
            XGBoost
          </button>
          <button id="model-rf-btn" onclick="setMatchModel('random_forest')" class="px-3.5 py-1.5 text-xs font-semibold rounded-lg text-slate-400 hover:text-white">
            Random Forest
          </button>
        </div>
      </div>

      <!-- Match Selectors Card -->
      <div class="grid grid-cols-1 sm:grid-cols-5 gap-3 items-center">
        <div class="sm:col-span-2">
          <label class="block text-xs font-bold uppercase tracking-wider text-slate-400 mb-1.5">Home Team</label>
          <select id="home-team-select" onchange="calculateMatch()" class="w-full px-3.5 py-2.5 rounded-xl border text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 font-semibold" style="background-color: #0B0F17; border-color: #1F293D;">
            {"".join(f'<option value="{t}" {"selected" if t == "Arsenal" else ""}>{t}</option>' for t in teams)}
          </select>
        </div>

        <div class="text-center font-black text-slate-500 text-sm sm:pt-6">VS</div>

        <div class="sm:col-span-2">
          <label class="block text-xs font-bold uppercase tracking-wider text-slate-400 mb-1.5">Away Team</label>
          <select id="away-team-select" onchange="calculateMatch()" class="w-full px-3.5 py-2.5 rounded-xl border text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 font-semibold" style="background-color: #0B0F17; border-color: #1F293D;">
            {"".join(f'<option value="{t}" {"selected" if t == "Chelsea" else ""}>{t}</option>' for t in teams)}
          </select>
        </div>
      </div>

      <!-- Quick Derby Shortcuts -->
      <div class="flex flex-wrap items-center gap-2 text-xs">
        <span class="text-slate-500 font-medium">Quick Fixtures:</span>
        <button onclick="setFixture('Arsenal', 'Chelsea')" class="px-2.5 py-1 rounded-lg border text-slate-300 hover:text-white hover:border-blue-500" style="background-color: #0B0F17; border-color: #1F293D;">Arsenal vs Chelsea</button>
        <button onclick="setFixture('Manchester City', 'Liverpool')" class="px-2.5 py-1 rounded-lg border text-slate-300 hover:text-white hover:border-blue-500" style="background-color: #0B0F17; border-color: #1F293D;">Man City vs Liverpool</button>
        <button onclick="setFixture('Tottenham', 'Arsenal')" class="px-2.5 py-1 rounded-lg border text-slate-300 hover:text-white hover:border-blue-500" style="background-color: #0B0F17; border-color: #1F293D;">N. London Derby</button>
        <button onclick="setFixture('Liverpool', 'Manchester United')" class="px-2.5 py-1 rounded-lg border text-slate-300 hover:text-white hover:border-blue-500" style="background-color: #0B0F17; border-color: #1F293D;">Liverpool vs Man Utd</button>
      </div>

      <!-- Probabilities Display Card -->
      <div class="rounded-2xl p-5 sm:p-6 border text-center space-y-4" style="background: linear-gradient(135deg, rgba(17,24,39,0.95), rgba(30,41,59,0.95)); border-color: #1F293D;">
        <div class="flex items-center justify-between text-xs font-bold text-slate-400">
          <span id="prob-home-label">Arsenal Win</span>
          <span>Draw</span>
          <span id="prob-away-label">Chelsea Win</span>
        </div>

        <!-- 3-Segment Stacked Progress Bar -->
        <div class="w-full h-4 rounded-full flex overflow-hidden border" style="background-color: #0B0F17; border-color: #1F293D;">
          <div id="prob-home-bar" class="h-full transition-all duration-300" style="width: 55%; background: linear-gradient(90deg, #2563EB, #3B82F6);"></div>
          <div id="prob-draw-bar" class="h-full transition-all duration-300" style="width: 25%; background: #64748B;"></div>
          <div id="prob-away-bar" class="h-full transition-all duration-300" style="width: 20%; background: linear-gradient(90deg, #F59E0B, #EF4444);"></div>
        </div>

        <div class="grid grid-cols-3 gap-2 text-center pt-1">
          <div class="p-3 rounded-xl border" style="background-color: #0B0F17; border-color: #1F293D;">
            <div class="text-[10px] uppercase font-bold text-blue-400">Home Win</div>
            <div id="prob-home-val" class="text-xl sm:text-2xl font-black text-white mt-0.5">55.1%</div>
          </div>
          <div class="p-3 rounded-xl border" style="background-color: #0B0F17; border-color: #1F293D;">
            <div class="text-[10px] uppercase font-bold text-slate-400">Draw</div>
            <div id="prob-draw-val" class="text-xl sm:text-2xl font-black text-slate-300 mt-0.5">17.5%</div>
          </div>
          <div class="p-3 rounded-xl border" style="background-color: #0B0F17; border-color: #1F293D;">
            <div class="text-[10px] uppercase font-bold text-amber-400">Away Win</div>
            <div id="prob-away-val" class="text-xl sm:text-2xl font-black text-white mt-0.5">27.3%</div>
          </div>
        </div>

        <div class="pt-2">
          <span id="match-model-tag" class="text-xs uppercase font-extrabold tracking-wider text-blue-400">XGBoost Match Prediction</span>
          <div id="match-outcome-title" class="text-lg sm:text-2xl font-black text-white mt-1">
            Predicted: Arsenal Win (55.1%)
          </div>
        </div>
      </div>

      <!-- Comparative Team Stats Grid -->
      <div>
        <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Head-to-Head Rolling Metrics</h3>
        <div class="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center text-xs">
          
          <div class="p-3 rounded-xl border" style="background-color: #0B0F17; border-color: #1F293D;">
            <div class="text-[10px] uppercase font-bold text-slate-400">Form (Last 5 Pts)</div>
            <div class="text-base sm:text-lg font-black mt-1">
              <span id="stat-h-form" class="text-blue-400">4 pts</span>
              <span class="text-slate-600 mx-1">vs</span>
              <span id="stat-a-form" class="text-amber-400">3 pts</span>
            </div>
          </div>

          <div class="p-3 rounded-xl border" style="background-color: #0B0F17; border-color: #1F293D;">
            <div class="text-[10px] uppercase font-bold text-slate-400">Goals Scored/Game</div>
            <div class="text-base sm:text-lg font-black mt-1">
              <span id="stat-h-goals" class="text-blue-400">1.0</span>
              <span class="text-slate-600 mx-1">vs</span>
              <span id="stat-a-goals" class="text-amber-400">0.8</span>
            </div>
          </div>

          <div class="p-3 rounded-xl border" style="background-color: #0B0F17; border-color: #1F293D;">
            <div class="text-[10px] uppercase font-bold text-slate-400">Goals Conceded/Game</div>
            <div class="text-base sm:text-lg font-black mt-1">
              <span id="stat-h-conceded" class="text-blue-400">0.4</span>
              <span class="text-slate-600 mx-1">vs</span>
              <span id="stat-a-conceded" class="text-amber-400">1.2</span>
            </div>
          </div>

          <div class="p-3 rounded-xl border" style="background-color: #0B0F17; border-color: #1F293D;">
            <div class="text-[10px] uppercase font-bold text-slate-400">Avg Possession</div>
            <div class="text-base sm:text-lg font-black mt-1">
              <span id="stat-h-poss" class="text-blue-400">61.0%</span>
              <span class="text-slate-600 mx-1">vs</span>
              <span id="stat-a-poss" class="text-amber-400">58.0%</span>
            </div>
          </div>

          <div class="p-3 rounded-xl border col-span-2 sm:col-span-1" style="background-color: #0B0F17; border-color: #1F293D;">
            <div class="text-[10px] uppercase font-bold text-slate-400">Shots on Target</div>
            <div class="text-base sm:text-lg font-black mt-1">
              <span id="stat-h-sot" class="text-blue-400">6.2</span>
              <span class="text-slate-600 mx-1">vs</span>
              <span id="stat-a-sot" class="text-amber-400">5.4</span>
            </div>
          </div>

        </div>
      </div>

    </main>


    <!-- ======================================================== -->
    <!-- SECTION 3: PLAYER SCOUTING & SIMILARITY RECOMMENDER -->
    <!-- ======================================================== -->
    <main id="section-scout" class="rounded-2xl p-5 sm:p-7 shadow-2xl border space-y-6 hidden" style="background-color: #111827; border-color: #1F293D;">
      
      <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b pb-4 gap-3" style="border-color: #1F293D;">
        <div>
          <h2 class="text-lg sm:text-xl font-bold text-white flex items-center gap-2">
            AI Player Scouting & Similarity Recommender
            <span class="text-[10px] font-bold px-2 py-0.5 rounded-full bg-cyan-600/20 text-cyan-400 border border-cyan-500/30">
              COSINE SIMILARITY • K-MEANS
            </span>
          </h2>
          <p class="text-xs text-slate-400 mt-0.5">Discovers statistical twins and playing style archetypes across 441 Premier League players</p>
        </div>

        <div class="flex items-center gap-2">
          <a href="http://localhost:8501" target="_blank" class="px-3.5 py-1.5 text-xs font-semibold rounded-lg text-white border border-blue-500/40 hover:bg-blue-600/30 transition-all flex items-center gap-1.5 shadow" style="background: linear-gradient(135deg, #1E293B, #0F172A);">
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>Local Streamlit Dashboard (Port 8501) ↗</span>
          </a>
        </div>
      </div>

      <!-- Search Target Player Input & Filters -->
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div class="sm:col-span-2">
          <label class="block text-xs font-bold uppercase tracking-wider text-slate-400 mb-1.5">Select or Search Target Player</label>
          <select id="scout-player-select" onchange="updateScoutingView()" class="w-full px-3.5 py-2.5 rounded-xl border text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 font-semibold" style="background-color: #0B0F17; border-color: #1F293D;">
            {"".join(f'<option value="{p}" {"selected" if p == "Bukayo Saka" else ""}>{p}</option>' for p in all_player_names)}
          </select>
        </div>
        <div>
          <label class="block text-xs font-bold uppercase tracking-wider text-slate-400 mb-1.5">Filter Options</label>
          <div class="flex items-center gap-3 pt-2">
            <label class="text-xs text-slate-300 flex items-center gap-1.5 cursor-pointer">
              <input type="checkbox" id="scout-same-pos-check" onchange="updateScoutingView()" class="rounded border-slate-700 bg-slate-900 text-blue-600 focus:ring-0">
              <span>Same Position Only</span>
            </label>
          </div>
        </div>
      </div>

      <!-- Target Player Highlight Card -->
      <div id="scout-target-card" class="rounded-2xl p-5 border relative overflow-hidden" style="background: linear-gradient(135deg, rgba(37,99,235,0.1), rgba(17,24,39,0.95)); border-color: #1F293D;">
        <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div>
            <div class="flex items-center gap-2 mb-1">
              <span id="scout-target-team" class="px-2 py-0.5 rounded text-[11px] font-bold border text-blue-400 border-blue-500/30" style="background-color: rgba(37,99,235,0.15);">Arsenal</span>
              <span id="scout-target-pos" class="px-2 py-0.5 rounded text-[11px] font-bold border text-emerald-400 border-emerald-500/30" style="background-color: rgba(16,185,129,0.15);">RW</span>
              <span id="scout-target-age" class="text-xs text-slate-400">Age 23</span>
            </div>
            <h3 id="scout-target-name" class="text-xl sm:text-2xl font-black text-white">Bukayo Saka</h3>
            <p class="text-xs text-slate-400 mt-1">
              Tactical Archetype: <strong id="scout-target-style" class="text-cyan-400">Creative Winger / Attacking Spark</strong>
            </p>
          </div>
          <div class="sm:text-right">
            <div class="text-[10px] uppercase font-bold text-slate-400">Market Valuation</div>
            <div id="scout-target-val" class="text-2xl sm:text-3xl font-black text-emerald-400">€140.0M</div>
          </div>
        </div>
      </div>

      <!-- Closest Statistical Matches -->
      <div>
        <div class="flex justify-between items-center mb-3">
          <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400">
            Closest Statistical Matches (Cosine Similarity)
          </h3>
          <span class="text-xs text-slate-500">Multidimensional feature vector comparison</span>
        </div>

        <div id="scout-results-container" class="space-y-3"></div>
      </div>

    </main>

  </div>

  <script>
    // Embedded Model Artifacts
    const PLAYER_MODELS = {json.dumps(player_models_summary)};
    const MATCH_PREDICTIONS = {json.dumps(match_predictions)};
    const TEAM_STATS = {json.dumps(team_stats)};
    const ALL_PLAYERS = {json.dumps(players_data)};
    const SCOUTING_SIMILARITY = {json.dumps(scouting_similarity)};
    const PLAYER_STYLES = {json.dumps(player_styles)};

    // Navigation Switcher
    function switchNav(tab) {{
      const playerSec = document.getElementById("section-player");
      const matchSec = document.getElementById("section-match");
      const scoutSec = document.getElementById("section-scout");
      const playerNavBtn = document.getElementById("nav-player-btn");
      const matchNavBtn = document.getElementById("nav-match-btn");
      const scoutNavBtn = document.getElementById("nav-scout-btn");

      // Reset all buttons
      [playerNavBtn, matchNavBtn, scoutNavBtn].forEach(b => {{
        b.style.background = "transparent";
        b.classList.remove("text-white");
        b.classList.add("text-slate-400");
      }});

      // Hide all sections
      playerSec.classList.add("hidden");
      matchSec.classList.add("hidden");
      scoutSec.classList.add("hidden");

      if (tab === "player") {{
        playerSec.classList.remove("hidden");
        playerNavBtn.style.background = "linear-gradient(135deg, #2563EB, #1D4ED8)";
        playerNavBtn.classList.add("text-white");
        playerNavBtn.classList.remove("text-slate-400");
      }} else if (tab === "match") {{
        matchSec.classList.remove("hidden");
        matchNavBtn.style.background = "linear-gradient(135deg, #2563EB, #1D4ED8)";
        matchNavBtn.classList.add("text-white");
        matchNavBtn.classList.remove("text-slate-400");
        calculateMatch();
      }} else if (tab === "scout") {{
        scoutSec.classList.remove("hidden");
        scoutNavBtn.style.background = "linear-gradient(135deg, #2563EB, #1D4ED8)";
        scoutNavBtn.classList.add("text-white");
        scoutNavBtn.classList.remove("text-slate-400");
        updateScoutingView();
      }}
    }}

    // -------------------------------------------------------------
    // MATCH OUTCOME PREDICTOR LOGIC
    // -------------------------------------------------------------
    let currentMatchModel = "xgboost";

    function setMatchModel(m) {{
      currentMatchModel = m;
      const xgbBtn = document.getElementById("model-xgb-btn");
      const rfBtn = document.getElementById("model-rf-btn");
      if (m === "xgboost") {{
        xgbBtn.style.background = "linear-gradient(135deg, #2563EB, #1D4ED8)";
        xgbBtn.classList.add("text-white");
        xgbBtn.classList.remove("text-slate-400");
        rfBtn.style.background = "transparent";
        rfBtn.classList.remove("text-white");
        rfBtn.classList.add("text-slate-400");
        document.getElementById("match-model-tag").innerText = "XGBoost Match Prediction";
      }} else {{
        rfBtn.style.background = "linear-gradient(135deg, #2563EB, #1D4ED8)";
        rfBtn.classList.add("text-white");
        rfBtn.classList.remove("text-slate-400");
        xgbBtn.style.background = "transparent";
        xgbBtn.classList.remove("text-white");
        xgbBtn.classList.add("text-slate-400");
        document.getElementById("match-model-tag").innerText = "Random Forest Match Prediction";
      }}
      calculateMatch();
    }}

    function setFixture(h, a) {{
      document.getElementById("home-team-select").value = h;
      document.getElementById("away-team-select").value = a;
      calculateMatch();
    }}

    function calculateMatch() {{
      const home = document.getElementById("home-team-select").value;
      const away = document.getElementById("away-team-select").value;

      if (home === away) {{
        document.getElementById("match-outcome-title").innerText = "Select two different clubs";
        return;
      }}

      const key = `${{home}}_vs_${{away}}`;
      const predObj = MATCH_PREDICTIONS[key];
      const modelData = predObj ? predObj[currentMatchModel] : null;

      let probs = {{ home_win: 50.0, draw: 25.0, away_win: 25.0 }};
      let outcome = "Home Win (H)";

      if (modelData) {{
        probs = modelData.probabilities;
        outcome = modelData.outcome;
      }}

      // Labels
      document.getElementById("prob-home-label").innerText = `${{home}} Win`;
      document.getElementById("prob-away-label").innerText = `${{away}} Win`;
      document.getElementById("prob-home-val").innerText = `${{probs.home_win.toFixed(1)}}%`;
      document.getElementById("prob-draw-val").innerText = `${{probs.draw.toFixed(1)}}%`;
      document.getElementById("prob-away-val").innerText = `${{probs.away_win.toFixed(1)}}%`;

      document.getElementById("prob-home-bar").style.width = `${{probs.home_win}}%`;
      document.getElementById("prob-draw-bar").style.width = `${{probs.draw}}%`;
      document.getElementById("prob-away-bar").style.width = `${{probs.away_win}}%`;

      const winTeam = probs.home_win > probs.away_win ? home : away;
      const isDraw = Math.abs(probs.home_win - probs.away_win) < 4.0;
      document.getElementById("match-outcome-title").innerText = isDraw 
        ? `Predicted: Close Match / Draw (${{probs.draw.toFixed(1)}}%)` 
        : `Predicted: ${{winTeam}} Win (${{Math.max(probs.home_win, probs.away_win).toFixed(1)}}%)`;

      // Stats comparison
      const hStat = TEAM_STATS[home] || {{ form_points_last_5: 7, goals_scored_avg: 1.5, goals_conceded_avg: 1.2, possession_avg: 50, sot_avg: 5 }};
      const aStat = TEAM_STATS[away] || {{ form_points_last_5: 7, goals_scored_avg: 1.5, goals_conceded_avg: 1.2, possession_avg: 50, sot_avg: 5 }};

      document.getElementById("stat-h-form").innerText = `${{hStat.form_points_last_5}} pts`;
      document.getElementById("stat-a-form").innerText = `${{aStat.form_points_last_5}} pts`;
      document.getElementById("stat-h-goals").innerText = hStat.goals_scored_avg.toFixed(1);
      document.getElementById("stat-a-goals").innerText = aStat.goals_scored_avg.toFixed(1);
      document.getElementById("stat-h-conceded").innerText = hStat.goals_conceded_avg.toFixed(1);
      document.getElementById("stat-a-conceded").innerText = aStat.goals_conceded_avg.toFixed(1);
      document.getElementById("stat-h-poss").innerText = `${{hStat.possession_avg.toFixed(1)}}%`;
      document.getElementById("stat-a-poss").innerText = `${{aStat.possession_avg.toFixed(1)}}%`;
      document.getElementById("stat-h-sot").innerText = hStat.sot_avg.toFixed(1);
      document.getElementById("stat-a-sot").innerText = aStat.sot_avg.toFixed(1);
    }}

    // -------------------------------------------------------------
    // PLAYER SCOUTING LOGIC
    // -------------------------------------------------------------
    function updateScoutingView() {{
      const targetName = document.getElementById("scout-player-select").value;
      const samePosOnly = document.getElementById("scout-same-pos-check").checked;
      
      const target = ALL_PLAYERS.find(p => p.player_name === targetName);
      if (!target) return;

      const styleInfo = PLAYER_STYLES[targetName] || {{ style: "Versatile Profile", cluster: 0 }};

      document.getElementById("scout-target-name").innerText = target.player_name;
      document.getElementById("scout-target-team").innerText = target.team;
      document.getElementById("scout-target-pos").innerText = target.position;
      document.getElementById("scout-target-age").innerText = `Age ${{target.age}}`;
      document.getElementById("scout-target-style").innerText = styleInfo.style;
      document.getElementById("scout-target-val").innerText = `€${{target.market_value_eur_m}}M`;

      // Render Similar Matches
      let matches = SCOUTING_SIMILARITY[targetName] || [];
      if (samePosOnly) {{
        matches = matches.filter(m => m.position === target.position);
      }}

      const container = document.getElementById("scout-results-container");
      container.innerHTML = "";

      if (matches.length === 0) {{
        container.innerHTML = `<div class="p-6 text-center text-slate-400 text-xs">No similar players match the selected filters.</div>`;
        return;
      }}

      matches.forEach((m, idx) => {{
        const pct = m.similarity_pct;
        const color = pct >= 95 ? "#34D399" : pct >= 90 ? "#38BDF8" : "#F59E0B";
        
        const card = document.createElement("div");
        card.className = "p-4 rounded-xl border transition-all hover:border-blue-500/50";
        card.style.backgroundColor = "#161F30";
        card.style.borderColor = "#1F293D";

        card.innerHTML = `
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-2">
            <div>
              <div class="flex items-center gap-2">
                <span class="font-bold text-white text-sm sm:text-base">#${{idx + 1}} ${{m.name}}</span>
                <span class="text-xs px-2 py-0.5 rounded font-semibold text-blue-400 bg-blue-950/60 border border-blue-800/50">${{m.position}}</span>
                <span class="text-xs px-2 py-0.5 rounded font-medium text-slate-300" style="background-color: #0B0F17; border: 1px solid #1F293D;">${{m.team}}</span>
                <span class="text-xs text-slate-400">Age ${{m.age}}</span>
              </div>
              <div class="text-xs text-slate-400 mt-1">
                Style: <span class="text-slate-200 font-medium">${{m.style}}</span> • Market Value: <span class="text-emerald-400 font-semibold">€${{m.market_val}}M</span>
              </div>
            </div>
            <div class="sm:text-right">
              <span class="text-lg font-black" style="color: ${{color}};">${{pct}}%</span>
              <span class="text-[10px] uppercase font-bold text-slate-400 ml-1">Similarity</span>
            </div>
          </div>
          <div class="w-full h-1.5 rounded-full overflow-hidden" style="background-color: #0B0F17;">
            <div class="h-full rounded-full" style="width: ${{pct}}%; background-color: ${{color}};"></div>
          </div>
          <div class="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-400 mt-2.5 pt-2 border-t" style="border-color: #1F293D;">
            <span>Goals: <strong class="text-white">${{m.goals}}</strong></span>
            <span>Assists: <strong class="text-white">${{m.assists}}</strong></span>
            <span>Chances Created: <strong class="text-white">${{m.chances}}</strong></span>
            <span>Tackles & Duels: <strong class="text-white">${{m.tackles_duels}}</strong></span>
          </div>
        `;
        container.appendChild(card);
      }});
    }}

    // -------------------------------------------------------------
    // PLAYER VALUATION PREDICTOR LOGIC
    // -------------------------------------------------------------
    let currentRole = "CF";
    let listRoleFilter = "";

    const ROLE_CONFIGS = {{
      "CF": {{
        name: "Central Forward",
        group: "CF_SS",
        sliders: [
          {{ id: "goals", label: "Goals Scored", min: 0, max: 35, val: 15, step: 1 }},
          {{ id: "assists", label: "Assists", min: 0, max: 20, val: 5, step: 1 }}
        ]
      }},
      "SS": {{
        name: "Second Striker",
        group: "CF_SS",
        sliders: [
          {{ id: "goals", label: "Goals Scored", min: 0, max: 30, val: 10, step: 1 }},
          {{ id: "assists", label: "Assists", min: 0, max: 20, val: 6, step: 1 }}
        ]
      }},
      "RW": {{
        name: "Right Winger",
        group: "WINGERS",
        sliders: [
          {{ id: "goals", label: "Goals Scored", min: 0, max: 25, val: 8, step: 1 }},
          {{ id: "assists", label: "Assists", min: 0, max: 20, val: 7, step: 1 }},
          {{ id: "chances_created", label: "Chances Created", min: 0, max: 120, val: 55, step: 2 }},
          {{ id: "dribbles_completed", label: "Dribbles Completed", min: 0, max: 120, val: 50, step: 2 }}
        ]
      }},
      "LW": {{
        name: "Left Winger",
        group: "WINGERS",
        sliders: [
          {{ id: "goals", label: "Goals Scored", min: 0, max: 25, val: 8, step: 1 }},
          {{ id: "assists", label: "Assists", min: 0, max: 20, val: 6, step: 1 }},
          {{ id: "chances_created", label: "Chances Created", min: 0, max: 120, val: 50, step: 2 }},
          {{ id: "dribbles_completed", label: "Dribbles Completed", min: 0, max: 120, val: 55, step: 2 }}
        ]
      }},
      "AMF": {{
        name: "Attacking Midfielder",
        group: "AMF",
        sliders: [
          {{ id: "goals", label: "Goals Scored", min: 0, max: 25, val: 6, step: 1 }},
          {{ id: "assists", label: "Assists", min: 0, max: 20, val: 8, step: 1 }},
          {{ id: "chances_created", label: "Chances Created", min: 0, max: 120, val: 65, step: 2 }}
        ]
      }},
      "CMF": {{
        name: "Central Midfielder",
        group: "CMF",
        sliders: [
          {{ id: "balls_recovered", label: "Balls Recovered", min: 20, max: 280, val: 140, step: 5 }},
          {{ id: "line_breaking_passes", label: "Line-Breaking Passes", min: 20, max: 300, val: 130, step: 5 }},
          {{ id: "pass_accuracy", label: "Pass Accuracy %", min: 65, max: 96, val: 86, step: 0.5 }}
        ]
      }},
      "DMF": {{
        name: "Defensive Midfielder",
        group: "DMF",
        sliders: [
          {{ id: "balls_recovered", label: "Balls Recovered", min: 30, max: 300, val: 180, step: 5 }},
          {{ id: "duels_won", label: "Duels Won", min: 20, max: 280, val: 150, step: 5 }},
          {{ id: "aerial_duels_won", label: "Aerial Duels Won", min: 5, max: 90, val: 40, step: 2 }},
          {{ id: "line_breaking_passes", label: "Line-Breaking Passes", min: 20, max: 300, val: 120, step: 5 }}
        ]
      }},
      "CB": {{
        name: "Centre Back",
        group: "CB",
        sliders: [
          {{ id: "duels_won", label: "Duels Won", min: 30, max: 250, val: 140, step: 5 }},
          {{ id: "successful_tackles", label: "Successful Tackles", min: 10, max: 100, val: 50, step: 2 }},
          {{ id: "aerial_duels_won", label: "Aerial Duels Won", min: 20, max: 150, val: 75, step: 2 }}
        ]
      }},
      "RB": {{
        name: "Right Back",
        group: "FULLBACK",
        sliders: [
          {{ id: "assists", label: "Assists", min: 0, max: 15, val: 3, step: 1 }},
          {{ id: "duels_won", label: "Duels Won", min: 30, max: 220, val: 120, step: 5 }},
          {{ id: "successful_tackles", label: "Successful Tackles", min: 15, max: 110, val: 55, step: 2 }},
          {{ id: "aerial_duels_won", label: "Aerial Duels Won", min: 5, max: 70, val: 28, step: 2 }}
        ]
      }},
      "LB": {{
        name: "Left Back",
        group: "FULLBACK",
        sliders: [
          {{ id: "assists", label: "Assists", min: 0, max: 15, val: 3, step: 1 }},
          {{ id: "duels_won", label: "Duels Won", min: 30, max: 220, val: 120, step: 5 }},
          {{ id: "successful_tackles", label: "Successful Tackles", min: 15, max: 110, val: 55, step: 2 }},
          {{ id: "aerial_duels_won", label: "Aerial Duels Won", min: 5, max: 70, val: 28, step: 2 }}
        ]
      }},
      "GK": {{
        name: "Goalkeeper",
        group: "GK",
        sliders: [
          {{ id: "saves", label: "Saves Made", min: 5, max: 160, val: 80, step: 2 }},
          {{ id: "penalties_saved", label: "Penalties Saved", min: 0, max: 4, val: 1, step: 1 }},
          {{ id: "pass_accuracy", label: "Passing Accuracy %", min: 50, max: 92, val: 80, step: 0.5 }}
        ]
      }}
    }};

    function switchPlayerTab(tab) {{
      const predView = document.getElementById("tab-player-predict");
      const lookView = document.getElementById("tab-player-lookup");
      const predBtn = document.getElementById("tab-predict-btn");
      const lookBtn = document.getElementById("tab-lookup-btn");

      if (tab === "predict") {{
        predView.classList.remove("hidden");
        lookView.classList.add("hidden");
        predBtn.style.background = "linear-gradient(135deg, #2563EB, #1D4ED8)";
        predBtn.classList.add("text-white");
        predBtn.classList.remove("text-slate-400");
        lookBtn.style.background = "transparent";
        lookBtn.classList.remove("text-white");
        lookBtn.classList.add("text-slate-400");
      }} else {{
        predView.classList.add("hidden");
        lookView.classList.remove("hidden");
        lookBtn.style.background = "linear-gradient(135deg, #2563EB, #1D4ED8)";
        lookBtn.classList.add("text-white");
        lookBtn.classList.remove("text-slate-400");
        predBtn.style.background = "transparent";
        predBtn.classList.remove("text-white");
        predBtn.classList.add("text-slate-400");
        filterPlayers();
      }}
    }}

    function selectRole(role) {{
      currentRole = role;
      document.querySelectorAll(".role-btn").forEach(btn => {{
        if (btn.getAttribute("data-role") === role) {{
          btn.style.backgroundColor = "#2563EB";
          btn.style.borderColor = "#3B82F6";
          btn.classList.add("text-white");
          btn.classList.remove("text-slate-400");
        }} else {{
          btn.style.backgroundColor = "#0B0F17";
          btn.style.borderColor = "#1F293D";
          btn.classList.remove("text-white");
          btn.classList.add("text-slate-400");
        }}
      }});

      const cfg = ROLE_CONFIGS[role];
      document.getElementById("pos-group-desc").innerText = `${{role}} (${{cfg.name}})`;
      buildDynamicSliders(cfg);
      calculatePlayer();
    }}

    function buildDynamicSliders(cfg) {{
      const container = document.getElementById("dynamic-sliders");
      container.innerHTML = "";
      cfg.sliders.forEach(s => {{
        const div = document.createElement("div");
        div.className = "p-4 rounded-xl border";
        div.style.backgroundColor = "#0B0F17";
        div.style.borderColor = "#1F293D";
        div.innerHTML = `
          <div class="flex justify-between items-center text-xs sm:text-sm mb-2">
            <span class="font-medium text-slate-400">${{s.label}}</span>
            <span id="${{s.id}}-val" class="font-black text-white px-2 py-0.5 rounded" style="background-color: #161F30;">${{s.val}}</span>
          </div>
          <input id="${{s.id}}-slider" type="range" min="${{s.min}}" max="${{s.max}}" step="${{s.step}}" value="${{s.val}}" class="w-full cursor-pointer" oninput="calculatePlayer()">
        `;
        container.appendChild(div);
      }});
    }}

    function calculatePlayer() {{
      const age = parseFloat(document.getElementById("age-slider").value);
      const mins = parseFloat(document.getElementById("mins-slider").value);
      document.getElementById("age-val").innerText = `${{age}} years`;
      document.getElementById("mins-val").innerText = `${{mins.toLocaleString()}} mins`;

      const cfg = ROLE_CONFIGS[currentRole];
      const model = PLAYER_MODELS[cfg.group];
      if (!model) return;

      const inputValues = {{ age: age, minutes_played: mins }};
      cfg.sliders.forEach(s => {{
        const el = document.getElementById(`${{s.id}}-slider`);
        if (el) {{
          const val = parseFloat(el.value);
          inputValues[s.id] = val;
          const displayEl = document.getElementById(`${{s.id}}-val`);
          if (displayEl) {{
            displayEl.innerText = s.id.includes("accuracy") ? `${{val}}%` : val;
          }}
        }}
      }});

      let val = model.intercept;
      const breakdownEl = document.getElementById("breakdown-container");
      breakdownEl.innerHTML = "";

      model.coefficients.forEach(c => {{
        const fVal = inputValues[c.feature] || 0;
        const impact = c.weight * fVal;
        val += impact;
      }});

      if (val < 2.0) val = 2.0;

      document.getElementById("predicted-val").innerText = val.toFixed(1);

      // Model metadata badge
      const r2Badge = document.createElement("span");
      r2Badge.className = "px-2 py-0.5 rounded text-[11px] text-blue-400 border border-blue-500/30";
      r2Badge.style.backgroundColor = "rgba(37,99,235,0.15)";
      r2Badge.innerText = `Sub-model R²: ${{(model.metrics.r2 * 100).toFixed(1)}}%`;
      breakdownEl.appendChild(r2Badge);

      const maeBadge = document.createElement("span");
      maeBadge.className = "px-2 py-0.5 rounded text-[11px] text-slate-300 border border-slate-700";
      maeBadge.style.backgroundColor = "#0B0F17";
      maeBadge.innerText = `MAE: ±€${{model.metrics.mae.toFixed(1)}}M`;
      breakdownEl.appendChild(maeBadge);
    }}

    function predictPlayerStats(p) {{
      let group = "CF_SS";
      if (["RW", "LW", "LMF", "RMF"].includes(p.position)) group = "WINGERS";
      else if (p.position === "AMF") group = "AMF";
      else if (p.position === "CMF") group = "CMF";
      else if (p.position === "DMF") group = "DMF";
      else if (p.position === "CB") group = "CB";
      else if (["RB", "LB"].includes(p.position)) group = "FULLBACK";
      else if (p.position === "GK") group = "GK";

      const model = PLAYER_MODELS[group];
      if (!model) return p.market_value_eur_m;

      let val = model.intercept;
      model.coefficients.forEach(c => {{
        const fVal = p[c.feature] !== undefined ? p[c.feature] : 0;
        val += c.weight * fVal;
      }});
      return val < 2.0 ? 2.0 : val;
    }}

    function setListRoleFilter(role) {{
      listRoleFilter = role;
      document.querySelectorAll(".list-role-btn").forEach(btn => {{
        if (btn.getAttribute("data-lrole") === role) {{
          btn.className = "list-role-btn px-2.5 py-1 rounded-lg font-bold bg-blue-600 text-white";
        }} else {{
          btn.className = "list-role-btn px-2.5 py-1 rounded-lg font-medium text-slate-400";
          btn.style.backgroundColor = "#0B0F17";
        }}
      }});
      filterPlayers();
    }}

    function getRoleSpecificChips(p) {{
      if (p.position === "GK") {{
        return `<span>${{p.saves}} saves</span><span>•</span><span>${{p.penalties_saved}} pens</span><span>•</span><span>${{p.pass_accuracy}}% pass</span>`;
      }}
      if (p.position === "CB") {{
        return `<span>${{p.duels_won}} duels</span><span>•</span><span>${{p.successful_tackles}} tackles</span><span>•</span><span>${{p.aerial_duels_won}} aerial</span>`;
      }}
      if (["RB", "LB"].includes(p.position)) {{
        return `<span>${{p.assists}} ast</span><span>•</span><span>${{p.duels_won}} duels</span><span>•</span><span>${{p.successful_tackles}} tackles</span>`;
      }}
      if (p.position === "DMF") {{
        return `<span>${{p.balls_recovered}} recov</span><span>•</span><span>${{p.duels_won}} duels</span><span>•</span><span>${{p.line_breaking_passes}} passes</span>`;
      }}
      if (p.position === "CMF") {{
        return `<span>${{p.balls_recovered}} recov</span><span>•</span><span>${{p.line_breaking_passes}} passes</span><span>•</span><span>${{p.pass_accuracy}}% acc</span>`;
      }}
      if (["RW", "LW"].includes(p.position)) {{
        return `<span>${{p.goals}}G / ${{p.assists}}A</span><span>•</span><span>${{p.chances_created}} chances</span><span>•</span><span>${{p.dribbles_completed}} dribbles</span>`;
      }}
      if (p.position === "AMF") {{
        return `<span>${{p.goals}}G / ${{p.assists}}A</span><span>•</span><span>${{p.chances_created}} chances</span>`;
      }}
      return `<span>${{p.goals}} goals</span><span>•</span><span>${{p.assists}} assists</span>`;
    }}

    function filterPlayers() {{
      const q = document.getElementById("player-search").value.toLowerCase().trim();
      const team = document.getElementById("team-filter").value;
      const list = document.getElementById("players-list");
      list.innerHTML = "";

      const filtered = ALL_PLAYERS.filter(p => {{
        const matchesQuery = p.player_name.toLowerCase().includes(q);
        const matchesTeam = !team || p.team === team;
        const matchesRole = !listRoleFilter || p.position === listRoleFilter;
        return matchesQuery && matchesTeam && matchesRole;
      }});

      document.getElementById("player-count").innerText = `Showing ${{filtered.length}} of ${{ALL_PLAYERS.length}} players`;

      if (filtered.length === 0) {{
        list.innerHTML = `<div class="p-8 text-center text-slate-400 text-sm">No players found matching your filters.</div>`;
        return;
      }}

      filtered.forEach(p => {{
        const pred = predictPlayerStats(p);
        const diff = p.market_value_eur_m - pred;
        const isBargain = diff < -5;
        const isPremium = diff > 5;
        const statusBadge = isBargain 
          ? `<span class="px-2 py-0.5 rounded text-[11px] font-bold border" style="background-color: rgba(16,185,129,0.15); color: #34D399; border-color: rgba(16,185,129,0.3);">Stat Bargain (-€${{Math.abs(diff).toFixed(1)}}M)</span>`
          : isPremium 
            ? `<span class="px-2 py-0.5 rounded text-[11px] font-bold border" style="background-color: rgba(245,158,11,0.15); color: #FBBF24; border-color: rgba(245,158,11,0.3);">Market Premium (+€${{diff.toFixed(1)}}M)</span>`
            : `<span class="px-2 py-0.5 rounded text-[11px] font-bold border" style="background-color: rgba(59,130,246,0.15); color: #60A5FA; border-color: rgba(59,130,246,0.3);">Fairly Valued</span>`;

        const statChips = getRoleSpecificChips(p);

        const card = document.createElement("div");
        card.className = "p-3.5 rounded-xl border transition-all duration-150 hover:border-blue-500/50 flex flex-col sm:flex-row sm:items-center justify-between gap-3";
        card.style.backgroundColor = "#161F30";
        card.style.borderColor = "#1F293D";
        
        card.innerHTML = `
          <div>
            <div class="flex items-center gap-2">
              <span class="font-bold text-sm sm:text-base text-white">${{p.player_name}}</span>
              <span class="text-xs px-2 py-0.5 rounded font-semibold text-blue-400 bg-blue-950/60 border border-blue-800/50">${{p.position}}</span>
              <span class="text-xs px-2 py-0.5 rounded font-medium text-slate-300" style="background-color: #0B0F17; border: 1px solid #1F293D;">${{p.team}}</span>
            </div>
            <div class="text-xs text-slate-400 mt-1 flex flex-wrap gap-x-2">
              <span>Age ${{p.age}}</span>
              <span>•</span>
              <span>${{p.minutes_played.toLocaleString()}}m</span>
              <span>•</span>
              ${{statChips}}
            </div>
          </div>
          <div class="sm:text-right flex sm:flex-col items-center sm:items-end justify-between gap-1.5 pt-2 sm:pt-0 border-t sm:border-t-0" style="border-color: #1F293D;">
            <div class="text-xs text-slate-300">
              Actual: <span class="font-bold text-white">€${{p.market_value_eur_m}}M</span> | Model: <span class="font-bold text-blue-400">€${{pred.toFixed(1)}}M</span>
            </div>
            <div>${{statusBadge}}</div>
          </div>
        `;
        list.appendChild(card);
      }});
    }}

    // Initial setup
    selectRole("CF");
  </script>
</body>
</html>
"""

with open(WEB_OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Generated unified web UI at: {WEB_OUTPUT_PATH}")

with open(WIDGET_OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Generated unified chat widget at: {WIDGET_OUTPUT_PATH}")
