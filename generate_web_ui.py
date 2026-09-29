"""
Unified Web UI Generator for Premier League AI Suite
Integrates:
1. Player Valuation Predictor (Role-specific Linear Regression for 441 players)
2. Match Outcome Predictor (XGBoost & Random Forest Win/Draw/Loss classifier)
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

WEB_OUTPUT_PATH = os.path.join(PROJECT_DIR, "index.html")
WIDGET_OUTPUT_PATH = r"C:\Users\Admin\.gemini\antigravity\brain\8d4c148a-7a62-48eb-8644-4a4628d38d4b\transfer_predictor_widget.html"

with open(PLAYERS_JSON_PATH, "r", encoding="utf-8") as f:
    players_data = json.load(f)

with open(MATCHES_JSON_PATH, "r", encoding="utf-8") as f:
    match_predictions = json.load(f)

with open(TEAM_STATS_JSON_PATH, "r", encoding="utf-8") as f:
    team_stats = json.load(f)

teams = sorted(list(team_stats.keys()))

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
  <title>Premier League AI Hub | Player Valuation & Match Predictor</title>
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
            Premier League AI Analytics Hub
          </h1>
          <p class="text-xs text-slate-400">Machine Learning tools for Player Valuations & Match Outcomes</p>
        </div>
      </div>

      <!-- Top Bar Functionality Tabs -->
      <nav class="flex p-1 rounded-xl border w-full sm:w-auto justify-center" style="background-color: #0B0F17; border-color: #1F293D;">
        <button id="nav-player-btn" onclick="switchNav('player')" class="flex-1 sm:flex-none px-4 py-2 text-xs sm:text-sm font-bold rounded-lg text-white transition-all shadow" style="background: linear-gradient(135deg, #2563EB, #1D4ED8);">
          ⚽ Player Valuation
        </button>
        <button id="nav-match-btn" onclick="switchNav('match')" class="flex-1 sm:flex-none px-4 py-2 text-xs sm:text-sm font-bold rounded-lg text-slate-400 hover:text-white transition-all">
          🏆 Match Outcome Predictor
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
            Premier League Match Outcome Predictor
            <span class="text-[10px] font-bold px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              XGBOOST & RANDOM FOREST
            </span>
          </h2>
          <p class="text-xs text-slate-400 mt-0.5">Trained on 760 historical matches with rolling form, shots on target, possession & home advantage</p>
        </div>

        <!-- Model Selector Toggle -->
        <div class="flex p-1 rounded-xl border self-start sm:self-auto" style="background-color: #0B0F17; border-color: #1F293D;">
          <button id="model-xgb-btn" onclick="setMatchModel('xgboost')" class="px-3.5 py-1.5 text-xs font-bold rounded-lg text-white transition-all shadow" style="background: linear-gradient(135deg, #2563EB, #1D4ED8);">
            ⚡ XGBoost
          </button>
          <button id="model-rf-btn" onclick="setMatchModel('random_forest')" class="px-3.5 py-1.5 text-xs font-bold rounded-lg text-slate-400 hover:text-white transition-all">
            🌲 Random Forest
          </button>
        </div>
      </div>

      <!-- Quick Fixture Launcher -->
      <div>
        <label class="text-xs font-bold uppercase tracking-wider text-slate-400 block mb-2">⚡ Quick Marquee Fixtures</label>
        <div class="flex flex-wrap gap-2">
          <button onclick="setFixture('Arsenal', 'Chelsea')" class="px-3 py-1.5 rounded-lg border text-xs font-semibold text-slate-300 hover:text-white hover:border-blue-500 transition" style="background-color: #0B0F17; border-color: #1F293D;">Arsenal vs Chelsea</button>
          <button onclick="setFixture('Manchester City', 'Liverpool')" class="px-3 py-1.5 rounded-lg border text-xs font-semibold text-slate-300 hover:text-white hover:border-blue-500 transition" style="background-color: #0B0F17; border-color: #1F293D;">Man City vs Liverpool</button>
          <button onclick="setFixture('Tottenham', 'Arsenal')" class="px-3 py-1.5 rounded-lg border text-xs font-semibold text-slate-300 hover:text-white hover:border-blue-500 transition" style="background-color: #0B0F17; border-color: #1F293D;">Tottenham vs Arsenal</button>
          <button onclick="setFixture('Aston Villa', 'Manchester United')" class="px-3 py-1.5 rounded-lg border text-xs font-semibold text-slate-300 hover:text-white hover:border-blue-500 transition" style="background-color: #0B0F17; border-color: #1F293D;">Aston Villa vs Man United</button>
          <button onclick="setFixture('Newcastle', 'West Ham')" class="px-3 py-1.5 rounded-lg border text-xs font-semibold text-slate-300 hover:text-white hover:border-blue-500 transition" style="background-color: #0B0F17; border-color: #1F293D;">Newcastle vs West Ham</button>
        </div>
      </div>

      <!-- Team Selectors -->
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 items-center">
        <!-- Home Team -->
        <div class="p-4 rounded-xl border relative" style="background-color: #0B0F17; border-color: #1F293D;">
          <span class="text-[10px] font-extrabold uppercase tracking-widest text-blue-400 block mb-1">🏠 Home Team (Home Advantage)</span>
          <select id="home-team-select" class="w-full px-3 py-2 rounded-lg border text-white font-bold text-base focus:outline-none focus:ring-2 focus:ring-blue-500" style="background-color: #161F30; border-color: #25334D;" onchange="calculateMatch()">
            {"".join(f'<option value="{t}" {"selected" if t=="Arsenal" else ""}>{t}</option>' for t in teams)}
          </select>
        </div>

        <!-- Away Team -->
        <div class="p-4 rounded-xl border relative" style="background-color: #0B0F17; border-color: #1F293D;">
          <span class="text-[10px] font-extrabold uppercase tracking-widest text-amber-400 block mb-1">✈️ Away Team (Visiting)</span>
          <select id="away-team-select" class="w-full px-3 py-2 rounded-lg border text-white font-bold text-base focus:outline-none focus:ring-2 focus:ring-blue-500" style="background-color: #161F30; border-color: #25334D;" onchange="calculateMatch()">
            {"".join(f'<option value="{t}" {"selected" if t=="Chelsea" else ""}>{t}</option>' for t in teams)}
          </select>
        </div>
      </div>

      <!-- Live Match Prediction Display Card -->
      <div class="rounded-2xl p-6 border text-center relative overflow-hidden" style="background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(15,23,42,0.98), rgba(16,185,129,0.18)); border-color: #2563EB;">
        <span id="match-model-tag" class="text-xs uppercase font-extrabold tracking-widest text-blue-400">XGBoost Match Prediction</span>
        
        <div id="match-outcome-title" class="text-2xl sm:text-4xl font-black text-white my-3 tracking-tight">
          Predicted Outcome: Home Win (H)
        </div>

        <!-- Probability Bars Gauge -->
        <div class="space-y-3 max-w-xl mx-auto mt-5">
          <!-- Home Win -->
          <div>
            <div class="flex justify-between text-xs font-bold mb-1">
              <span id="prob-home-label" class="text-blue-400">Arsenal Win</span>
              <span id="prob-home-val" class="text-blue-400 font-extrabold">57.2%</span>
            </div>
            <div class="w-full h-3 rounded-full bg-slate-900 border border-slate-800 overflow-hidden">
              <div id="prob-home-bar" class="h-full rounded-full transition-all duration-500" style="width: 57.2%; background: linear-gradient(90deg, #2563EB, #3B82F6);"></div>
            </div>
          </div>

          <!-- Draw -->
          <div>
            <div class="flex justify-between text-xs font-bold mb-1">
              <span class="text-slate-400">Draw</span>
              <span id="prob-draw-val" class="text-slate-300 font-extrabold">17.5%</span>
            </div>
            <div class="w-full h-3 rounded-full bg-slate-900 border border-slate-800 overflow-hidden">
              <div id="prob-draw-bar" class="h-full rounded-full transition-all duration-500" style="width: 17.5%; background: #64748B;"></div>
            </div>
          </div>

          <!-- Away Win -->
          <div>
            <div class="flex justify-between text-xs font-bold mb-1">
              <span id="prob-away-label" class="text-amber-400">Chelsea Win</span>
              <span id="prob-away-val" class="text-amber-400 font-extrabold">25.3%</span>
            </div>
            <div class="w-full h-3 rounded-full bg-slate-900 border border-slate-800 overflow-hidden">
              <div id="prob-away-bar" class="h-full rounded-full transition-all duration-500" style="width: 25.3%; background: linear-gradient(90deg, #F59E0B, #EF4444);"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Comparative Head-to-Head Stats Cards -->
      <div>
        <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Head-to-Head Recent Form & Performance Metrics</h3>
        <div class="grid grid-cols-2 sm:grid-cols-5 gap-3 text-center">
          
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

  </div>

  <script>
    // Embedded Model Artifacts
    const PLAYER_MODELS = {json.dumps(player_models_summary)};
    const MATCH_PREDICTIONS = {json.dumps(match_predictions)};
    const TEAM_STATS = {json.dumps(team_stats)};
    const ALL_PLAYERS = {json.dumps(players_data)};

    // Navigation Switcher
    function switchNav(tab) {{
      const playerSec = document.getElementById("section-player");
      const matchSec = document.getElementById("section-match");
      const playerNavBtn = document.getElementById("nav-player-btn");
      const matchNavBtn = document.getElementById("nav-match-btn");

      if (tab === "player") {{
        playerSec.classList.remove("hidden");
        matchSec.classList.add("hidden");
        playerNavBtn.style.background = "linear-gradient(135deg, #2563EB, #1D4ED8)";
        playerNavBtn.classList.add("text-white");
        playerNavBtn.classList.remove("text-slate-400");
        matchNavBtn.style.background = "transparent";
        matchNavBtn.classList.remove("text-white");
        matchNavBtn.classList.add("text-slate-400");
      }} else {{
        matchSec.classList.remove("hidden");
        playerSec.classList.add("hidden");
        matchNavBtn.style.background = "linear-gradient(135deg, #2563EB, #1D4ED8)";
        matchNavBtn.classList.add("text-white");
        matchNavBtn.classList.remove("text-slate-400");
        playerNavBtn.style.background = "transparent";
        playerNavBtn.classList.remove("text-white");
        playerNavBtn.classList.add("text-slate-400");
        calculateMatch();
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
    // PLAYER VALUATION LOGIC
    // -------------------------------------------------------------
    const ROLE_TO_GROUP = {{
      "CF": "CF_SS", "SS": "CF_SS", "RW": "WINGERS", "LW": "WINGERS",
      "AMF": "AMF", "CMF": "CMF", "DMF": "DMF", "CB": "CB",
      "RB": "FULLBACK", "LB": "FULLBACK", "GK": "GK"
    }};

    const METRIC_DEFINITIONS = {{
      "goals": {{ label: "Goals Scored", min: 0, max: 36, step: 1, default: 15, unit: "goals", color: "#34D399" }},
      "assists": {{ label: "Assists Provided", min: 0, max: 22, step: 1, default: 6, unit: "assists", color: "#60A5FA" }},
      "chances_created": {{ label: "Chances Created", min: 5, max: 130, step: 1, default: 65, unit: "chances", color: "#38BDF8" }},
      "dribbles_completed": {{ label: "Dribbles Completed", min: 5, max: 110, step: 1, default: 55, unit: "dribbles", color: "#C084FC" }},
      "balls_recovered": {{ label: "Balls Recovered", min: 30, max: 260, step: 5, default: 160, unit: "recoveries", color: "#F472B6" }},
      "line_breaking_passes": {{ label: "Line-Breaking Passes", min: 30, max: 290, step: 5, default: 150, unit: "passes", color: "#FBBF24" }},
      "pass_accuracy": {{ label: "Pass Accuracy (%)", min: 60, max: 96, step: 0.5, default: 85, unit: "%", color: "#A3E635" }},
      "duels_won": {{ label: "Duels Won", min: 30, max: 240, step: 5, default: 150, unit: "duels", color: "#FB923C" }},
      "aerial_duels_won": {{ label: "Aerial Duels Won", min: 10, max: 130, step: 2, default: 65, unit: "aerials", color: "#E879F9" }},
      "successful_tackles": {{ label: "Successful Tackles", min: 10, max: 105, step: 2, default: 55, unit: "tackles", color: "#2DD4BF" }},
      "saves": {{ label: "Saves Made", min: 10, max: 160, step: 2, default: 95, unit: "saves", color: "#818CF8" }},
      "penalties_saved": {{ label: "Penalties Saved", min: 0, max: 5, step: 1, default: 1, unit: "pens", color: "#F87171" }}
    }};

    let currentRole = "CF";
    let listRoleFilter = "";

    function selectRole(role) {{
      currentRole = role;
      document.querySelectorAll(".role-btn").forEach(btn => {{
        if (btn.dataset.role === role) {{
          btn.style.backgroundColor = "#2563EB";
          btn.style.borderColor = "#3B82F6";
          btn.classList.remove("text-slate-400");
          btn.classList.add("text-white");
        }} else {{
          btn.style.backgroundColor = "#0B0F17";
          btn.style.borderColor = "#1F293D";
          btn.classList.remove("text-white");
          btn.classList.add("text-slate-400");
        }}
      }});

      const groupKey = ROLE_TO_GROUP[role] || "CF_SS";
      const model = PLAYER_MODELS[groupKey];
      document.getElementById("pos-group-desc").innerText = `${{role}} (${{model.name}})`;

      renderDynamicSliders(model.features);
      calculatePlayer();
    }}

    function renderDynamicSliders(features) {{
      const container = document.getElementById("dynamic-sliders");
      container.innerHTML = "";
      const roleFeatures = features.filter(f => f !== "age" && f !== "minutes_played");

      roleFeatures.forEach(feat => {{
        const def = METRIC_DEFINITIONS[feat] || {{ label: feat, min: 0, max: 100, step: 1, default: 50, unit: "", color: "#3B82F6" }};
        const box = document.createElement("div");
        box.className = "p-4 rounded-xl border";
        box.style.backgroundColor = "#0B0F17";
        box.style.borderColor = "#1F293D";

        box.innerHTML = `
          <div class="flex justify-between items-center text-xs sm:text-sm mb-2">
            <span class="font-medium text-slate-400">${{def.label}}</span>
            <span id="${{feat}}-val" class="font-black px-2 py-0.5 rounded" style="color: ${{def.color}}; background-color: #161F30;">
              ${{def.default}} ${{def.unit}}
            </span>
          </div>
          <input id="${{feat}}-slider" type="range" min="${{def.min}}" max="${{def.max}}" step="${{def.step}}" value="${{def.default}}" class="w-full cursor-pointer" oninput="calculatePlayer()">
        `;
        container.appendChild(box);
      }});
    }}

    function calculatePlayer() {{
      const age = parseFloat(document.getElementById("age-slider").value);
      const mins = parseFloat(document.getElementById("mins-slider").value);
      document.getElementById("age-val").innerText = age + " years";
      document.getElementById("mins-val").innerText = mins.toLocaleString() + " mins";

      const groupKey = ROLE_TO_GROUP[currentRole] || "CF_SS";
      const model = PLAYER_MODELS[groupKey];

      let total = model.intercept;
      const breakdownItems = [
        `<span class="px-2.5 py-1 rounded-lg border text-slate-300" style="background-color: #0B0F17; border-color: #1F293D;">Base: €${{model.intercept.toFixed(1)}}M</span>`
      ];

      model.coefficients.forEach(coef => {{
        const feat = coef.feature;
        const weight = coef.weight;
        let featVal = 0;

        if (feat === "age") featVal = age;
        else if (feat === "minutes_played") featVal = mins;
        else {{
          const el = document.getElementById(`${{feat}}-slider`);
          if (el) {{
            featVal = parseFloat(el.value);
            const def = METRIC_DEFINITIONS[feat];
            const valSpan = document.getElementById(`${{feat}}-val`);
            if (valSpan && def) valSpan.innerText = `${{featVal}} ${{def.unit}}`;
          }}
        }}

        const impact = featVal * weight;
        total += impact;

        if (feat !== "minutes_played") {{
          const sign = impact >= 0 ? "+" : "";
          const color = impact >= 0 ? "#34D399" : "#F87171";
          const bg = impact >= 0 ? "rgba(16,185,129,0.15)" : "rgba(239,68,68,0.15)";
          breakdownItems.push(
            `<span class="px-2.5 py-1 rounded-lg border text-[11px]" style="background-color: ${{bg}}; color: ${{color}}; border-color: rgba(255,255,255,0.1);">${{sign}}€${{impact.toFixed(1)}}M (${{feat.replace(/_/g, " ")}})</span>`
          );
        }}
      }});

      total = Math.max(0.5, total);
      document.getElementById("predicted-val").innerText = total.toFixed(1);
      document.getElementById("breakdown-container").innerHTML = breakdownItems.join("");
    }}

    function switchPlayerTab(tab) {{
      if (tab === "predict") {{
        document.getElementById("tab-player-predict").classList.remove("hidden");
        document.getElementById("tab-player-lookup").classList.add("hidden");
        document.getElementById("tab-predict-btn").style.background = "linear-gradient(135deg, #2563EB, #1D4ED8)";
        document.getElementById("tab-predict-btn").classList.add("text-white");
        document.getElementById("tab-predict-btn").classList.remove("text-slate-400");
        document.getElementById("tab-lookup-btn").style.background = "transparent";
        document.getElementById("tab-lookup-btn").classList.remove("text-white");
        document.getElementById("tab-lookup-btn").classList.add("text-slate-400");
      }} else {{
        document.getElementById("tab-player-predict").classList.add("hidden");
        document.getElementById("tab-player-lookup").classList.remove("hidden");
        document.getElementById("tab-lookup-btn").style.background = "linear-gradient(135deg, #2563EB, #1D4ED8)";
        document.getElementById("tab-lookup-btn").classList.add("text-white");
        document.getElementById("tab-lookup-btn").classList.remove("text-slate-400");
        document.getElementById("tab-predict-btn").style.background = "transparent";
        document.getElementById("tab-predict-btn").classList.remove("text-white");
        document.getElementById("tab-predict-btn").classList.add("text-slate-400");
        filterPlayers();
      }}
    }}

    function predictPlayerStats(p) {{
      const groupKey = ROLE_TO_GROUP[p.position] || "CF_SS";
      const model = PLAYER_MODELS[groupKey];
      let val = model.intercept;
      model.coefficients.forEach(coef => {{
        const v = p[coef.feature] !== undefined ? p[coef.feature] : 0;
        val += v * coef.weight;
      }});
      return Math.max(0.5, val);
    }}

    function setListRoleFilter(role) {{
      listRoleFilter = role;
      document.querySelectorAll(".list-role-btn").forEach(btn => {{
        if (btn.dataset.lrole === role) {{
          btn.className = "list-role-btn px-2.5 py-1 rounded-lg font-bold bg-blue-600 text-white";
          btn.style.backgroundColor = "#2563EB";
        }} else {{
          btn.className = "list-role-btn px-2.5 py-1 rounded-lg font-medium text-slate-400";
          btn.style.backgroundColor = "#0B0F17";
        }}
      }});
      filterPlayers();
    }}

    function getRoleSpecificChips(p) {{
      const pos = p.position;
      if (pos === "GK") return `<span class="text-indigo-400 font-semibold">${{p.saves}} Saves</span> • <span class="text-red-400 font-semibold">${{p.penalties_saved}} Pens Saved</span> • <span class="text-green-400 font-semibold">${{p.pass_accuracy}}% Pass</span>`;
      if (pos === "CB") return `<span class="text-orange-400 font-semibold">${{p.duels_won}} Duels</span> • <span class="text-teal-400 font-semibold">${{p.successful_tackles}} Tackles</span> • <span class="text-purple-400 font-semibold">${{p.aerial_duels_won}} Aerials</span>`;
      if (pos === "RB" || pos === "LB") return `<span class="text-blue-400 font-semibold">${{p.assists}} Assists</span> • <span class="text-orange-400 font-semibold">${{p.duels_won}} Duels</span> • <span class="text-teal-400 font-semibold">${{p.successful_tackles}} Tackles</span>`;
      if (pos === "DMF") return `<span class="text-amber-400 font-semibold">${{p.line_breaking_passes}} Line Breaks</span> • <span class="text-pink-400 font-semibold">${{p.balls_recovered}} Recoveries</span> • <span class="text-orange-400 font-semibold">${{p.duels_won}} Duels</span>`;
      if (pos === "CMF") return `<span class="text-amber-400 font-semibold">${{p.line_breaking_passes}} Line Breaks</span> • <span class="text-pink-400 font-semibold">${{p.balls_recovered}} Recoveries</span> • <span class="text-lime-400 font-semibold">${{p.pass_accuracy}}% Pass</span>`;
      if (pos === "AMF") return `<span class="text-sky-400 font-semibold">${{p.chances_created}} Chances Created</span> • <span class="text-emerald-400 font-semibold">${{p.goals}} Goals</span>`;
      if (pos === "RW" || pos === "LW") return `<span class="text-emerald-400 font-semibold">${{p.goals}}G / ${{p.assists}}A</span> • <span class="text-sky-400 font-semibold">${{p.chances_created}} Chances</span> • <span class="text-purple-400 font-semibold">${{p.dribbles_completed}} Dribbles</span>`;
      return `<span class="text-emerald-400 font-semibold">${{p.goals}} Goals</span> • <span class="text-blue-400 font-semibold">${{p.assists}} Assists</span>`;
    }}

    function filterPlayers() {{
      const q = (document.getElementById("player-search").value || "").toLowerCase().trim();
      const selectedTeam = document.getElementById("team-filter").value;
      const list = document.getElementById("players-list");
      list.innerHTML = "";

      const filtered = ALL_PLAYERS.filter(p => {{
        const matchesName = p.player_name.toLowerCase().includes(q) || p.team.toLowerCase().includes(q);
        const matchesTeam = !selectedTeam || p.team === selectedTeam;
        const matchesRole = !listRoleFilter || p.position === listRoleFilter;
        return matchesName && matchesTeam && matchesRole;
      }});

      document.getElementById("player-count").innerText = `Showing ${{filtered.length}} of ${{ALL_PLAYERS.length}} players`;

      if (filtered.length === 0) {{
        list.innerHTML = `<div class="text-sm text-slate-400 p-8 text-center border rounded-xl" style="background-color: #0B0F17; border-color: #1F293D;">No matching players found.</div>`;
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
