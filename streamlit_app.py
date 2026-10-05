"""
PremPredict - Player Scouting & Similarity Dashboard
---------------------------------------------------
Interactive Streamlit scouting suite powered by Scikit-Learn:
- Cosine Similarity player matching (e.g. Find players similar to Bukayo Saka)
- K-Means tactical style clustering & archetypes
- 2D PCA Style Space projection & Matplotlib radar comparison charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from model.scouting_engine import PlayerScoutingEngine

# Page Configuration
st.set_page_config(
    page_title="PremPredict | Player Scouting & Similarity Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Obsidian & Electric Blue Theme)
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0B0F17;
        color: #F8FAFC;
    }
    
    /* Headers & Text */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F293D;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #3B82F6, #2563EB);
        border: none;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }
    
    /* Cards / Containers */
    .scout-card {
        background: #111827;
        border: 1px solid #1F293D;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-blue {
        background-color: rgba(37, 99, 235, 0.2);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .badge-emerald {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    """Cache scouting engine initialization."""
    return PlayerScoutingEngine()


engine = load_engine()
all_players = engine.get_all_players()

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/football2--v1.png", width=64)
    st.title("PremPredict Scouting")
    st.caption("AI-Powered Player Style & Similarity Engine")
    st.markdown("---")
    
    # Target Player Selection
    default_index = all_players.index("Bukayo Saka") if "Bukayo Saka" in all_players else 0
    selected_player_name = st.selectbox(
        "🎯 Select Target Player",
        options=all_players,
        index=default_index,
        help="Choose any Premier League player to discover their statistical twins."
    )
    
    top_n = st.slider("Number of Similar Matches", min_value=3, max_value=15, value=5, step=1)
    
    st.markdown("### ⚙️ Search Filters")
    same_position = st.checkbox("Strict Same Position Only", value=False, help="Match only players in identical registered positions (e.g. RW to RW)")
    exclude_team = st.checkbox("Exclude Teammates", value=False, help="Filter out players currently at the same club")
    
    st.markdown("---")
    st.markdown("### 🧠 ML Tech Stack")
    st.markdown("""
    - **Similarity**: Cosine Vector Similarity
    - **Clustering**: K-Means ($k=6$ Archetypes)
    - **Dimensionality**: 2D PCA Projection
    - **Features**: 12 Tactical & Output Metrics
    """)
    st.caption("PremPredict v3.0 | 441 Premier League Players")


# ----------------- MAIN CONTENT -----------------
target_player = engine.get_player(selected_player_name)

if target_player is not None:
    # Header Banner
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #111827 0%, #16243E 100%); padding: 24px; border-radius: 16px; border: 1px solid #1F293D; margin-bottom: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div>
                <span class="badge badge-blue">{target_player['team']}</span>
                <span class="badge badge-emerald">{target_player['position']}</span>
                <span class="badge badge-blue">Age {target_player['age']}</span>
                <h1 style="margin: 8px 0 4px 0; font-size: 2.2rem; letter-spacing: -0.5px;">{target_player['player_name']}</h1>
                <p style="margin: 0; color: #94A3B8; font-size: 1rem;">
                    Tactical Archetype: <strong style="color: #38BDF8;">{target_player['playing_style']}</strong>
                </p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Transfer Valuation</div>
                <div style="font-size: 2.4rem; font-weight: 800; color: #34D399;">€{target_player['market_value_eur_m']}M</div>
                <div style="font-size: 0.85rem; color: #64748B;">{target_player['minutes_played']} Minutes Played</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Target Player Quick Stats Grid
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Goals", int(target_player['goals']))
    with col2:
        st.metric("Assists", int(target_player['assists']))
    with col3:
        st.metric("Chances Created", int(target_player['chances_created']))
    with col4:
        st.metric("Dribbles", int(target_player['dribbles_completed']))
    with col5:
        st.metric("Balls Recovered", int(target_player['balls_recovered']))
    with col6:
        st.metric("Pass Accuracy", f"{target_player['pass_accuracy']}%")

    st.markdown("---")

    # Get Similar Players
    similar_df = engine.find_similar_players(
        selected_player_name,
        top_n=top_n,
        same_position_only=same_position,
        exclude_same_team=exclude_team
    )

    # Tabs for Dashboard Views
    tab_matches, tab_comparison, tab_clusters, tab_pca = st.tabs([
        "👥 Top Similar Players",
        "📊 Statistical Radar Comparison",
        "🧭 Playing Style Archetypes (K-Means)",
        "🗺️ 2D Tactical Style Space (PCA)"
    ])

    # ---------------- TAB 1: SIMILAR PLAYERS ----------------
    with tab_matches:
        st.markdown(f"### Closest Statistical Matches to **{target_player['player_name']}**")
        st.caption(f"Ranked by Cosine Similarity across 12 standardized tactical metrics ({'All Positions' if not same_position else target_player['position'] + ' Only'}).")
        
        for idx, (_, row) in enumerate(similar_df.iterrows(), 1):
            pct = row['similarity_pct']
            bar_color = "#34D399" if pct >= 95 else "#38BDF8" if pct >= 90 else "#F59E0B"
            
            with st.container():
                st.markdown(f"""
                <div class="scout-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 8px;">
                        <div>
                            <span style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-right: 8px;">#{idx} {row['player_name']}</span>
                            <span class="badge badge-blue">{row['team']}</span>
                            <span class="badge badge-emerald">{row['position']}</span>
                            <span style="font-size: 0.85rem; color: #94A3B8; margin-left: 6px;">Age {row['age']}</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 1.4rem; font-weight: 800; color: {bar_color};">{pct}%</span>
                            <span style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; margin-left: 4px;">Similarity</span>
                        </div>
                    </div>
                    <div style="background-color: #1F293D; border-radius: 9999px; height: 8px; width: 100%; margin-bottom: 12px; overflow: hidden;">
                        <div style="background-color: {bar_color}; height: 100%; width: {pct}%; border-radius: 9999px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; flex-wrap: wrap; gap: 8px;">
                        <span><strong>Archetype:</strong> {row['playing_style']}</span>
                        <span><strong>Market Value:</strong> €{row['market_value_eur_m']}M</span>
                        <span><strong>Goals:</strong> {row['goals']}</span>
                        <span><strong>Assists:</strong> {row['assists']}</span>
                        <span><strong>Chances:</strong> {row['chances_created']}</span>
                        <span><strong>Tackles/Duels:</strong> {row['successful_tackles'] + row['duels_won']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ---------------- TAB 2: RADAR COMPARISON ----------------
    with tab_comparison:
        st.markdown("### Head-to-Head Tactical Comparison")
        st.caption("Compare the target player against any similar candidate using normalized tactical radar charts.")
        
        candidate_names = similar_df['player_name'].tolist()
        comp_player_name = st.selectbox("Select Player to Compare Against:", options=candidate_names, index=0)
        comp_player = engine.get_player(comp_player_name)
        
        if comp_player is not None:
            # Metrics to display on radar
            radar_features = [
                ('goals', 'Goals'),
                ('assists', 'Assists'),
                ('chances_created', 'Chances Created'),
                ('dribbles_completed', 'Dribbles'),
                ('balls_recovered', 'Recoveries'),
                ('successful_tackles', 'Tackles'),
                ('duels_won', 'Duels Won'),
                ('pass_accuracy', 'Pass Acc %')
            ]
            
            categories = [f[1] for f in radar_features]
            raw_keys = [f[0] for f in radar_features]
            
            # Percentile/Normalized Scaling (relative to max in entire dataset)
            p1_values = []
            p2_values = []
            for key in raw_keys:
                max_val = max(engine.df[key].max(), 1.0)
                p1_values.append((target_player[key] / max_val) * 100)
                p2_values.append((comp_player[key] / max_val) * 100)
            
            # Close the polygon loop
            p1_values += p1_values[:1]
            p2_values += p2_values[:1]
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            # Matplotlib Radar Plot
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), facecolor='#111827')
            ax.set_facecolor('#111827')
            
            # Draw one axe per variable and add labels
            plt.xticks(angles[:-1], categories, color='#94A3B8', size=10, weight='bold')
            ax.tick_params(axis='x', pad=15)
            
            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="#475569", size=8)
            plt.ylim(0, 105)
            ax.grid(color='#1F293D', linestyle='--', linewidth=0.8)
            
            # Plot Player 1 (Target)
            ax.plot(angles, p1_values, color='#3B82F6', linewidth=2.5, linestyle='solid', label=f"{target_player['player_name']} (Target)")
            ax.fill(angles, p1_values, color='#3B82F6', alpha=0.25)
            
            # Plot Player 2 (Candidate)
            ax.plot(angles, p2_values, color='#10B981', linewidth=2.5, linestyle='solid', label=f"{comp_player['player_name']} (Candidate)")
            ax.fill(angles, p2_values, color='#10B981', alpha=0.25)
            
            ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')
            
            r_col1, r_col2 = st.columns([1.2, 1])
            with r_col1:
                st.pyplot(fig)
            with r_col2:
                st.markdown(f"#### Stat Breakdown: {target_player['player_name']} vs {comp_player['player_name']}")
                comp_table = []
                for key, label in radar_features:
                    comp_table.append({
                        "Metric": label,
                        target_player['player_name']: target_player[key],
                        comp_player['player_name']: comp_player[key],
                        "Delta": round(target_player[key] - comp_player[key], 1)
                    })
                st.dataframe(pd.DataFrame(comp_table), use_container_width=True, hide_index=True)

    # ---------------- TAB 3: K-MEANS CLUSTERS ----------------
    with tab_clusters:
        st.markdown("### K-Means Playing Style Clusters & Archetypes")
        st.caption("Unsupervised K-Means clustering algorithm partitions 441 Premier League players into 6 distinct tactical playing styles based on multidimensional feature profiles.")
        
        clusters_summary = engine.get_clusters_summary()
        for _, c_row in clusters_summary.iterrows():
            c_id = c_row['cluster']
            is_target_cluster = (c_id == target_player['cluster'])
            highlight_border = "border: 2px solid #3B82F6;" if is_target_cluster else "border: 1px solid #1F293D;"
            badge_text = "⭐ TARGET PLAYER'S STYLE" if is_target_cluster else f"Cluster {c_id}"
            
            st.markdown(f"""
            <div class="scout-card" style="{highlight_border}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <h4 style="margin: 0; color: #FFFFFF;">{c_row['playing_style']}</h4>
                    <span class="badge {'badge-blue' if is_target_cluster else 'badge-emerald'}">{badge_text}</span>
                </div>
                <div style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 6px;">
                    <strong>Total Players in Cluster:</strong> {c_row['player_count']} | 
                    <strong>Avg Market Value:</strong> €{c_row['avg_market_value']:.1f}M
                </div>
                <div style="font-size: 0.85rem; color: #CBD5E1;">
                    <strong>Notable Players:</strong> {c_row['top_players']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---------------- TAB 4: 2D PCA PROJECTION ----------------
    with tab_pca:
        st.markdown("### 2D Tactical Style Space (Principal Component Analysis)")
        st.caption("Dimensionality reduction compresses 12 statistical variables down into 2 orthogonal axes representing Attacking Output (PC1) vs Defensive Solidity & Game Control (PC2).")
        
        pca_fig, ax_pca = plt.subplots(figsize=(10, 6), facecolor='#111827')
        ax_pca.set_facecolor('#111827')
        
        # Color palette for clusters
        palette = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']
        
        for c in range(engine.n_clusters):
            c_data = engine.df[engine.df['cluster'] == c]
            label = engine.cluster_labels.get(c, f"Cluster {c}")
            ax_pca.scatter(
                c_data['pca_x'], c_data['pca_y'],
                label=label,
                alpha=0.45,
                s=40,
                color=palette[c % len(palette)],
                edgecolors='none'
            )
            
        # Highlight similar players
        ax_pca.scatter(
            similar_df['pca_x'], similar_df['pca_y'],
            color='#34D399',
            s=120,
            edgecolors='#FFFFFF',
            linewidths=1.5,
            label='Top Similar Matches',
            zorder=4
        )
        
        # Highlight target player
        ax_pca.scatter(
            [target_player['pca_x']], [target_player['pca_y']],
            color='#EF4444',
            s=220,
            marker='*',
            edgecolors='#FFFFFF',
            linewidths=1.5,
            label=f"Target: {target_player['player_name']}",
            zorder=5
        )
        
        ax_pca.set_title("Premier League Player Archetypes Map (PCA Space)", color='#FFFFFF', fontsize=12, pad=12, weight='bold')
        ax_pca.set_xlabel("Principal Component 1 (Attacking Threat & Output)", color='#94A3B8', fontsize=10)
        ax_pca.set_ylabel("Principal Component 2 (Defensive Workrate & Duels)", color='#94A3B8', fontsize=10)
        ax_pca.tick_params(colors='#64748B')
        for spine in ax_pca.spines.values():
            spine.set_color('#1F293D')
        ax_pca.grid(color='#1F293D', linestyle='--', linewidth=0.5, alpha=0.7)
        ax_pca.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC', loc='upper right', fontsize=8)
        
        st.pyplot(pca_fig)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748B; font-size: 0.85rem;'>PremPredict Intelligence Suite • Built with Scikit-Learn, Streamlit, and Pandas • 2026</div>", unsafe_allow_html=True)
