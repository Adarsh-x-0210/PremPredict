"""
PremPredict - Player Scouting & Similarity Recommender Engine
-------------------------------------------------------------
Identifies statistically similar players across the Premier League
using cosine similarity, Euclidean distance, PCA, and K-Means clustering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class PlayerScoutingEngine:
    def __init__(self, data_path: str = "data/pl_players_dataset.csv", n_clusters: int = 6):
        self.data_path = data_path
        self.n_clusters = n_clusters
        self.df = pd.read_csv(data_path)
        
        # Core statistical feature vectors (per-match normalized or totals)
        self.stat_cols = [
            'goals', 'assists', 'chances_created', 'dribbles_completed',
            'balls_recovered', 'line_breaking_passes', 'pass_accuracy',
            'duels_won', 'aerial_duels_won', 'successful_tackles',
            'saves', 'penalties_saved'
        ]
        
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = None
        self._fit_engine()

    def _fit_engine(self):
        """Fit scaler, K-Means clustering, and 2D PCA representation."""
        # Scale the stats matrix
        self.X_scaled = self.scaler.fit_transform(self.df[self.stat_cols])
        
        # Fit K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = self.kmeans.fit_predict(self.X_scaled)
        
        # Label clusters with human-readable tactical archetypes
        self.cluster_labels = self._generate_cluster_labels()
        self.df['playing_style'] = self.df['cluster'].map(self.cluster_labels)
        
        # Fit 2D PCA for visual mapping
        self.pca = PCA(n_components=2, random_state=42)
        pca_coords = self.pca.fit_transform(self.X_scaled)
        self.df['pca_x'] = pca_coords[:, 0]
        self.df['pca_y'] = pca_coords[:, 1]
        
        # Precompute full cosine similarity matrix
        self.similarity_matrix = cosine_similarity(self.X_scaled)

    def _generate_cluster_labels(self) -> dict:
        """Assign intuitive tactical labels to clusters based on feature centroids."""
        cluster_means = self.df.groupby('cluster')[self.stat_cols].mean()
        labels = {}
        for c in range(self.n_clusters):
            means = cluster_means.loc[c]
            if means['saves'] > 20:
                labels[c] = "Goalkeeper / Shot Stopper"
            elif means['successful_tackles'] > 25 or means['aerial_duels_won'] > 40:
                labels[c] = "Defensive Stopper / Ball Winner"
            elif means['chances_created'] > 25 or means['dribbles_completed'] > 25:
                labels[c] = "Creative Winger / Attacking Spark"
            elif means['goals'] > 8:
                labels[c] = "Goal Poacher / Lethal Finisher"
            elif means['balls_recovered'] > 60 or means['line_breaking_passes'] > 50:
                labels[c] = "Playmaker / Deep Midfield Engine"
            else:
                labels[c] = "Box-to-Box / Balanced Operator"
        return labels

    def get_player(self, player_name: str):
        """Find player row by name (case-insensitive substring)."""
        match = self.df[self.df['player_name'].str.lower() == player_name.lower()]
        if match.empty:
            match = self.df[self.df['player_name'].str.contains(player_name, case=False, na=False)]
        return match.iloc[0] if not match.empty else None

    def find_similar_players(
        self,
        player_name: str,
        top_n: int = 5,
        same_position_only: bool = False,
        exclude_same_team: bool = False
    ) -> pd.DataFrame:
        """
        Find the most statistically similar players to a target player.
        Returns a DataFrame with similarity scores (0% to 100%), stats, and styling.
        """
        target = self.get_player(player_name)
        if target is None:
            raise ValueError(f"Player '{player_name}' not found in Premier League dataset.")
        
        target_idx = target.name
        sim_scores = self.similarity_matrix[target_idx]
        
        # Build results DataFrame
        results = self.df.copy()
        results['similarity_pct'] = (sim_scores * 100).round(1)
        
        # Exclude the target player himself
        results = results[results.index != target_idx]
        
        # Optional filters
        if same_position_only:
            results = results[results['position'] == target['position']]
        if exclude_same_team:
            results = results[results['team'] != target['team']]
            
        results = results.sort_values(by='similarity_pct', ascending=False)
        return results.head(top_n)

    def get_all_players(self):
        """Return list of all player names for dropdowns/search."""
        return sorted(self.df['player_name'].tolist())

    def get_clusters_summary(self):
        """Summary of each tactical style cluster."""
        return self.df.groupby(['cluster', 'playing_style']).agg(
            player_count=('player_name', 'count'),
            top_players=('player_name', lambda x: ", ".join(x.head(4))),
            avg_market_value=('market_value_eur_m', 'mean')
        ).reset_index()


if __name__ == "__main__":
    engine = PlayerScoutingEngine()
    print(f"Scouting Engine initialized with {len(engine.df)} players.")
    print("\nTest: Top 5 players most similar to Bukayo Saka:")
    sims = engine.find_similar_players("Bukayo Saka", top_n=5)
    for _, row in sims.iterrows():
        print(f"  - {row['player_name']} ({row['team']} | {row['position']}): {row['similarity_pct']}% match | Style: {row['playing_style']}")
