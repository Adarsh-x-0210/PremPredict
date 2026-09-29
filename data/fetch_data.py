"""
Data Fetcher Module for Premier League Statistics
Uses the `requests` library to interact with football-data.org API
and provides utilities to load or enrich player statistics.
"""

import os
import requests
import pandas as pd

BASE_URL = "https://api.football-data.org/v4"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(DATA_DIR, "pl_players_dataset.csv")


def fetch_premier_league_scorers(api_key: str = None, limit: int = 20) -> list:
    """
    Fetch top scorers from football-data.org using the requests library.
    
    Args:
        api_key: Optional API key. If not provided, reads from FOOTBALL_DATA_API_KEY env var.
        limit: Number of top scorers to retrieve (max 50).
        
    Returns:
        List of player stats dictionaries or empty list if request fails.
    """
    token = api_key or os.environ.get("FOOTBALL_DATA_API_KEY", "")
    headers = {"X-Auth-Token": token} if token else {}
    
    endpoint = f"{BASE_URL}/competitions/PL/scorers"
    params = {"limit": limit}
    
    print(f"[requests] GET {endpoint} (limit={limit})...")
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            scorers = data.get("scorers", [])
            print(f"[requests] Successfully fetched {len(scorers)} scorers from football-data.org!")
            return scorers
        elif response.status_code == 403:
            print("[requests] HTTP 403: API token required or invalid for football-data.org.")
        elif response.status_code == 429:
            print("[requests] HTTP 429: Rate limit reached. Free tier allows 10 requests/minute.")
        else:
            print(f"[requests] HTTP {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[requests] Connection error: {e}")
        
    return []


def fetch_premier_league_teams(api_key: str = None) -> list:
    """
    Fetch Premier League teams and squad members using requests.
    """
    token = api_key or os.environ.get("FOOTBALL_DATA_API_KEY", "")
    headers = {"X-Auth-Token": token} if token else {}
    
    endpoint = f"{BASE_URL}/competitions/PL/teams"
    print(f"[requests] GET {endpoint}...")
    try:
        response = requests.get(endpoint, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            teams = data.get("teams", [])
            print(f"[requests] Successfully fetched {len(teams)} teams!")
            return teams
        else:
            print(f"[requests] HTTP {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[requests] Connection error: {e}")
        
    return []


def parse_scorers_to_df(scorers_data: list) -> pd.DataFrame:
    """
    Parses raw scorers JSON from football-data.org into a pandas DataFrame.
    """
    records = []
    for item in scorers_data:
        player = item.get("player", {})
        team = item.get("team", {})
        
        records.append({
            "player_name": player.get("name"),
            "team": team.get("shortName") or team.get("name"),
            "position": player.get("position", "Forward"),
            "goals": item.get("goals", 0),
            "assists": item.get("assists") or 0,
            "played_matches": item.get("playedMatches", 0),
        })
    return pd.DataFrame(records)


def load_dataset(csv_path: str = None) -> pd.DataFrame:
    """
    Loads the Premier League dataset into a pandas DataFrame.
    If csv_path is None, loads the default curated CSV.
    """
    path = csv_path or DEFAULT_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")
        
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    print("Testing data loader...")
    df = load_dataset()
    print(f"Loaded {len(df)} players from {DEFAULT_CSV}")
    print(df.head(5))
