# src/test_api.py
import os
import requests
api_key = os.getenv("ODDS_API_KEY")
if not api_key:
    print("ODDS_API_KEY not set")
    exit(1)
# Simple test endpoint (adjust sport/region if needed)
url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={api_key}&regions=uk&markets=h2h"
try:
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        print("Odds API OK")
        exit(0)
    else:
        print("Odds API error:", r.status_code, r.text)
        exit(2)
except Exception as e:
    print("Odds API request failed:", e)
    exit(3)
