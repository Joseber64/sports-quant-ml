import os
import requests

def main():
    api_key = os.getenv("ODDS_API_KEY")
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={api_key}"
    response = requests.get(url)

    try:
        data = response.json()
    except Exception:
        print("Error al decodificar respuesta:", response.text)
        return

    print("Respuesta de Odds API:")
    print(data)

if __name__ == "__main__":
    main()
