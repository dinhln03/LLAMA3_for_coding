from bs4 import BeautifulSoup
from spotipy.oauth2 import SpotifyOAuth

import requests
import spotipy

SPOTIFY_CLIENT_ID = "YOUR_SPOTIFY_CLIENT_ID"
SPOTIFY_CLIENT_SECRET = "YOUR_SPOTIFY_CLIENT_SECRET"

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri="https://www.example.com",
        scope="playlist-modify-private",
        show_dialog=True,
        cache_path="token.txt"
    )
)

user_id = sp.current_user()["id"]

travel_date = input("Which year do you want to travel to? Type the date in this format YYYY-MM-DD:")
travel_year = travel_date[:4]
billboard_url = f"https://www.billboard.com/charts/hot-100/{travel_date}"
response = requests.get(billboard_url)
soup = BeautifulSoup(response.text, "html.parser")

song_names = [name.getText() for name in soup.select(".chart-element__information__song")]
song_artists = [name.getText() for name in soup.select(".chart-element__information__artist")]

songs = [{
    "artist": song_artists[i],
    "name": song_names[i]
} for i in range(len(song_artists))]

print(songs)

song_urls = []

for song in songs:
    sp_song = sp.search(f"track:{song['name']} year:{travel_year}", type="track")
    try:
        url = sp_song["tracks"]["items"][0]["uri"]
        song_urls.append(url)
    except IndexError:
        print(f"{song['name']} doesn't exist in Spotify. Skipped.")

playlist = sp.user_playlist_create(user=user_id, name=f"{travel_date} Billboard 100", public=False)
sp.playlist_add_items(playlist_id=playlist["id"], items=song_urls)
