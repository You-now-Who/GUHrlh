from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import httpx
import os
from typing import Optional
import secrets
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import lyricsgenius

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Spotify Lyrics API")

# CORS middleware to allow React frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Spotify API credentials - should be set as environment variables
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/callback")

# Genius API access token for lyrics
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN", "")

# Initialize Genius client (will be created lazily)
genius_client = None
executor = ThreadPoolExecutor(max_workers=2)

# In-memory storage for access tokens (use Redis or database in production)
user_tokens = {}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None


class CurrentlyPlaying(BaseModel):
    track_name: str
    artist_name: str
    album_name: Optional[str] = None
    album_art: Optional[str] = None
    is_playing: bool
    progress_ms: int
    duration_ms: int


class LyricsResponse(BaseModel):
    lyrics: str
    track_name: str
    artist_name: str


@app.get("/")
async def root():
    return {"message": "Spotify Lyrics API"}


@app.get("/debug/config")
async def debug_config():
    """Debug endpoint to check configuration (without exposing secrets)"""
    return {
        "client_id_set": bool(SPOTIFY_CLIENT_ID),
        "client_secret_set": bool(SPOTIFY_CLIENT_SECRET),
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "genius_token_set": bool(GENIUS_ACCESS_TOKEN),
    }


@app.get("/login")
async def login():
    """Initiate Spotify OAuth flow"""
    if not SPOTIFY_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Spotify Client ID not configured")
    
    if not SPOTIFY_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Spotify Client Secret not configured")
    
    state = secrets.token_urlsafe(32)
    scope = "user-read-currently-playing user-read-playback-state"
    
    # URL encode the redirect URI
    from urllib.parse import quote
    encoded_redirect_uri = quote(SPOTIFY_REDIRECT_URI, safe='')
    
    auth_url = (
        f"https://accounts.spotify.com/authorize?"
        f"response_type=code&"
        f"client_id={SPOTIFY_CLIENT_ID}&"
        f"scope={scope}&"
        f"redirect_uri={encoded_redirect_uri}&"
        f"state={state}"
    )
    
    print(f"[DEBUG] Redirect URI: {SPOTIFY_REDIRECT_URI}")
    print(f"[DEBUG] Auth URL: {auth_url}")
    
    return RedirectResponse(url=auth_url)


@app.get("/callback")
async def callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    """Handle Spotify OAuth callback"""
    print(f"[DEBUG] Callback received - code: {code is not None}, state: {state is not None}, error: {error}")
    
    if error:
        print(f"[ERROR] Spotify OAuth error: {error}")
        return RedirectResponse(
            url=f"http://localhost:3000?error={error}"
        )
    
    if not code:
        print("[ERROR] No authorization code received")
        raise HTTPException(status_code=400, detail="No authorization code received")
    
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print("[ERROR] Spotify credentials not configured")
        raise HTTPException(status_code=500, detail="Spotify credentials not configured")
    
    print(f"[DEBUG] Using redirect URI: {SPOTIFY_REDIRECT_URI}")
    
    async with httpx.AsyncClient() as client:
        # Exchange code for access token
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": SPOTIFY_REDIRECT_URI,
        }
        
        auth_header = f"Basic {__encode_client_credentials(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)}"
        
        print("[DEBUG] Requesting access token from Spotify...")
        response = await client.post(
            "https://accounts.spotify.com/api/token",
            data=token_data,
            headers={"Authorization": auth_header},
        )
        
        print(f"[DEBUG] Token response status: {response.status_code}")
        
        if response.status_code != 200:
            error_detail = response.text
            print(f"[ERROR] Failed to get access token: {error_detail}")
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to get access token: {error_detail}"
            )
        
        token_info = response.json()
        access_token = token_info["access_token"]
        refresh_token = token_info.get("refresh_token")
        
        print("[DEBUG] Access token received, fetching user info...")
        
        # Get user info to create a session
        user_response = await client.get(
            "https://api.spotify.com/v1/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        
        if user_response.status_code != 200:
            print(f"[ERROR] Failed to get user info: {user_response.status_code}")
            raise HTTPException(status_code=400, detail="Failed to get user info")
        
        user_id = user_response.json()["id"]
        print(f"[DEBUG] User authenticated: {user_id}")
        
        user_tokens[user_id] = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": token_info.get("expires_in", 3600),
        }
        
        # Redirect to frontend with token
        frontend_url = f"http://localhost:3000?user_id={user_id}"
        print(f"[DEBUG] Redirecting to frontend: {frontend_url}")
        return RedirectResponse(url=frontend_url)


@app.get("/currently-playing/{user_id}")
async def get_currently_playing(user_id: str):
    """Get the currently playing track for a user"""
    if user_id not in user_tokens:
        raise HTTPException(status_code=401, detail="User not authenticated")
    
    access_token = user_tokens[user_id]["access_token"]
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.spotify.com/v1/me/player/currently-playing",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        
        if response.status_code == 204:
            return {"is_playing": False, "message": "No track currently playing"}
        
        if response.status_code != 200:
            # Try to refresh token if expired
            if response.status_code == 401:
                await __refresh_user_token(user_id)
                access_token = user_tokens[user_id]["access_token"]
                response = await client.get(
                    "https://api.spotify.com/v1/me/player/currently-playing",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to get currently playing track"
                )
        
        data = response.json()
        item = data.get("item")
        
        if not item:
            return {"is_playing": False, "message": "No track currently playing"}
        
        track_name = item["name"]
        artists = [artist["name"] for artist in item["artists"]]
        artist_name = ", ".join(artists)
        album_name = item["album"]["name"]
        album_art = item["album"]["images"][0]["url"] if item["album"]["images"] else None
        
        return CurrentlyPlaying(
            track_name=track_name,
            artist_name=artist_name,
            album_name=album_name,
            album_art=album_art,
            is_playing=data.get("is_playing", False),
            progress_ms=data.get("progress_ms", 0),
            duration_ms=item.get("duration_ms", 0),
        )


@app.get("/lyrics/{track_name}/{artist_name}")
async def get_lyrics(track_name: str, artist_name: str):
    """Get lyrics for a track"""
    # Try LRC (synchronized lyrics) first
    lrc_data = await __get_lrc_lyrics(track_name, artist_name)
    if lrc_data and lrc_data.get("lines"):
        return LyricsResponse(
            lyrics=lrc_data.get("text", ""),
            track_name=track_name,
            artist_name=artist_name,
            synced=True,
            lines=lrc_data.get("lines"),
        )
    
    # Try Genius API
    if GENIUS_ACCESS_TOKEN:
        lyrics = await __get_genius_lyrics(track_name, artist_name)
        if lyrics:
            return LyricsResponse(
                lyrics=lyrics,
                track_name=track_name,
                artist_name=artist_name,
                synced=False,
            )
    
    # Fallback to Lyrics.ovh (free, no API key needed)
    lyrics = await __get_lyrics_ovh(track_name, artist_name)
    
    if not lyrics:
        raise HTTPException(
            status_code=404,
            detail=f"Lyrics not found for {track_name} by {artist_name}"
        )
    
    return LyricsResponse(
        lyrics=lyrics,
        track_name=track_name,
        artist_name=artist_name,
        synced=False,
    )


def __get_genius_lyrics_sync(track_name: str, artist_name: str) -> Optional[str]:
    """Synchronous function to get lyrics from Genius API using lyricsgenius"""
    global genius_client
    
    try:
        # Initialize Genius client if not already done
        if genius_client is None:
            if not GENIUS_ACCESS_TOKEN:
                return None
            genius_client = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)
            # Remove section headers and other metadata from lyrics
            genius_client.remove_section_headers = True
            genius_client.skip_non_songs = True
        
        # Clean up artist name (take first artist if multiple)
        artist = artist_name.split(",")[0].strip()
        
        # Search for the song
        song = genius_client.search_song(track_name, artist)
        
        if song and song.lyrics:
            # Clean up the lyrics - remove metadata at the end
            lyrics = song.lyrics
            # Remove common Genius metadata patterns
            if "You might also like" in lyrics:
                lyrics = lyrics.split("You might also like")[0]
            if "Embed" in lyrics and lyrics.count("Embed") > 1:
                # Remove embed information
                parts = lyrics.split("Embed")
                if len(parts) > 1:
                    lyrics = parts[0].strip()
            
            return lyrics.strip()
        
        return None
    except Exception as e:
        print(f"Error fetching Genius lyrics: {e}")
        return None


async def __get_genius_lyrics(track_name: str, artist_name: str) -> Optional[str]:
    """Get lyrics from Genius API (async wrapper)"""
    try:
        # Run the synchronous lyricsgenius call in a thread pool
        loop = asyncio.get_event_loop()
        lyrics = await loop.run_in_executor(
            executor,
            __get_genius_lyrics_sync,
            track_name,
            artist_name
        )
        return lyrics
    except Exception as e:
        print(f"Error in async Genius lyrics fetch: {e}")
        return None


async def __get_lyrics_ovh(track_name: str, artist_name: str) -> Optional[str]:
    """Get lyrics from Lyrics.ovh (free API)"""
    try:
        async with httpx.AsyncClient() as client:
            # Clean up artist name (take first artist if multiple)
            artist = artist_name.split(",")[0].strip()
            
            url = f"https://api.lyrics.ovh/v1/{artist}/{track_name}"
            response = await client.get(url, timeout=5.0)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("lyrics", "")
            
            return None
    except Exception:
        return None


def __encode_client_credentials(client_id: str, client_secret: str) -> str:
    """Encode client credentials for Basic Auth"""
    import base64
    credentials = f"{client_id}:{client_secret}"
    return base64.b64encode(credentials.encode()).decode()


async def __refresh_user_token(user_id: str):
    """Refresh the access token for a user"""
    if user_id not in user_tokens:
        raise HTTPException(status_code=401, detail="User not authenticated")
    
    refresh_token = user_tokens[user_id].get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token available")
    
    async with httpx.AsyncClient() as client:
        token_data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        
        auth_header = f"Basic {__encode_client_credentials(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)}"
        
        response = await client.post(
            "https://accounts.spotify.com/api/token",
            data=token_data,
            headers={"Authorization": auth_header},
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to refresh token")
        
        token_info = response.json()
        user_tokens[user_id]["access_token"] = token_info["access_token"]
        if "refresh_token" in token_info:
            user_tokens[user_id]["refresh_token"] = token_info["refresh_token"]

