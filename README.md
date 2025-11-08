# 🎵 Live Spotify Lyrics

A real-time web application that displays lyrics for whatever you're currently listening to on Spotify. Built with FastAPI backend and React frontend.

## Features

- 🔐 **Spotify OAuth Integration** - Secure login with your Spotify account
- 🎵 **Real-time Track Detection** - Automatically detects what you're playing
- 📝 **Live Lyrics Display** - Shows lyrics for the current track
- 🔄 **Auto-updates** - Lyrics update automatically when the track changes
- 🎨 **Beautiful UI** - Modern, responsive design

## Prerequisites

- Python 3.8+
- Node.js 14+
- Spotify Developer Account (for API credentials)
- Genius API Access Token (for lyrics) - Get one at [Genius API Clients](https://genius.com/api-clients)

## Setup Instructions

### 1. Spotify API Setup

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Note down your **Client ID** and **Client Secret**
4. Add `http://127.0.0.1:8000/callback` to your app's Redirect URIs

### 2. Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Get a Genius API Client Access Token (no OAuth callbacks needed):
   - Go to [Genius API Clients](https://genius.com/api-clients)
   - Sign up/login and create a new API client
   - Click "Generate Access Token" to get your Client Access Token
   - This token allows access to public data (lyrics) without OAuth callbacks
   - Note: You only need OAuth if you want user-specific data, which we don't need for lyrics

5. Create a `.env` file in the `backend` directory:
   ```env
   SPOTIFY_CLIENT_ID=your_spotify_client_id_here
   SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
   SPOTIFY_REDIRECT_URI=http://127.0.0.1:8000/callback
   GENIUS_ACCESS_TOKEN=your_genius_access_token_here
   ```

6. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

   The backend will run on `http://localhost:8000`

### 3. Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm start
   ```

   The frontend will run on `http://localhost:3000`

## Usage

1. Make sure both backend and frontend servers are running
2. Open `http://localhost:3000` in your browser
3. Click "Login with Spotify"
4. Authorize the application
5. Start playing a song on Spotify (desktop app, web player, or mobile)
6. The lyrics will automatically appear!

## How It Works

1. **Authentication**: Uses Spotify OAuth 2.0 to authenticate users
2. **Track Detection**: Polls Spotify's "Currently Playing" API every 2 seconds
3. **Lyrics Fetching**: 
   - First tries Genius API using a **Client Access Token** (simple token, no OAuth callbacks needed)
   - Falls back to Lyrics.ovh (free, no API key needed)
4. **Real-time Updates**: Automatically fetches new lyrics when track changes

**Note on Genius API**: We use a Client Access Token (not OAuth), which is perfect for fetching public lyrics data. You get this token directly from the Genius API Clients page - no OAuth flow or callbacks required!

## Project Structure

```
GUHRL3/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Environment variables (create this)
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── components/       # React components
│   │   └── services/        # API service functions
│   └── package.json         # Node dependencies
└── README.md
```

## API Endpoints

- `GET /login` - Initiate Spotify OAuth flow
- `GET /callback` - Handle OAuth callback
- `GET /currently-playing/{user_id}` - Get currently playing track
- `GET /lyrics/{track_name}/{artist_name}` - Get lyrics for a track

## Troubleshooting

### "No track currently playing"
- Make sure Spotify is actively playing a song
- The track must be playing on a device where you're logged in

### "Lyrics not found"
- Some tracks may not have lyrics available
- Try a popular song to test

### CORS errors
- Make sure the backend is running on port 8000
- Check that CORS is properly configured in `main.py`

### Authentication issues
- Verify your Spotify Client ID and Secret are correct
- Make sure the redirect URI matches exactly in Spotify Dashboard

## Technologies Used

- **Backend**: FastAPI, Python, httpx, lyricsgenius
- **Frontend**: React, JavaScript
- **APIs**: Spotify Web API, Genius API, Lyrics.ovh

## License

This project is open source and available for personal use.

## Notes

- The app requires Spotify Premium for the "Currently Playing" API to work reliably
- Free Spotify accounts may have limited access
- Lyrics availability depends on the lyrics provider's database

