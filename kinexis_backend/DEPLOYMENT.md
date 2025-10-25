# Kinexis Deployment Guide

## Architecture
- **Frontend**: Static HTML/CSS/JS deployed to Vercel (kinexis.fit)
- **Backend**: Flask + WebSocket API (needs separate deployment)

## Frontend Deployment to Vercel

### Prerequisites
1. Vercel account (create at vercel.com)
2. Domain kinexis.fit configured in Vercel

### Steps to Deploy Frontend

1. **Login to Vercel**
   ```bash
   vercel login
   ```

2. **Deploy to Vercel**
   ```bash
   vercel --prod
   ```

   When prompted:
   - Set up and deploy: Y
   - Which scope: Select your account
   - Link to existing project? N (first time) or Y (updates)
   - Project name: kinexis
   - Directory: ./kinexis/frontend
   - Override settings? N

3. **Configure Domain**
   - Go to https://vercel.com/dashboard
   - Select your project
   - Go to Settings → Domains
   - Add kinexis.fit
   - Follow DNS configuration instructions

## Backend Deployment Options

Since your backend uses WebSockets (Socket.IO), you need a service that supports persistent connections:

### Option 1: Render (Recommended - Free tier available)
1. Create account at render.com
2. New → Web Service
3. Connect GitHub repo
4. Configure:
   - Name: kinexis-backend
   - Environment: Python 3
   - Build: `pip install -r requirements.txt`
   - Start: `python app.py`
5. Deploy
6. Copy the URL (e.g., https://kinexis-backend.onrender.com)

### Option 2: Railway
1. Create account at railway.app
2. New Project → Deploy from GitHub
3. Select repository
4. Railway auto-detects Python
5. Deploy
6. Generate domain in Settings

### Option 3: Fly.io
1. Install flyctl CLI
2. Run `fly launch` in backend directory
3. Follow prompts
4. Deploy with `fly deploy`

## Update Frontend Configuration

After deploying backend, update `kinexis/frontend/config.js`:

```javascript
const CONFIG = {
  BACKEND_URL: 'https://YOUR-BACKEND-URL.com', // Replace with actual URL
  WS_URL: 'wss://YOUR-BACKEND-URL.com'
};
```

Then redeploy frontend:
```bash
vercel --prod
```

## Environment Variables

For production, set these in your backend hosting service:
- `FLASK_ENV=production`
- `PORT=5000` (or as required by host)

## Testing Production

1. Visit https://kinexis.fit
2. Click "Start Session"
3. Verify connection to backend
4. Test PT exercise tracking

## Troubleshooting

- **CORS errors**: Update `app.py` to include your frontend domain in CORS origins
- **WebSocket connection failed**: Ensure backend supports WebSocket upgrade
- **404 on routes**: Check vercel.json routing configuration