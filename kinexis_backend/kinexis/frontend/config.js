// Configuration for backend API endpoint
// In production, this will point to your deployed backend
// For now, using localhost for development

const CONFIG = {
  // Backend API URL
  BACKEND_URL: window.location.hostname === 'localhost'
    ? 'http://localhost:5001'
    : 'https://kinexis-backend.onrender.com',

  // WebSocket URL for real-time communication
  WS_URL: window.location.hostname === 'localhost'
    ? 'ws://localhost:5001'
    : 'wss://kinexis-backend.onrender.com'
};

// Make config globally available
window.KINEXIS_CONFIG = CONFIG;