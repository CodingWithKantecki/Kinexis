// Configuration for backend API endpoint
// In production, this will point to your deployed backend
// For now, using localhost for development

const CONFIG = {
  // Update this to your deployed backend URL when ready
  // For example: 'https://kinexis-backend.onrender.com' or 'https://api.kinexis.fit'
  BACKEND_URL: window.location.hostname === 'localhost'
    ? 'http://localhost:5001'
    : 'https://kinexis-backend.onrender.com', // Replace with your actual backend URL

  // WebSocket URL (usually same as backend)
  WS_URL: window.location.hostname === 'localhost'
    ? 'ws://localhost:5001'
    : 'wss://kinexis-backend.onrender.com' // Replace with your actual backend URL
};

// Make config globally available
window.KINEXIS_CONFIG = CONFIG;