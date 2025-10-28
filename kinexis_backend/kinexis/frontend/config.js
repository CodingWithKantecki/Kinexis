// Configuration for backend API endpoint
// Automatically detects the environment and uses the appropriate backend

const CONFIG = {
  // Backend API URL
  // Use Render backend by default (production)
  // Only use localhost when explicitly running on localhost
  BACKEND_URL: (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://localhost:5001'
    : 'https://kinexis-backend.onrender.com',

  // WebSocket URL for real-time communication
  WS_URL: (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'ws://localhost:5001'
    : 'wss://kinexis-backend.onrender.com'
};

// Make config globally available
window.KINEXIS_CONFIG = CONFIG;

// Log the configuration for debugging
console.log('Kinexis Config:', {
  hostname: window.location.hostname,
  protocol: window.location.protocol,
  backendURL: CONFIG.BACKEND_URL,
  wsURL: CONFIG.WS_URL
});