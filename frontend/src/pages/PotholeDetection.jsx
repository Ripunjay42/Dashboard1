import React, { useState, useEffect, useRef } from 'react';

const PotholeDetection = () => {
  const [isActive, setIsActive] = useState(false);
  const [loading, setLoading] = useState(true);  // Start with loading true
  const [loadingMessage, setLoadingMessage] = useState('Initializing camera...');
  const [error, setError] = useState(null);
  const [potholeDetected, setPotholeDetected] = useState(false);
  const imgRef = useRef(null);
  const statusIntervalRef = useRef(null);
  const API_URL = 'http://localhost:5000/api/pothole';

  useEffect(() => {
    // Auto-start detection when component mounts (camera is pre-warmed on backend)
    startDetection();
    
    // Cleanup on unmount
    return () => {
      if (isActive) {
        stopDetection();
      }
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
    };
  }, []);

  useEffect(() => {
    // Poll for pothole detection status when active
    if (isActive) {
      statusIntervalRef.current = setInterval(async () => {
        try {
          const response = await fetch(`${API_URL}/status`);
          const data = await response.json();
          setPotholeDetected(data.pothole_detected);
        } catch (err) {
          console.error('Error checking status:', err);
        }
      }, 500); // Check every 500ms (reduced from 200ms to avoid too many requests)
    } else {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
      setPotholeDetected(false);
    }
    
    return () => {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
    };
  }, [isActive]);

  const startDetection = async () => {
    setLoading(true);
    setError(null);
    try {
      setLoadingMessage('Starting detection...');
      
      const response = await fetch(`${API_URL}/start`, {
        method: 'POST',
      });
      const data = await response.json();
      
      if (data.status === 'success') {
        setIsActive(true);
        setLoading(false);
        setLoadingMessage('');
      } else {
        setError(data.message || 'Failed to start detection');
        setLoading(false);
        setLoadingMessage('');
      }
    } catch (err) {
      setError('Error connecting to server. Make sure Flask backend is running on port 5000.');
      setLoading(false);
      setLoadingMessage('');
    }
  };

  const stopDetection = async () => {
    try {
      await fetch(`${API_URL}/stop`, {
        method: 'POST',
      });
      setIsActive(false);
      setPotholeDetected(false);
    } catch (err) {
      console.error('Error stopping detection:', err);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            UC 2.5 Pothole Identification System
          </h1>
          <p className="text-gray-600">
            Real-time pothole detection using deep learning
          </p>
        </div>

        {/* Status Bar */}
        <div className="bg-white rounded-lg shadow-md p-4 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className={`w-3 h-3 rounded-full ${isActive ? 'bg-green-500 animate-pulse' : 'bg-gray-300'}`}></div>
              <span className="text-sm font-medium text-gray-700">
                {loading ? loadingMessage : isActive ? 'Camera Active' : 'Camera Inactive'}
              </span>
            </div>
            
            {isActive && (
              <div className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                potholeDetected ? 'bg-red-100 border-2 border-red-400' : 'bg-green-100 border-2 border-green-300'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  potholeDetected ? 'bg-red-500 animate-pulse' : 'bg-green-500'
                }`}></div>
                <span className={`text-sm font-semibold ${
                  potholeDetected ? 'text-red-700' : 'text-green-700'
                }`}>
                  {potholeDetected ? '⚠️ Pothole Detected!' : '✓ Road Clear'}
                </span>
              </div>
            )}
          </div>

          {error && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-700 text-sm">{error}</p>
              <button 
                onClick={startDetection}
                className="mt-2 text-sm text-red-600 hover:text-red-800 underline"
              >
                Retry Connection
              </button>
            </div>
          )}
        </div>

        {/* Video Feed */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Live Camera Feed</h2>
          
          {loading ? (
            <div className="bg-gray-100 rounded-lg h-80 max-w-3xl mx-auto flex items-center justify-center">
              <div className="text-center">
                <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-teal-500 mx-auto mb-4"></div>
                <p className="text-gray-500 text-lg">Starting camera...</p>
                <p className="text-gray-400 text-sm mt-2">Please wait</p>
              </div>
            </div>
          ) : isActive ? (
            <div className="relative bg-gray-900 rounded-lg overflow-hidden max-w-3xl mx-auto">
              <img
                ref={imgRef}
                src={`${API_URL}/video_feed?t=${Date.now()}`}
                alt="Pothole Detection Feed"
                className="w-full h-auto"
                style={{
                  imageRendering: 'auto',
                  maxWidth: '100%',
                  maxHeight: '500px',
                  height: 'auto',
                  objectFit: 'contain'
                }}
                loading="eager"
                onError={(e) => {
                  console.error('Error loading video feed');
                  setError('Failed to load video feed. Check if camera is connected.');
                }}
              />
              
              {/* Detection Alert - Only show when pothole detected */}
              {potholeDetected && (
                <div className="absolute top-4 right-4 bg-red-600 bg-opacity-95 text-white px-4 py-3 rounded-lg shadow-lg animate-pulse">
                  <div className="flex items-center gap-2">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <span className="font-bold text-lg">POTHOLE</span>
                  </div>
                </div>
              )}
              
              {/* Info Legend */}
              <div className="absolute bottom-4 left-4 bg-black bg-opacity-75 text-white p-3 rounded-lg text-xs">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-3 h-3 bg-red-500"></div>
                  <span>Detected Pothole</span>
                </div>
                <div className="flex items-center gap-2 text-gray-300 mt-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span>Live • Real-time Detection</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-gray-100 rounded-lg h-80 max-w-3xl mx-auto flex items-center justify-center">
              <div className="text-center">
                <svg
                  className="w-16 h-16 text-gray-400 mx-auto mb-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
                <p className="text-gray-500 text-lg">No active video feed</p>
                <p className="text-gray-400 text-sm mt-2">Click "Start Detection" to begin</p>
              </div>
            </div>
          )}
        </div>

        {/* Info Section */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="font-semibold text-gray-700 mb-2">Model</h3>
            <p className="text-sm text-gray-600">U-Net with ResNet50 Encoder</p>
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="font-semibold text-gray-700 mb-2">Detection Method</h3>
            <p className="text-sm text-gray-600">Semantic Segmentation</p>
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="font-semibold text-gray-700 mb-2">Framework</h3>
            <p className="text-sm text-gray-600">PyTorch + OpenCV</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PotholeDetection;
