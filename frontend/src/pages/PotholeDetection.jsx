import React, { useState, useEffect, useRef } from 'react';

const PotholeDetection = ({ onBack }) => {
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
    <div className="min-h-screen bg-black flex items-center justify-center p-4">
      <div className="relative w-full max-w-[1600px]">
        {/* Top Status Bar - Same as Dashboard */}
        <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 w-full max-w-5xl">
          <div className="bg-gray-900 border-2 border-gray-700 rounded-t-3xl px-6 py-3">
            <div className="flex items-center justify-between gap-4">
              <button className="flex items-center gap-2 px-4 py-2 rounded-lg text-gray-300">
                <span className="text-xl">🎯</span>
                <div className="text-left">
                  <div className="text-xs text-gray-400">Dashboard</div>
                </div>
              </button>
              <button className="flex items-center gap-2 px-4 py-2 rounded-lg text-gray-300">
                <span className="text-xl">🔋</span>
                <div className="text-left">
                  <div className="text-xs text-gray-400">Battery</div>
                  <div className="text-sm font-bold">46%</div>
                </div>
              </button>
              <button className="flex items-center gap-2 px-4 py-2 rounded-lg text-gray-300">
                <span className="text-xl">❄️</span>
                <div className="text-left">
                  <div className="text-xs text-gray-400">Climate</div>
                  <div className="text-sm font-bold">-12°C</div>
                </div>
              </button>
              <button className="flex items-center gap-2 px-4 py-2 rounded-lg text-gray-300">
                <span className="text-xl">⛽</span>
                <div className="text-left">
                  <div className="text-xs text-gray-400">Fuel</div>
                  <div className="text-sm font-bold">130km</div>
                </div>
              </button>
              <button className="flex items-center gap-2 px-4 py-2 rounded-lg text-gray-300">
                <span className="text-xl">📍</span>
                <div className="text-left">
                  <div className="text-xs text-gray-400">Trip</div>
                  <div className="text-sm font-bold">21341km</div>
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* Main Container - Matching Dashboard Layout */}
        <div className="flex items-center justify-center mt-10">
          <div className="w-full max-w-5xl">
            <div className="bg-gray-900 border-4 border-gray-700 rounded-3xl overflow-hidden p-6" style={{ height: '600px' }}>
              <div className="h-full flex items-center gap-6">
                {/* Center Display - Video Feed (Full Width without gauges) */}
                <div className="flex-1 h-full">
                  <div className="bg-gray-800 border-2 border-gray-600 rounded-2xl h-full overflow-hidden relative">
                    <div className="h-full overflow-y-auto p-4">
                      {/* Header with Home Button */}
                      <div className="mb-3 text-center relative">
                        {/* Home Button */}
                        <button
                          onClick={onBack}
                          className="absolute left-0 top-0 flex items-center gap-2 px-3 py-2 bg-gray-900 hover:bg-gray-700 text-gray-300 hover:text-white rounded-lg transition-colors border border-gray-600"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                          </svg>
                          <span className="text-sm font-semibold">Home</span>
                        </button>
                        <h1 className="text-xl font-bold text-white mb-1">
                          UC 2.5 Pothole Detection
                        </h1>
                        <p className="text-gray-400 text-xs">
                          Real-time detection using deep learning
                        </p>
                      </div>

                      {/* Status Bar */}
                      <div className="bg-gray-900 border border-gray-600 rounded-lg p-2 mb-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-500 animate-pulse' : 'bg-gray-600'}`}></div>
                            <span className="text-xs font-medium text-gray-300">
                              {loading ? loadingMessage : isActive ? 'Camera Active' : 'Camera Inactive'}
                            </span>
                          </div>
                          
                          {isActive && (
                            <div className={`flex items-center gap-2 px-3 py-1 rounded-lg transition-all ${
                              potholeDetected ? 'bg-red-900 border border-red-500' : 'bg-green-900 border border-green-500'
                            }`}>
                              <div className={`w-1.5 h-1.5 rounded-full ${
                                potholeDetected ? 'bg-red-500 animate-pulse' : 'bg-green-500'
                              }`}></div>
                              <span className={`text-xs font-semibold ${
                                potholeDetected ? 'text-red-300' : 'text-green-300'
                              }`}>
                                {potholeDetected ? '⚠️ Pothole!' : '✓ Clear'}
                              </span>
                            </div>
                          )}
                        </div>

                        {error && (
                          <div className="mt-2 p-2 bg-red-900 border border-red-700 rounded-lg">
                            <p className="text-red-300 text-xs">{error}</p>
                            <button 
                              onClick={startDetection}
                              className="mt-1 text-xs text-red-400 hover:text-red-200 underline"
                            >
                              Retry
                            </button>
                          </div>
                        )}
                      </div>

                      {/* Video Feed */}
                      <div className="bg-gray-900 border border-gray-700 rounded-lg overflow-hidden">
                        {loading ? (
                          <div className="bg-black h-96 flex items-center justify-center">
                            <div className="text-center">
                              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-500 mx-auto mb-3"></div>
                              <p className="text-gray-300 text-sm">Starting camera...</p>
                            </div>
                          </div>
                        ) : isActive ? (
                          <div className="relative bg-black">
                            <img
                              ref={imgRef}
                              src={`${API_URL}/video_feed?t=${Date.now()}`}
                              alt="Pothole Detection Feed"
                              className="w-full h-auto"
                              style={{
                                imageRendering: 'auto',
                                maxHeight: '380px',
                                objectFit: 'contain'
                              }}
                              loading="eager"
                              onError={(e) => {
                                console.error('Error loading video feed');
                                setError('Failed to load video feed.');
                              }}
                            />
                            
                            {potholeDetected && (
                              <div className="absolute top-2 right-2 bg-red-600 text-white px-3 py-1.5 rounded-lg shadow-lg animate-pulse">
                                <div className="flex items-center gap-1.5">
                                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                  </svg>
                                  <span className="font-bold text-sm">POTHOLE</span>
                                </div>
                              </div>
                            )}
                            
                            <div className="absolute bottom-2 left-2 bg-black bg-opacity-75 text-white p-2 rounded text-xs border border-gray-700">
                              <div className="flex items-center gap-1.5">
                                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                                <span>Live Detection</span>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="bg-black h-96 flex items-center justify-center">
                            <div className="text-center">
                              <svg
                                className="w-12 h-12 text-gray-600 mx-auto mb-3"
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
                              <p className="text-gray-400 text-sm">Camera Inactive</p>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Info Bar - Same as Dashboard */}
        <div className="mt-8 flex justify-center">
          <div className="bg-gray-900 border-2 border-gray-700 rounded-b-3xl px-8 py-3">
            <div className="flex items-center gap-8 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-gray-300">SYSTEM ONLINE</span>
              </div>
              <div className="text-gray-500">|</div>
              <div className="text-gray-300">
                {new Date().toLocaleTimeString()}
              </div>
              <div className="text-gray-500">|</div>
              <div className="text-gray-300">
                {new Date().toLocaleDateString()}
              </div>
              <div className="text-gray-500">|</div>
              <div className="text-gray-300">
                <span className="text-cyan-400 font-bold">P</span> PARK
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PotholeDetection;
