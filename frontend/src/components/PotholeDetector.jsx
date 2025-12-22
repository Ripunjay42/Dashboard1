import React, { useState, useEffect, useRef } from 'react';

const PotholeDetector = ({ onBack }) => {
  const [isActive, setIsActive] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingMessage, setLoadingMessage] = useState('Initializing cameras...');
  const [error, setError] = useState(null);
  const [topPotholeDetected, setTopPotholeDetected] = useState(false);
  const [bottomPotholeDetected, setBottomPotholeDetected] = useState(false);
  const [feedKey, setFeedKey] = useState(Date.now()); // Cache buster for video feeds
  const [viewMode, setViewMode] = useState('top'); // 'top', 'bottom', or 'both' - default to top for less Jetson load
  const statusIntervalRef = useRef(null);
  const topImgRef = useRef(null);
  const bottomImgRef = useRef(null);
  const API_URL = 'http://localhost:5000/api/pothole';
  const hasStartedRef = useRef(false);
  const isMountedRef = useRef(true); // Track if component is still mounted

  useEffect(() => {
    isMountedRef.current = true;
    
    // Start detection only once when component mounts
    if (!hasStartedRef.current) {
      hasStartedRef.current = true;
      startDetection('top'); // Start with top camera by default
    }
    
    // Cleanup on unmount
    return () => {
      isMountedRef.current = false;
      
      // Clear streams FIRST to release browser connections
      if (topImgRef.current) {
        topImgRef.current.src = '';
      }
      if (bottomImgRef.current) {
        bottomImgRef.current.src = '';
      }
      
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
        statusIntervalRef.current = null;
      }
      
      // Stop detection on backend (fire and forget - Dashboard handles proper cleanup)
      fetch(`${API_URL}/stop`, { method: 'POST' }).catch(() => {});
    };
  }, []);

  useEffect(() => {
    // Poll for pothole detection status when active
    if (isActive) {
      statusIntervalRef.current = setInterval(async () => {
        try {
          const response = await fetch(`${API_URL}/status`);
          const data = await response.json();
          setTopPotholeDetected(data.top_pothole_detected || false);
          setBottomPotholeDetected(data.bottom_pothole_detected || false);
        } catch (err) {
          console.error('Error checking status:', err);
        }
      }, 500);
    } else {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
      setTopPotholeDetected(false);
      setBottomPotholeDetected(false);
    }
    
    return () => {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
    };
  }, [isActive]);

  const startDetection = async (cameraMode = 'top') => {
    setLoading(true);
    setError(null);
    try {
      setLoadingMessage(`Starting ${cameraMode === 'both' ? 'dual' : cameraMode} camera detection...`);
      
      const response = await fetch(`${API_URL}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera: cameraMode })
      });
      const data = await response.json();
      
      if (!isMountedRef.current) return; // Component unmounted during fetch
      
      if (data.status === 'success') {
        // Generate new cache key to force fresh feed connection
        setFeedKey(Date.now());
        setViewMode(cameraMode);
        setIsActive(true);
        setLoading(false);
        setLoadingMessage('');
      } else {
        setError(data.message || 'Failed to start detection');
        setLoading(false);
        setLoadingMessage('');
      }
    } catch (err) {
      if (!isMountedRef.current) return;
      setError('Error connecting to server. Please try again.');
      setLoading(false);
      setLoadingMessage('');
    }
  };

  const stopDetection = async () => {
    // Clear image sources FIRST to release browser HTTP connections
    if (topImgRef.current) {
      topImgRef.current.src = '';
    }
    if (bottomImgRef.current) {
      bottomImgRef.current.src = '';
    }
    
    // Small delay to ensure browser releases connections
    await new Promise(resolve => setTimeout(resolve, 100));
    
    try {
      await fetch(`${API_URL}/stop`, {
        method: 'POST',
      });
      setIsActive(false);
      setTopPotholeDetected(false);
      setBottomPotholeDetected(false);
    } catch (err) {
      console.error('Error stopping detection:', err);
    }
  };

  // Handle view mode change - switch camera on backend
  const handleViewModeChange = async (newMode) => {
    if (newMode === viewMode || !isActive) return;
    
    console.log(`Switching camera view from ${viewMode} to ${newMode}...`);
    
    // Clear current feeds
    if (topImgRef.current) topImgRef.current.src = '';
    if (bottomImgRef.current) bottomImgRef.current.src = '';
    
    setLoading(true);
    setLoadingMessage(`Switching to ${newMode === 'both' ? 'dual' : newMode} camera...`);
    
    try {
      // Stop current detection
      await fetch(`${API_URL}/stop`, { method: 'POST' });
      await new Promise(resolve => setTimeout(resolve, 500)); // Wait for camera release
      
      // Start with new mode
      const response = await fetch(`${API_URL}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera: newMode })
      });
      const data = await response.json();
      
      if (data.status === 'success') {
        setFeedKey(Date.now()); // Force new feed
        setViewMode(newMode);
        setLoading(false);
        console.log(`✓ Switched to ${newMode} camera`);
      } else {
        setError(data.message || 'Failed to switch camera');
        setLoading(false);
      }
    } catch (err) {
      setError('Error switching camera');
      setLoading(false);
      console.error('Camera switch error:', err);
    }
  };

  return (
    <div className="h-full flex flex-col p-4">
      {/* Header with view mode toggle */}
      <div className="flex items-center justify-between mb-4">
        <div className="w-32"></div>
        <h2 className="text-xl font-bold text-white">Pothole Detection</h2>
        
        {/* View Mode Toggle */}
        <div className="flex gap-1 bg-gray-900 rounded-lg p-1">
          <button
            onClick={() => handleViewModeChange('top')}
            className={`px-3 py-1 rounded text-sm transition-all ${
              viewMode === 'top'
                ? 'bg-orange-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
            disabled={loading}
          >
            Top
          </button>
          <button
            onClick={() => handleViewModeChange('bottom')}
            className={`px-3 py-1 rounded text-sm transition-all ${
              viewMode === 'bottom'
                ? 'bg-orange-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
            disabled={loading}
          >
            Bottom
          </button>
          <button
            onClick={() => handleViewModeChange('both')}
            className={`px-3 py-1 rounded text-sm transition-all ${
              viewMode === 'both'
                ? 'bg-orange-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
            disabled={loading}
          >
            Both
          </button>
        </div>
      </div>

      {/* Video Feed Container */}
      <div className="flex-1 flex items-center justify-center gap-4 bg-black rounded-lg overflow-hidden relative p-2">
        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 z-10">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-orange-500 border-t-transparent mb-4"></div>
            <p className="text-white text-lg">{loadingMessage}</p>
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center justify-center text-center p-8">
            <div className="text-red-500 text-4xl mb-4">⚠️</div>
            <p className="text-white text-lg mb-2">Error</p>
            <p className="text-gray-400 text-sm">{error}</p>
            <button
              onClick={() => startDetection(viewMode)}
              className="mt-4 px-4 py-2 bg-orange-600 hover:bg-orange-500 text-white rounded-lg transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {!loading && !error && (
          <>
            {/* Top Section - Shows feed when viewMode is 'top' or 'both' */}
            <div className="flex-1 relative border-2 border-gray-700 rounded-lg overflow-hidden">
              <div className="absolute top-2 left-2 bg-black/70 px-3 py-1 rounded-lg z-10">
                <span className="text-white text-sm font-bold">TOP VIEW</span>
              </div>
              {(viewMode === 'top' || viewMode === 'both') ? (
                <img
                  ref={topImgRef}
                  src={`${API_URL}/top_feed?t=${feedKey}`}
                  alt="Top Camera - Pothole Detection"
                  className="w-full h-full object-contain"
                  style={{ display: 'block' }}
                  onError={(e) => console.error('Top feed error:', e)}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-gray-900">
                  <div className="text-center">
                    <div className="text-gray-500 text-4xl mb-2">▲</div>
                    <p className="text-gray-500 text-sm">Switch to Top Camera</p>
                  </div>
                </div>
              )}
              {(viewMode === 'top' || viewMode === 'both') && topPotholeDetected && (
                <div className="absolute top-2 right-2 z-20">
                  <div className="bg-red-600 text-white px-4 py-2 rounded-lg shadow-2xl flex items-center gap-2 animate-pulse">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <span className="font-bold text-sm">POTHOLE!</span>
                  </div>
                </div>
              )}
            </div>

            {/* Bottom Section - Shows feed when viewMode is 'bottom' or 'both' */}
            <div className="flex-1 relative border-2 border-gray-700 rounded-lg overflow-hidden">
              <div className="absolute top-2 left-2 bg-black/70 px-3 py-1 rounded-lg z-10">
                <span className="text-white text-sm font-bold">BOTTOM VIEW</span>
              </div>
              {(viewMode === 'bottom' || viewMode === 'both') ? (
                <img
                  ref={bottomImgRef}
                  src={`${API_URL}/bottom_feed?t=${feedKey}`}
                  alt="Bottom Camera - Pothole Detection"
                  className="w-full h-full object-contain"
                  style={{ display: 'block' }}
                  onError={(e) => console.error('Bottom feed error:', e)}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-gray-900">
                  <div className="text-center">
                    <div className="text-gray-500 text-4xl mb-2">▼</div>
                    <p className="text-gray-500 text-sm">Switch to Bottom Camera</p>
                  </div>
                </div>
              )}
              {(viewMode === 'bottom' || viewMode === 'both') && bottomPotholeDetected && (
                <div className="absolute top-2 right-2 z-20">
                  <div className="bg-red-600 text-white px-4 py-2 rounded-lg shadow-2xl flex items-center gap-2 animate-pulse">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <span className="font-bold text-sm">POTHOLE!</span>
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* Status Bar */}
      <div className="mt-4 flex items-center justify-center gap-4 flex-wrap">
        <div className="flex items-center gap-2 px-4 py-2 bg-gray-900 rounded-lg">
          <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
          <span className="text-sm text-gray-300">
            {isActive ? `Detection Active (${viewMode === 'both' ? 'Dual' : viewMode === 'top' ? 'Top' : 'Bottom'})` : 'Detection Inactive'}
          </span>
        </div>
        {isActive && (
          <div className="px-4 py-2 bg-gray-900 rounded-lg">
            <span className="text-sm text-gray-300">Real-time Analysis</span>
          </div>
        )}
        {(topPotholeDetected || bottomPotholeDetected) && (
          <div className="px-4 py-2 bg-red-900/50 rounded-lg border border-red-500">
            <span className="text-sm text-red-300 font-semibold">
              ⚠️ {topPotholeDetected && bottomPotholeDetected ? 'Both Detected' : topPotholeDetected ? 'Top Alert' : 'Bottom Alert'}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default PotholeDetector;
