import React, { useState, useEffect, useRef } from 'react';

const BlindSpotDetector = ({ onBack }) => {
  const [isActive, setIsActive] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingMessage, setLoadingMessage] = useState('Initializing cameras...');
  const [error, setError] = useState(null);
  const [leftDanger, setLeftDanger] = useState(false);
  const [rightDanger, setRightDanger] = useState(false);
  const [feedKey, setFeedKey] = useState(Date.now()); // Cache buster for video feeds
  const [viewMode, setViewMode] = useState('left'); // 'left', 'right', or 'both' - default to left for less Jetson load
  const statusIntervalRef = useRef(null);
  const leftImgRef = useRef(null);
  const rightImgRef = useRef(null);
  const API_URL = 'http://localhost:5000/api/blindspot';
  const hasStartedRef = useRef(false);
  const isMountedRef = useRef(true); // Track if component is still mounted

  useEffect(() => {
    isMountedRef.current = true;
    
    // Start detection only once when component mounts
    if (!hasStartedRef.current) {
      hasStartedRef.current = true;
      startDetection();
    }
    
    // Cleanup on unmount
    return () => {
      isMountedRef.current = false;
      
      // Clear streams FIRST to release browser connections
      if (leftImgRef.current) {
        leftImgRef.current.src = '';
      }
      if (rightImgRef.current) {
        rightImgRef.current.src = '';
      }
      
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
      
      // Fire-and-forget stop - Dashboard handles proper cleanup
      fetch(`${API_URL}/stop`, { method: 'POST' }).catch(() => {});
    };
  }, []);

  useEffect(() => {
    // Poll for blind spot detection status when active
    if (isActive) {
      statusIntervalRef.current = setInterval(async () => {
        try {
          const response = await fetch(`${API_URL}/status`);
          const data = await response.json();
          setLeftDanger(data.left_danger || false);
          setRightDanger(data.right_danger || false);
        } catch (err) {
          console.error('Error fetching status:', err);
        }
      }, 500);
    } else {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
      setLeftDanger(false);
      setRightDanger(false);
    }
    
    return () => {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
    };
  }, [isActive]);

  const startDetection = async (cameraMode = 'left') => {
    setLoading(true);
    setError(null);
    try {
      setLoadingMessage('Starting blind spot detection...');
      
      // Pass camera mode to backend - only opens one camera for less Jetson load
      const response = await fetch(`${API_URL}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera: cameraMode })
      });
      const data = await response.json();
      
      if (data.status === 'success') {
        // Only update state if still mounted
        if (!isMountedRef.current) return;
        
        // Generate new cache key to force fresh feed connections
        setFeedKey(Date.now());
        setIsActive(true);
        setLoading(false);
        setLoadingMessage('');
      } else {
        if (!isMountedRef.current) return;
        setError(data.message || 'Failed to start detection');
        setLoading(false);
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
    // This prevents browser hang on Jetson when switching tabs
    if (leftImgRef.current) {
      leftImgRef.current.src = '';
    }
    if (rightImgRef.current) {
      rightImgRef.current.src = '';
    }
    
    // Small delay to ensure browser releases connections
    await new Promise(resolve => setTimeout(resolve, 100));
    
    try {
      await fetch(`${API_URL}/stop`, {
        method: 'POST',
      });
      setIsActive(false);
      setLeftDanger(false);
      setRightDanger(false);
    } catch (err) {
      console.error('Error stopping detection:', err);
    }
  };

  // Handle view mode change - switch camera on backend
  const handleViewModeChange = async (newMode) => {
    if (newMode === viewMode) return; // No change needed
    
    setViewMode(newMode);
    setLoading(true);
    setLoadingMessage('Switching camera...');
    
    // Clear both feeds
    if (leftImgRef.current) leftImgRef.current.src = '';
    if (rightImgRef.current) rightImgRef.current.src = '';
    
    try {
      // Stop current detection
      await fetch(`${API_URL}/stop`, { method: 'POST' });
      await new Promise(resolve => setTimeout(resolve, 300));
      
      // Start with new camera mode
      const response = await fetch(`${API_URL}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera: newMode })
      });
      const data = await response.json();
      
      if (data.status === 'success') {
        setFeedKey(Date.now());
        setLoading(false);
      } else {
        setError(data.message || 'Failed to switch camera');
        setLoading(false);
      }
    } catch (err) {
      setError('Error switching camera');
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col p-4">
      {/* Header with View Mode Toggle */}
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xl font-bold text-white">Blind Spot Detection</h2>
        
        {/* View Mode Toggle - Left, Both, or Right */}
        <div className="flex items-center gap-1 bg-gray-800 rounded-lg p-1">
          <button
            onClick={() => handleViewModeChange('left')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              viewMode === 'left'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            Left
          </button>
          <button
            onClick={() => handleViewModeChange('both')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              viewMode === 'both'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            Both
          </button>
          <button
            onClick={() => handleViewModeChange('right')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              viewMode === 'right'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            Right
          </button>
        </div>
      </div>

      {/* Video Feed Container */}
      <div className="flex-1 flex gap-2 bg-black rounded-lg overflow-hidden relative">
        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 z-10">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-cyan-500 border-t-transparent mb-4"></div>
            <p className="text-white text-lg">{loadingMessage}</p>
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center justify-center text-center p-8 w-full">
            <div className="text-red-500 text-4xl mb-4">⚠️</div>
            <p className="text-white text-lg mb-2">Error</p>
            <p className="text-gray-400 text-sm">{error}</p>
            <button
              onClick={startDetection}
              className="mt-4 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {!loading && !error && (
          <>
            {/* Left Section - Shows feed when viewMode is 'left' or 'both' */}
            <div className="flex-1 relative border-2 border-gray-700 rounded-lg overflow-hidden">
              <div className="absolute top-2 left-2 bg-black/70 px-3 py-1 rounded-lg z-10">
                <span className="text-white text-sm font-bold">LEFT MIRROR</span>
              </div>
              {(viewMode === 'left' || viewMode === 'both') ? (
                <img
                  ref={leftImgRef}
                  src={`${API_URL}/left_feed?t=${feedKey}`}
                  alt="Left Blind Spot Feed"
                  className="w-full h-full object-contain"
                  style={{ display: 'block' }}
                  onError={(e) => console.error('Left feed error:', e)}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-gray-900">
                  <div className="text-center">
                    <div className="text-gray-500 text-4xl mb-2">◀</div>
                    <p className="text-gray-500 text-sm">Switch to Left Camera</p>
                  </div>
                </div>
              )}
              {(viewMode === 'left' || viewMode === 'both') && leftDanger && (
                <div className="absolute top-2 right-2 z-20">
                  <div className="bg-red-600 text-white px-4 py-2 rounded-lg shadow-2xl flex items-center gap-2 animate-pulse">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <span className="font-bold text-sm">DANGER!</span>
                  </div>
                </div>
              )}
            </div>

            {/* Right Section - Shows feed when viewMode is 'right' or 'both' */}
            <div className="flex-1 relative border-2 border-gray-700 rounded-lg overflow-hidden">
              <div className="absolute top-2 left-2 bg-black/70 px-3 py-1 rounded-lg z-10">
                <span className="text-white text-sm font-bold">RIGHT MIRROR</span>
              </div>
              {(viewMode === 'right' || viewMode === 'both') ? (
                <img
                  ref={rightImgRef}
                  src={`${API_URL}/right_feed?t=${feedKey}`}
                  alt="Right Blind Spot Feed"
                  className="w-full h-full object-contain"
                  style={{ display: 'block' }}
                  onError={(e) => console.error('Right feed error:', e)}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-gray-900">
                  <div className="text-center">
                    <div className="text-gray-500 text-4xl mb-2">▶</div>
                    <p className="text-gray-500 text-sm">Switch to Right Camera</p>
                  </div>
                </div>
              )}
              {(viewMode === 'right' || viewMode === 'both') && rightDanger && (
                <div className="absolute top-2 right-2 z-20">
                  <div className="bg-red-600 text-white px-4 py-2 rounded-lg shadow-2xl flex items-center gap-2 animate-pulse">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <span className="font-bold text-sm">DANGER!</span>
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* Status Bar */}
      <div className="mt-3 flex items-center justify-center gap-3 flex-wrap">
        <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-900 rounded-lg">
          <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
          <span className="text-xs text-gray-300">
            {isActive ? 'Detection Active' : 'Detection Inactive'}
          </span>
        </div>
        {isActive && (
          <div className="px-3 py-1.5 bg-gray-900 rounded-lg">
            <span className="text-xs text-gray-300">
              {viewMode === 'both' ? 'Both Cameras Active' : viewMode === 'left' ? 'Left Camera Active' : 'Right Camera Active'}
            </span>
          </div>
        )}
        {/* Show danger warning for active camera(s) */}
        {((viewMode === 'left' || viewMode === 'both') && leftDanger) && (
          <div className="px-3 py-1.5 bg-red-900/50 rounded-lg border border-red-500">
            <span className="text-xs text-red-300 font-semibold">⚠️ Left Side Warning</span>
          </div>
        )}
        {((viewMode === 'right' || viewMode === 'both') && rightDanger) && (
          <div className="px-3 py-1.5 bg-red-900/50 rounded-lg border border-red-500">
            <span className="text-xs text-red-300 font-semibold">⚠️ Right Side Warning</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default BlindSpotDetector;
