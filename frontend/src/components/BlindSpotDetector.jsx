import React, { useState, useEffect, useRef } from 'react';

const BlindSpotDetector = ({ onBack }) => {
  const [leftActive, setLeftActive] = useState(false);
  const [rightActive, setRightActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [error, setError] = useState(null);
  const [leftDanger, setLeftDanger] = useState(false);
  const [rightDanger, setRightDanger] = useState(false);
  const statusIntervalRef = useRef(null);
  const API_URL = 'http://localhost:5000/api/blindspot';

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopDetection();
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
    };
  }, []);

  useEffect(() => {
    // Poll for blind spot detection status when any camera is active
    if (leftActive || rightActive) {
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
  }, [leftActive, rightActive]);

  const toggleCamera = async (camera) => {
    const isLeft = camera === 'left';
    const currentlyActive = isLeft ? leftActive : rightActive;
    
    if (currentlyActive) {
      // Turn off this camera
      await stopCameraDetection(camera);
    } else {
      // Turn on this camera (will stop the other one first since only one can run at a time)
      await startDetection(camera);
    }
  };

  const startDetection = async (camera) => {
    // Stop any existing detection first
    if (leftActive || rightActive) {
      await stopDetection();
    }
    
    setLoading(true);
    setError(null);
    setLoadingMessage(`Starting ${camera} camera...`);
    
    try {
      const response = await fetch(`${API_URL}/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ camera: camera }),
      });
      const data = await response.json();
      
      if (data.status === 'success') {
        if (camera === 'left') {
          setLeftActive(true);
          setRightActive(false);
        } else {
          setRightActive(true);
          setLeftActive(false);
        }
        setLoading(false);
        setLoadingMessage('');
      } else {
        setError(data.message || 'Failed to start detection');
        setLoading(false);
      }
    } catch (err) {
      setError('Error connecting to server. Please try again.');
      setLoading(false);
      setLoadingMessage('');
    }
  };

  const stopCameraDetection = async (camera) => {
    try {
      await fetch(`${API_URL}/stop`, {
        method: 'POST',
      });
      if (camera === 'left') {
        setLeftActive(false);
      } else {
        setRightActive(false);
      }
      setLeftDanger(false);
      setRightDanger(false);
    } catch (err) {
      console.error('Error stopping detection:', err);
    }
  };

  const stopDetection = async () => {
    try {
      await fetch(`${API_URL}/stop`, {
        method: 'POST',
      });
      setLeftActive(false);
      setRightActive(false);
      setLeftDanger(false);
      setRightDanger(false);
    } catch (err) {
      console.error('Error stopping detection:', err);
    }
  };

  return (
    <div className="h-full flex flex-col p-4">
      {/* Header */}
      <div className="flex items-center justify-center mb-4">
        <h2 className="text-xl font-bold text-white">Blind Spot Detection</h2>
      </div>

      {/* Video Feed Container - Both cameras side by side */}
      <div className="flex-1 flex gap-2 bg-black rounded-lg overflow-hidden relative">
        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 z-10">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-cyan-500 border-t-transparent mb-4"></div>
            <p className="text-white text-lg">{loadingMessage}</p>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-8 bg-black/80 z-10">
            <div className="text-red-500 text-4xl mb-4">⚠️</div>
            <p className="text-white text-lg mb-2">Error</p>
            <p className="text-gray-400 text-sm">{error}</p>
            <button
              onClick={() => setError(null)}
              className="mt-4 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg transition-colors"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Left Camera Feed */}
        <div className="flex-1 relative border-2 border-gray-700 rounded-lg overflow-hidden">
          <div className="absolute top-2 left-2 bg-black/70 px-3 py-1 rounded-lg z-10 flex items-center gap-2">
            <span className="text-white text-sm font-bold">LEFT MIRROR</span>
            <button
              onClick={() => toggleCamera('left')}
              className={`px-2 py-1 rounded text-xs font-bold transition-all ${
                leftActive 
                  ? 'bg-green-600 text-white' 
                  : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
              }`}
            >
              {leftActive ? 'ON' : 'OFF'}
            </button>
          </div>
          
          {leftActive ? (
            <>
              <img
                src={`${API_URL}/left_feed?t=${Date.now()}`}
                alt="Left Blind Spot Feed"
                className="w-full h-full object-contain"
                style={{ display: 'block' }}
              />
              {leftDanger && (
                <div className="absolute top-2 right-2 z-20">
                  <div className="bg-red-600 text-white px-4 py-2 rounded-lg shadow-2xl flex items-center gap-2 animate-pulse">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <span className="font-bold text-sm">DANGER!</span>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gray-900">
              <div className="text-center">
                <svg className="w-16 h-16 text-gray-600 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <p className="text-gray-500 text-sm">Camera Off</p>
                <button
                  onClick={() => toggleCamera('left')}
                  className="mt-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-sm transition-colors"
                >
                  Turn On
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right Camera Feed */}
        <div className="flex-1 relative border-2 border-gray-700 rounded-lg overflow-hidden">
          <div className="absolute top-2 left-2 bg-black/70 px-3 py-1 rounded-lg z-10 flex items-center gap-2">
            <span className="text-white text-sm font-bold">RIGHT MIRROR</span>
            <button
              onClick={() => toggleCamera('right')}
              className={`px-2 py-1 rounded text-xs font-bold transition-all ${
                rightActive 
                  ? 'bg-green-600 text-white' 
                  : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
              }`}
            >
              {rightActive ? 'ON' : 'OFF'}
            </button>
          </div>
          
          {rightActive ? (
            <>
              <img
                src={`${API_URL}/right_feed?t=${Date.now()}`}
                alt="Right Blind Spot Feed"
                className="w-full h-full object-contain"
                style={{ display: 'block' }}
              />
              {rightDanger && (
                <div className="absolute top-2 right-2 z-20">
                  <div className="bg-red-600 text-white px-4 py-2 rounded-lg shadow-2xl flex items-center gap-2 animate-pulse">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <span className="font-bold text-sm">DANGER!</span>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gray-900">
              <div className="text-center">
                <svg className="w-16 h-16 text-gray-600 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <p className="text-gray-500 text-sm">Camera Off</p>
                <button
                  onClick={() => toggleCamera('right')}
                  className="mt-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-sm transition-colors"
                >
                  Turn On
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Status Bar */}
      <div className="mt-4 flex items-center justify-center gap-4">
        <div className="flex items-center gap-2 px-4 py-2 bg-gray-900 rounded-lg">
          <div className={`w-2 h-2 rounded-full ${(leftActive || rightActive) ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
          <span className="text-sm text-gray-300">
            {(leftActive || rightActive) ? 'Detection Active' : 'Detection Inactive'}
          </span>
        </div>
        {(leftActive || rightActive) && (
          <div className="px-4 py-2 bg-gray-900 rounded-lg">
            <span className="text-sm text-gray-300">
              {leftActive ? 'Left' : 'Right'} Camera Active
            </span>
          </div>
        )}
        {(leftDanger || rightDanger) && (
          <div className="px-4 py-2 bg-red-900/50 rounded-lg border border-red-500">
            <span className="text-sm text-red-300 font-semibold">
              ⚠️ {leftDanger && rightDanger ? 'Both Sides' : leftDanger ? 'Left Side' : 'Right Side'} Warning
            </span>
          </div>
        )}
        <div className="text-xs text-gray-500">
          (Only one camera at a time)
        </div>
      </div>
    </div>
  );
};

export default BlindSpotDetector;
