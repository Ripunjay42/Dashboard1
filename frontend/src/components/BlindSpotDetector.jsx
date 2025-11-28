import React, { useState, useEffect, useRef } from 'react';

const BlindSpotDetector = ({ onBack }) => {
  const [isActive, setIsActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [error, setError] = useState(null);
  const [leftDanger, setLeftDanger] = useState(false);
  const [rightDanger, setRightDanger] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState(null); // 'left' or 'right' or null
  const statusIntervalRef = useRef(null);
  const API_URL = 'http://localhost:5000/api/blindspot';

  // Cleanup on unmount
  useEffect(() => {
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
    // Poll for blind spot detection status when active
    if (isActive && selectedCamera) {
      statusIntervalRef.current = setInterval(async () => {
        try {
          const response = await fetch(`${API_URL}/status`);
          const data = await response.json();
          if (selectedCamera === 'left') {
            setLeftDanger(data.left_danger || false);
          } else {
            setRightDanger(data.right_danger || false);
          }
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
  }, [isActive, selectedCamera]);

  const startDetection = async (camera) => {
    // Stop any existing detection first
    if (isActive) {
      await stopDetection();
    }
    
    setSelectedCamera(camera);
    setLoading(true);
    setError(null);
    setLoadingMessage(`Starting ${camera} camera detection...`);
    
    try {
      const response = await fetch(`${API_URL}/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ camera: camera }), // Send which camera to use
      });
      const data = await response.json();
      
      if (data.status === 'success') {
        setIsActive(true);
        setLoading(false);
        setLoadingMessage('');
      } else {
        setError(data.message || 'Failed to start detection');
        setLoading(false);
        setSelectedCamera(null);
      }
    } catch (err) {
      setError('Error connecting to server. Please try again.');
      setLoading(false);
      setLoadingMessage('');
      setSelectedCamera(null);
    }
  };

  const stopDetection = async () => {
    try {
      await fetch(`${API_URL}/stop`, {
        method: 'POST',
      });
      setIsActive(false);
      setLeftDanger(false);
      setRightDanger(false);
      setSelectedCamera(null);
    } catch (err) {
      console.error('Error stopping detection:', err);
    }
  };

  const switchCamera = async (newCamera) => {
    if (newCamera === selectedCamera) return;
    await startDetection(newCamera);
  };

  // Camera selection screen
  if (!selectedCamera && !loading) {
    return (
      <div className="h-full flex flex-col p-4">
        {/* Header */}
        <div className="flex items-center justify-center mb-6">
          <h2 className="text-xl font-bold text-white">Blind Spot Detection</h2>
        </div>

        {/* Camera Selection */}
        <div className="flex-1 flex items-center justify-center">
          <div className="flex gap-8">
            {/* Left Camera Button */}
            <button
              onClick={() => startDetection('left')}
              className="group flex flex-col items-center gap-4 p-8 bg-gray-800/80 hover:bg-gray-700/80 border-2 border-gray-600 hover:border-cyan-500 rounded-2xl transition-all duration-300 hover:scale-105"
            >
              <div className="w-24 h-24 flex items-center justify-center bg-gray-900 rounded-full border-2 border-gray-600 group-hover:border-cyan-500 transition-colors">
                <svg className="w-12 h-12 text-gray-400 group-hover:text-cyan-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                </svg>
              </div>
              <div className="text-center">
                <h3 className="text-lg font-bold text-white group-hover:text-cyan-400 transition-colors">Left Mirror</h3>
                <p className="text-sm text-gray-400 mt-1">Monitor left blind spot</p>
              </div>
            </button>

            {/* Right Camera Button */}
            <button
              onClick={() => startDetection('right')}
              className="group flex flex-col items-center gap-4 p-8 bg-gray-800/80 hover:bg-gray-700/80 border-2 border-gray-600 hover:border-cyan-500 rounded-2xl transition-all duration-300 hover:scale-105"
            >
              <div className="w-24 h-24 flex items-center justify-center bg-gray-900 rounded-full border-2 border-gray-600 group-hover:border-cyan-500 transition-colors">
                <svg className="w-12 h-12 text-gray-400 group-hover:text-cyan-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                </svg>
              </div>
              <div className="text-center">
                <h3 className="text-lg font-bold text-white group-hover:text-cyan-400 transition-colors">Right Mirror</h3>
                <p className="text-sm text-gray-400 mt-1">Monitor right blind spot</p>
              </div>
            </button>
          </div>
        </div>

        {/* Info Text */}
        <div className="text-center text-gray-500 text-sm mt-4">
          <p>Select a camera to start blind spot detection</p>
          <p className="text-xs mt-1">Only one camera runs at a time for optimal performance</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col p-4">
      {/* Header with camera switcher */}
      <div className="flex items-center justify-between mb-4">
        {/* Camera Switch Buttons */}
        <div className="flex gap-2">
          <button
            onClick={() => switchCamera('left')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
              selectedCamera === 'left'
                ? 'bg-cyan-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            ← Left
          </button>
          <button
            onClick={() => switchCamera('right')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
              selectedCamera === 'right'
                ? 'bg-cyan-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Right →
          </button>
        </div>
        
        <h2 className="text-xl font-bold text-white">
          {selectedCamera === 'left' ? 'Left' : 'Right'} Blind Spot
        </h2>
        
        {/* Stop Button */}
        <button
          onClick={stopDetection}
          className="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-sm font-semibold transition-colors"
        >
          Stop
        </button>
      </div>

      {/* Video Feed Container */}
      <div className="flex-1 flex bg-black rounded-lg overflow-hidden relative">
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
              onClick={() => startDetection(selectedCamera || 'left')}
              className="mt-4 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {!loading && !error && selectedCamera && (
          <div className="flex-1 relative border-2 border-gray-700 rounded-lg overflow-hidden">
            <div className="absolute top-2 left-2 bg-black/70 px-3 py-1 rounded-lg z-10">
              <span className="text-white text-sm font-bold">
                {selectedCamera === 'left' ? 'LEFT MIRROR' : 'RIGHT MIRROR'}
              </span>
            </div>
            <img
              src={`${API_URL}/${selectedCamera}_feed?t=${Date.now()}`}
              alt={`${selectedCamera} Blind Spot Feed`}
              className="w-full h-full object-contain"
              style={{ display: 'block' }}
            />
            {((selectedCamera === 'left' && leftDanger) || (selectedCamera === 'right' && rightDanger)) && (
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
        )}
      </div>

      {/* Status Bar */}
      <div className="mt-4 flex items-center justify-center gap-4">
        <div className="flex items-center gap-2 px-4 py-2 bg-gray-900 rounded-lg">
          <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
          <span className="text-sm text-gray-300">
            {isActive ? 'Detection Active' : 'Detection Inactive'}
          </span>
        </div>
        {isActive && (
          <div className="px-4 py-2 bg-gray-900 rounded-lg">
            <span className="text-sm text-gray-300">
              {selectedCamera === 'left' ? 'Left' : 'Right'} Camera
            </span>
          </div>
        )}
        {((selectedCamera === 'left' && leftDanger) || (selectedCamera === 'right' && rightDanger)) && (
          <div className="px-4 py-2 bg-red-900/50 rounded-lg border border-red-500">
            <span className="text-sm text-red-300 font-semibold">
              ⚠️ {selectedCamera === 'left' ? 'Left' : 'Right'} Side Warning
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default BlindSpotDetector;
