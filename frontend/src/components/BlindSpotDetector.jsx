import React, { useState, useEffect, useRef } from 'react';

const BlindSpotDetector = ({ onBack }) => {
  const [isActive, setIsActive] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingMessage, setLoadingMessage] = useState('Initializing cameras...');
  const [error, setError] = useState(null);
  const [leftDanger, setLeftDanger] = useState(false);
  const [rightDanger, setRightDanger] = useState(false);
  const statusIntervalRef = useRef(null);
  const API_URL = 'http://localhost:5000/api/blindspot';
  const hasStartedRef = useRef(false);

  useEffect(() => {
    // Start detection only once when component mounts
    if (!hasStartedRef.current) {
      hasStartedRef.current = true;
      startDetection();
    }
    
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

  const startDetection = async () => {
    setLoading(true);
    setError(null);
    try {
      setLoadingMessage('Starting blind spot detection...');
      
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
      }
    } catch (err) {
      setError('Error connecting to server. Please try again.');
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
      setLeftDanger(false);
      setRightDanger(false);
    } catch (err) {
      console.error('Error stopping detection:', err);
    }
  };

  return (
    <div className="h-full flex flex-col p-4">
      {/* Header with back button */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-white">Blind Spot Detection</h2>
        <div className="w-20"></div>
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
            {/* Left Camera Feed */}
            <div className="flex-1 relative border-2 border-gray-700 rounded-lg overflow-hidden">
              <div className="absolute top-2 left-2 bg-black/70 px-3 py-1 rounded-lg z-10">
                <span className="text-white text-sm font-bold">LEFT MIRROR</span>
              </div>
              <img
                src={`${API_URL}/left_feed`}
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
            </div>

            {/* Right Camera Feed */}
            <div className="flex-1 relative border-2 border-gray-700 rounded-lg overflow-hidden">
              <div className="absolute top-2 left-2 bg-black/70 px-3 py-1 rounded-lg z-10">
                <span className="text-white text-sm font-bold">RIGHT MIRROR</span>
              </div>
              <img
                src={`${API_URL}/right_feed`}
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
            </div>
          </>
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
            <span className="text-sm text-gray-300">Real-time Analysis</span>
          </div>
        )}
        {(leftDanger || rightDanger) && (
          <div className="px-4 py-2 bg-red-900/50 rounded-lg border border-red-500">
            <span className="text-sm text-red-300 font-semibold">
              ⚠️ {leftDanger && rightDanger ? 'Both Sides' : leftDanger ? 'Left Side' : 'Right Side'} Warning
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default BlindSpotDetector;
