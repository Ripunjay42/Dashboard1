import React, { useState, useEffect, useRef } from 'react';

const PotholeDetector = ({ onBack }) => {
  const [isActive, setIsActive] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingMessage, setLoadingMessage] = useState('Initializing camera...');
  const [error, setError] = useState(null);
  const [potholeDetected, setPotholeDetected] = useState(false);
  const statusIntervalRef = useRef(null);
  const imgRef = useRef(null);
  const API_URL = 'http://localhost:5000/api/pothole';
  const hasStartedRef = useRef(false);

  useEffect(() => {
    // Start detection only once when component mounts
    if (!hasStartedRef.current) {
      hasStartedRef.current = true;
      startDetection();
    }
    
    // Cleanup on unmount
    return () => {
      // Clear stream FIRST to release browser connection
      if (imgRef.current) {
        imgRef.current.src = '';
      }
      
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
      
      // Then stop detection on backend
      if (isActive) {
        stopDetection();
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
      }, 500);
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
      setError('Error connecting to server. Please try again.');
      setLoading(false);
      setLoadingMessage('');
    }
  };

  const stopDetection = async () => {
    // Clear image source FIRST to release browser HTTP connection
    // This prevents browser hang on Jetson when switching tabs
    if (imgRef.current) {
      imgRef.current.src = '';
    }
    
    // Small delay to ensure browser releases connection
    await new Promise(resolve => setTimeout(resolve, 100));
    
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
    <div className="h-full flex flex-col p-4">
      {/* Header with back button */}
      <div className="flex items-center justify-center mb-4">

        <h2 className="text-xl font-bold text-white">Pothole Detection</h2>
        <div className="w-16"></div> {/* Spacer for centering */}
      </div>

      {/* Video Feed Container */}
      <div className="flex-1 flex items-center justify-center bg-black rounded-lg overflow-hidden relative">
        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 z-10">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-cyan-500 border-t-transparent mb-4"></div>
            <p className="text-white text-lg">{loadingMessage}</p>
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center justify-center text-center p-8">
            <div className="text-red-500 text-4xl mb-4"></div>
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
            <img
              ref={imgRef}
              src={`${API_URL}/video_feed`}
              alt="Pothole Detection Feed"
              className="max-w-full max-h-full object-contain"
              style={{ display: 'block' }}
            />
            
            {/* Pothole Detection Alert Overlay */}
            {potholeDetected && (
              <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20">
                <div className="bg-red-600 text-white px-6 py-3 rounded-lg shadow-2xl flex items-center gap-3 animate-pulse">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <span className="font-bold text-lg">POTHOLE DETECTED!</span>
                </div>
              </div>
            )}
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
        {potholeDetected && (
          <div className="px-4 py-2 bg-red-900/50 rounded-lg border border-red-500">
            <span className="text-sm text-red-300 font-semibold">⚠️ Warning Active</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default PotholeDetector;
