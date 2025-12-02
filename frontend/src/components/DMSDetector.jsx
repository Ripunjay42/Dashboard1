import React, { useState, useEffect, useRef } from 'react';
import { TbArrowLeft } from 'react-icons/tb';

const DMSDetector = ({ onBack }) => {
  const [isActive, setIsActive] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingMessage, setLoadingMessage] = useState('Initializing DMS camera...');
  const [error, setError] = useState(null);
  const [isDrowsy, setIsDrowsy] = useState(false);
  const [isYawning, setIsYawning] = useState(false);
  const [earValue, setEarValue] = useState(0);
  const [yawnValue, setYawnValue] = useState(0);
  const statusIntervalRef = useRef(null);
  const API_URL = 'http://localhost:5000/api/dms';
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
    // Poll for DMS detection status when active
    if (isActive) {
      statusIntervalRef.current = setInterval(async () => {
        try {
          const response = await fetch(`${API_URL}/status`);
          const data = await response.json();
          setIsDrowsy(data.is_drowsy);
          setIsYawning(data.is_yawning);
          setEarValue(data.ear_value || 0);
          setYawnValue(data.yawn_value || 0);
        } catch (err) {
          console.error('Status poll error:', err);
        }
      }, 300); // Poll every 300ms for responsive alerts
    } else {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
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
      setLoadingMessage('Starting DMS detection...');
      const response = await fetch(`${API_URL}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      
      if (response.ok && data.status === 'success') {
        setIsActive(true);
        setLoading(false);
      } else {
        throw new Error(data.message || 'Failed to start DMS');
      }
    } catch (err) {
      console.error('DMS start error:', err);
      setError(err.message);
      setLoading(false);
    }
  };

  const stopDetection = async () => {
    try {
      await fetch(`${API_URL}/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      setIsActive(false);
      setIsDrowsy(false);
      setIsYawning(false);
    } catch (err) {
      console.error('DMS stop error:', err);
    }
  };

  return (
    <div className="h-full flex flex-col p-4">
      {/* Header */}
      <div className="flex items-center justify-center mb-4">
        <h2 className="text-xl font-bold text-white">Driver Monitoring System</h2>
        <div className="w-16"></div>
      </div>

      {/* Video Feed Container */}
      <div className="flex-1 flex items-center justify-center bg-black rounded-lg overflow-hidden relative">
        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 z-10">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent mb-4"></div>
            <p className="text-white text-lg">{loadingMessage}</p>
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center justify-center text-center p-8">
            <div className="text-red-500 text-4xl mb-4">⚠️</div>
            <p className="text-white text-lg mb-2">Error</p>
            <p className="text-gray-400 text-sm">{error}</p>
            <button
              onClick={startDetection}
              className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {!loading && !error && (
          <>
            <img
              src={`${API_URL}/video_feed`}
              alt="DMS Detection Feed"
              className="max-w-full max-h-full object-contain"
              style={{ display: 'block' }}
            />
            
            {/* Drowsiness Alert Overlay */}
            {/* {isDrowsy && (
              <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20">
                <div className="bg-red-600 text-white px-6 py-3 rounded-lg shadow-2xl flex items-center gap-3 animate-pulse">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <span className="font-bold text-lg">DROWSINESS DETECTED!</span>
                </div>
              </div>
            )} */}

            {/* Yawning Alert Overlay */}
            {/* {isYawning && !isDrowsy && (
              <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20">
                <div className="bg-orange-500 text-white px-6 py-3 rounded-lg shadow-2xl flex items-center gap-3 animate-pulse">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="font-bold text-lg">YAWNING DETECTED!</span>
                </div>
              </div>
            )} */}
          </>
        )}
      </div>

      {/* Status Bar */}
      <div className="mt-4 flex flex-wrap items-center justify-center gap-4">
        <div className="flex items-center gap-2 px-4 py-2 bg-gray-900 rounded-lg">
          <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
          <span className="text-sm text-gray-300">
            {isActive ? 'DMS Active' : 'DMS Inactive'}
          </span>
        </div>
        
        {isActive && (
          <>
            <div className="px-4 py-2 bg-gray-900 rounded-lg">
              <span className="text-sm text-gray-300">EAR: {earValue.toFixed(2)}</span>
            </div>
            <div className="px-4 py-2 bg-gray-900 rounded-lg">
              <span className="text-sm text-gray-300">Yawn: {yawnValue.toFixed(2)}</span>
            </div>
          </>
        )}
        
        {isDrowsy && (
          <div className="px-4 py-2 bg-red-900/50 rounded-lg border border-red-500">
            <span className="text-sm text-red-300 font-semibold">😴 Drowsy Alert!</span>
          </div>
        )}
        
        {isYawning && !isDrowsy && (
          <div className="px-4 py-2 bg-orange-900/50 rounded-lg border border-orange-500">
            <span className="text-sm text-orange-300 font-semibold">🥱 Yawn Alert!</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default DMSDetector;
