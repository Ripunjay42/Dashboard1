import React, { useState, useEffect } from 'react';
import Speedometer from '../components/Speedometer';
import BatteryMeter from '../components/BatteryMeter';
import PotholeDetector from '../components/PotholeDetector';
import BlindSpotDetector from '../components/BlindSpotDetector';
import FeatureBar from '../components/FeatureBar';
import StatusBar from '../components/StatusBar';
import VehicleIndicators from '../components/VehicleIndicators';
import Car3DView from '../components/Car3DView';

const Dashboard = ({ onSelectUseCase }) => {
  const [time, setTime] = useState(new Date());
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [activeTab, setActiveTab] = useState('home');
  const [activeFeature, setActiveFeature] = useState(null); // Track which feature is active
  const [speed, setSpeed] = useState(0);
  const [battery, setBattery] = useState(100); // Battery percentage (0-100) - Start at full charge
  const [isThrottling, setIsThrottling] = useState(false);
  const [leftTurnActive, setLeftTurnActive] = useState(false);
  const [rightTurnActive, setRightTurnActive] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Cleanup: Stop any active feature when component unmounts
  useEffect(() => {
    return () => {
      if (activeFeature) {
        stopActiveFeature(activeFeature);
      }
    };
  }, [activeFeature]);

  // Throttle effect - increase speed while holding, decrease when released
  // Battery drains slightly when driving fast
  useEffect(() => {
    let throttleInterval;
    let releaseInterval;
    let batteryInterval;

    if (isThrottling) {
      // Increase speed while throttling
      throttleInterval = setInterval(() => {
        setSpeed(prev => Math.min(prev + 5, 180));
      }, 50);

      // Drain battery slightly when accelerating
      batteryInterval = setInterval(() => {
        setBattery(prev => Math.max(prev - 0.1, 0));
      }, 1000);
    } else {
      // Gradually decrease speed to 0 when throttle is released
      releaseInterval = setInterval(() => {
        setSpeed(prev => {
          const newSpeed = prev - 8;
          return newSpeed < 0 ? 0 : newSpeed;
        });
      }, 50);
    }

    return () => {
      clearInterval(throttleInterval);
      clearInterval(releaseInterval);
      clearInterval(batteryInterval);
    };
  }, [isThrottling]);

  // Keyboard controls for speed, RPM, and turn signals
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowUp' && !e.repeat) {
        e.preventDefault();
        setIsThrottling(true);
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSpeed(prev => Math.max(prev - 10, 0));
      } else if (e.key === 'ArrowRight' && !e.repeat) {
        e.preventDefault();
        setRightTurnActive(true);
        // Only turn signal - no speed changes
      } else if (e.key === 'ArrowLeft' && !e.repeat) {
        e.preventDefault();
        setLeftTurnActive(true);
        // Only turn signal - no speed changes
      }
    };

    const handleKeyUp = (e) => {
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setIsThrottling(false);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        setRightTurnActive(false);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setLeftTurnActive(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  // Stop active feature before switching to a new one
  const stopActiveFeature = async (currentFeature) => {
    const API_BASE = 'http://localhost:5000/api';
    
    try {
      if (currentFeature === 'pothole') {
        console.log('Stopping pothole detection...');
        await fetch(`${API_BASE}/pothole/stop`, { method: 'POST' });
      } else if (currentFeature === 'blindspot') {
        console.log('Stopping blind spot detection...');
        await fetch(`${API_BASE}/blindspot/stop`, { method: 'POST' });
      }
    } catch (error) {
      console.error(`Error stopping ${currentFeature}:`, error);
    }
  };

  const handleFeatureClick = async (featureId) => {
    // If switching to a different feature, stop the current one first
    if (activeFeature && activeFeature !== featureId) {
      await stopActiveFeature(activeFeature);
    }
    
    // Set the new active feature
    if (featureId === 'pothole') {
      setActiveFeature('pothole');
      // Also call the parent's onSelectUseCase if needed
      if (onSelectUseCase) {
        onSelectUseCase('pothole');
      }
    } else if (featureId === 'blindspot') {
      setActiveFeature('blindspot');
      if (onSelectUseCase) {
        onSelectUseCase('blindspot');
      }
    } else if (featureId === null) {
      // Home button clicked - go back to main view
      setActiveFeature(null);
      if (onSelectUseCase) {
        onSelectUseCase(null);
      }
    }
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-2 sm:p-4">
      {/* Oval Dashboard Container */}
  <div className="relative w-full max-w-[1200px] aspect-16/8">
        {/* Outer Decorative Border with Car Dashboard Shape */}
        <div className="absolute inset-0" style={{
          borderRadius: '45% 45% 40% 40% / 35% 35% 30% 30%',
          border: '4px solid #1f2937',
          boxShadow: '0 0 40px rgba(6, 182, 212, 0.3), inset 0 0 50px rgba(0, 0, 0, 0.5)',
          background: 'linear-gradient(180deg, rgba(17, 24, 39, 0.4) 0%, rgba(0, 0, 0, 0.8) 100%)'
        }}></div>

        {/* Middle decorative ring */}
        <div className="absolute inset-3" style={{
          borderRadius: '45% 45% 40% 40% / 35% 35% 30% 30%',
          border: '1px solid rgba(6, 182, 212, 0.2)',
          pointerEvents: 'none'
        }}></div>

        {/* Inner Content Container */}
        <div className="relative w-full h-full flex items-center justify-center px-1 sm:px-8 py-2 sm:py-8">
          {/* Vehicle Indicators */}
          <VehicleIndicators 
            leftTurnActive={leftTurnActive} 
            rightTurnActive={rightTurnActive} 
          />

          {/* Main Content Area - Vertically stacked with bars */}
          <div className="flex flex-col items-center gap-2 w-full max-w-[1600px]">
            {/* Top Feature Buttons */}
            <FeatureBar activeFeature={activeFeature} onFeatureClick={handleFeatureClick} />

            {/* Main Dashboard Container - Meters and Center Display */}
            <div className="flex flex-col lg:flex-row items-center justify-center gap-0 w-full">
              {/* Left - Speedometer - Always visible */}
              <div className="w-full max-w-[380px] shrink-0 flex justify-center">
                <Speedometer value={speed} />
              </div>

              {/* Center Display - Responsive */}
              <div className="w-full max-w-[800px] shrink-0 flex justify-center">
                <div className="bg-gray-900/90 backdrop-blur-sm border-4 border-gray-700 rounded-3xl overflow-hidden p-2 shadow-2xl w-full" style={{ minHeight: '320px', height: '480px', maxWidth: '900px' }}>
                  <div className="h-full w-full">
                    <div className="bg-gray-900 border-2 rounded-2xl h-full w-full overflow-hidden relative">
                      {/* Show Pothole Detection when active */}
                      {activeFeature === 'pothole' ? (
                        <PotholeDetector onBack={async () => {
                          await stopActiveFeature('pothole');
                          setActiveFeature(null);
                        }} />
                      ) : activeFeature === 'blindspot' ? (
                        <BlindSpotDetector onBack={async () => {
                          await stopActiveFeature('blindspot');
                          setActiveFeature(null);
                        }} />
                      ) : (
                        <Car3DView />
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Right - Battery Meter - Always visible */}
              <div className="w-full max-w-[380px] shrink-0 flex justify-center">
                {/* <BatteryMeter value={battery} /> */}
              </div>
            </div>

            {/* Bottom Status Bar */}
            <StatusBar time={time} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
