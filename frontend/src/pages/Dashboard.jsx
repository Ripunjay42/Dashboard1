import React, { useState, useEffect } from 'react';
import Speedometer from '../components/Speedometer';
import BatteryMeter from '../components/BatteryMeter';
import PotholeDetector from '../components/PotholeDetector';
import BlindSpotDetector from '../components/BlindSpotDetector';
import DMSDetector from '../components/DMSDetector';
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
  const [pirAlert, setPirAlert] = useState(0);
  const [mqttConnected, setMqttConnected] = useState(false);
  const [useMqtt, setUseMqtt] = useState(false); // Toggle between MQTT and keyboard control - OFF by default
  const [tripDistance, setTripDistance] = useState(1000); // Trip distance in km, starts at 1000
  const [isSwitching, setIsSwitching] = useState(false); // Prevent rapid concurrent switching
  const [previousFeature, setPreviousFeature] = useState(null); // Track previous feature during transition

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Update trip distance based on speed
  useEffect(() => {
    if (speed === 0) return; // Don't update if not moving

    const updateInterval = setInterval(() => {
      setTripDistance(prev => {
        // Calculate distance: speed (km/h) / 3600 (seconds per hour) = km per second
        const distancePerSecond = speed / 3600;
        const newDistance = prev + distancePerSecond;
        
        // Reset to 1000 if exceeds 10000
        if (newDistance >= 10000) {
          return 1000;
        }
        
        return newDistance;
      });
    }, 1000); // Update every second

    return () => clearInterval(updateInterval);
  }, [speed]);

  // Start/Stop MQTT service based on toggle
  useEffect(() => {
    const startMqtt = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/mqtt/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ broker: '10.42.0.1', port: 1883 })
        });
        const data = await response.json();
        if (data.status === 'success') {
          console.log('✓ MQTT service started');
          setMqttConnected(true);
        }
      } catch (error) {
        console.error('✗ Failed to start MQTT service:', error);
        setMqttConnected(false);
      }
    };

    const stopMqtt = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/mqtt/stop', {
          method: 'POST'
        });
        const data = await response.json();
        if (data.status === 'success') {
          console.log('✓ MQTT service stopped');
          // Reset all MQTT-controlled values to defaults
          setMqttConnected(false);
          setSpeed(0);
          setLeftTurnActive(false);
          setRightTurnActive(false);
          setPirAlert(0);
        }
      } catch (error) {
        console.error('✗ Failed to stop MQTT service:', error);
      }
    };

    if (useMqtt) {
      startMqtt();
    } else {
      // Explicitly stop MQTT and reset state when toggle is OFF
      stopMqtt();
    }

    return () => {
      // Cleanup: Stop MQTT when component unmounts
      if (useMqtt) {
        fetch('http://localhost:5000/api/mqtt/stop', { method: 'POST' }).catch(console.error);
      }
    };
  }, [useMqtt]); // Run when useMqtt changes

  // Poll MQTT state periodically
  useEffect(() => {
    if (!useMqtt) return; // Skip polling if not using MQTT

    const pollMqttState = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/mqtt/state');
        const data = await response.json();
        
        if (data.status === 'success' && data.data) {
          const mqttData = data.data;
          
          // Update speed from MQTT (ADC value)
          if (mqttData.speed !== undefined) {
            setSpeed(mqttData.speed);
          }
          
          // Update indicators from MQTT
          if (mqttData.left_indicator !== undefined) {
            setLeftTurnActive(mqttData.left_indicator === 1);
          }
          if (mqttData.right_indicator !== undefined) {
            setRightTurnActive(mqttData.right_indicator === 1);
          }
          
          // Update PIR alert
          if (mqttData.pir_alert !== undefined) {
            setPirAlert(mqttData.pir_alert);
          }
          
          setMqttConnected(mqttData.connected || false);
        }
      } catch (error) {
        console.error('Error polling MQTT state:', error);
        setMqttConnected(false);
      }
    };

    // Poll every 200ms for responsive updates
    const interval = setInterval(pollMqttState, 200);
    
    return () => clearInterval(interval);
  }, [useMqtt]);

  // Cleanup: Stop any active feature when component unmounts
  useEffect(() => {
    return () => {
      if (activeFeature) {
        stopActiveFeature(activeFeature);
      }
    };
  }, [activeFeature]);

  // BROWSER REFRESH/CLOSE: Send cleanup beacon to stop all cameras
  useEffect(() => {
    const handleBeforeUnload = () => {
      // Use sendBeacon for reliable delivery even during page unload
      const API_BASE = 'http://localhost:5000/api';
      navigator.sendBeacon(`${API_BASE}/cleanup/force`);
      console.log('🔄 Browser refresh/close - sent cleanup beacon');
    };

    // Listen for page unload (refresh, close, navigate away)
    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  // Throttle effect - increase speed while holding, decrease when released
  // Battery drains slightly when driving fast
  // ONLY active when NOT using MQTT control
  useEffect(() => {
    if (useMqtt) return; // Skip keyboard throttle control if using MQTT

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
  }, [isThrottling, useMqtt]);

  // Keyboard controls for speed, RPM, and turn signals
  // ONLY active when NOT using MQTT control
  useEffect(() => {
    if (useMqtt) return; // Skip keyboard control if using MQTT

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
  }, [useMqtt]);

  // Stop active feature before switching to a new one
  const stopActiveFeature = async (currentFeature) => {
    const API_BASE = 'http://localhost:5000/api';
    
    try {
      let response;
      
      // Use AbortController with timeout to prevent hanging on slow stop
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // 3 second timeout
      
      try {
        if (currentFeature === 'pothole') {
          console.log('Stopping pothole detection...');
          response = await fetch(`${API_BASE}/pothole/stop`, { 
            method: 'POST',
            signal: controller.signal
          });
        } else if (currentFeature === 'blindspot') {
          console.log('Stopping blind spot detection...');
          response = await fetch(`${API_BASE}/blindspot/stop`, { 
            method: 'POST',
            signal: controller.signal
          });
        } else if (currentFeature === 'dms') {
          console.log('Stopping DMS detection...');
          response = await fetch(`${API_BASE}/dms/stop`, { 
            method: 'POST',
            signal: controller.signal
          });
        }
        clearTimeout(timeoutId);
      } catch (abortError) {
        console.log(`${currentFeature} stop request timed out or aborted`);
        clearTimeout(timeoutId);
      }
      
      // Wait for response to confirm stop is complete
      if (response) {
        try {
          await response.json();
        } catch (e) {
          // Ignore JSON parse errors on aborted requests
        }
      }
      
      // Longer delay on Jetson to ensure camera is fully released
      // This prevents "camera busy" errors when switching
      await new Promise(resolve => setTimeout(resolve, 500));
      console.log(`✓ ${currentFeature} stopped and camera released`);
    } catch (error) {
      console.error(`Error stopping ${currentFeature}:`, error);
      // Even on error, wait a bit before trying to open new camera
      await new Promise(resolve => setTimeout(resolve, 300));
    }
  };

  // Stop ALL cameras and trigger garbage collection when going home
  const stopAllCamerasAndCleanup = async () => {
    const API_BASE = 'http://localhost:5000/api';
    console.log('🏠 Home button clicked - stopping all cameras and cleaning up...');
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
    
    try {
      // Call the dedicated cleanup endpoint that stops all cameras and forces GC
      const response = await fetch(`${API_BASE}/cleanup/force`, { 
        method: 'POST',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        const result = await response.json();
        console.log('✓ Cleanup result:', result);
      }
      
      // Additional delay for complete cleanup on Jetson
      await new Promise(resolve => setTimeout(resolve, 500));
      
      console.log('✓ All cameras stopped and cleaned up');
    } catch (error) {
      console.error('Error during cleanup:', error);
      clearTimeout(timeoutId);
      
      // Fallback: try individual stops
      const stopPromises = [];
      
      stopPromises.push(
        fetch(`${API_BASE}/pothole/stop`, { method: 'POST' })
          .catch(err => console.log('Pothole stop:', err.message))
      );
      
      stopPromises.push(
        fetch(`${API_BASE}/blindspot/stop`, { method: 'POST' })
          .catch(err => console.log('Blindspot stop:', err.message))
      );
      
      stopPromises.push(
        fetch(`${API_BASE}/dms/stop`, { method: 'POST' })
          .catch(err => console.log('DMS stop:', err.message))
      );
      
      await Promise.allSettled(stopPromises);
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  };

  const handleFeatureClick = async (featureId) => {
    // Prevent rapid concurrent switching
    if (isSwitching) {
      console.log('⏳ Switch in progress, please wait...');
      return;
    }
    
    // If clicking home button (featureId === null)
    if (featureId === null) {
      setIsSwitching(true);
      
      // INSTANT UI switch - don't wait for backend cleanup
      setActiveFeature(null);
      setPreviousFeature(null);
      
      // Reset all states for fresh start (no browser refresh)
      setSpeed(0);
      setBattery(100);
      setLeftTurnActive(false);
      setRightTurnActive(false);
      setPirAlert(0);
      setTripDistance(1000);
      
      if (onSelectUseCase) {
        onSelectUseCase(null);
      }
      
      // Note: Cleanup is already sent via beacon in FeatureBar.jsx
      // No need to call stopAllCamerasAndCleanup() here - it causes double cleanup
      
      // Allow next switch after brief delay
      setTimeout(() => setIsSwitching(false), 200);
      return;
    }
    
    // If clicking the same feature, do nothing
    if (activeFeature === featureId) return;
    
    const currentFeature = activeFeature;
    setIsSwitching(true);
    
    // IMPORTANT: Stop previous feature COMPLETELY before starting new one
    // This prevents camera conflicts on Jetson
    if (currentFeature) {
      // Store current feature as previous - UI will keep showing it during transition
      setPreviousFeature(currentFeature);
      
      // Set activeFeature to null to unmount the component
      // This releases the browser's MJPEG connection
      setActiveFeature(null);
      
      // Wait a frame for React to unmount the component
      await new Promise(resolve => setTimeout(resolve, 50));
      
      // Now stop the backend service
      await stopActiveFeature(currentFeature);
      
      // Additional delay to ensure camera is fully released
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    
    // Now set new feature and allow new component to mount
    setActiveFeature(featureId);
    setPreviousFeature(null); // Clear previous - new feature is now showing
    
    if (onSelectUseCase) {
      onSelectUseCase(featureId);
    }
    
    // Allow next switch after the new feature starts loading
    setTimeout(() => setIsSwitching(false), 500);
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-2 sm:p-4">
      {/* Oval Dashboard Container */}
    <div className="relative w-full max-w-[1320px] aspect-16/8">
        {/* Outer Decorative Border with Car Dashboard Shape */}
        <div className="absolute inset-0" style={{
            borderRadius: '45% 45% 40% 40% / 35% 35% 30% 30%',
            border: '4.4px solid #1f2937',
            boxShadow: '0 0 44px rgba(6, 182, 212, 0.3), inset 0 0 55px rgba(0, 0, 0, 0.5)',
            background: 'linear-gradient(180deg, rgba(17, 24, 39, 0.4) 0%, rgba(0, 0, 0, 0.8) 100%)'
        }}></div>

        {/* Middle decorative ring */}
        <div className="absolute inset-3" style={{
            borderRadius: '45% 45% 40% 40% / 35% 35% 30% 30%',
            border: '1.1px solid rgba(6, 182, 212, 0.2)',
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
            <div className="flex flex-col items-center gap-2 w-full max-w-[1760px]">
            {/* Top Feature Buttons */}
            <FeatureBar activeFeature={activeFeature} onFeatureClick={handleFeatureClick} />

            {/* Main Dashboard Container - Meters and Center Display */}
            <div className="flex flex-col lg:flex-row items-center justify-center gap-0 w-full">
              {/* Left - Speedometer - Always visible */}
                <div className="w-full max-w-[418px] shrink-0 flex justify-center">
                <Speedometer value={speed} />
              </div>

              {/* Center Display - Responsive */}
                <div className="w-full max-w-[880px] shrink-0 flex justify-center">
                  <div className="bg-gray-900/90 backdrop-blur-sm border-4 border-gray-700 rounded-3xl overflow-hidden p-2 shadow-2xl w-full" style={{ minHeight: '352px', height: '528px', maxWidth: '990px' }}>
                  <div className="h-full w-full">
                    <div className="bg-gray-900 border-2 rounded-2xl h-full w-full overflow-hidden relative">
                      {/* Show loading overlay during feature transition */}
                      {isSwitching && previousFeature && !activeFeature && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-900 z-20">
                          <div className="animate-spin rounded-full h-12 w-12 border-4 border-cyan-500 border-t-transparent mb-4"></div>
                          <p className="text-white text-lg">Switching...</p>
                        </div>
                      )}
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
                      ) : activeFeature === 'dms' ? (
                        <DMSDetector onBack={async () => {
                          await stopActiveFeature('dms');
                          setActiveFeature(null);
                        }} />
                      ) : !isSwitching ? (
                        <Car3DView pirAlert={pirAlert} />
                      ) : null}
                    </div>
                  </div>
                </div>
              </div>

              {/* Right - Battery Meter - Always visible */}
                <div className="w-full max-w-[418px] shrink-0 flex justify-center">
                <BatteryMeter value={battery} />
              </div>
            </div>

            {/* Bottom Status Bar */}
              <StatusBar 
                time={time} 
                mqttConnected={mqttConnected}
                useMqtt={useMqtt}
                onToggleMqtt={() => setUseMqtt(!useMqtt)}
                tripDistance={tripDistance}
                sizeMultiplier={1.1}
              />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
