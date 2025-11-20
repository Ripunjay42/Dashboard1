import React, { useState, useEffect } from 'react';
import Speedometer from '../components/Speedometer';
import RPMMeter from '../components/RPMMeter';
import PotholeDetector from '../components/PotholeDetector';

const Dashboard = ({ onSelectUseCase }) => {
  const [time, setTime] = useState(new Date());
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [activeTab, setActiveTab] = useState('home');
  const [activeFeature, setActiveFeature] = useState(null); // Track which feature is active
  const [speed, setSpeed] = useState(0);
  const [rpm, setRpm] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const useCases = [
    {
      id: 'pothole',
      name: 'UC 2.5',
      title: 'Pothole Detection',
      category: 'Safety',
      status: 'active',
      icon: '🛣️'
    },
    {
      id: 'threewheeler',
      name: 'UC 2.1',
      title: 'Three Wheeler',
      category: 'Safety',
      status: 'inactive',
      icon: '🛺'
    },
    {
      id: 'blindspot',
      name: 'UC 2.2',
      title: 'Blind Spot',
      category: 'Safety',
      status: 'inactive',
      icon: '👁️'
    },
    {
      id: 'chassis',
      name: 'UC 2.3',
      title: 'Chassis',
      category: 'Safety',
      status: 'inactive',
      icon: '⚙️'
    },
    {
      id: 'dms',
      name: 'UC 2.4',
      title: 'Driver Monitor',
      category: 'Safety',
      status: 'inactive',
      icon: '👤'
    },
    {
      id: 'throttle',
      name: 'UC 2.6',
      title: 'Throttle',
      category: 'Control',
      status: 'inactive',
      icon: '🎮'
    },
    {
      id: 'cruise',
      name: 'UC 2.8',
      title: 'Cruise Control',
      category: 'Control',
      status: 'inactive',
      icon: '🚗'
    },
    {
      id: 'drivetrain',
      name: 'UC 2.11',
      title: 'Drive Train',
      category: 'Control',
      status: 'inactive',
      icon: '⚡'
    },
    {
      id: 'simulation',
      name: 'UC 2.7',
      title: 'Simulation',
      category: 'Smart',
      status: 'inactive',
      icon: '🖥️'
    },
    {
      id: 'security',
      name: 'UC 2.9',
      title: 'Security',
      category: 'Smart',
      status: 'inactive',
      icon: '🔒'
    },
    {
      id: 'battery',
      name: 'UC 2.10',
      title: 'Battery',
      category: 'Smart',
      status: 'inactive',
      icon: '🔋'
    },
    {
      id: 'hpc',
      name: 'UC 2.12',
      title: 'HPC',
      category: 'Smart',
      status: 'inactive',
      icon: '💻'
    }
  ];

  const categories = ['Safety', 'Control', 'Smart'];

  const filteredUseCases = selectedCategory
    ? useCases.filter(uc => uc.category === selectedCategory)
    : useCases;

  // Top buttons - Key features
  const featureButtons = [
    { id: 'pothole', label: 'Pothole', icon: '🛣️', status: 'active' },
    { id: 'blindspot', label: 'Blind Spot', icon: '👁️', status: 'inactive' },
    { id: 'dms', label: 'DMS', icon: '👤', status: 'inactive' },
    { id: 'cruise', label: 'Cruise', icon: '🚗', status: 'inactive' },
    { id: 'security', label: 'Security', icon: '🔒', status: 'inactive' }
  ];

  const handleFeatureClick = (featureId) => {
    if (featureId === 'pothole') {
      setActiveFeature('pothole');
      // Also call the parent's onSelectUseCase if needed
      if (onSelectUseCase) {
        onSelectUseCase('pothole');
      }
    }
  };

  // Bottom status items
  const bottomStatusItems = [
    // { id: 'battery', label: 'Battery', icon: '🔋', value: '46%' },
    { id: 'ac', label: 'Climate', icon: '❄️', value: '-12°C' },
    { id: 'fuel', label: 'Fuel', icon: '⛽', value: '130km' },
    { id: 'distance', label: 'Trip', icon: '📍', value: '21341km' }
  ];

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-4">
      {/* Oval Dashboard Container */}
      <div className="relative" style={{ 
        width: '92%', 
        maxWidth: '1800px',
        aspectRatio: '16/7'
      }}>
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
        <div className="relative w-full h-full flex items-center justify-center" style={{ padding: '25px 50px' }}>
          
          {/* Main Content Area - Vertically stacked with bars */}
          <div className="flex flex-col items-center gap-3 w-full" style={{ maxWidth: '1600px' }}>
            
            {/* Top Feature Buttons - Aligned with main component */}
            <div className="bg-gray-900/90 backdrop-blur-sm border-2 border-gray-700 rounded-3xl px-6 py-3 shadow-xl">
              <div className="flex items-center justify-center gap-3">
                {featureButtons.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => item.status === 'active' && handleFeatureClick(item.id)}
                    disabled={item.status !== 'active'}
                    className={`flex items-center gap-2 px-5 py-2 rounded-lg transition-all ${
                      item.status === 'active'
                        ? activeFeature === item.id
                          ? 'bg-gray-900 text-white'
                          : 'bg-gray-900 text-white hover:bg-gray-800 cursor-pointer'
                        : 'bg-gray-900 text-white cursor-not-allowed opacity-50'
                    }`}
                  >
                    <span className="text-xl">{item.icon}</span>
                    <div className="text-left">
                      <div className="text-sm font-bold">{item.label}</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Main Dashboard Container - Meters and Center Display */}
            <div className="flex items-center justify-center gap-6">
              {/* Left - Speedometer - Always visible */}
              <div className="flex-0">
                <Speedometer value={speed} />
              </div>

          {/* Center Display with fixed size */}
          <div className="flex-0" style={{ width: '900px' }}>
            <div className="bg-gray-900/90 backdrop-blur-sm border-4 border-gray-700 rounded-3xl overflow-hidden p-2 shadow-2xl" style={{ height: '480px', width: '900px' }}>
              <div className="h-full w-full">
                <div className="bg-gray-900 border-2 rounded-2xl h-full w-full overflow-hidden relative">
                  {/* Show Pothole Detection when active */}
                  {activeFeature === 'pothole' ? (
                    <PotholeDetector onBack={() => setActiveFeature(null)} />
                  ) : (
                    /* Default Car Dashboard View */
                    <div className="h-full flex flex-col items-center justify-center p-8">
                      {/* Car Silhouette */}
                      <div className="mb-8">
                        <svg width="400" height="200" viewBox="0 0 400 200" className="text-cyan-400">
                          {/* Car Body */}
                          <path d="M 80 120 L 60 140 L 60 160 L 340 160 L 340 140 L 320 120 L 280 80 L 120 80 Z" 
                            fill="none" stroke="currentColor" strokeWidth="3" strokeLinejoin="round"/>
                          {/* Windshield */}
                          <path d="M 140 80 L 160 100 L 240 100 L 260 80" 
                            fill="none" stroke="currentColor" strokeWidth="3"/>
                          {/* Windows */}
                          <path d="M 160 100 L 170 120 L 230 120 L 240 100" 
                            fill="none" stroke="currentColor" strokeWidth="2"/>
                          {/* Left Wheel */}
                          <circle cx="120" cy="160" r="20" fill="none" stroke="currentColor" strokeWidth="3"/>
                          <circle cx="120" cy="160" r="10" fill="none" stroke="currentColor" strokeWidth="2"/>
                          {/* Right Wheel */}
                          <circle cx="280" cy="160" r="20" fill="none" stroke="currentColor" strokeWidth="3"/>
                          <circle cx="280" cy="160" r="10" fill="none" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                      </div>
                      
                      {/* Dashboard Title */}
                      <h1 className="text-4xl font-bold text-white mb-3">VEHICLE DASHBOARD</h1>
                      <p className="text-gray-400 text-lg mb-6">Experience Centre - Smart Vehicle System</p>
                      
                      {/* Status Indicators */}
                      <div className="flex gap-6 mt-4">
                        <div className="text-center">
                          <div className="text-3xl mb-2">✓</div>
                          <div className="text-sm text-cyan-400 font-semibold">ALL SYSTEMS</div>
                          <div className="text-xs text-gray-500">OPERATIONAL</div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl mb-2">🔋</div>
                          <div className="text-sm text-green-400 font-semibold">BATTERY</div>
                          <div className="text-xs text-gray-500">46% CHARGED</div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl mb-2">🛡️</div>
                          <div className="text-sm text-blue-400 font-semibold">SAFETY</div>
                          <div className="text-xs text-gray-500">ACTIVE</div>
                        </div>
                      </div>
                      
                      {/* Info Text */}
                      <div className="mt-8 text-center">
                        <p className="text-gray-500 text-sm">
                          Select a feature from the top menu to get started
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

              {/* Right - RPM Meter - Always visible */}
              <div className="flex-shrink-0">
                <RPMMeter value={rpm} />
              </div>
            </div>

            {/* Bottom Status Bar - Aligned with main component */}
            <div className="bg-gray-900/90 backdrop-blur-sm border-2 border-gray-700 rounded-3xl px-8 py-3 shadow-xl">
              <div className="flex items-center gap-6 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-gray-300">SYSTEM ONLINE</span>
                </div>
                <div className="text-gray-500">|</div>
                {bottomStatusItems.map((item, index) => (
                  <React.Fragment key={item.id}>
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{item.icon}</span>
                      <div>
                        <div className="text-xs text-gray-400">{item.label}</div>
                        <div className="text-sm font-bold text-white">{item.value}</div>
                      </div>
                    </div>
                    {index < bottomStatusItems.length - 1 && <div className="text-gray-500">|</div>}
                  </React.Fragment>
                ))}
                <div className="text-gray-500">|</div>
                <div className="text-gray-300">
                  {time.toLocaleTimeString()}
                </div>
                <div className="text-gray-500">|</div>
                <div className="text-gray-300">
                  {time.toLocaleDateString()}
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
    </div>
  );
};

export default Dashboard;
