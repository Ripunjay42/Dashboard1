import React, { useState } from 'react';
import { TbRoad, TbEye, TbHome } from 'react-icons/tb';
import { GiCarSeat, GiSpeedometer } from 'react-icons/gi';
import { IoShieldCheckmark } from 'react-icons/io5';
import { FaCar, FaTruck, FaBus } from 'react-icons/fa';
import ConfigDropdown from './ConfigDropdown';

const FeatureBar = ({ activeFeature, onFeatureClick }) => {



  const algoButtons = [
    { 
      id: 'pothole', 
      label: 'Pothole', 
      icon: <TbRoad className="w-5 h-5 text-orange-400" />, 
      status: 'active' 
    },
    { 
      id: 'blindspot', 
      label: 'Blind Spot', 
      icon: <TbEye className="w-5 h-5 text-purple-400" />, 
      status: 'active' 
    },
    { 
      id: 'dms', 
      label: 'DMS', 
      icon: <GiCarSeat className="w-5 h-5 text-blue-400" />, 
      status: 'active' 
    },
  ];

  const featureButtons = [
    { 
      id: 'cruise', 
      label: 'Cruise', 
      icon: <GiSpeedometer className="w-5 h-5 text-green-400" />, 
      status: 'active' 
    },
    { 
      id: 'security', 
      label: 'Security', 
      icon: <IoShieldCheckmark className="w-5 h-5 text-cyan-400" />, 
      status: 'active' 
    }
  ];

  const handleAlgoSelect = (algoId) => {
    const algo = algoButtons.find(a => a.id === algoId);
    if (algo && algo.status === 'active') {
      onFeatureClick(algoId);
    }
  };

  const handleFeatureSelect = (featureId) => {
    onFeatureClick(featureId);
  };

  const handleHomeClick = () => {
    onFeatureClick(null); // Go to home/main page
  };

  // Main view with home button + all features
  return (
    <div className="relative w-full max-w-[800px] mx-auto">
      {/* Main Feature Bar */}
      <div className="bg-gray-900/90 backdrop-blur-sm border-2 border-gray-700 rounded-3xl px-1 sm:px-1 py-2 sm:py-3 shadow-xl">
        <div className="flex flex-wrap items-center justify-center gap-1 sm:gap-1">
          {/* Home Button */}
          <button
            onClick={handleHomeClick}
            className={`flex items-center gap-2 px-3 sm:px-4 py-2 rounded-lg transition-all text-xs sm:text-sm text-white hover:bg-gray-800`}
          >
            <TbHome className="w-5 h-5" />
            <div className="text-left">
              <div className="font-bold text-md">Home</div>
            </div>
          </button>

          <div className="text-gray-500">|</div>

          {/* Algorithm Buttons */}
          {algoButtons.map((algo) => (
            <button
              key={algo.id}
              onClick={() => algo.status === 'active' && handleAlgoSelect(algo.id)}
              disabled={algo.status !== 'active'}
              className={`flex items-center gap-2 px-3 sm:px-5 py-2 rounded-lg transition-all text-xs sm:text-sm ${
                algo.status === 'active'
                  ? activeFeature === algo.id
                    ? 'bg-gray-700 text-white'
                    : 'bg-gray-900 text-white hover:bg-gray-800 cursor-pointer'
                  : 'bg-gray-900 text-white cursor-not-allowed opacity-50'
              }`}
            >
              <span className="text-xl">{algo.icon}</span>
              <div className="text-left">
                <div className="font-bold text-md">{algo.label}</div>
              </div>
            </button>
          ))}

          <div className="text-gray-500">|</div>

          {/* Other Feature Buttons */}
          {featureButtons.map((item) => (
            <button
              key={item.id}
              onClick={() => item.status === 'active' && handleFeatureSelect(item.id)}
              disabled={item.status !== 'active'}
              className={`flex items-center gap-2 px-3 sm:px-5 py-2 rounded-lg transition-all text-xs sm:text-sm ${
                item.status === 'active'
                  ? activeFeature === item.id
                    ? 'bg-gray-700 text-white'
                    : 'bg-gray-900 text-white hover:bg-gray-800 cursor-pointer'
                  : 'bg-gray-900 text-white cursor-not-allowed opacity-50'
              }`}
            >
              <span className="text-xl">{item.icon}</span>
              <div className="text-left">
                <div className="font-bold text-md">{item.label}</div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FeatureBar;
