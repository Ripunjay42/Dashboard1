import React from 'react';
import { TbRoad, TbEye } from 'react-icons/tb';
import { GiCarSeat, GiSpeedometer } from 'react-icons/gi';
import { IoShieldCheckmark } from 'react-icons/io5';

const FeatureBar = ({ activeFeature, onFeatureClick }) => {
  const featureButtons = [
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

  return (
    <div className="bg-gray-900/90 backdrop-blur-sm border-2 border-gray-700 rounded-3xl px-2 sm:px-6 py-2 sm:py-3 shadow-xl w-full max-w-[750px] mx-auto">
      <div className="flex flex-wrap items-center justify-center gap-1 sm:gap-1">
        {featureButtons.map((item) => (
          <button
            key={item.id}
            onClick={() => item.status === 'active' && onFeatureClick(item.id)}
            disabled={item.status !== 'active'}
            className={`flex items-center gap-2 px-3 sm:px-5 py-2 rounded-lg transition-all text-xs sm:text-sm ${
              item.status === 'active'
                ? activeFeature === item.id
                  ? 'bg-gray-900 text-white'
                  : 'bg-gray-900 text-white hover:bg-gray-800 cursor-pointer'
                : 'bg-gray-900 text-white cursor-not-allowed opacity-50'
            }`}
          >
            <span className="text-xl">{item.icon}</span>
            <div className="text-left">
              <div className="font-bold">{item.label}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default FeatureBar;
