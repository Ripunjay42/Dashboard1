import React, { useState } from 'react';
import { TbRoad, TbEye, TbCar } from 'react-icons/tb';
import { GiCarSeat, GiSpeedometer } from 'react-icons/gi';
import { IoShieldCheckmark } from 'react-icons/io5';
import { FaCar, FaTruck, FaBus } from 'react-icons/fa';
import ConfigDropdown from './ConfigDropdown';

const FeatureBar = ({ activeFeature, onFeatureClick }) => {
  const [selectionMode, setSelectionMode] = useState(null); // null, 'vehicle', 'camera', 'algo'
  const [selectedVehicle, setSelectedVehicle] = useState(null);
  const [selectedCameraCount, setSelectedCameraCount] = useState(null);
  const [showDropdown, setShowDropdown] = useState(false);

  const vehicleTypes = [
    { id: 'car', label: 'Car', icon: <FaCar className="w-5 h-5" /> },
    { id: 'truck', label: 'Truck', icon: <FaTruck className="w-5 h-5" />, disabled: true },
    { id: 'bus', label: 'Bus', icon: <FaBus className="w-5 h-5" />, disabled: true },
  ];

  const cameraCounts = [
    { id: '1', label: '1 Camera', count: 1, disabled: true },
    { id: '2', label: '2 Cameras', count: 2, disabled: true },
    { id: '3', label: '3 Cameras', count: 3 },
  ];

  const algoButtons = [
    { 
      id: 'pothole', 
      label: 'Pothole', 
      icon: <TbRoad className="w-4 h-4 text-orange-400" />, 
      status: 'active' 
    },
    { 
      id: 'blindspot', 
      label: 'Blind Spot', 
      icon: <TbEye className="w-4 h-4 text-purple-400" />, 
      status: 'active' 
    },
    { 
      id: 'dms', 
      label: 'DMS', 
      icon: <GiCarSeat className="w-4 h-4 text-blue-400" />, 
      status: 'active' 
    },
  ];

  const featureButtons = [
    { 
      id: 'cruise', 
      label: 'Cruise', 
      icon: <GiSpeedometer className="w-4 h-4 text-green-400" />, 
      status: 'active' 
    },
    { 
      id: 'security', 
      label: 'Security', 
      icon: <IoShieldCheckmark className="w-4 h-4 text-cyan-400" />, 
      status: 'active' 
    }
  ];

  const handleAlgoSelect = (algoId) => {
    const algo = algoButtons.find(a => a.id === algoId);
    if (algo && algo.status === 'active') {
      onFeatureClick(algoId);
      setSelectionMode(null);
      setShowDropdown(false);
    }
  };

  const handleFeatureSelect = (featureId) => {
    onFeatureClick(featureId);
  };

  const handleConfigClick = () => {
    setShowDropdown(!showDropdown);
    if (!showDropdown) {
      setSelectionMode('vehicle');
    } else {
      setSelectionMode(null);
    }
  };

  const handleCloseDropdown = () => {
    setSelectionMode(null);
    setShowDropdown(false);
  };

  // Main view with config button + all features
  return (
    <div className="relative w-full max-w-[750px] mx-auto">
      {/* Main Feature Bar */}
      <div className="bg-gray-900/90 backdrop-blur-sm border-2 border-gray-700 rounded-3xl px-2 sm:px-2 py-2 sm:py-3 shadow-xl">
        <div className="flex flex-wrap items-center justify-center gap-1 sm:gap-1">
          {/* Vehicle Setup Button */}
          <button
            onClick={handleConfigClick}
            className={`flex items-center gap-2 px-3 sm:px-4 py-2 rounded-lg transition-all text-xs sm:text-sm text-white ${
              showDropdown ? 'bg-gray-800' : 'hover:bg-gray-800'
            }`}
          >
            <TbCar className="w-4 h-4" />
            <div className="text-left">
              <div className="font-bold">Vehicle</div>
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
                <div className="font-bold">{algo.label}</div>
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
                <div className="font-bold">{item.label}</div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Dropdown Panel */}
      {showDropdown && (
        <ConfigDropdown
          selectionMode={selectionMode}
          setSelectionMode={setSelectionMode}
          selectedVehicle={selectedVehicle}
          setSelectedVehicle={setSelectedVehicle}
          selectedCameraCount={selectedCameraCount}
          setSelectedCameraCount={setSelectedCameraCount}
          vehicleTypes={vehicleTypes}
          cameraCounts={cameraCounts}
          algoButtons={algoButtons}
          activeFeature={activeFeature}
          onAlgoSelect={handleAlgoSelect}
          onClose={handleCloseDropdown}
        />
      )}
    </div>
  );
};

export default FeatureBar;
