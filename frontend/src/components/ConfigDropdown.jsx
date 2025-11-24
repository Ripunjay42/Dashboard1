import React from 'react';
import { TbCamera } from 'react-icons/tb';
import { FaCar } from 'react-icons/fa';

const ConfigDropdown = ({ 
  selectionMode, 
  setSelectionMode,
  selectedVehicle,
  setSelectedVehicle,
  selectedCameraCount,
  setSelectedCameraCount,
  vehicleTypes,
  cameraCounts,
  algoButtons,
  activeFeature,
  onAlgoSelect,
  onClose 
}) => {

  const handleVehicleSelect = (vehicleId) => {
    setSelectedVehicle(vehicleId);
    setSelectionMode('camera');
  };

  const handleCameraSelect = (cameraId) => {
    setSelectedCameraCount(cameraId);
    setSelectionMode('algo');
  };

  return (
    <div className="absolute top-full left-1/2 translate mt-2 z-50 w-full animate-slideDown max-w-1/2">
      <div className="bg-gray-900/95 backdrop-blur-sm border-2 border-gray-700 rounded-2xl px-4 py-4 shadow-2xl">
        {/* Vehicle Selection View */}
        {selectionMode === 'vehicle' && (
          <div className="flex flex-col gap-3 animate-fadeIn">
            <div className="flex items-center justify-between mb-2">
              <button
                onClick={onClose}
                className="flex items-center gap-1 px-3 py-1 text-xs text-gray-400 hover:text-white transition-colors"
              >
                Close
              </button>
              <h3 className="text-sm font-bold text-white">Select Vehicle</h3>
              <div className="w-16"></div>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-5">
              {vehicleTypes.map((vehicle) => (
                <button
                  key={vehicle.id}
                  onClick={() => !vehicle.disabled && handleVehicleSelect(vehicle.id)}
                  disabled={vehicle.disabled}
                  className={`flex flex-col items-center px-1 py-1 rounded-lg transition-all duration-300 transform hover:scale-110 ${
                    vehicle.disabled
                      ? ' text-gray-500 cursor-not-allowed opacity-50'
                      : selectedVehicle === vehicle.id
                      ? 'text-white scale-110'
                      : 'text-white hover:bg-gray-700 hover:border-gray-500 cursor-pointer'
                  }`}
                >
                  <span className="text-3xl">{vehicle.icon}</span>
                  <div className="text-sm font-bold">{vehicle.label}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Camera Count Selection View */}
        {selectionMode === 'camera' && (
          <div className="flex flex-col gap-3 animate-fadeIn">
            <div className="flex items-center justify-between mb-2">
              <button
                onClick={() => setSelectionMode('vehicle')}
                className="flex items-center gap-1 px-3 py-1 text-xs text-gray-400 hover:text-white transition-colors"
              >
                ← Back
              </button>
              <h3 className="text-sm font-bold text-white">Select Camera</h3>
              <button
                onClick={onClose}
                className="flex items-center gap-1 px-3 py-1 text-xs text-gray-400 hover:text-white transition-colors"
              >
                Close
              </button>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-5">
              {cameraCounts.map((camera) => (
                <button
                  key={camera.id}
                  onClick={() => !camera.disabled && handleCameraSelect(camera.id)}
                  disabled={camera.disabled}
                  className={`flex flex-col items-center px-1 py-1 rounded-lg transition-all duration-300 transform hover:scale-110 ${
                    camera.disabled
                      ? 'text-gray-500 cursor-not-allowed opacity-50'
                      : selectedCameraCount === camera.id
                      ? 'text-white scale-110'
                      : 'text-white hover:bg-gray-700 cursor-pointer'
                  }`}
                >
                  <TbCamera className="w-8 h-8" />
                  <div className="text-sm font-bold">{camera.label}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Algorithm Selection View */}
        {selectionMode === 'algo' && (
          <div className="flex flex-col gap-3 animate-fadeIn">
            <div className="flex items-center justify-between mb-2">
              <button
                onClick={() => setSelectionMode('camera')}
                className="flex items-center gap-1 px-3 py-1 text-xs text-gray-400 hover:text-white transition-colors"
              >
                ← Back
              </button>
              <h3 className="text-sm font-bold text-white">Select Algorithm</h3>
              <button
                onClick={onClose}
                className="flex items-center gap-1 px-3 py-1 text-xs text-gray-400 hover:text-white transition-colors"
              >
                Close
              </button>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-5">
              {algoButtons.map((algo) => (
                <button
                  key={algo.id}
                  onClick={() => onAlgoSelect(algo.id)}
                  disabled={algo.status !== 'active'}
                  className={`flex flex-col items-center px-1 py-1 rounded-lg transition-all duration-300 transform hover:scale-110 ${
                    algo.status === 'active'
                      ? activeFeature === algo.id
                        ? 'text-white scale-110'
                        : 'text-white hover:bg-gray-700 cursor-pointer'
                      : 'text-gray-500 cursor-not-allowed opacity-50'
                  }`}
                >
                  <span className="text-3xl">{algo.icon}</span>
                  <div className="text-center">
                    <div className="text-sm font-bold">{algo.label}</div>
                    {algo.status !== 'active' && (
                      <div className="text-xs text-gray-500">Coming Soon</div>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ConfigDropdown;
