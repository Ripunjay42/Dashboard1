import React from 'react';
import { 
  MdAcUnit, 
  MdLocalGasStation, 
  MdMyLocation, 
  MdAccessTime, 
  MdCalendarToday
} from 'react-icons/md';
import { TbManualGearbox } from 'react-icons/tb';

const StatusBar = ({ time, mqttConnected = false, useMqtt = false, onToggleMqtt }) => {
  const bottomStatusItems = [
    { 
      id: 'ac', 
      label: 'Climate', 
      icon: <MdAcUnit className="w-4 h-4 text-cyan-400" />, 
      value: '22°C' 
    },
    { 
      id: 'distance', 
      label: 'Trip', 
      icon: <MdMyLocation className="w-4 h-4 text-green-400" />, 
      value: '2134km' 
    }
  ];

  return (
    <div className="bg-gray-900/90 backdrop-blur-sm border-2 border-gray-700 rounded-3xl px-2 sm:px-2 py-2 sm:py-3 shadow-xl w-full max-w-[800px] mx-auto">
      <div className="flex flex-wrap items-center gap-2 sm:gap-4 text-xs sm:text-sm justify-center">
        {/* Simple Toggle Switch */}
        <div 
          className="relative inline-flex items-center cursor-pointer"
          onClick={onToggleMqtt}
          title={useMqtt ? "MQTT Mode (Click for Keyboard)" : "Keyboard Mode (Click for MQTT)"}
        >
          <div className={`w-12 h-6 rounded-full transition-colors duration-300 ${
            useMqtt ? (mqttConnected ? 'bg-transparent' : 'bg-transparent') : 'bg-transparent'
          }`}>
            <div className={`absolute top-1 left-1 w-4 h-4 rounded-full transition-transform duration-300 ${
              useMqtt ? (mqttConnected ? 'bg-green-500 translate-x-6' : 'bg-yellow-300 translate-x-6') : 'translate-x-0 bg-red-500'
            }`}></div>
          </div>
        </div>
        <div className="text-gray-500">|</div>
        
        {bottomStatusItems.map((item, index) => (
          <React.Fragment key={item.id}>
            <div className="flex items-center gap-1">
              <span className="text-md">{item.icon}</span>
              <div>
                <div className="text-xs text-gray-400">{item.label}</div>
                <div className="text-xs font-bold text-white">{item.value}</div>
              </div>
            </div>
            {index < bottomStatusItems.length - 1 && <div className="text-gray-500">|</div>}
          </React.Fragment>
        ))}
        <div className="text-gray-500">|</div>
        <div className="flex items-center gap-2 text-gray-300">
          <MdAccessTime className="w-4 h-4 text-blue-400" />
          {time.toLocaleTimeString()}
        </div>
        <div className="text-gray-500">|</div>
        <div className="flex items-center gap-2 text-gray-300">
          <MdCalendarToday className="w-4 h-4 text-purple-400" />
          {time.toLocaleDateString()}
        </div>
        <div className="text-gray-500">|</div>
        <div className="flex items-center gap-2 text-gray-300">
          <TbManualGearbox className="w-4 h-4 text-orange-400" />
          PARK
        </div>
      </div>
    </div>
  );
};

export default StatusBar;
