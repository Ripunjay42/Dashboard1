import React from 'react';
import { 
  MdAcUnit, 
  MdLocalGasStation, 
  MdMyLocation, 
  MdAccessTime, 
  MdCalendarToday,
  MdWifi,
  MdWifiOff,
  MdWarning
} from 'react-icons/md';
import { TbManualGearbox } from 'react-icons/tb';

const StatusBar = ({ time, mqttConnected = false, pirAlert = 0, useMqtt = true, onToggleMqtt }) => {
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
        {/* MQTT Status Indicator */}
        <div 
          className="flex items-center gap-1 cursor-pointer hover:opacity-80 transition-opacity"
          onClick={onToggleMqtt}
          title={useMqtt ? (mqttConnected ? "MQTT Connected (Click to use Keyboard)" : "MQTT Disconnected") : "Using Keyboard (Click to use MQTT)"}
        >
          {useMqtt ? (
            mqttConnected ? (
              <MdWifi className="w-4 h-4 text-green-400 animate-pulse" />
            ) : (
              <MdWifiOff className="w-4 h-4 text-red-400" />
            )
          ) : (
            <div className="w-4 h-4 text-yellow-400 font-bold text-xs flex items-center justify-center">KB</div>
          )}
          <div>
            <div className="text-xs text-gray-400">
              {useMqtt ? "MQTT" : "Keyboard"}
            </div>
            <div className={`text-xs font-bold ${useMqtt ? (mqttConnected ? 'text-green-400' : 'text-red-400') : 'text-yellow-400'}`}>
              {useMqtt ? (mqttConnected ? 'ON' : 'OFF') : 'ON'}
            </div>
          </div>
        </div>
        <div className="text-gray-500">|</div>
        
        {/* PIR Alert Indicator */}
        {pirAlert === 1 && (
          <>
            <div className="flex items-center gap-1 animate-pulse">
              <MdWarning className="w-5 h-5 text-red-500" />
              <div>
                <div className="text-xs text-gray-400">Alert</div>
                <div className="text-xs font-bold text-red-500">PIR DETECTED</div>
              </div>
            </div>
            <div className="text-gray-500">|</div>
          </>
        )}
        
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
