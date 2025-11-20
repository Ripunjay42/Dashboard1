import React from 'react';

const Speedometer = ({ value, max = 300 }) => {
  const angle = (value / max) * 240 - 120;
  
  return (
    <div className="relative w-80 h-80">
      <div className="absolute inset-0 rounded-full border-8 border-gray-700"></div>
      
      {/* Numbers around the speedometer */}
      <div className="absolute inset-0">
        {[0, 40, 80, 120, 160, 200, 240, 280].map((mark) => {
          const markAngle = (mark / max) * 240 - 120;
          const angleInRadians = (markAngle * Math.PI) / 180;
          const radiusPixels = 130; // Distance from center in pixels
          const centerX = 160; // Half of 256px (w-64)
          const centerY = 160;
          const x = centerX + radiusPixels * Math.sin(angleInRadians);
          const y = centerY - radiusPixels * Math.cos(angleInRadians);
          const isRed = mark >= 240;
          
          return (
            <div
              key={`num-${mark}`}
              className="absolute text-xs font-bold"
              style={{
                left: `${x}px`,
                top: `${y}px`,
                transform: 'translate(-50%, -50%)',
                color: isRed ? '#ef4444' : '#9ca3af'
              }}
            >
              {mark}
            </div>
          );
        })}
      </div>
      
      {/* Tick marks */}
      <div className="absolute inset-4 rounded-full border-2 border-gray-600">
        {[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300].map((mark) => {
          const markAngle = (mark / max) * 240 - 120;
          const isRed = mark >= 240;
          const isMajor = mark % 40 === 0;
          return (
            <div
              key={mark}
              className="absolute origin-bottom"
              style={{
                left: '50%',
                top: '10%',
                width: isMajor ? '2px' : '1px',
                height: isMajor ? '12px' : '8px',
                transform: `translateX(-50%) rotate(${markAngle}deg)`,
                backgroundColor: isRed ? '#ef4444' : '#9ca3af'
              }}
            />
          );
        })}
      </div>
      
      {/* Center display */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <div className="text-6xl font-bold text-white">{value}</div>
          <div className="text-sm text-gray-400">km/h</div>
        </div>
      </div>
      
      {/* Needle */}
      <div
        className="absolute bottom-1/2 left-1/2 w-1 h-24 bg-cyan-400 origin-bottom transition-transform duration-500"
        style={{
          transform: `translateX(-50%) rotate(${angle}deg)`,
          boxShadow: '0 0 10px rgba(34, 211, 238, 0.8)'
        }}
      >
        <div className="absolute top-0 left-1/2 w-3 h-3 bg-cyan-400 rounded-full -translate-x-1/2"></div>
      </div>
    </div>
  );
};

export default Speedometer;
