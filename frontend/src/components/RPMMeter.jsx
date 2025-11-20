import React from 'react';

const RPMMeter = ({ value, max = 8000 }) => {
  const angle = (value / max) * 240 - 120;
  
  return (
    <div className="relative w-80 h-80">
      <div className="absolute inset-0 rounded-full border-8 border-gray-700"></div>
      
      {/* Numbers around the RPM meter */}
      <div className="absolute inset-0">
        {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((mark) => {
          const markValue = mark * 1000;
          const markAngle = (markValue / max) * 240 - 120;
          const angleInRadians = (markAngle * Math.PI) / 180;
          const radiusPixels = 130; // Distance from center in pixels
          const centerX = 160; // Half of 256px (w-80)
          const centerY = 160;
          const x = centerX + radiusPixels * Math.sin(angleInRadians);
          const y = centerY - radiusPixels * Math.cos(angleInRadians);
          const isRed = mark >= 6;
          
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
        {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((mark) => {
          const markValue = mark * 1000;
          const markAngle = (markValue / max) * 240 - 120;
          const isRed = mark >= 6;
          return (
            <div key={mark}>
              <div
                className="absolute w-0.5 h-4 origin-bottom"
                style={{
                  left: '50%',
                  top: '10%',
                  transform: `translateX(-50%) rotate(${markAngle}deg)`,
                  backgroundColor: isRed ? '#ef4444' : '#9ca3af'
                }}
              />
            </div>
          );
        })}
      </div>
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <div className="text-5xl font-bold text-white">{Math.floor(value / 1000)}</div>
          <div className="text-xs text-gray-400 mt-1">x1000 RPM</div>
        </div>
      </div>
      <div
        className="absolute bottom-1/2 left-1/2 w-1 h-24 bg-orange-500 origin-bottom transition-transform duration-500"
        style={{
          transform: `translateX(-50%) rotate(${angle}deg)`,
          boxShadow: '0 0 10px rgba(249, 115, 22, 0.8)'
        }}
      >
        <div className="absolute top-0 left-1/2 w-3 h-3 bg-orange-500 rounded-full -translate-x-1/2"></div>
      </div>
    </div>
  );
};

export default RPMMeter;
