import React from 'react';
import { MdWarning } from 'react-icons/md';
import carImage from '../assets/car_3d.png';

const Car3DView = ({ pirAlert = 0 }) => {
  return (
    <div className="h-full w-full flex items-center justify-center relative bg-gray-900 overflow-hidden">
      {/* 3D Road Background with Extended Perspective */}
      <div className="absolute inset-0 flex items-center justify-center" style={{
        perspective: '3500px',
        perspectiveOrigin: 'center center'
      }}>
        {/* Extended Road Surface with 3D Transform */}
        <div 
          className="relative w-full h-[1200%] bg-gray-900"
          style={{
            transform: 'rotateX(78deg) translateZ(-500px) translateY(-50%)',
            transformOrigin: 'center center',
            transformStyle: 'preserve-3d'
          }}
        >
          {/* Road Grid Pattern for 3D Effect */}
          <div className="absolute inset-0 opacity-5" style={{
            backgroundImage: `
              linear-gradient(to bottom, transparent 0%, rgba(255,255,255,0.1) 50%, transparent 100%),
              repeating-linear-gradient(
                0deg,
                transparent,
                transparent 35px,
                rgba(255, 255, 255, 0.15) 35px,
                rgba(255, 255, 255, 0.15) 40px
              )
            `
          }}></div>

          {/* Three Lane Lines Container */}
          <div className="absolute inset-0 flex justify-center items-center gap-0">
            {/* Left Lane Line */}
            <div className="absolute left-[20%] top-0 bottom-0 w-0.5 flex flex-col justify-start py-0 gap-px">
              {[...Array(200)].map((_, i) => (
                <div 
                  key={`left-${i}`}
                  className="w-full bg-white"
                  style={{
                    height: `${1.2 + i * 0.3}px`,
                    opacity: 0.99 - (i * 0.004)
                  }}
                ></div>
              ))}
            </div>

            {/* Center Lane Line */}
            <div className="absolute left-1/2 -translate-x-1/2 top-0 bottom-0 w-1 flex flex-col justify-start py-0 gap-px">
              {[...Array(200)].map((_, i) => (
                <div 
                  key={`center-${i}`}
                  className="w-full bg-yellow-400"
                  style={{
                    height: `${1.5 + i * 0.4}px`,
                    opacity: 1 - (i * 0.004)
                  }}
                ></div>
              ))}
            </div>

            {/* Right Lane Line */}
            <div className="absolute right-[20%] top-0 bottom-0 w-0.5 flex flex-col justify-start py-0 gap-px">
              {[...Array(200)].map((_, i) => (
                <div 
                  key={`right-${i}`}
                  className="w-full bg-white"
                  style={{
                    height: `${1.2 + i * 0.3}px`,
                    opacity: 0.99 - (i * 0.004)
                  }}
                ></div>
              ))}
            </div>
          </div>

        </div>
      </div>

      {/* Subtle Grid Background */}
      <div className="absolute inset-0 opacity-3">
        <div className="h-full w-full" style={{
          backgroundImage: `
            linear-gradient(rgba(128, 128, 128, 0.2) 1px, transparent 1px),
            linear-gradient(90deg, rgba(128, 128, 128, 0.2) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px'
        }}></div>
      </div>

      {/* Car Container with Clean 3D Positioning */}
      <div className="relative z-10 flex items-center justify-center h-full pt-16">
        {/* Car Image with Clean 3D Positioning */}
        <div 
          className="relative"
          style={{
            transform: 'perspective(1000px) rotateX(-2deg) translateZ(10px)',
            transformOrigin: 'center center'
          }}
        >
          <img 
            src={carImage} 
            alt="Vehicle View" 
            className="w-auto h-24 sm:h-28 lg:h-[115px] object-contain relative z-10"
          />
        </div>
      </div>

      {/* Status Display */}
      {pirAlert === 1 && (
        <div className="absolute top-6 left-1/2 -translate-x-1/2 text-center z-20">
          <div className="backdrop-blur-md bg-red-900/80 px-6 py-3 rounded-full border-2 border-red-500 shadow-2xl animate-pulse">
            <div className="flex items-center gap-3">
              <MdWarning className="w-6 h-6 text-red-400 animate-bounce" />
              <span className="text-red-200 text-sm sm:text-base font-bold tracking-widest">
                Motion Detected in Vehicle
              </span>
              <MdWarning className="w-6 h-6 text-red-400 animate-bounce" />
            </div>
          </div>
        </div>
      )}

      {/* Corner Frame Decorations */}
      <div className="absolute top-4 left-4 w-12 h-12 border-l-2 border-t-2 border-gray-500/30 rounded-tl-lg"></div>
      <div className="absolute top-4 right-4 w-12 h-12 border-r-2 border-t-2 border-gray-500/30 rounded-tr-lg"></div>
      <div className="absolute bottom-4 left-4 w-12 h-12 border-l-2 border-b-2 border-gray-500/30 rounded-bl-lg"></div>
      <div className="absolute bottom-4 right-4 w-12 h-12 border-r-2 border-b-2 border-gray-500/30 rounded-br-lg"></div>
    </div>
  );
};

export default Car3DView;
