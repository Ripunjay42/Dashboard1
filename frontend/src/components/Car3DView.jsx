import React from 'react';
import { MdWarning } from 'react-icons/md';
import carImage from '../assets/car_3d.png';

const Car3DView = ({ pirAlert = 0 }) => {
  return (
    <div className="h-full w-full flex items-center justify-center relative bg-gray-900 overflow-hidden">
      
      {/* 2D Road with Perspective Effect using Gradients and Clip-path */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Road surface - trapezoid shape using clip-path for perspective effect */}
        <div 
          className="absolute bottom-0 left-1/2 w-full h-[100%]"
          style={{
            background: 'linear-gradient(to bottom, #1a1a2e 0%, #0f0f1a 100%)',
            clipPath: 'polygon(35% 0%, 65% 0%, 100% 100%, 0% 100%)',
            WebkitClipPath: 'polygon(35% 0%, 65% 0%, 100% 100%, 0% 100%)',
            transform: 'translateX(-50%)'
          }}
        >
          {/* Road texture overlay */}
          <div 
            className="absolute inset-0 opacity-10"
            style={{
              background: 'repeating-linear-gradient(to bottom, transparent 0px, transparent 20px, rgba(255,255,255,0.05) 20px, rgba(255,255,255,0.05) 22px)'
            }}
          ></div>
        </div>

        {/* Left lane line - parallel to road left edge (35% top to 0% bottom) */}
        <div 
          className="absolute top-0 left-1/2 h-full"
          style={{
            width: '70%',
            transform: 'translateX(-50%)',
            clipPath: 'polygon(35.9% 0%, 36% 0%, 1.2% 100%, 1% 100%)',
            WebkitClipPath: 'polygon(35.9% 0%, 36% 0%, 1.2% 100%, 1% 100%)',
            background: 'linear-gradient(to bottom, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.5) 100%)'
          }}
        ></div>

        {/* Right lane line - parallel to road right edge (65% top to 100% bottom) */}
        <div 
          className="absolute top-0 left-1/2 h-full"
          style={{
            width: '70%',
            transform: 'translateX(-50%)',
            clipPath: 'polygon(63.9% 0%, 64% 0%, 99% 100%, 98.8% 100%)',
            WebkitClipPath: 'polygon(63.9% 0%, 64% 0%, 99% 100%, 98.8% 100%)',
            background: 'linear-gradient(to bottom, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.5) 100%)'
          }}
        ></div>

        {/* Center lane line (yellow dashed) - perspective simulation */}
        <div 
          className="absolute bottom-0 left-1/2 h-[100%] flex flex-col items-center justify-end"
          style={{
            transform: 'translateX(-50%)'
          }}
        >
          {/* Dashed line segments that get larger towards bottom */}
          {[...Array(12)].map((_, i) => (
            <div 
              key={`center-dash-${i}`}
              className="bg-yellow-400"
              style={{
                width: `${2 + i * 0.1}px`,
                height: `${8 + i * 3}px`,
                marginBottom: `${6 + i * 2}px`,
                opacity: 0.9 - (i * 0.05),
                borderRadius: '1px'
              }}
            ></div>
          ))}
        </div>

        {/* Horizon glow effect */}
        <div 
          className="absolute top-[25%] left-1/2 w-[60%] h-[15%]"
          style={{
            transform: 'translateX(-50%)',
            background: 'radial-gradient(ellipse at center, rgba(59, 130, 246, 0.15) 0%, transparent 70%)',
            filter: 'blur(20px)',
            WebkitFilter: 'blur(20px)'
          }}
        ></div>

        {/* Road edge shadows for depth */}
        <div 
          className="absolute bottom-0 left-0 w-full h-[100%] pointer-events-none"
          style={{
            background: 'linear-gradient(90deg, rgba(0,0,0,0.5) 0%, transparent 20%, transparent 80%, rgba(0,0,0,0.5) 100%)'
          }}
        ></div>
      </div>

      {/* Subtle Grid Background */}
      <div className="absolute inset-0 opacity-5">
        <div 
          className="h-full w-full" 
          style={{
            backgroundImage: `
              linear-gradient(rgba(128, 128, 128, 0.3) 1px, transparent 1px),
              linear-gradient(90deg, rgba(128, 128, 128, 0.3) 1px, transparent 1px)
            `,
            backgroundSize: '40px 40px'
          }}
        ></div>
      </div>

      {/* Car Container */}
      <div className="relative z-10 flex items-center justify-center h-full pt-16">
        {/* Car shadow for depth effect */}
        <div 
          className="absolute bottom-[15%] left-1/2 w-32 h-6 bg-black/30 rounded-full blur-md"
          style={{
            transform: 'translateX(-50%)'
          }}
        ></div>
        
        {/* Car Image */}
        <div className="relative">
          <img 
            src={carImage} 
            alt="Vehicle View" 
            className="w-auto h-24 sm:h-28 lg:h-[115px] object-contain relative z-10"
          />
        </div>
      </div>

      {/* Status Display */}
      {pirAlert === 1 && (
        <div className="absolute top-6 left-1/2 text-center z-20" style={{ transform: 'translateX(-50%)' }}>
          <div className="bg-red-900/80 px-6 py-3 rounded-full border-2 border-red-500 shadow-2xl animate-pulse">
            <div className="flex items-center gap-3">
              <MdWarning className="w-6 h-6 text-red-400 animate-bounce" />
              <span className="text-red-200 text-sm sm:text-base font-bold tracking-widest">
                Caution: Under-Vehicle Activity Detected
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
