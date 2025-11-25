import React from 'react';
import { 
  TbMist, 
  TbBulb, 
  TbArrowBigLeftLinesFilled, 
  TbArrowBigRightLinesFilled 
} from 'react-icons/tb';
import { MdBrightnessMedium } from 'react-icons/md';
import { RiCarWashingFill } from 'react-icons/ri';
import { BsFillExclamationOctagonFill } from 'react-icons/bs';
import { PiSeatbeltFill } from 'react-icons/pi';

const VehicleIndicators = ({ leftTurnActive = false, rightTurnActive = false }) => {
  return (
    <>
      {/* Left Side Indicators - Top Left Arc */}
      <div className="absolute left-10 sm:left-28 top-16 sm:top-0 flex items-center gap-3 sm:gap-4 z-10">
        {/* Left Turn Signal */}
        <div className="flex flex-col items-center" title="Left Turn Signal">
          <TbArrowBigLeftLinesFilled 
            className={`text-xl sm:text-4xl transition-all duration-150 ${
              leftTurnActive 
                ? 'text-green-400 drop-shadow-[0_0_10px_rgba(74,222,128,0.8)] animate-pulse' 
                : 'text-gray-400'
            }`} 
          />
        </div>
        
        {/* Fog Light Front */}
        {/* <div className="flex flex-col items-center" title="Fog Light Front">
          <TbMist className="text-gray-400 text-xl sm:text-2xl" />
        </div> */}
        <div className="flex flex-col items-center" title="Parking Brake">
          <div className="text-gray-400 text-base sm:text-xl font-bold border-2 border-gray-400 rounded-full w-6 h-6 sm:w-8 sm:h-8 flex items-center justify-center">
            P
          </div>
        </div>
        
        {/* High Beam */}
        {/* <div className="flex flex-col items-center" title="High Beam">
          <TbBulb className="text-gray-400 text-xl sm:text-2xl" />
        </div> */}
        
        {/* Low Beam */}
        {/* <div className="flex flex-col items-center" title="Low Beam">
          <MdBrightnessMedium className="text-gray-400 text-xl sm:text-2xl" />
        </div> */}
      </div>

      {/* Right Side Indicators - Top Right Arc */}
      <div className="absolute right-12 sm:right-28 top-16 sm:top-0 flex items-center gap-3 sm:gap-4 z-10">
        
        {/* Seatbelt */}
        <div className="flex flex-col items-center" title="Seatbelt">
          <PiSeatbeltFill className="text-gray-400 text-xl sm:text-3xl" />
        </div>
        
        {/* Right Turn Signal */}
        <div className="flex flex-col items-center" title="Right Turn Signal">
          <TbArrowBigRightLinesFilled 
            className={`text-xl sm:text-4xl transition-all duration-150 ${
              rightTurnActive 
                ? 'text-green-400 drop-shadow-[0_0_10px_rgba(74,222,128,0.8)] animate-pulse' 
                : 'text-gray-400'
            }`} 
          />
        </div>
      </div>
    </>
  );
};

export default VehicleIndicators;
