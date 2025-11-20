import React from 'react';

const Navbar = () => {
  return (
    <nav className="bg-white border-b border-gray-200 px-6 py-4 fixed top-0 left-0 right-0 z-30">
      <div className="flex items-center justify-between">
        {/* <div className="flex items-center gap-4"> */}
          <div className="flex items-center gap-2">
            {/* <div className="w-10 h-10 bg-teal-500 rounded-full flex items-center justify-center">
              <span className="text-white font-bold text-xl">CDAC</span>
            </div> */}
            <span className="text-2xl font-bold text-gray-800">Experience Centre Dashboard</span>
          </div>
        </div>
      {/* </div> */}
    </nav>
  );
};

export default Navbar;
