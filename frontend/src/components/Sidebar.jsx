import React, { useState } from 'react';

const Sidebar = ({ onSelectItem }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeMenu, setActiveMenu] = useState(null);

  const menuItems = [
    {
      id: 'safety',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
        </svg>
      ),
      label: 'Safety & Detection',
      shortLabel: 'S',
      subItems: [
        { label: 'UC 2.1 Three wheeler demonstration', id: null },
        { label: 'UC 2.2 Blind spot detection - rear', id: null },
        { label: 'UC 2.3 Under chassis blind spot detection', id: null },
        { label: 'UC 2.4 Driver monitoring system', id: null },
        { label: 'UC 2.5 Pothole identification system', id: 'pothole' }
      ]
    },
    {
      id: 'control',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
        </svg>
      ),
      label: 'Control Systems',
      shortLabel: 'C',
      subItems: [
        { label: 'UC 2.6 Electronic throttle board control', id: null },
        { label: 'UC 2.8 Adaptive Cruise control', id: null },
        { label: 'UC 2.11 Electric Drive Train Control', id: null }
      ]
    },
    {
      id: 'smart',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      ),
      label: 'Smart Systems',
      shortLabel: 'M',
      subItems: [
        { label: 'UC 2.7 Interactive simulations', id: null },
        { label: 'UC 2.9 Automotive security', id: null },
        { label: 'UC 2.10 Battery Monitoring System', id: null },
        { label: 'UC 2.12 Integration using HPC', id: null }
      ]
    }
  ];

  const handleMenuClick = (menuId) => {
    if (!isExpanded) {
      setIsExpanded(true);
      setActiveMenu(menuId);
    } else {
      setActiveMenu(activeMenu === menuId ? null : menuId);
    }
  };

  return (
    <>
      {/* Collapsed Sidebar */}
      <div className="fixed left-0 top-16 h-[calc(100vh-4rem)] bg-gray-50 border-r border-gray-200 transition-all duration-300 z-10 flex flex-col">
        {/* Main Menu Icons */}
        <div className="flex flex-col items-center py-4 space-y-2">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="w-12 h-12 flex items-center justify-center text-gray-600 hover:bg-gray-200 rounded-lg transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>

        <div className="border-t border-gray-200 my-2"></div>

        {/* Menu Items */}
        <div className="flex flex-col items-center space-y-1 px-2">
          {menuItems.map((item) => (
            <button
              key={item.id}
              onClick={() => handleMenuClick(item.id)}
              className={`w-12 h-12 flex items-center justify-center rounded-lg transition-colors ${
                activeMenu === item.id
                  ? 'bg-teal-500 text-white'
                  : 'text-gray-600 hover:bg-gray-200'
              }`}
              title={item.label}
            >
              {item.icon}
            </button>
          ))}
        </div>
      </div>

      {/* Expanded Sidebar */}
      <div
        className={`fixed left-16 top-16 h-[calc(100vh-4rem)] bg-white border-r border-gray-200 transition-all duration-300 z-5 overflow-hidden ${
          isExpanded ? 'w-64 opacity-100' : 'w-0 opacity-0'
        }`}
      >
        <div className={`p-4 mt-4 w-64 ${isExpanded ? '' : 'invisible'}`}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 bg-teal-500 rounded-full flex items-center justify-center">
                <span className="text-white font-bold">EC</span>
              </div>
              <span className="font-semibold text-gray-800">Experience Centre</span>
            </div>
            <button
              onClick={() => {
                setIsExpanded(false);
                setActiveMenu(null);
              }}
              className="text-gray-400 hover:text-gray-600"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
          </div>

          <div className="border-t border-gray-200 mb-4"></div>

          {/* Menu Categories */}
          <div className="space-y-2">
            {menuItems.map((item) => (
              <div key={item.id}>
                <button
                  onClick={() => setActiveMenu(activeMenu === item.id ? null : item.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
                    activeMenu === item.id
                      ? 'bg-teal-50 text-teal-600'
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <span className={activeMenu === item.id ? 'text-teal-600' : 'text-gray-600'}>
                    {item.icon}
                  </span>
                  <span className="flex-1 text-left font-medium text-sm">{item.label}</span>
                  <svg
                    className={`w-4 h-4 transition-transform ${
                      activeMenu === item.id ? 'rotate-180' : ''
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>

                {/* Sub Items */}
                {activeMenu === item.id && (
                  <div className="ml-9 mt-1 space-y-1">
                    {item.subItems.map((subItem, index) => (
                      <button
                        key={index}
                        onClick={() => subItem.id && onSelectItem && onSelectItem(subItem.id)}
                        className={`w-full text-left px-3 py-2 text-sm text-gray-600 hover:bg-gray-50 hover:text-teal-600 rounded-lg transition-colors ${
                          subItem.id ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'
                        }`}
                        disabled={!subItem.id}
                      >
                        {subItem.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Overlay for mobile */}
      {isExpanded && (
        <div
          className="fixed inset-0 bg-black bg-opacity-25 z-5 lg:hidden"
          onClick={() => {
            setIsExpanded(false);
            setActiveMenu(null);
          }}
        ></div>
      )}
    </>
  );
};

export default Sidebar;
