import { useState } from 'react'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import PotholeDetection from './pages/PotholeDetection'
import './App.css'

function App() {
  const [currentPage, setCurrentPage] = useState(null);

  const renderContent = () => {
    switch (currentPage) {
      case 'pothole':
        return <PotholeDetection />;
      default:
        return (
          <div className="flex items-center justify-center h-screen">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-700 mb-2">Welcome to Experience Centre Dashboard</h2>
              <p className="text-gray-500">Select a use case from the sidebar to begin</p>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <Sidebar onSelectItem={setCurrentPage} />
      <div className="ml-16 pt-16">
        {renderContent()}
      </div>
    </div>
  )
}

export default App
