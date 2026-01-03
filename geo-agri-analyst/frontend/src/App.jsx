import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import axios from 'axios'
import MapComponent from './components/MapComponent'
import ResultsPanel from './components/ResultsPanel'
import AnalyticsPage from './components/AnalyticsPage'
import SettingsPage from './components/SettingsPage'
import SearchBar from './components/SearchBar'

// Navigation Bar Component
function NavigationBar({ currentPath }) {
  return (
    <nav className="fixed top-4 left-4 right-4 z-50 glass rounded-xl lg:rounded-2xl border border-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.3)]">
      <div className="w-full px-4 sm:px-6 lg:px-8 py-3 lg:py-4">
        <div className="flex items-center justify-between">
          {/* Logo and Brand */}
          <div className="flex items-center space-x-2 lg:space-x-3">
            <div className="w-8 h-8 lg:w-10 lg:h-10 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-full flex items-center justify-center shadow-[0_0_20px_rgba(16,185,129,0.3)]">
              <span className="text-white text-lg lg:text-xl font-bold">🌍</span>
            </div>
            <div>
              <h1 className="text-lg lg:text-xl font-bold gradient-text">Geo-Agri Analyst</h1>
              <p className="text-xs text-gray-400 hidden sm:block">AI-Powered Agriculture</p>
            </div>
          </div>

          {/* Navigation Links */}
          <div className="hidden lg:flex items-center space-x-4 xl:space-x-6">
            <Link 
              to="/"
              className={`glass-button px-3 py-2 xl:px-4 xl:py-2 rounded-lg text-white hover:text-emerald-400 transition-colors duration-300 flex items-center space-x-2 ${currentPath === '/' ? 'text-emerald-400' : ''}`}
            >
              <span>🗺️</span>
              <span className="text-sm">Map</span>
            </Link>
            <Link 
              to="/analytics"
              className={`glass-button px-3 py-2 xl:px-4 xl:py-2 rounded-lg text-white hover:text-blue-400 transition-colors duration-300 flex items-center space-x-2 ${currentPath === '/analytics' ? 'text-blue-400' : ''}`}
            >
              <span>📊</span>
              <span className="text-sm">Analytics</span>
            </Link>
            <Link 
              to="/settings"
              className={`glass-button px-3 py-2 xl:px-4 xl:py-2 rounded-lg text-white hover:text-purple-400 transition-colors duration-300 flex items-center space-x-2 ${currentPath === '/settings' ? 'text-purple-400' : ''}`}
            >
              <span>⚙️</span>
              <span className="text-sm">Settings</span>
            </Link>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center space-x-2 lg:space-x-3">
            {/* Status Indicator */}
            <div className="hidden md:flex items-center space-x-2 glass px-2 py-1 lg:px-3 lg:py-2 rounded-lg">
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse shadow-[0_0_10px_rgba(16,185,129,0.5)]"></div>
              <span className="text-xs text-gray-300">Built by Sudarshan Hegde</span>
            </div>
            
            {/* Mobile Menu Button */}
            <button className="lg:hidden glass-button p-2 rounded-lg text-white hover:text-emerald-400 transition-colors duration-300">
              <span className="text-lg">☰</span>
            </button>
          </div>
        </div>
      </div>
    </nav>
  )
}

function HomePage({ onAnalyze }) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [predictionData, setPredictionData] = useState(null)
  const [selectedPos, setSelectedPos] = useState(null)
  const [polygonPoints, setPolygonPoints] = useState([])
  const [selectionMode, setSelectionMode] = useState('point')
  const [mapRef, setMapRef] = useState(null)

  const handleLocationSearch = (lat, lng, displayName) => {
    // This will be called from SearchBar
    console.log('🔍 Search location selected:', displayName, { lat, lng });
    setSelectedPos({ lat, lng });
  }

  const handleAnalyze = async (analysisData) => {
    const hasValidSelection = analysisData && 
      ((analysisData.type === 'point' && analysisData.position) ||
       (analysisData.type === 'polygon' && analysisData.points && analysisData.points.length >= 3))

    if (!hasValidSelection) {
      setError('Please select a location or create a polygon on the map first')
      return
    }

    setIsLoading(true)
    setError(null)
    setPredictionData(null)

    try {
      let requestData
      
      if (analysisData.type === 'polygon') {
        // Convert polygon points to the correct format
        const polygonPointsArray = analysisData.points.map(p => [p.lat, p.lng]);
        const avgLat = analysisData.points.reduce((sum, point) => sum + point.lat, 0) / analysisData.points.length;
        const avgLng = analysisData.points.reduce((sum, point) => sum + point.lng, 0) / analysisData.points.length;
        
        requestData = {
          type: 'polygon',
          points: polygonPointsArray,
          lat: avgLat,
          lng: avgLng
        }
        console.log('📊 Sending polygon analysis request:', requestData);
      } else {
        requestData = {
          type: 'point',
          lat: analysisData.position.lat,
          lng: analysisData.position.lng
        }
        console.log('📍 Sending point analysis request:', requestData);
      }

      const response = await axios.post('http://localhost:8000/api/v1/analyze', requestData)
      const newAnalysis = {
        ...requestData,
        results: response.data,
        timestamp: new Date().toISOString()
      }
      
      setPredictionData(response.data)
      onAnalyze(newAnalysis)
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed: Could not retrieve satellite data')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 pt-32 lg:pt-36 overflow-hidden" style={{ paddingBottom: '70px' }}>
      {/* Add spacing for search bar */}
      <div className="mb-24 lg:mb-16"></div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 xl:grid-cols-4 gap-4 lg:gap-6 h-full" style={{ paddingLeft: '5px', paddingRight: '1.5rem' }}>
        <div className="lg:col-span-2 xl:col-span-3 relative h-full">
          {/* Search Bar - Positioned above map card */}
          <div className="absolute -top-20 lg:-top-24 left-1/2 transform -translate-x-1/2 w-11/12 max-w-2xl z-[9999]">
            <SearchBar onLocationSelect={handleLocationSearch} />
          </div>
          
          <div className="glass-card rounded-xl lg:rounded-2xl overflow-hidden hover:shadow-2xl transition-all duration-500 h-full w-full">
            <div className="absolute inset-0">
              <MapComponent 
                selectedPos={selectedPos}
                setSelectedPos={setSelectedPos}
                onAnalyze={handleAnalyze}
                isLoading={isLoading}
                onLocationSearch={handleLocationSearch}
                polygonPoints={polygonPoints}
                setPolygonPoints={setPolygonPoints}
                selectionMode={selectionMode}
                setSelectionMode={setSelectionMode}
              />
            </div>
          </div>
        </div>

        <div className="lg:col-span-1 xl:col-span-1 flex flex-col gap-4 lg:gap-6 h-full overflow-hidden">
          <div className="glass-card rounded-xl lg:rounded-2xl p-4 lg:p-6 flex-shrink-0">
            <div className="flex items-center space-x-3 mb-4 lg:mb-6">
              <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse"></div>
              <h3 className="text-lg lg:text-xl font-bold text-white">Analysis Control</h3>
            </div>
            
            <div className="space-y-3 lg:space-y-4">
              {/* Selection Info Display */}
              {selectionMode === 'point' && selectedPos && (
                <div className="glass rounded-lg p-3 lg:p-4">
                  <p className="text-xs lg:text-sm text-gray-400 mb-1">📍 Selected Point</p>
                  <p className="text-sm lg:text-base text-white font-mono">
                    {selectedPos.lat.toFixed(4)}, {selectedPos.lng.toFixed(4)}
                  </p>
                </div>
              )}
              
              {selectionMode === 'polygon' && polygonPoints.length > 0 && (
                <div className="glass rounded-lg p-3 lg:p-4">
                  <p className="text-xs lg:text-sm text-gray-400 mb-1">🔷 Polygon Selection</p>
                  <p className="text-sm lg:text-base text-white font-mono">
                    {polygonPoints.length} point{polygonPoints.length !== 1 ? 's' : ''} selected
                  </p>
                  {polygonPoints.length >= 3 && (
                    <p className="text-xs text-emerald-400 mt-1">✓ Ready to analyze</p>
                  )}
                  {polygonPoints.length < 3 && polygonPoints.length > 0 && (
                    <p className="text-xs text-yellow-400 mt-1">Need {3 - polygonPoints.length} more point{3 - polygonPoints.length !== 1 ? 's' : ''}</p>
                  )}
                </div>
              )}
              
              {/* Unified Start Analysis Button */}
              <button
                onClick={() => {
                  if (selectionMode === 'polygon' && polygonPoints.length >= 3) {
                    handleAnalyze({ type: 'polygon', points: polygonPoints });
                  } else if (selectionMode === 'point' && selectedPos) {
                    handleAnalyze({ type: 'point', position: selectedPos });
                  }
                }}
                disabled={(
                  (selectionMode === 'point' && !selectedPos) ||
                  (selectionMode === 'polygon' && polygonPoints.length < 3) ||
                  isLoading
                )}
                className="w-full glass-button bg-gradient-to-r from-emerald-500 to-blue-500 hover:from-emerald-600 hover:to-blue-600 text-white font-bold py-3 lg:py-4 px-4 lg:px-6 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:shadow-[0_0_30px_rgba(16,185,129,0.5)] text-sm lg:text-base"
              >
                {isLoading ? (
                  <div className="flex items-center justify-center space-x-2">
                    <div className="w-4 h-4 lg:w-5 lg:h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Analyzing...</span>
                  </div>
                ) : (
                  <div className="flex items-center justify-center space-x-2">
                    <span>{selectionMode === 'polygon' ? '🔷' : '🔍'}</span>
                    <span>
                      {selectionMode === 'polygon' 
                        ? `Analyze Polygon${polygonPoints.length >= 3 ? ` (${polygonPoints.length} points)` : ''}` 
                        : 'Start Analysis'}
                    </span>
                  </div>
                )}
              </button>
            </div>
          </div>

          <div className="glass-card rounded-xl lg:rounded-2xl p-4 lg:p-6 flex-1 overflow-y-auto min-h-0">
            <ResultsPanel 
              isLoading={isLoading}
              error={error}
              data={predictionData}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

function App() {
  const [analyticsHistory, setAnalyticsHistory] = useState([])
  const [currentPath, setCurrentPath] = useState(window.location.pathname)
  const [appSettings, setAppSettings] = useState({
    mapSettings: {
      defaultZoom: 13,
      defaultCenter: { lat: 20.5937, lng: 78.9629 },
      satelliteView: true,
    },
    analysisSettings: {
      autoAnalyze: false,
      saveHistory: true,
      historyLimit: 50,
      showCoordinates: true,
    },
    displaySettings: {
      darkMode: true,
      showTips: true,
      language: 'en',
      distanceUnit: 'km',
    }
  })

  useEffect(() => {
    const handleLocationChange = () => {
      setCurrentPath(window.location.pathname)
    }
    window.addEventListener('popstate', handleLocationChange)
    return () => window.removeEventListener('popstate', handleLocationChange)
  }, [])

  const handleNewAnalysis = (analysis) => {
    setAnalyticsHistory(prev => [...prev, analysis])
  }

  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black text-white">
        {/* Animated background elements */}
        <div className="fixed inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-gray-800 rounded-full opacity-3 animate-pulse-glow"></div>
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gray-700 rounded-full opacity-3 animate-float"></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gray-900 rounded-full opacity-2 animate-pulse-glow"></div>
        </div>

        <NavigationBar currentPath={currentPath} />

        <Routes>
          <Route 
            path="/" 
            element={<HomePage onAnalyze={handleNewAnalysis} />} 
          />
          <Route 
            path="/analytics" 
            element={<AnalyticsPage history={analyticsHistory} settings={appSettings} />} 
          />
          <Route 
            path="/settings" 
            element={<SettingsPage settings={appSettings} onSettingsChange={(newSettings) => setAppSettings(newSettings)} />} 
          />
        </Routes>
      </div>
    </Router>
  )
}

export default App