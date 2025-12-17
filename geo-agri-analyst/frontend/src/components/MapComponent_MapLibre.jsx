import { useEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'

const MapComponent = ({ onAnalyze, isLoading }) => {
  const mapContainer = useRef(null)
  const map = useRef(null)
  const [selectionMode, setSelectionMode] = useState('point') // 'point' or 'polygon'
  const [selectedPos, setSelectedPos] = useState(null)
  const [polygonPoints, setPolygonPoints] = useState([])
  const [locationName, setLocationName] = useState('')
  const [loadingLocation, setLoadingLocation] = useState(false)
  const markers = useRef([])
  const polygonLayer = useRef(null)

  useEffect(() => {
    if (map.current) return // initialize map only once
    
    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: 'https://demotiles.maplibre.org/globe.json', // 3D globe style
      center: [78.9629, 20.5937], // India center [lng, lat]
      zoom: 4,
      projection: 'globe', // Enable globe projection
      attributionControl: false
    })

    // Add navigation controls
    map.current.addControl(new maplibregl.NavigationControl(), 'top-right')
    
    // Add attribution
    map.current.addControl(
      new maplibregl.AttributionControl({
        compact: true,
        customAttribution: '© MapLibre | © OpenStreetMap contributors'
      }),
      'bottom-right'
    )

    // Handle map clicks
    map.current.on('click', (e) => {
      const { lng, lat } = e.lngLat
      
      if (selectionMode === 'point') {
        // Clear previous markers
        clearMarkers()
        
        // Add new marker
        const marker = new maplibregl.Marker({ color: '#3b82f6' })
          .setLngLat([lng, lat])
          .addTo(map.current)
        
        markers.current.push(marker)
        setSelectedPos({ lat, lng })
        fetchLocationName(lat, lng)
      } else if (selectionMode === 'polygon') {
        // Add point to polygon
        const newPoints = [...polygonPoints, [lng, lat]]
        setPolygonPoints(newPoints)
        
        // Add marker for point
        const marker = new maplibregl.Marker({ color: '#10b981' })
          .setLngLat([lng, lat])
          .setPopup(new maplibregl.Popup().setText(`Point ${newPoints.length}`))
          .addTo(map.current)
        
        markers.current.push(marker)
        
        // Draw polygon if we have 3+ points
        if (newPoints.length >= 3) {
          drawPolygon(newPoints)
          const avgLat = newPoints.reduce((sum, p) => sum + p[1], 0) / newPoints.length
          const avgLng = newPoints.reduce((sum, p) => sum + p[0], 0) / newPoints.length
          fetchLocationName(avgLat, avgLng)
        }
      }
    })

    // Add fog/atmosphere for globe effect
    map.current.on('style.load', () => {
      map.current.setFog({
        color: 'rgb(186, 210, 235)', // Lower atmosphere
        'high-color': 'rgb(36, 92, 223)', // Upper atmosphere
        'horizon-blend': 0.02, // Atmosphere thickness (default 0.2 at low zooms)
        'space-color': 'rgb(11, 11, 25)', // Background color
        'star-intensity': 0.6 // Background star brightness (default 0.35 at low zoooms )
      })
    })

    return () => map.current.remove()
  }, [])

  // Update polygon when points change
  useEffect(() => {
    if (selectionMode === 'polygon' && polygonPoints.length >= 3) {
      drawPolygon(polygonPoints)
    }
  }, [polygonPoints, selectionMode])

  const clearMarkers = () => {
    markers.current.forEach(marker => marker.remove())
    markers.current = []
    
    // Remove polygon layer if exists
    if (map.current && polygonLayer.current) {
      if (map.current.getLayer('polygon-fill')) {
        map.current.removeLayer('polygon-fill')
      }
      if (map.current.getLayer('polygon-outline')) {
        map.current.removeLayer('polygon-outline')
      }
      if (map.current.getSource('polygon')) {
        map.current.removeSource('polygon')
      }
      polygonLayer.current = null
    }
  }

  const drawPolygon = (points) => {
    if (!map.current) return

    // Close the polygon by adding first point at end
    const coordinates = [...points, points[0]]

    const geojson = {
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [coordinates]
      }
    }

    // Remove existing polygon layers
    if (map.current.getLayer('polygon-fill')) {
      map.current.removeLayer('polygon-fill')
    }
    if (map.current.getLayer('polygon-outline')) {
      map.current.removeLayer('polygon-outline')
    }
    if (map.current.getSource('polygon')) {
      map.current.removeSource('polygon')
    }

    // Add new polygon
    map.current.addSource('polygon', {
      type: 'geojson',
      data: geojson
    })

    map.current.addLayer({
      id: 'polygon-fill',
      type: 'fill',
      source: 'polygon',
      paint: {
        'fill-color': '#10b981',
        'fill-opacity': 0.2
      }
    })

    map.current.addLayer({
      id: 'polygon-outline',
      type: 'line',
      source: 'polygon',
      paint: {
        'line-color': '#10b981',
        'line-width': 3
      }
    })

    polygonLayer.current = true
  }

  const fetchLocationName = async (lat, lng) => {
    setLoadingLocation(true)
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&zoom=10`
      )
      const data = await response.json()
      const name = data.display_name || `${lat.toFixed(4)}, ${lng.toFixed(4)}`
      setLocationName(name)
    } catch (error) {
      console.error('Error fetching location name:', error)
      setLocationName(`${lat.toFixed(4)}, ${lng.toFixed(4)}`)
    } finally {
      setLoadingLocation(false)
    }
  }

  const handleClearPolygon = () => {
    setPolygonPoints([])
    setSelectedPos(null)
    setLocationName('')
    clearMarkers()
  }

  const handleAnalyze = () => {
    if (selectionMode === 'polygon' && polygonPoints.length >= 3) {
      // Convert [lng, lat] to [lat, lng] for backend
      const points = polygonPoints.map(p => [p[1], p[0]])
      onAnalyze({ type: 'polygon', points })
    } else if (selectionMode === 'point' && selectedPos) {
      onAnalyze({ type: 'point', position: selectedPos })
    }
  }

  return (
    <div className="relative h-full w-full min-h-[400px] sm:min-h-[500px] lg:min-h-[600px]">
      {/* Map Container */}
      <div ref={mapContainer} className="w-full h-full rounded-xl overflow-hidden" />

      {/* Mode Selector */}
      <div className="absolute top-4 left-4 z-10">
        <div className="glass rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <button
              onClick={() => {
                setSelectionMode('point')
                handleClearPolygon()
              }}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                selectionMode === 'point'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              📍 Point
            </button>
            <button
              onClick={() => {
                setSelectionMode('polygon')
                handleClearPolygon()
              }}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                selectionMode === 'polygon'
                  ? 'bg-emerald-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              🔷 Polygon
            </button>
          </div>
          {selectionMode === 'polygon' && polygonPoints.length > 0 && (
            <button
              onClick={handleClearPolygon}
              className="w-full px-3 py-1.5 bg-red-500 hover:bg-red-600 text-white rounded-md text-sm font-medium transition-all"
            >
              Clear ({polygonPoints.length} points)
            </button>
          )}
        </div>
      </div>

      {/* Instructions */}
      <div className="absolute top-4 right-20 z-10">
        <div className="glass rounded-lg px-4 py-2">
          <p className="text-white text-sm">
            {selectionMode === 'point' 
              ? '📍 Click anywhere on the globe to select a location' 
              : '🔷 Click points on the globe to create a polygon (min 3 points)'}
          </p>
        </div>
      </div>

      {/* Location Info */}
      {locationName && (
        <div className="absolute bottom-20 left-4 z-10">
          <div className="glass rounded-lg px-4 py-3 max-w-sm">
            <div className="flex items-start space-x-2">
              <span className="text-2xl">📍</span>
              <div className="flex-1">
                <p className="text-white font-medium text-sm mb-1">Selected Location</p>
                {loadingLocation ? (
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 border border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                    <span className="text-gray-300 text-xs">Loading...</span>
                  </div>
                ) : (
                  <p className="text-gray-300 text-xs">{locationName}</p>
                )}
                {selectedPos && (
                  <p className="text-gray-400 text-xs mt-1">
                    {selectedPos.lat.toFixed(6)}, {selectedPos.lng.toFixed(6)}
                  </p>
                )}
                {selectionMode === 'polygon' && polygonPoints.length >= 3 && (
                  <p className="text-emerald-300 text-xs mt-1">
                    {polygonPoints.length} points selected
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Analyze Button */}
      {((selectionMode === 'point' && selectedPos) || 
        (selectionMode === 'polygon' && polygonPoints.length >= 3)) && (
        <div className="absolute bottom-4 left-4 z-10">
          <button
            onClick={handleAnalyze}
            disabled={isLoading}
            className="glass-button text-white font-medium py-3 px-6 rounded-lg transition-all duration-300 hover:scale-105 shadow-lg"
          >
            {isLoading ? (
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Analyzing...</span>
              </div>
            ) : (
              <span>🔍 Analyze Location</span>
            )}
          </button>
        </div>
      )}

      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-black bg-opacity-50 z-[1001] flex items-center justify-center rounded-xl">
          <div className="glass rounded-xl p-8 text-center">
            <div className="w-12 h-12 border-3 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-white font-medium text-base">Processing satellite imagery...</p>
            <p className="text-gray-300 text-sm mt-1">This may take a few moments</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default MapComponent
