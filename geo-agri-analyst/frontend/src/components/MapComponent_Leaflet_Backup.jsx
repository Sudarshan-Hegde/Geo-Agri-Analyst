import { MapContainer, TileLayer, Marker, Popup, Polygon, useMapEvents, useMap, CircleMarker } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import { useState, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
})

// Custom glowing icon for selected location
const createGlowingIcon = () => {
  return L.divIcon({
    className: 'custom-div-icon',
    html: `
      <div style="
        position: relative;
        width: 30px;
        height: 30px;
      ">
        <div style="
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 20px;
          height: 20px;
          background: linear-gradient(135deg, #3b82f6, #10b981);
          border-radius: 50%;
          box-shadow: 
            0 0 20px rgba(59, 130, 246, 0.6),
            0 0 40px rgba(59, 130, 246, 0.3),
            inset 0 2px 4px rgba(255, 255, 255, 0.3);
          animation: pulse-glow 2s ease-in-out infinite;
        "></div>
        <div style="
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 8px;
          height: 8px;
          background: white;
          border-radius: 50%;
          box-shadow: 0 0 10px rgba(255, 255, 255, 0.8);
        "></div>
      </div>
    `,
    iconSize: [30, 30],
    iconAnchor: [15, 15],
    popupAnchor: [0, -15],
  })
}

// Custom icon for polygon points
const createPolygonPointIcon = (index) => {
  return L.divIcon({
    className: 'custom-polygon-point',
    html: `
      <div style="
        position: relative;
        width: 25px;
        height: 25px;
      ">
        <div style="
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 18px;
          height: 18px;
          background: linear-gradient(135deg, #10b981, #059669);
          border-radius: 4px;
          box-shadow: 
            0 0 15px rgba(16, 185, 129, 0.6),
            0 0 30px rgba(16, 185, 129, 0.3);
          border: 2px solid white;
        "></div>
        <div style="
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          color: white;
          font-size: 10px;
          font-weight: bold;
          text-shadow: 0 1px 2px rgba(0,0,0,0.5);
        ">${index + 1}</div>
      </div>
    `,
    iconSize: [25, 25],
    iconAnchor: [12, 12],
    popupAnchor: [0, -12],
  })
}

// Component to handle map clicks for both point and polygon selection
function MapClickHandler({ setSelectedPos, polygonPoints, setPolygonPoints, selectionMode }) {
  useMapEvents({
    click: (e) => {
      if (selectionMode === 'polygon') {
        // Add point to polygon
        const newPoint = [e.latlng.lat, e.latlng.lng];
        setPolygonPoints(prev => [...prev, newPoint]);
        setSelectedPos(null); // Clear single point selection
      } else {
        // Single point selection
        setSelectedPos({ lat: e.latlng.lat, lng: e.latlng.lng });
        setPolygonPoints([]); // Clear polygon points
      }
    }
  })
  return null
}

// Component to track map center while scrolling/panning
function MapCenterTracker({ onCenterChange }) {
  const map = useMap();
  const debounceTimer = useRef(null);

  useEffect(() => {
    const updateCenter = () => {
      const center = map.getCenter();
      onCenterChange(center.lat, center.lng);
    };

    // Initial center
    updateCenter();

    const handleMoveEnd = () => {
      // Debounce to avoid too many API calls
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
      debounceTimer.current = setTimeout(() => {
        updateCenter();
      }, 500);
    };

    map.on('moveend', handleMoveEnd);
    
    return () => {
      map.off('moveend', handleMoveEnd);
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, [map, onCenterChange]);

  return null;
}

// Semantic Zoom Labels - Shows different information based on zoom level
function SemanticZoomLabels() {
  const map = useMap();
  const [zoomLevel, setZoomLevel] = useState(map.getZoom());
  const [labels, setLabels] = useState([]);

  // Define label data for different zoom levels
  const labelData = {
    // Zoom 1-4: Continents and major regions
    continent: [
      { position: [28.6139, 77.2090], name: 'India', minZoom: 1, maxZoom: 4 },
      { position: [35.8617, 104.1954], name: 'China', minZoom: 1, maxZoom: 4 },
      { position: [37.0902, -95.7129], name: 'United States', minZoom: 1, maxZoom: 4 },
      { position: [-14.2350, -51.9253], name: 'Brazil', minZoom: 1, maxZoom: 4 },
      { position: [51.1657, 10.4515], name: 'Germany', minZoom: 1, maxZoom: 4 },
    ],
    // Zoom 5-7: States/Provinces
    state: [
      // India states
      { position: [19.0760, 72.8777], name: 'Maharashtra', minZoom: 5, maxZoom: 7 },
      { position: [28.7041, 77.1025], name: 'Delhi', minZoom: 5, maxZoom: 7 },
      { position: [13.0827, 80.2707], name: 'Tamil Nadu', minZoom: 5, maxZoom: 7 },
      { position: [12.9716, 77.5946], name: 'Karnataka', minZoom: 5, maxZoom: 7 },
      { position: [26.8467, 80.9462], name: 'Uttar Pradesh', minZoom: 5, maxZoom: 7 },
      { position: [22.2587, 71.1924], name: 'Gujarat', minZoom: 5, maxZoom: 7 },
      { position: [30.7333, 76.7794], name: 'Punjab', minZoom: 5, maxZoom: 7 },
      { position: [23.6102, 85.2799], name: 'Jharkhand', minZoom: 5, maxZoom: 7 },
    ],
    // Zoom 8-10: Major cities
    city: [
      // India cities
      { position: [28.6139, 77.2090], name: 'New Delhi', minZoom: 8, maxZoom: 10 },
      { position: [19.0760, 72.8777], name: 'Mumbai', minZoom: 8, maxZoom: 10 },
      { position: [12.9716, 77.5946], name: 'Bengaluru', minZoom: 8, maxZoom: 10 },
      { position: [13.0827, 80.2707], name: 'Chennai', minZoom: 8, maxZoom: 10 },
      { position: [17.3850, 78.4867], name: 'Hyderabad', minZoom: 8, maxZoom: 10 },
      { position: [22.5726, 88.3639], name: 'Kolkata', minZoom: 8, maxZoom: 10 },
      { position: [23.0225, 72.5714], name: 'Ahmedabad', minZoom: 8, maxZoom: 10 },
      { position: [18.5204, 73.8567], name: 'Pune', minZoom: 8, maxZoom: 10 },
    ],
    // Zoom 11-13: Districts/Neighborhoods
    district: [
      { position: [28.6517, 77.2219], name: 'Connaught Place', minZoom: 11, maxZoom: 13 },
      { position: [19.0176, 72.8561], name: 'Andheri', minZoom: 11, maxZoom: 13 },
      { position: [12.9352, 77.6245], name: 'Whitefield', minZoom: 11, maxZoom: 13 },
      { position: [13.0569, 80.2425], name: 'T. Nagar', minZoom: 11, maxZoom: 13 },
    ],
    // Zoom 14+: Local landmarks (shown when very zoomed in)
    landmark: [
      { position: [28.6129, 77.2295], name: 'India Gate', minZoom: 14, maxZoom: 20 },
      { position: [18.9220, 72.8347], name: 'Gateway of India', minZoom: 14, maxZoom: 20 },
      { position: [12.9767, 77.5993], name: 'MG Road', minZoom: 14, maxZoom: 20 },
    ]
  };

  useEffect(() => {
    const updateZoom = () => {
      const currentZoom = map.getZoom();
      setZoomLevel(currentZoom);

      // Determine which labels to show based on zoom level
      const visibleLabels = [];
      Object.values(labelData).forEach(category => {
        category.forEach(label => {
          if (currentZoom >= label.minZoom && currentZoom <= label.maxZoom) {
            visibleLabels.push(label);
          }
        });
      });
      
      setLabels(visibleLabels);
    };

    map.on('zoomend', updateZoom);
    updateZoom(); // Initial call

    return () => {
      map.off('zoomend', updateZoom);
    };
  }, [map]);

  // Determine label style based on zoom level
  const getLabelStyle = (zoom) => {
    if (zoom <= 4) {
      return {
        fontSize: '16px',
        fontWeight: 'bold',
        padding: '8px 12px',
        background: 'rgba(0, 0, 0, 0.85)',
        color: '#60A5FA', // blue-400
        textShadow: '0 0 4px rgba(0,0,0,0.9), 0 0 8px rgba(0,0,0,0.8)',
      };
    } else if (zoom <= 7) {
      return {
        fontSize: '14px',
        fontWeight: '600',
        padding: '6px 10px',
        background: 'rgba(0, 0, 0, 0.85)',
        color: '#34D399', // emerald-400
        textShadow: '0 0 4px rgba(0,0,0,0.9), 0 0 8px rgba(0,0,0,0.8)',
      };
    } else if (zoom <= 10) {
      return {
        fontSize: '13px',
        fontWeight: '500',
        padding: '5px 8px',
        background: 'rgba(0, 0, 0, 0.8)',
        color: '#FBBF24', // amber-400
        textShadow: '0 0 3px rgba(0,0,0,0.9), 0 0 6px rgba(0,0,0,0.8)',
      };
    } else if (zoom <= 13) {
      return {
        fontSize: '12px',
        fontWeight: '500',
        padding: '4px 7px',
        background: 'rgba(0, 0, 0, 0.8)',
        color: '#A78BFA', // violet-400
        textShadow: '0 0 3px rgba(0,0,0,0.9), 0 0 6px rgba(0,0,0,0.8)',
      };
    } else {
      return {
        fontSize: '11px',
        fontWeight: '400',
        padding: '3px 6px',
        background: 'rgba(0, 0, 0, 0.75)',
        color: '#F472B6', // pink-400
        textShadow: '0 0 3px rgba(0,0,0,0.9), 0 0 6px rgba(0,0,0,0.8)',
      };
    }
  };

  const labelStyle = getLabelStyle(zoomLevel);

  return (
    <>
      {labels.map((label, index) => (
        <CircleMarker
          key={`${label.name}-${index}`}
          center={label.position}
          radius={0}
          pathOptions={{ opacity: 0, fillOpacity: 0 }}
        >
          <Popup
            permanent
            closeButton={false}
            className="semantic-zoom-label"
            offset={[0, 0]}
            autoPan={false}
          >
            <div
              style={{
                ...labelStyle,
                borderRadius: '6px',
                textAlign: 'center',
                whiteSpace: 'nowrap',
                boxShadow: '0 3px 12px rgba(0,0,0,0.6)',
                border: '2px solid rgba(255,255,255,0.3)',
                pointerEvents: 'none',
                userSelect: 'none',
              }}
            >
              {label.name}
            </div>
          </Popup>
        </CircleMarker>
      ))}
    </>
  );
}

// Location name display component - wrapper that uses useMap hook
function LocationNameDisplay({ locationName, loadingLocation, selectedPos, polygonPoints, selectionMode, mapCenterLocation, loadingMapCenter }) {
  // This component must be inside MapContainer to use useMap
  const map = useMap();
  
  const hasSelection = selectedPos || polygonPoints.length >= 3;
  const displayLocation = hasSelection ? locationName : mapCenterLocation;
  const isLoading = hasSelection ? loadingLocation : loadingMapCenter;
  const icon = hasSelection ? (selectionMode === 'polygon' ? '🔷' : '📍') : '🗺️';
  const textColor = hasSelection ? 'text-white' : 'text-gray-200';

  console.log('LocationNameDisplay render:', {
    hasSelection,
    displayLocation,
    isLoading,
    locationName,
    mapCenterLocation
  });

  if (!displayLocation && !isLoading) return null;

  return (
    <div className="leaflet-bottom leaflet-center" style={{
      position: 'absolute',
      bottom: '30px',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 1000,
      pointerEvents: 'none',
      display: 'flex',
      justifyContent: 'center'
    }}>
      <div className="glass rounded-lg px-4 py-2 shadow-lg" style={{ 
        pointerEvents: 'auto',
        maxWidth: 'calc(100vw - 100px)',
        width: 'auto'
      }}>
        {isLoading ? (
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
            <span className="text-gray-300 text-sm">Loading...</span>
          </div>
        ) : displayLocation ? (
          <div className="flex items-center space-x-2">
            <span className="text-lg">{icon}</span>
            <span className={`${textColor} font-medium text-sm`} style={{
              maxWidth: '400px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap'
            }}>{displayLocation}</span>
          </div>
        ) : null}
      </div>
    </div>
  );
}

// Mode selector for point vs polygon selection
function ModeSelector({ selectionMode, setSelectionMode, onClearPolygon }) {
  return (
    <div className="absolute top-16 left-4 lg:top-20 lg:left-6 z-[1000] space-y-2">
      <div className="glass rounded-lg p-3">
        <h4 className="text-white font-medium text-sm mb-3">Selection Mode</h4>
        <div className="space-y-2">
          <button
            onClick={() => setSelectionMode('point')}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all duration-300 ${
              selectionMode === 'point'
                ? 'bg-blue-500 text-white shadow-[0_0_10px_rgba(59,130,246,0.5)]'
                : 'text-gray-300 hover:text-white hover:bg-white/10'
            }`}
          >
            📍 Point Selection
          </button>
          <button
            onClick={() => setSelectionMode('polygon')}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all duration-300 ${
              selectionMode === 'polygon'
                ? 'bg-emerald-500 text-white shadow-[0_0_10px_rgba(16,185,129,0.5)]'
                : 'text-gray-300 hover:text-white hover:bg-white/10'
            }`}
          >
            🔷 Polygon Selection
          </button>
        </div>
        {selectionMode === 'polygon' && (
          <button
            onClick={onClearPolygon}
            className="w-full mt-2 px-3 py-2 text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-all duration-300"
          >
            🗑️ Clear Polygon
          </button>
        )}
      </div>
    </div>
  )
}

// Floating analyze button component that appears on the map near the selection
function AnalyzeButtonOnMap({ onAnalyze, isLoading, selectedPos, polygonPoints, selectionMode }) {
  const map = useMap();
  const hasSelection = selectedPos || (polygonPoints.length >= 3);
  
  if (!hasSelection) return null;

  const getButtonText = () => {
    if (isLoading) return "Analyzing...";
    if (selectionMode === 'polygon') return `Analyze Polygon (${polygonPoints.length} points)`;
    return "Analyze Point";
  };

  // Calculate position based on selection
  let position;
  if (selectionMode === 'polygon' && polygonPoints.length >= 3) {
    // Use centroid of polygon
    const avgLat = polygonPoints.reduce((sum, p) => sum + p[0], 0) / polygonPoints.length;
    const avgLng = polygonPoints.reduce((sum, p) => sum + p[1], 0) / polygonPoints.length;
    position = [avgLat, avgLng];
  } else if (selectedPos) {
    position = [selectedPos.lat, selectedPos.lng];
  }

  if (!position) return null;

  // Convert lat/lng to pixel coordinates
  const point = map.latLngToContainerPoint(position);
  
  return createPortal(
    <div 
      style={{
        position: 'absolute',
        left: `${point.x}px`,
        top: `${point.y - 80}px`, // Position above the marker
        transform: 'translateX(-50%)',
        zIndex: 1000,
        pointerEvents: 'auto'
      }}
    >
      <button
        onClick={(e) => {
          e.stopPropagation();
          onAnalyze();
        }}
        disabled={isLoading}
        className={`glass-button text-white font-semibold py-2 px-4 lg:py-3 lg:px-6 rounded-lg lg:rounded-xl text-sm lg:text-base transition-all duration-300 shadow-2xl ${
          isLoading 
            ? 'opacity-50 cursor-not-allowed'
            : 'hover:scale-105 hover:glow-blue'
        }`}
      >
        {isLoading ? (
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 lg:w-5 lg:h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            <span>{getButtonText()}</span>
          </div>
        ) : (
          <div className="flex items-center space-x-2">
            <span>{selectionMode === 'polygon' ? '🔷' : '🔍'}</span>
            <span>{getButtonText()}</span>
          </div>
        )}
      </button>
    </div>,
    map.getContainer()
  );
}

// Instructions overlay
function MapInstructions({ selectedPos, polygonPoints, selectionMode }) {
  const hasSelection = selectedPos || polygonPoints.length > 0;
  
  if (hasSelection) return null;

  return (
    <div className="absolute bottom-4 left-4 lg:bottom-6 lg:left-6 z-[1000] glass rounded-lg lg:rounded-xl p-3 lg:p-4 max-w-xs">
      <div className="flex items-center space-x-2 lg:space-x-3">
        <div className="w-6 h-6 lg:w-8 lg:h-8 bg-gradient-to-r from-blue-500 to-emerald-500 rounded-full flex items-center justify-center">
          <span className="text-white text-xs lg:text-sm">💡</span>
        </div>
        <div>
          {selectionMode === 'polygon' ? (
            <>
              <p className="text-white font-medium text-xs lg:text-sm">Click to place polygon points</p>
              <p className="text-gray-300 text-xs">Need at least 3 points to analyze</p>
            </>
          ) : (
            <>
              <p className="text-white font-medium text-xs lg:text-sm">Click anywhere on the map</p>
              <p className="text-gray-300 text-xs">to select a location for analysis</p>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function MapComponent({ selectedPos, setSelectedPos, onAnalyze, isLoading, onLocationSearch, polygonPoints, setPolygonPoints, selectionMode, setSelectionMode }) {
  const [locationName, setLocationName] = useState('');
  const [loadingLocation, setLoadingLocation] = useState(false);
  const [mapCenterLocation, setMapCenterLocation] = useState('');
  const [loadingMapCenter, setLoadingMapCenter] = useState(false);
  const [searchedLocation, setSearchedLocation] = useState(null);
  const mapRef = useRef(null);

  // Component to handle flying to searched location
  const MapController = () => {
    const map = useMap();
    
    useEffect(() => {
      mapRef.current = map;
    }, [map]);

    useEffect(() => {
      if (searchedLocation && map) {
        map.flyTo([searchedLocation.lat, searchedLocation.lng], 12, {
          duration: 2
        });
      }
    }, [searchedLocation, map]);

    // Handle map resize when container dimensions change
    useEffect(() => {
      const resizeObserver = new ResizeObserver(() => {
        if (map) {
          // Invalidate size and redraw map
          setTimeout(() => {
            map.invalidateSize();
          }, 100);
        }
      });

      const container = map.getContainer().parentElement;
      if (container) {
        resizeObserver.observe(container);
      }

      return () => {
        resizeObserver.disconnect();
      };
    }, [map]);

    return null;
  };

  const handleLocationSearch = (lat, lng, displayName) => {
    setSearchedLocation({ lat, lng, displayName });
    // Optionally set as selected position
    setSelectedPos({ lat, lng });
    setPolygonPoints([]);
    setSelectionMode('point');
    
    // Call parent callback if provided
    if (onLocationSearch) {
      onLocationSearch(lat, lng, displayName);
    }
  };

  // Fetch location name for map center (while scrolling)
  const fetchLocationName = async (lat, lng, isMapCenter = false) => {
    if (isMapCenter) {
      setLoadingMapCenter(true);
    } else {
      setLoadingLocation(true);
    }
    
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&zoom=10&addressdetails=1`,
        {
          headers: {
            'Accept-Language': 'en'
          }
        }
      );
      
      const data = await response.json();
      
      if (data && data.address) {
        // Build a readable location name
        const parts = [];
        if (data.address.city) parts.push(data.address.city);
        else if (data.address.town) parts.push(data.address.town);
        else if (data.address.village) parts.push(data.address.village);
        else if (data.address.county) parts.push(data.address.county);
        
        if (data.address.state) parts.push(data.address.state);
        if (data.address.country) parts.push(data.address.country);
        
        const name = parts.join(', ') || data.display_name;
        
        console.log('Fetched location name:', name, 'isMapCenter:', isMapCenter);
        
        if (isMapCenter) {
          setMapCenterLocation(name);
        } else {
          setLocationName(name);
        }
      } else {
        if (isMapCenter) {
          setMapCenterLocation('');
        } else {
          setLocationName('Location Name Unavailable');
        }
      }
    } catch (error) {
      console.error('Error fetching location name:', error);
      if (isMapCenter) {
        setMapCenterLocation('');
      } else {
        setLocationName('');
      }
    } finally {
      if (isMapCenter) {
        setLoadingMapCenter(false);
      } else {
        setLoadingLocation(false);
      }
    }
  };

  // Handle map center changes (while scrolling/panning)
  const handleMapCenterChange = (lat, lng) => {
    fetchLocationName(lat, lng, true);
  };

  // Fetch location name when position changes (for selected points)
  useEffect(() => {
    if (selectedPos && selectionMode === 'point') {
      fetchLocationName(selectedPos.lat, selectedPos.lng, false);
    } else if (selectionMode === 'polygon' && polygonPoints.length >= 3) {
      // For polygon, use centroid
      const avgLat = polygonPoints.reduce((sum, p) => sum + p[0], 0) / polygonPoints.length;
      const avgLng = polygonPoints.reduce((sum, p) => sum + p[1], 0) / polygonPoints.length;
      fetchLocationName(avgLat, avgLng, false);
    } else {
      setLocationName('');
    }
  }, [selectedPos, polygonPoints, selectionMode]);

  const handleClearPolygon = () => {
    setPolygonPoints([]);
    setSelectedPos(null);
  };

  const handleAnalyze = () => {
    if (selectionMode === 'polygon' && polygonPoints.length >= 3) {
      // Pass polygon data for analysis
      onAnalyze({ type: 'polygon', points: polygonPoints });
    } else if (selectionMode === 'point' && selectedPos) {
      // Pass single point data for analysis
      onAnalyze({ type: 'point', position: selectedPos });
    }
  };

  return (
    <div className="relative h-full w-full min-h-[400px] sm:min-h-[500px] lg:min-h-[600px]">
      <MapContainer 
        center={[20.5937, 78.9629]} // India center
        zoom={5}
        style={{ height: '100%', width: '100%' }}
        className="w-full h-full"
        attributionControl={false}
      >
        {/* Satellite imagery tile layer */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
        />
        
        <MapController />
        <MapCenterTracker onCenterChange={handleMapCenterChange} />
        <SemanticZoomLabels />
        
        <MapClickHandler 
          setSelectedPos={setSelectedPos}
          polygonPoints={polygonPoints}
          setPolygonPoints={setPolygonPoints}
          selectionMode={selectionMode}
        />
        
        {/* Single point marker */}
        {selectedPos && selectionMode === 'point' && (
          <Marker position={[selectedPos.lat, selectedPos.lng]} icon={createGlowingIcon()}>
            <Popup
              className="dark-popup"
              closeButton={false}
            >
              <div className="text-center p-2">
                <div className="mb-3">
                  <h3 className="text-white font-semibold mb-2">📍 Selected Location</h3>
                  {loadingLocation ? (
                    <div className="flex items-center justify-center space-x-2 mb-2">
                      <div className="w-3 h-3 border border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                      <span className="text-gray-300 text-sm">Loading location...</span>
                    </div>
                  ) : locationName ? (
                    <div className="mb-2">
                      <p className="text-blue-300 font-medium text-sm">{locationName}</p>
                    </div>
                  ) : null}
                  <div className="text-gray-300 text-sm space-y-1">
                    <p><strong>Lat:</strong> {selectedPos.lat.toFixed(6)}</p>
                    <p><strong>Lng:</strong> {selectedPos.lng.toFixed(6)}</p>
                  </div>
                </div>
                <button
                  onClick={handleAnalyze}
                  disabled={isLoading}
                  className="glass-button text-white text-sm font-medium py-2 px-4 rounded-lg transition-all duration-300 hover:scale-105"
                >
                  {isLoading ? (
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 border border-white border-t-transparent rounded-full animate-spin"></div>
                      <span>Analyzing...</span>
                    </div>
                  ) : (
                    'Analyze this Location'
                  )}
                </button>
              </div>
            </Popup>
          </Marker>
        )}

        {/* Polygon points markers */}
        {selectionMode === 'polygon' && polygonPoints.map((point, index) => (
          <Marker 
            key={index} 
            position={point} 
            icon={createPolygonPointIcon(index)}
          >
            <Popup closeButton={false}>
              <div className="text-center p-2">
                <h4 className="text-white font-medium">Point {index + 1}</h4>
                <p className="text-gray-300 text-sm">
                  {point[0].toFixed(6)}, {point[1].toFixed(6)}
                </p>
              </div>
            </Popup>
          </Marker>
        ))}

        {/* Polygon shape */}
        {selectionMode === 'polygon' && polygonPoints.length >= 3 && (
          <Polygon
            positions={polygonPoints}
            pathOptions={{
              color: '#10b981',
              fillColor: '#10b981',
              fillOpacity: 0.2,
              weight: 3,
            }}
          >
            <Popup closeButton={false}>
              <div className="text-center p-2">
                <h3 className="text-white font-semibold mb-2">🔷 Selected Polygon</h3>
                {loadingLocation ? (
                  <div className="flex items-center justify-center space-x-2 mb-2">
                    <div className="w-3 h-3 border border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
                    <span className="text-gray-300 text-sm">Loading location...</span>
                  </div>
                ) : locationName ? (
                  <div className="mb-2">
                    <p className="text-emerald-300 font-medium text-sm">{locationName}</p>
                  </div>
                ) : null}
                <p className="text-gray-300 text-sm mb-3">
                  {polygonPoints.length} points selected
                </p>
                <button
                  onClick={handleAnalyze}
                  disabled={isLoading}
                  className="glass-button text-white text-sm font-medium py-2 px-4 rounded-lg transition-all duration-300 hover:scale-105"
                >
                  {isLoading ? (
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 border border-white border-t-transparent rounded-full animate-spin"></div>
                      <span>Analyzing...</span>
                    </div>
                  ) : (
                    'Analyze this Area'
                  )}
                </button>
              </div>
            </Popup>
          </Polygon>
        )}
        
        <LocationNameDisplay 
          locationName={locationName}
          loadingLocation={loadingLocation}
          selectedPos={selectedPos}
          polygonPoints={polygonPoints}
          selectionMode={selectionMode}
          mapCenterLocation={mapCenterLocation}
          loadingMapCenter={loadingMapCenter}
        />
        
        <AnalyzeButtonOnMap 
          onAnalyze={handleAnalyze}
          isLoading={isLoading}
          selectedPos={selectedPos}
          polygonPoints={polygonPoints}
          selectionMode={selectionMode}
        />
      </MapContainer>

      {/* Overlays */}
      <ModeSelector 
        selectionMode={selectionMode}
        setSelectionMode={setSelectionMode}
        onClearPolygon={handleClearPolygon}
      />
      <MapInstructions 
        selectedPos={selectedPos}
        polygonPoints={polygonPoints}
        selectionMode={selectionMode}
      />
      
      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-black bg-opacity-50 z-[1001] flex items-center justify-center">
          <div className="glass rounded-xl lg:rounded-2xl p-6 lg:p-8 text-center">
            <div className="w-10 h-10 lg:w-12 lg:h-12 border-3 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-3 lg:mb-4"></div>
            <p className="text-white font-medium text-sm lg:text-base">Processing satellite imagery...</p>
            <p className="text-gray-300 text-xs lg:text-sm mt-1">This may take a few moments</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default MapComponent