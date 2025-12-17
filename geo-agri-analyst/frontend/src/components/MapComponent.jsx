import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

const MapComponent = ({ onAnalyze, isLoading, selectedPos, setSelectedPos, polygonPoints, setPolygonPoints, selectionMode, setSelectionMode }) => {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [currentLayer, setCurrentLayer] = useState('satellite');
  const [currentLocation, setCurrentLocation] = useState('Loading location...');
  const [mapCenter, setMapCenter] = useState({ lat: 20.5937, lng: 78.9629 });
  const markersRef = useRef([]);
  const polygonLayerRef = useRef(null);

  // Function to create base style with political boundaries overlay
  const createMapStyle = (baseLayer) => {
    const baseStyles = {
      satellite: {
        source: {
          type: 'raster',
          tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
          tileSize: 256,
          attribution: '© Esri'
        },
        layer: {
          id: 'satellite-tiles',
          type: 'raster',
          source: 'satellite-source'
        }
      },
      streets: {
        source: {
          type: 'raster',
          tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
          tileSize: 256,
          attribution: '© OpenStreetMap contributors'
        },
        layer: {
          id: 'streets-tiles',
          type: 'raster',
          source: 'streets-source'
        }
      },
      terrain: {
        source: {
          type: 'raster',
          tiles: ['https://tile.opentopomap.org/{z}/{x}/{y}.png'],
          tileSize: 256,
          attribution: '© OpenTopoMap contributors'
        },
        layer: {
          id: 'terrain-tiles',
          type: 'raster',
          source: 'terrain-source'
        }
      },
      dark: {
        source: {
          type: 'raster',
          tiles: ['https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png'],
          tileSize: 256,
          attribution: '© CARTO'
        },
        layer: {
          id: 'dark-tiles',
          type: 'raster',
          source: 'dark-source'
        }
      }
    };

    const selectedStyle = baseStyles[baseLayer];
    
    // Create complete style with political boundaries overlay and globe projection
    return {
      version: 8,
      glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
      projection: {
        type: 'globe'
      },
      sources: {
        [selectedStyle.layer.source]: selectedStyle.source,
        'countries': {
          type: 'geojson',
          data: 'https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson'
        }
      },
      layers: [
        selectedStyle.layer,
        // Country borders
        {
          id: 'country-boundaries',
          type: 'line',
          source: 'countries',
          paint: {
            'line-color': '#ff6b6b',
            'line-width': [
              'interpolate',
              ['linear'],
              ['zoom'],
              0, 1,
              5, 2,
              10, 3
            ],
            'line-opacity': 0.8
          }
        },
        // Country names
        {
          id: 'country-labels',
          type: 'symbol',
          source: 'countries',
          layout: {
            'text-field': ['get', 'ADMIN'],
            'text-size': [
              'interpolate',
              ['linear'],
              ['zoom'],
              0, 8,
              5, 12,
              10, 16
            ],
            'text-font': ['Open Sans Regular', 'Arial Unicode MS Regular']
          },
          paint: {
            'text-color': '#ffffff',
            'text-halo-color': '#000000',
            'text-halo-width': 2,
            'text-halo-blur': 1
          }
        }
      ]
    };
  };

  useEffect(() => {
    if (map.current) return; // Initialize map only once

    console.log('🗺️ Initializing MapLibre GL map with globe projection...');

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: createMapStyle('satellite'), // Start with satellite + boundaries
      center: [78.9629, 20.5937],
      zoom: 1.5,
      bearing: 0,
      antialias: true
    });

    // Add navigation controls
    map.current.addControl(new maplibregl.NavigationControl(), 'top-right');
    map.current.addControl(new maplibregl.FullscreenControl(), 'top-right');
    
    // Add atmosphere effect for globe
    map.current.on('style.load', () => {
      map.current.setFog({
        color: 'rgb(186, 210, 235)', // Lower atmosphere
        'high-color': 'rgb(36, 92, 223)', // Upper atmosphere
        'horizon-blend': 0.02, // Atmosphere thickness
        'space-color': 'rgb(11, 11, 25)', // Background space color
        'star-intensity': 0.6 // Stars intensity
      });
      console.log('✅ Atmosphere/fog applied to globe');
    });

    map.current.on('load', () => {
      setMapLoaded(true);
      console.log('✅ Map loaded successfully with globe projection!');
      console.log('Map projection:', map.current.getProjection());
      updateLocation(map.current.getCenter());
    });

    // Update location when map moves
    map.current.on('moveend', () => {
      const center = map.current.getCenter();
      setMapCenter({ lat: center.lat, lng: center.lng });
      updateLocation(center);
    });

    map.current.on('error', (e) => {
      console.error('❌ Map error:', e);
    });

    return () => {
      if (map.current) {
        clearMarkers();
        clearPolygon();
        map.current.remove();
        map.current = null;
      }
    };
  }, []);

  // Handle external location selection (from search bar)
  useEffect(() => {
    if (selectedPos && map.current && mapLoaded) {
      console.log('🗺️ Flying to searched location:', selectedPos);
      
      // Fly to the selected location
      map.current.flyTo({
        center: [selectedPos.lng, selectedPos.lat],
        zoom: 12,
        duration: 2000,
        essential: true
      });
      
      // Add marker if in point mode
      if (selectionMode === 'point') {
        clearMarkers();
        addMarker([selectedPos.lng, selectedPos.lat], selectedPos);
      }
    }
  }, [selectedPos, mapLoaded]);

  const changeMapStyle = (styleType) => {
    if (!map.current) return;
    
    setCurrentLayer(styleType);
    
    // Get current map state
    const currentCenter = map.current.getCenter();
    const currentZoom = map.current.getZoom();
    const currentBearing = map.current.getBearing();
    const currentPitch = map.current.getPitch();
    
    // Store current markers and polygon data
    const hasPolygon = polygonPoints.length >= 3;
    
    // Apply new style with political boundaries
    map.current.setStyle(createMapStyle(styleType));
    
    // Restore map state and fog after style loads
    map.current.once('styledata', () => {
      map.current.jumpTo({
        center: currentCenter,
        zoom: currentZoom,
        bearing: currentBearing,
        pitch: currentPitch
      });
      
      // Re-apply fog effect
      map.current.setFog({
        color: 'rgb(186, 210, 235)',
        'high-color': 'rgb(36, 92, 223)',
        'horizon-blend': 0.02,
        'space-color': 'rgb(11, 11, 25)',
        'star-intensity': 0.6
      });
      
      // Re-add polygon if it existed
      if (hasPolygon && selectionMode === 'polygon') {
        setTimeout(() => {
          drawPolygon(polygonPoints);
        }, 100);
      }
    });
  };

  const updateLocation = async (center) => {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${center.lat}&lon=${center.lng}&zoom=10`
      );
      const data = await response.json();
      
      if (data.address) {
        const { city, town, village, county, state, country } = data.address;
        const locationName = city || town || village || county || state || country || 'Unknown Location';
        const region = state && country ? `${state}, ${country}` : country || '';
        setCurrentLocation(region ? `${locationName}, ${region}` : locationName);
      } else {
        setCurrentLocation(`${center.lat.toFixed(4)}, ${center.lng.toFixed(4)}`);
      }
    } catch (error) {
      console.error('Error fetching location:', error);
      setCurrentLocation(`${center.lat.toFixed(4)}, ${center.lng.toFixed(4)}`);
    }
  };

  const handleLocationSearch = async (locationName) => {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(locationName)}`
      );
      const data = await response.json();
      
      if (data && data.length > 0) {
        const { lat, lon } = data[0];
        
        if (map.current) {
          map.current.flyTo({
            center: [parseFloat(lon), parseFloat(lat)],
            zoom: 12,
            duration: 2000
          });
          
          if (selectionMode === 'point') {
            const position = { lat: parseFloat(lat), lng: parseFloat(lon) };
            setSelectedPos(position);
            addMarker([parseFloat(lon), parseFloat(lat)], position);
          }
        }
      }
    } catch (error) {
      console.error('Error searching location:', error);
    }
  };

  const handleMapClick = (e) => {
    if (!map.current) return;

    const { lng, lat } = e.lngLat;
    
    console.log('🖱️ Map clicked! Mode:', selectionMode, 'Position:', { lat, lng });

    if (selectionMode === 'point') {
      // Clear previous markers and set new point
      clearMarkers();
      const position = { lat, lng };
      setSelectedPos(position);
      addMarker([lng, lat], position);
      console.log('✅ Point selected:', position);
    } else if (selectionMode === 'polygon') {
      // Add point to polygon
      const newPoints = [...polygonPoints, { lat, lng }];
      console.log('📍 Adding polygon point:', { lat, lng });
      console.log('📊 Previous points:', polygonPoints.length, '→ New points:', newPoints.length);
      setPolygonPoints(newPoints);
      addMarker([lng, lat], { lat, lng }, true);
      console.log('✅ Polygon point added. Total points:', newPoints.length);
    } else {
      console.warn('⚠️ Unknown selection mode:', selectionMode);
    }
  };

  const clearMarkers = () => {
    markersRef.current.forEach(marker => marker.remove());
    markersRef.current = [];
  };

  const clearPolygon = () => {
    if (!map.current) return;
    
    try {
      // Remove all polygon layers
      if (map.current.getLayer('polygon-outline-inner')) {
        map.current.removeLayer('polygon-outline-inner');
      }
      if (map.current.getLayer('polygon-outline')) {
        map.current.removeLayer('polygon-outline');
      }
      if (map.current.getLayer('polygon-fill')) {
        map.current.removeLayer('polygon-fill');
      }
      if (map.current.getSource('polygon')) {
        map.current.removeSource('polygon');
      }
      console.log('✅ Polygon cleared');
    } catch (error) {
      console.error('Error clearing polygon:', error);
    }
  };

  const addMarker = (lngLat, position, isPolygonPoint = false) => {
    if (!map.current) return;

    // Create marker element
    const el = document.createElement('div');
    el.className = isPolygonPoint ? 'polygon-marker' : 'point-marker';
    el.style.width = '16px';
    el.style.height = '16px';
    el.style.borderRadius = '50%';
    el.style.backgroundColor = isPolygonPoint ? '#10b981' : '#3b82f6';
    el.style.border = '3px solid white';
    el.style.boxShadow = '0 0 10px rgba(0,0,0,0.5)';
    el.style.cursor = 'pointer';

    const marker = new maplibregl.Marker({ element: el })
      .setLngLat(lngLat)
      .setPopup(
        new maplibregl.Popup({ offset: 25 })
          .setHTML(
            `<div style="color: white; background: rgba(0,0,0,0.9); padding: 8px; border-radius: 4px;">
              <strong style="color: ${isPolygonPoint ? '#10b981' : '#3b82f6'}">${isPolygonPoint ? '🔷 Polygon Point' : '📍 Selected Point'}</strong><br/>
              Lat: ${position.lat.toFixed(6)}<br/>
              Lng: ${position.lng.toFixed(6)}
            </div>`
          )
      )
      .addTo(map.current);

    markersRef.current.push(marker);
    return marker;
  };

  const drawPolygon = (points) => {
    if (!map.current || points.length < 3) {
      console.log('⚠️ Cannot draw polygon: need at least 3 points, have:', points.length);
      return;
    }

    try {
      console.log('🎨 Drawing polygon with', points.length, 'points:', points);

      // Clear existing polygon first
      clearPolygon();

      // Wait a frame to ensure clear is complete
      requestAnimationFrame(() => {
        try {
          // Double check map is still available and loaded
          if (!map.current) {
            console.log('❌ Map instance lost');
            return;
          }
          
          if (!map.current.isStyleLoaded()) {
            console.log('⚠️ Map style not loaded, retrying in 200ms...');
            setTimeout(() => drawPolygon(points), 200);
            return;
          }

          // Create GeoJSON for the polygon
          const coordinates = points.map(p => [p.lng, p.lat]);
          coordinates.push(coordinates[0]); // Close the polygon

          const geojson = {
            type: 'Feature',
            geometry: {
              type: 'Polygon',
              coordinates: [coordinates]
            }
          };

          console.log('📐 Polygon GeoJSON:', geojson);

          // Add source
          map.current.addSource('polygon', {
            type: 'geojson',
            data: geojson
          });

          // Add fill layer with enhanced visibility and glow effect
          map.current.addLayer({
            id: 'polygon-fill',
            type: 'fill',
            source: 'polygon',
            paint: {
              'fill-color': '#10b981',
              'fill-opacity': 0.5
            }
          });

          // Add outline layer with glow
          map.current.addLayer({
            id: 'polygon-outline',
            type: 'line',
            source: 'polygon',
            paint: {
              'line-color': '#10b981',
              'line-width': 5,
              'line-opacity': 1,
              'line-blur': 2
            }
          });

          // Add inner outline for more definition
          map.current.addLayer({
            id: 'polygon-outline-inner',
            type: 'line',
            source: 'polygon',
            paint: {
              'line-color': '#ffffff',
              'line-width': 2,
              'line-opacity': 0.8
            }
          });

          console.log('✅ Polygon drawn successfully!');
        } catch (innerError) {
          console.error('❌ Error in drawPolygon inner:', innerError);
        }
      });
    } catch (error) {
      console.error('❌ Error drawing polygon:', error);
    }
  };

  // Update click handler when selectionMode or polygonPoints change
  useEffect(() => {
    if (!map.current || !mapLoaded) return;
    
    // Remove old handler
    map.current.off('click', handleMapClick);
    // Add new handler with current closure
    map.current.on('click', handleMapClick);
    
    console.log('🔄 Click handler updated for mode:', selectionMode);
    
    return () => {
      if (map.current) {
        map.current.off('click', handleMapClick);
      }
    };
  }, [selectionMode, polygonPoints, selectedPos, mapLoaded]);

  // Clear polygon when switching to point mode
  useEffect(() => {
    console.log('🔄 Selection mode changed to:', selectionMode);
    if (selectionMode === 'point') {
      clearPolygon();
      if (polygonPoints.length > 0) {
        console.log('🗑️ Clearing', polygonPoints.length, 'polygon points');
        setPolygonPoints([]);
      }
    } else if (selectionMode === 'polygon') {
      clearMarkers();
      if (selectedPos) {
        console.log('🗑️ Clearing selected point');
        setSelectedPos(null);
      }
    }
  }, [selectionMode]);

  // Redraw polygon when points change
  useEffect(() => {
    console.log('🔄 Polygon points changed:', polygonPoints.length, 'Mode:', selectionMode);
    if (selectionMode === 'polygon' && polygonPoints.length >= 3 && map.current) {
      // Small delay to ensure state is updated
      const timer = setTimeout(() => {
        drawPolygon(polygonPoints);
      }, 100);
      return () => clearTimeout(timer);
    } else if (selectionMode === 'polygon' && polygonPoints.length < 3 && polygonPoints.length > 0) {
      console.log('ℹ️ Need', 3 - polygonPoints.length, 'more point(s) to draw polygon');
      clearPolygon();
    }
  }, [polygonPoints, selectionMode]);

  const handleAnalyze = () => {
    if (onAnalyze && selectedPoint) {
      onAnalyze({ point: selectedPoint });
    }
  };

  return (
    <div className="w-full h-full">
      {/* Map Container with Layer Switcher */}
      <div className="relative w-full h-full">
        <div 
          ref={mapContainer} 
          className="w-full h-full rounded-xl overflow-hidden"
          style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}
        />
        
        {/* Layer Switcher */}
        <div className="absolute top-4 left-4 z-10 bg-gray-900/95 border border-blue-500/30 rounded-lg p-2 flex gap-2 backdrop-blur-sm">
          <button 
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all border ${
              currentLayer === 'satellite' 
                ? 'bg-blue-500/20 border-blue-500 text-blue-400 shadow-lg shadow-blue-500/30' 
                : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:bg-blue-500/10 hover:border-blue-500/40 hover:text-blue-300'
            }`}
            onClick={() => changeMapStyle('satellite')}
            title="Satellite View"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/>
              <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>
            </svg>
            <span className="text-xs font-medium">Satellite</span>
          </button>
          <button 
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all border ${
              currentLayer === 'streets' 
                ? 'bg-blue-500/20 border-blue-500 text-blue-400 shadow-lg shadow-blue-500/30' 
                : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:bg-blue-500/10 hover:border-blue-500/40 hover:text-blue-300'
            }`}
            onClick={() => changeMapStyle('streets')}
            title="Streets View"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
            </svg>
            <span className="text-xs font-medium">Streets</span>
          </button>
          <button 
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all border ${
              currentLayer === 'terrain' 
                ? 'bg-blue-500/20 border-blue-500 text-blue-400 shadow-lg shadow-blue-500/30' 
                : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:bg-blue-500/10 hover:border-blue-500/40 hover:text-blue-300'
            }`}
            onClick={() => changeMapStyle('terrain')}
            title="Terrain View"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
            </svg>
            <span className="text-xs font-medium">Terrain</span>
          </button>
          <button 
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all border ${
              currentLayer === 'dark' 
                ? 'bg-blue-500/20 border-blue-500 text-blue-400 shadow-lg shadow-blue-500/30' 
                : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:bg-blue-500/10 hover:border-blue-500/40 hover:text-blue-300'
            }`}
            onClick={() => changeMapStyle('dark')}
            title="Dark View"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>
            <span className="text-xs font-medium">Dark</span>
          </button>
        </div>

        {/* Selection Mode Toggle - Bottom Left */}
        <div className="absolute bottom-4 left-4 z-10 bg-gray-900/95 border border-emerald-500/30 rounded-lg p-2 flex gap-2 backdrop-blur-sm">
          <button 
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all border ${
              selectionMode === 'point' 
                ? 'bg-emerald-500/20 border-emerald-500 text-emerald-400 shadow-lg shadow-emerald-500/30' 
                : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:bg-emerald-500/10 hover:border-emerald-500/40 hover:text-emerald-300'
            }`}
            onClick={() => {
              console.log('🔘 Point mode button clicked');
              setSelectionMode('point');
            }}
            title="Point Selection"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
              <circle cx="12" cy="10" r="3"/>
            </svg>
            <span className="text-xs font-medium">Point</span>
          </button>
          <button 
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all border ${
              selectionMode === 'polygon' 
                ? 'bg-emerald-500/20 border-emerald-500 text-emerald-400 shadow-lg shadow-emerald-500/30' 
                : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:bg-emerald-500/10 hover:border-emerald-500/40 hover:text-emerald-300'
            }`}
            onClick={() => {
              console.log('🔷 Polygon mode button clicked');
              setSelectionMode('polygon');
            }}
            title="Polygon Selection"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2l9 4.9V17L12 22l-9-4.9V7z"/>
            </svg>
            <span className="text-xs font-medium">Polygon</span>
          </button>
          {selectionMode === 'polygon' && polygonPoints.length > 0 && (
            <button 
              className="flex items-center gap-2 px-3 py-2 rounded-lg transition-all border bg-red-500/20 border-red-500 text-red-400 hover:bg-red-500/30"
              onClick={() => {
                setPolygonPoints([]);
                clearMarkers();
                clearPolygon();
              }}
              title="Clear Polygon"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
              <span className="text-xs font-medium">Clear ({polygonPoints.length})</span>
            </button>
          )}
        </div>

        {/* Location Info Panel */}
        <div className="absolute bottom-4 right-4 z-10 bg-gray-900/95 border border-blue-500/30 rounded-lg p-4 backdrop-blur-sm min-w-[200px]">
          <h3 className="text-sm font-semibold text-blue-400 mb-3">Location Info</h3>
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Latitude:</span>
              <span className="text-blue-300 font-mono">{mapCenter.lat.toFixed(6)}°</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Longitude:</span>
              <span className="text-blue-300 font-mono">{mapCenter.lng.toFixed(6)}°</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Zoom Level:</span>
              <span className="text-blue-300 font-mono">{map.current ? map.current.getZoom().toFixed(2) : '1.50'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MapComponent;
