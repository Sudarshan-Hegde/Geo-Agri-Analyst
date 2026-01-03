import { useEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'

const MapComponent = ({ onAnalyze, isLoading }) => {
  const mapContainer = useRef(null)
  const map = useRef(null)
  const [lng] = useState(78.9629)
  const [lat] = useState(20.5937)
  const [zoom] = useState(4)

  useEffect(() => {
    if (map.current) return

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: 'https://demotiles.maplibre.org/style.json',
      center: [lng, lat],
      zoom: zoom
    })

    map.current.on('load', () => {
      console.log('Map loaded!')
    })
  }, [lng, lat, zoom])

  return (
    <div className="map-wrap" style={{ position: 'relative', width: '100%', height: '600px' }}>
      <div ref={mapContainer} style={{ position: 'absolute', width: '100%', height: '100%' }} />
    </div>
  )
}

export default MapComponent
