import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
import uvicorn
import math
import logging

# Reduce httpx logging noise
logging.getLogger("httpx").setLevel(logging.WARNING)

# Relative imports for package structure
from weather_service import get_agricultural_climate_summary
from huggingface_service import get_hf_service
from crop_history_service import get_crop_history_service
from crop_suggestion_service import get_crop_suggestion_service
from polygon_utils import (
    generate_grid_samples,
    estimate_polygon_area_km2,
    determine_optimal_zoom,
    aggregate_predictions
)

app = FastAPI(title="Geo-Agri Analyst API", version="1.0.0")

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for coordinates
class Coords(BaseModel):
    lat: float
    lng: float

class AnalysisRequest(BaseModel):
    type: str  # 'point' or 'polygon'
    lat: Optional[float] = None
    lng: Optional[float] = None
    points: Optional[List[List[float]]] = None  # For polygon points [[lat, lng], ...]

# Tiny placeholder images (1x1 pixels) as base64 strings
# Red 1x1 pixel for "before" image
IMG_LR_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhqsxYAAAAABJRU5ErkJggg=="

# Green 1x1 pixel for "after" image  
IMG_SR_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60eADgAAAABJRU5ErkJggg=="

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth (in kilometers)
    using the Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of point 1 in decimal degrees
        lat2, lon2: Latitude and longitude of point 2 in decimal degrees
        
    Returns:
        Distance in kilometers
    """
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    distance = R * c
    return distance

def calculate_polygon_area(points):
    """
    Calculate the area of a polygon using the Shoelace formula with geodesic corrections.
    
    Args:
        points: List of [lat, lng] coordinates
        
    Returns:
        Area in hectares
    """
    if len(points) < 3:
        return 0.0
    
    # Convert lat/lng to approximate Cartesian coordinates (in meters)
    # Using Equirectangular projection (good for small areas)
    n = len(points)
    
    # Calculate centroid for reference point
    center_lat = sum(p[0] for p in points) / n
    center_lon = sum(p[1] for p in points) / n
    
    # Convert to meters using approximation
    # At the equator: 1 degree ≈ 111.32 km
    # Longitude varies with latitude: cos(lat) * 111.32 km
    lat_to_m = 111320.0  # meters per degree latitude
    lon_to_m = 111320.0 * math.cos(math.radians(center_lat))  # meters per degree longitude
    
    # Convert points to Cartesian coordinates
    x_coords = [(p[1] - center_lon) * lon_to_m for p in points]
    y_coords = [(p[0] - center_lat) * lat_to_m for p in points]
    
    # Apply Shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x_coords[i] * y_coords[j]
        area -= x_coords[j] * y_coords[i]
    
    area = abs(area) / 2.0  # Area in square meters
    
    # Convert to hectares (1 hectare = 10,000 square meters)
    area_hectares = area / 10000.0
    
    return round(area_hectares, 2)

def calculate_polygon_perimeter(points):
    """
    Calculate the perimeter of a polygon using Haversine distance for each edge.
    
    Args:
        points: List of [lat, lng] coordinates
        
    Returns:
        Perimeter in kilometers
    """
    if len(points) < 2:
        return 0.0
    
    perimeter = 0.0
    n = len(points)
    
    for i in range(n):
        # Get current point and next point (wrapping around to first point at the end)
        p1 = points[i]
        p2 = points[(i + 1) % n]
        
        # Calculate distance between consecutive points
        distance = haversine_distance(p1[0], p1[1], p2[0], p2[1])
        perimeter += distance
    
    return round(perimeter, 3)

@app.get("/")
async def root():
    return {"message": "Geo-Agri Analyst API - Enhanced with Live Weather Data", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint - also checks HuggingFace Space availability"""
    hf_service = get_hf_service()
    hf_healthy = await hf_service.check_health()
    
    return {
        "status": "healthy", 
        "mode": "live_backend_with_weather_and_ml",
        "services": {
            "weather": "available",
            "huggingface_ml": "available" if hf_healthy else "unavailable (using fallback)"
        }
    }

@app.post("/api/v1/weather")
async def get_weather_data(coords: Coords):
    """
    Dedicated endpoint for fetching weather/climate data.
    
    Args:
        coords: Coordinates object with lat and lng
        
    Returns:
        Agricultural climate analysis data
    """
    try:
        weather_data = await get_agricultural_climate_summary(coords.lat, coords.lng)
        if weather_data:
            return {
                "status": "success",
                "coordinates": {"lat": coords.lat, "lng": coords.lng},
                "data": weather_data
            }
        else:
            return {
                "status": "error",
                "message": "Unable to fetch weather data for the specified coordinates",
                "coordinates": {"lat": coords.lat, "lng": coords.lng}
            }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Weather service error: {str(e)}",
            "coordinates": {"lat": coords.lat, "lng": coords.lng}
        }

@app.post("/api/v1/crop-history")
async def get_crop_history_data(coords: Coords):
    """
    Dedicated endpoint for fetching historical crop and land use data.
    
    Args:
        coords: Coordinates object with lat and lng
        
    Returns:
        Historical crop data including NDVI trends, seasonal patterns, and land use analysis
    """
    try:
        crop_history_service = get_crop_history_service()
        history_data = await crop_history_service.get_crop_history(coords.lat, coords.lng, years=5)
        
        if history_data:
            return {
                "status": "success",
                "coordinates": {"lat": coords.lat, "lng": coords.lng},
                "data": history_data
            }
        else:
            return {
                "status": "error",
                "message": "Unable to fetch crop history for the specified coordinates",
                "coordinates": {"lat": coords.lat, "lng": coords.lng}
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Crop history service error: {str(e)}",
            "coordinates": {"lat": coords.lat, "lng": coords.lng}
        }

class CropSuggestionRequest(BaseModel):
    lat: float
    lng: float
    land_class: str
    weather_data: Optional[dict] = None
    crop_history: Optional[dict] = None
    farm_size_hectares: float = 1.0
    risk_tolerance: str = "medium"  # low, medium, high

@app.post("/api/v1/crop-suggestions")
async def get_crop_suggestions_endpoint(request: CropSuggestionRequest):
    """
    Get profit-optimized crop suggestions based on location and conditions.
    
    Args:
        request: CropSuggestionRequest with location, land class, and preferences
        
    Returns:
        Ranked list of crop suggestions with profitability analysis
    """
    try:
        crop_service = get_crop_suggestion_service()
        suggestions = await crop_service.get_crop_suggestions(
            lat=request.lat,
            lng=request.lng,
            land_class=request.land_class,
            weather_data=request.weather_data,
            crop_history=request.crop_history,
            farm_size_hectares=request.farm_size_hectares,
            risk_tolerance=request.risk_tolerance
        )
        
        if suggestions:
            return {
                "status": "success",
                "data": suggestions
            }
        else:
            return {
                "status": "error",
                "message": "Unable to generate crop suggestions"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Crop suggestion service error: {str(e)}"
        }

@app.post("/api/v1/analyze")
async def analyze_location(request: Union[Coords, AnalysisRequest]):
    """
    Enhanced analysis endpoint that combines satellite image analysis with weather/climate data.
    Supports both point and polygon analysis with multi-image sampling for polygons.
    
    Args:
        request: Either Coords (for backward compatibility) or AnalysisRequest
        
    Returns:
        Comprehensive analysis results including land classification and climate data
    """
    
    # Handle both old Coords format and new AnalysisRequest format
    if hasattr(request, 'type'):
        # New AnalysisRequest format
        analysis_type = request.type
        lat, lng = request.lat, request.lng
        points = request.points
        
        if analysis_type == "polygon" and points:
            num_points = len(points)
            print(f"Received polygon analysis with {num_points} points")
            # Calculate area (simplified)
            if num_points >= 3:
                print(f"Polygon points: {points}")
                # For polygon analysis, use centroid coordinates for weather data
                if lat is None or lng is None:
                    # Calculate centroid if not provided
                    lat = sum(point[0] for point in points) / len(points)
                    lng = sum(point[1] for point in points) / len(points)
        else:
            print(f"Received point analysis at: {lat}, {lng}")
    else:
        # Old Coords format (backward compatibility)
        analysis_type = "point"
        lat, lng = request.lat, request.lng
        points = None
        print(f"Received point analysis at: {lat}, {lng}")
    
    hf_service = get_hf_service()
    
    # For polygon analysis, use multi-image sampling
    if analysis_type == "polygon" and points and len(points) >= 3:
        print("🔄 Starting multi-image polygon analysis...")
        
        # Calculate polygon area
        area_km2 = estimate_polygon_area_km2(points)
        print(f"📏 Polygon area: {area_km2:.4f} km²")
        
        # Determine optimal zoom level based on area
        optimal_zoom = determine_optimal_zoom(area_km2, is_polygon=True)
        print(f"🔍 Using zoom level {optimal_zoom} for polygon analysis")
        
        # Generate grid sample points
        sample_coords = generate_grid_samples(points, max_samples=50, min_samples=5)
        print(f"📍 Generated {len(sample_coords)} sample points across polygon")
        
        # Get predictions for all sample points
        batch_predictions = await hf_service.predict_batch(sample_coords, zoom=optimal_zoom)
        
        # Aggregate predictions
        aggregated = aggregate_predictions(batch_predictions)
        
        # Use first sample's images for display (representative)
        before_img_b64 = batch_predictions[0].get("before_image_b64", IMG_LR_B64) if batch_predictions else IMG_LR_B64
        after_img_b64 = batch_predictions[0].get("after_image_b64", IMG_SR_B64) if batch_predictions else IMG_SR_B64
        
        # Build comprehensive area info
        area_hectares = calculate_polygon_area(points)
        perimeter_km = calculate_polygon_perimeter(points)
        
        area_info = {
            "total_points": len(points),
            "estimated_area_hectares": area_hectares,
            "estimated_area_km2": area_km2,
            "perimeter_km": perimeter_km,
            "dominant_land_type": aggregated["dominant_class"],
            "centroid_coordinates": {"lat": lat, "lng": lng},
            "sample_count": aggregated["sample_count"],
            "zoom_level_used": optimal_zoom,
            "class_distribution": aggregated["class_distribution"]
        }
        
        # Fetch weather data for centroid
        weather_data = None
        try:
            print(f"Fetching weather data for centroid: {lat}, {lng}")
            weather_data = await get_agricultural_climate_summary(lat, lng)
        except Exception as e:
            print(f"Error fetching weather data: {str(e)}")
        
        # Fetch crop history for centroid
        crop_history = None
        try:
            print(f"Fetching crop history for centroid: {lat}, {lng}")
            crop_history_service = get_crop_history_service()
            crop_history = await crop_history_service.get_crop_history(lat, lng, years=5)
            print("✅ Successfully retrieved crop history")
        except Exception as e:
            print(f"Error fetching crop history: {str(e)}")
        
        # Fetch crop suggestions for polygon
        crop_suggestions = None
        try:
            print(f"Generating crop suggestions for polygon...")
            crop_service = get_crop_suggestion_service()
            crop_suggestions = await crop_service.get_crop_suggestions(
                lat=lat,
                lng=lng,
                land_class=aggregated["dominant_class"],
                weather_data=weather_data,
                crop_history=crop_history,
                farm_size_hectares=area_hectares,
                risk_tolerance="medium"
            )
            print("✅ Successfully generated crop suggestions")
        except Exception as e:
            print(f"Error generating crop suggestions: {str(e)}")
        
        # Combine results
        result = {
            "land_class": aggregated["dominant_class"],
            "confidence": aggregated["confidence"],
            "before_image_b64": before_img_b64,
            "after_image_b64": after_img_b64,
            "top_predictions": dict(list(aggregated["class_distribution"].items())[:5]),
            "analysis_type": "polygon",
            "coordinates": {"lat": lat, "lng": lng},
            "area_info": area_info,
            "weather_data": weather_data,
            "crop_history": crop_history,
            "crop_suggestions": crop_suggestions,
            "ml_source": "multi-sample-aggregated",
            "detailed_samples": batch_predictions[:10]  # Include first 10 detailed samples
        }
        
        print(f"✅ Polygon analysis complete: {aggregated['dominant_class']} ({aggregated['confidence']:.2%} confidence)")
        
    else:
        # Single point analysis
        print(f"🔄 Starting single-point analysis at ({lat}, {lng})")
        
        # Use lower zoom for single point to capture more context
        point_zoom = determine_optimal_zoom(0, is_polygon=False)
        print(f"🔍 Using zoom level {point_zoom} for point analysis")
        
        # Get prediction with custom zoom
        from satellite_service import get_satellite_service
        satellite_svc = get_satellite_service()
        image = satellite_svc.get_satellite_image(lat, lng, size=30, zoom=point_zoom)
        
        ml_results = await hf_service.predict(lat, lng, image=image)
        
        # Extract results
        selected_class = ml_results.get("land_class", "Unknown")
        confidence = ml_results.get("confidence", 0.0)
        before_img_b64 = ml_results.get("before_image_b64", IMG_LR_B64)
        after_img_b64 = ml_results.get("after_image_b64", IMG_SR_B64)
        top_predictions = ml_results.get("predictions", {})
        
        # Fetch weather data
        weather_data = None
        try:
            print(f"Fetching weather data for coordinates: {lat}, {lng}")
            weather_data = await get_agricultural_climate_summary(lat, lng)
        except Exception as e:
            print(f"Error fetching weather data: {str(e)}")
        
        # Fetch crop history
        crop_history = None
        try:
            print(f"Fetching crop history for coordinates: {lat}, {lng}")
            crop_history_service = get_crop_history_service()
            crop_history = await crop_history_service.get_crop_history(lat, lng, years=5)
            print("✅ Successfully retrieved crop history")
        except Exception as e:
            print(f"Error fetching crop history: {str(e)}")
        
        # Fetch crop suggestions for point analysis
        crop_suggestions = None
        try:
            print(f"Generating crop suggestions for point...")
            crop_service = get_crop_suggestion_service()
            crop_suggestions = await crop_service.get_crop_suggestions(
                lat=lat,
                lng=lng,
                land_class=selected_class,
                weather_data=weather_data,
                crop_history=crop_history,
                farm_size_hectares=1.0,  # Default 1 hectare for point analysis
                risk_tolerance="medium"
            )
            print("✅ Successfully generated crop suggestions")
        except Exception as e:
            print(f"Error generating crop suggestions: {str(e)}")
        
        area_info = None
        if analysis_type == "polygon" and points:
            # Single point polygon (too small for multi-sampling)
            area_hectares = calculate_polygon_area(points)
            perimeter_km = calculate_polygon_perimeter(points)
            
            area_info = {
                "total_points": len(points),
                "estimated_area_hectares": area_hectares,
                "perimeter_km": perimeter_km,
                "dominant_land_type": selected_class,
                "centroid_coordinates": {"lat": lat, "lng": lng},
                "note": "Polygon too small for multi-sampling, using centroid analysis"
            }
        
        # Combine results
        result = {
            "land_class": selected_class,
            "confidence": confidence,
            "before_image_b64": before_img_b64,
            "after_image_b64": after_img_b64,
            "top_predictions": top_predictions,
            "analysis_type": analysis_type,
            "coordinates": {"lat": lat, "lng": lng},
            "area_info": area_info,
            "weather_data": weather_data,
            "crop_history": crop_history,
            "crop_suggestions": crop_suggestions,
            "ml_source": ml_results.get("source", "unknown"),
            "zoom_level_used": point_zoom
        }
        
        print(f"✅ Point analysis complete: {selected_class} ({confidence:.2%} confidence)")
    
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)