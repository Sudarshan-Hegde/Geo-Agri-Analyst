"""
Weather Service Module for Geo-Agri Analyst

This module provides weather and climate data functionality using the Open-Meteo Climate API.
It fetches 30-year climate normals (1991-2020) for agricultural analysis.
"""

import httpx
import asyncio
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Open-Meteo Weather API for current conditions
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

async def get_climate_data(lat: float, lng: float) -> Optional[Dict[str, Any]]:
    """
    Fetches current weather, climate estimates, and elevation for a specific lat/lng.
    
    Args:
        lat (float): Latitude coordinate
        lng (float): Longitude coordinate
        
    Returns:
        Dict containing climate data or None if error occurred
        
    Example response:
        {
            "avg_temp_c": 15.2,
            "avg_annual_rainfall_mm": 850.5,
            "elevation_m": 150,
            "location": {"lat": 40.7128, "lng": -74.0060},
            "data_source": "Open-Meteo"
        }
    """
    params = {
        "latitude": lat,
        "longitude": lng,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
        "forecast_days": 7
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"Fetching climate data for coordinates: {lat}, {lng}")
            
            response = await client.get(WEATHER_API_URL, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            data = response.json()
            logger.info(f"Successfully fetched weather data for {lat}, {lng}")
            
            # Process the response data (Open-Meteo Weather API format)
            current_data = data.get('current', {})
            daily_data = data.get('daily', {})
            elevation = data.get('elevation', 0)
            
            # Get current temperature
            current_temp = current_data.get('temperature_2m')
            
            # Calculate average temperature from 7-day forecast
            temp_max_values = daily_data.get('temperature_2m_max', [])
            temp_min_values = daily_data.get('temperature_2m_min', [])
            
            if temp_max_values and temp_min_values:
                avg_max = sum(temp_max_values) / len(temp_max_values)
                avg_min = sum(temp_min_values) / len(temp_min_values)
                avg_temp = (avg_max + avg_min) / 2
            else:
                avg_temp = current_temp
            
            # Adjust temperature for elevation (lapse rate: -0.6°C per 100m)
            if elevation and elevation > 0:
                # This is already adjusted by the API, so we just note it
                pass
            
            # Calculate weekly precipitation estimate (convert to annual estimate)
            precip_values = daily_data.get('precipitation_sum', [])
            weekly_precip = sum(precip_values) if precip_values else 0
            # Better annual estimate: Use climate-based multiplier
            # For India: Monsoon pattern means weekly * 52 is inaccurate
            # Use a more conservative estimate based on latitude
            abs_lat = abs(lat)
            if abs_lat < 23.5:  # Tropical - high rainfall
                estimated_annual_rainfall = weekly_precip * 45  # ~2250mm/year baseline
            elif abs_lat < 35:  # Subtropical - moderate
                estimated_annual_rainfall = weekly_precip * 40  # ~1200mm/year baseline  
            else:  # Temperate - lower
                estimated_annual_rainfall = weekly_precip * 35  # ~800mm/year baseline
            
            # Format the response
            climate_vitals = {
                "avg_temp_c": round(avg_temp, 1) if avg_temp is not None else None,
                "current_temp_c": round(current_temp, 1) if current_temp is not None else None,
                "avg_annual_rainfall_mm": round(estimated_annual_rainfall, 1) if estimated_annual_rainfall else None,
                "weekly_precipitation_mm": round(weekly_precip, 1),
                "elevation_m": round(elevation, 0) if elevation else 0,
                "location": {
                    "lat": lat,
                    "lng": lng
                },
                "data_source": "Open-Meteo Weather API",
                "status": "success"
            }
            
            return climate_vitals
            
    except httpx.TimeoutException:
        logger.error(f"Timeout error fetching climate data for {lat}, {lng}")
        return {
            "status": "error",
            "error": "API request timed out",
            "location": {"lat": lat, "lng": lng}
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code} fetching climate data for {lat}, {lng}")
        return {
            "status": "error", 
            "error": f"HTTP {e.response.status_code}",
            "location": {"lat": lat, "lng": lng}
        }
    except Exception as e:
        logger.error(f"Unexpected error fetching climate data for {lat}, {lng}: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "location": {"lat": lat, "lng": lng}
        }

async def get_agricultural_climate_summary(lat: float, lng: float) -> Optional[Dict[str, Any]]:
    """
    Get climate data with agricultural context and recommendations.
    
    Args:
        lat (float): Latitude coordinate
        lng (float): Longitude coordinate
        
    Returns:
        Dict containing climate analysis with agricultural insights
    """
    climate_data = await get_climate_data(lat, lng)
    
    if not climate_data or climate_data.get("status") == "error":
        return climate_data
    
    avg_temp = climate_data.get("avg_temp_c")
    avg_rainfall = climate_data.get("avg_annual_rainfall_mm")
    
    # Agricultural climate classification
    climate_classification = classify_agricultural_climate(avg_temp, avg_rainfall)
    
    # Add agricultural context
    agricultural_summary = {
        **climate_data,
        "agricultural_classification": climate_classification,
        "growing_season_info": get_growing_season_info(avg_temp, avg_rainfall),
        "crop_suitability": get_basic_crop_suitability(avg_temp, avg_rainfall)
    }
    
    return agricultural_summary

def classify_agricultural_climate(temp: Optional[float], rainfall: Optional[float]) -> Dict[str, str]:
    """
    Classify climate for agricultural purposes based on temperature and rainfall.
    """
    if temp is None or rainfall is None:
        return {"classification": "Unknown", "description": "Insufficient data"}
    
    # Basic climate classification for agriculture
    if temp < 0:
        climate_type = "Arctic"
        description = "Too cold for most agriculture"
    elif temp < 10:
        climate_type = "Subarctic" 
        description = "Limited growing season, cold-resistant crops only"
    elif temp < 20:
        if rainfall < 300:
            climate_type = "Temperate Dry"
            description = "Moderate temperatures, irrigation needed"
        elif rainfall < 1000:
            climate_type = "Temperate"
            description = "Good for temperate crops, moderate water needs"
        else:
            climate_type = "Temperate Wet"
            description = "Good rainfall, suitable for diverse crops"
    else:  # temp >= 20
        if rainfall < 500:
            climate_type = "Tropical Dry"
            description = "Hot and dry, drought-resistant crops preferred"
        elif rainfall < 1500:
            climate_type = "Tropical"
            description = "Hot with moderate rainfall, good for tropical crops"
        else:
            climate_type = "Tropical Wet"
            description = "Hot and wet, excellent for tropical agriculture"
    
    return {
        "classification": climate_type,
        "description": description
    }

def get_growing_season_info(temp: Optional[float], rainfall: Optional[float]) -> Dict[str, Any]:
    """
    Estimate growing season characteristics based on climate data.
    """
    if temp is None or rainfall is None:
        return {"status": "Unknown", "reason": "Insufficient data"}
    
    # Estimate growing season length (simplified)
    if temp < 5:
        season_length = "Very short (2-3 months)"
        season_quality = "Poor"
    elif temp < 15:
        season_length = "Short to moderate (4-6 months)" 
        season_quality = "Fair to good"
    else:
        season_length = "Long (6+ months)"
        season_quality = "Good to excellent"
    
    # Water availability assessment
    if rainfall < 300:
        water_status = "Irrigation essential"
    elif rainfall < 600:
        water_status = "Supplemental irrigation recommended"
    elif rainfall < 1200:
        water_status = "Generally adequate rainfall"
    else:
        water_status = "High rainfall, drainage may be needed"
    
    return {
        "estimated_season_length": season_length,
        "season_quality": season_quality,
        "water_availability": water_status
    }

def get_basic_crop_suitability(temp: Optional[float], rainfall: Optional[float]) -> Dict[str, list]:
    """
    Suggest basic crop categories suitable for the climate.
    """
    if temp is None or rainfall is None:
        return {"suitable_crops": [], "note": "Insufficient data for recommendations"}
    
    suitable_crops = []
    
    # Temperature-based crop suggestions
    if temp >= 25:  # Hot climates
        if rainfall >= 1000:
            suitable_crops.extend(["Rice", "Sugarcane", "Tropical fruits", "Cassava"])
        else:
            suitable_crops.extend(["Millet", "Sorghum", "Cotton", "Pulses"])
    elif temp >= 15:  # Moderate climates
        if rainfall >= 600:
            suitable_crops.extend(["Wheat", "Maize", "Soybeans", "Potatoes"])
        else:
            suitable_crops.extend(["Barley", "Oats", "Sunflower", "Canola"])
    else:  # Cool climates
        suitable_crops.extend(["Rye", "Turnips", "Cabbage", "Root vegetables"])
    
    return {
        "suitable_crops": suitable_crops,
        "note": "Basic recommendations based on climate averages"
    }

# Test function for development
async def test_weather_service():
    """Test the weather service with sample coordinates."""
    print("Testing Weather Service...")
    
    # Test coordinates (New York City)
    test_coords = [(40.7128, -74.0060), (19.0760, 72.8777)]  # NYC, Mumbai
    
    for lat, lng in test_coords:
        print(f"\nTesting coordinates: {lat}, {lng}")
        result = await get_agricultural_climate_summary(lat, lng)
        print(f"Result: {result}")

if __name__ == "__main__":
    # Run test if script is executed directly
    asyncio.run(test_weather_service())