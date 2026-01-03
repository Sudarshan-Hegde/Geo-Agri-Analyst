"""
Test improved crop suggestion system
Tests different locations in India with varying climates
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.crop_suggestion_service import get_crop_suggestion_service
from app.weather_service import get_climate_data


async def test_location(name, lat, lng, expected_zone):
    """Test crop suggestions for a specific location"""
    print("\n" + "=" * 80)
    print(f"Testing: {name}")
    print(f"Coordinates: {lat}, {lng}")
    print("=" * 80)
    
    # Get weather data
    print("\n📡 Fetching weather data...")
    weather = await get_climate_data(lat, lng)
    
    if weather:
        print(f"✅ Temperature: {weather.get('avg_temp_c')}°C")
        print(f"✅ Rainfall: {weather.get('avg_annual_rainfall_mm')}mm/year")
        print(f"✅ Elevation: {weather.get('elevation_m')}m")
    
    # Get crop suggestions
    print("\n🌾 Generating crop suggestions...")
    crop_service = get_crop_suggestion_service()
    
    suggestions = await crop_service.get_crop_suggestions(
        lat=lat,
        lng=lng,
        land_class="Agricultural land",
        weather_data=weather,
        farm_size_hectares=1.0,
        risk_tolerance="medium"
    )
    
    if suggestions and "top_suggestions" in suggestions:
        print(f"\n📊 India Agro-Zone: {suggestions.get('climate_zone')} - {suggestions.get('top_suggestions')[0].get('india_zone') if suggestions.get('top_suggestions') else 'N/A'}")
        print(f"\n🏆 Top 5 Crop Recommendations:")
        print("-" * 80)
        
        for idx, crop in enumerate(suggestions["top_suggestions"][:5], 1):
            print(f"\n{idx}. {crop.get('crop_name', 'Unknown')}")
            print(f"   Suitability: {crop.get('suitability_percentage', 0)}%")
            print(f"   Expected Profit: ₹{crop.get('expected_profit_per_hectare_inr', 0):,.0f}/hectare")
            print(f"   ROI: {crop.get('roi_percentage', 0):.1f}%")
            print(f"   Growing Period: {crop.get('growing_period_months', 0)} months")
    else:
        print("❌ No suitable crops found!")
    
    return suggestions


async def main():
    """Run tests for different Indian locations"""
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "CROP SUGGESTION SYSTEM TEST - INDIA LOCATIONS" + " " * 16 + "║")
    print("╚" + "=" * 78 + "╝")
    
    test_locations = [
        ("Punjab (Upper Gangetic Plains)", 30.7333, 76.7794, "upper-gangetic-plains"),
        ("Kerala (Western Coast - High Rain)", 10.8505, 76.2711, "western-coastal-plains-ghats"),
        ("Rajasthan (Western Dry Region)", 26.9124, 75.7873, "western-dry-region"),
        ("Maharashtra (Western Plateau)", 19.0760, 72.8777, "western-plateau-hills"),
        ("Tamil Nadu (Eastern Coast)", 13.0827, 80.2707, "eastern-coastal-plains"),
    ]
    
    results = []
    for name, lat, lng, expected_zone in test_locations:
        result = await test_location(name, lat, lng, expected_zone)
        results.append((name, result))
        await asyncio.sleep(1)  # Rate limit
    
    # Summary
    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for name, result in results:
        if result and "top_suggestions" in result:
            top_crops = [c['crop_name'] for c in result['top_suggestions'][:3]]
            print(f"\n{name}:")
            print(f"  Top 3: {', '.join(top_crops)}")
        else:
            print(f"\n{name}: No suitable crops")
    
    print("\n✅ Test complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
