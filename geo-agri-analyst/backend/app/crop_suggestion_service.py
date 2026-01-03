"""
Crop Suggestion Service
Advanced algorithm for recommending most profitable crops based on:
- Soil conditions and land classification
- Climate data (temperature, rainfall, seasonality)
- Historical crop performance
- Market prices and profitability analysis
- Regional suitability and crop rotation
"""

import httpx
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging
import math

logger = logging.getLogger(__name__)


class CropSuggestionService:
    """
    Service for generating profit-optimized crop recommendations
    """
    
    # Currency conversion rate (1 USD = 83 INR as of 2024-2025)
    USD_TO_INR = 83.0
    
    # Comprehensive crop database with profitability metrics
    # All prices in INR (Indian Rupees)
    CROP_DATABASE = {
        # High-value cash crops
        "saffron": {
            "name": "Saffron",
            "category": "Spice",
            "avg_yield_kg_per_hectare": 8,
            "avg_market_price_inr_per_kg": 124500,  # ~$1500
            "investment_cost_inr_per_hectare": 996000,  # ~$12000
            "growing_months": 6,
            "optimal_temp_range": (15, 25),
            "min_rainfall_mm": 150,
            "max_rainfall_mm": 400,
            "soil_types": ["loamy", "sandy loam", "well-drained"],
            "climate_zones": ["temperate", "subtropical"],
            "water_requirement": "low",
            "labor_intensity": "high",
            "risk_level": "high"
        },
        "vanilla": {
            "name": "Vanilla",
            "category": "Spice",
            "avg_yield_kg_per_hectare": 400,
            "avg_market_price_inr_per_kg": 49800,
            "investment_cost_inr_per_hectare": 664000,
            "growing_months": 36,
            "optimal_temp_range": (25, 30),
            "min_rainfall_mm": 2500,
            "max_rainfall_mm": 3500,
            "soil_types": ["rich organic", "loamy", "well-drained"],
            "climate_zones": ["tropical"],
            "water_requirement": "high",
            "labor_intensity": "very high",
            "risk_level": "high"
        },
        
        # Premium vegetables
        "cherry_tomatoes": {
            "name": "Cherry Tomatoes (Premium)",
            "category": "Vegetable",
            "avg_yield_kg_per_hectare": 40000,
            "avg_market_price_inr_per_kg": 249,
            "investment_cost_inr_per_hectare": 415000,
            "growing_months": 4,
            "optimal_temp_range": (18, 24),
            "min_rainfall_mm": 500,
            "max_rainfall_mm": 700,
            "soil_types": ["loamy", "sandy loam", "fertile"],
            "climate_zones": ["temperate"],
            "water_requirement": "medium",
            "labor_intensity": "medium",
            "risk_level": "medium"
        },
        "bell_peppers": {
            "name": "Bell Peppers (Capsicum)",
            "category": "Vegetable",
            "avg_yield_kg_per_hectare": 35000,
            "avg_market_price_inr_per_kg": 207,
            "investment_cost_inr_per_hectare": 373500,
            "growing_months": 5,
            "optimal_temp_range": (20, 30),
            "min_rainfall_mm": 500,
            "max_rainfall_mm": 900,
            "soil_types": ["loamy", "well-drained"],
            "climate_zones": ["temperate", "subtropical", "tropical"],
            "water_requirement": "medium",
            "labor_intensity": "medium",
            "risk_level": "low"
        },
        "broccoli": {
            "name": "Broccoli",
            "category": "Vegetable",
            "avg_yield_kg_per_hectare": 20000,
            "avg_market_price_inr_per_kg": 166,
            "investment_cost_inr_per_hectare": 249000,
            "growing_months": 3,
            "optimal_temp_range": (15, 23),
            "min_rainfall_mm": 400,
            "max_rainfall_mm": 700,
            "soil_types": ["fertile", "loamy", "well-drained"],
            "climate_zones": ["temperate"],
            "water_requirement": "medium",
            "labor_intensity": "medium",
            "risk_level": "low"
        },
        
        # Staple crops - reliable income
        "wheat": {
            "name": "Wheat",
            "category": "Grain",
            "avg_yield_kg_per_hectare": 3000,
            "avg_market_price_inr_per_kg": 24,
            "investment_cost_inr_per_hectare": 41500,
            "growing_months": 6,
            "optimal_temp_range": (12, 25),
            "min_rainfall_mm": 300,
            "max_rainfall_mm": 750,
            "soil_types": ["loamy", "clay loam", "fertile"],
            "climate_zones": ["temperate", "subtropical"],
            "water_requirement": "low",
            "labor_intensity": "low",
            "risk_level": "low"
        },
        "rice": {
            "name": "Rice",
            "category": "Grain",
            "avg_yield_kg_per_hectare": 4500,
            "avg_market_price_inr_per_kg": 37,
            "investment_cost_inr_per_hectare": 66400,
            "growing_months": 5,
            "optimal_temp_range": (20, 35),
            "min_rainfall_mm": 1000,
            "max_rainfall_mm": 2500,
            "soil_types": ["clay", "clay loam", "waterlogged"],
            "climate_zones": ["tropical", "subtropical"],
            "water_requirement": "very high",
            "labor_intensity": "medium",
            "risk_level": "low"
        },
        "corn": {
            "name": "Corn (Maize)",
            "category": "Grain",
            "avg_yield_kg_per_hectare": 5000,
            "avg_market_price_inr_per_kg": 20,
            "investment_cost_inr_per_hectare": 49800,
            "growing_months": 4,
            "optimal_temp_range": (18, 32),
            "min_rainfall_mm": 500,
            "max_rainfall_mm": 1000,
            "soil_types": ["loamy", "fertile", "well-drained"],
            "climate_zones": ["temperate", "subtropical", "tropical"],
            "water_requirement": "medium",
            "labor_intensity": "low",
            "risk_level": "low"
        },
        
        # Oil crops
        "sunflower": {
            "name": "Sunflower",
            "category": "Oilseed",
            "avg_yield_kg_per_hectare": 2000,
            "avg_market_price_inr_per_kg": 49,
            "investment_cost_inr_per_hectare": 33200,
            "growing_months": 4,
            "optimal_temp_range": (20, 30),
            "min_rainfall_mm": 400,
            "max_rainfall_mm": 700,
            "soil_types": ["loamy", "sandy loam", "well-drained"],
            "climate_zones": ["temperate", "subtropical"],
            "water_requirement": "low",
            "labor_intensity": "low",
            "risk_level": "low"
        },
        "canola": {
            "name": "Canola (Rapeseed)",
            "category": "Oilseed",
            "avg_yield_kg_per_hectare": 2500,
            "avg_market_price_inr_per_kg": 45,
            "investment_cost_inr_per_hectare": 37350,
            "growing_months": 6,
            "optimal_temp_range": (10, 20),
            "min_rainfall_mm": 400,
            "max_rainfall_mm": 650,
            "soil_types": ["loamy", "well-drained"],
            "climate_zones": ["temperate"],
            "water_requirement": "medium",
            "labor_intensity": "low",
            "risk_level": "low"
        },
        
        # Fruits - high value
        "strawberries": {
            "name": "Strawberries",
            "category": "Fruit",
            "avg_yield_kg_per_hectare": 25000,
            "avg_market_price_inr_per_kg": 415,
            "investment_cost_inr_per_hectare": 664000,
            "growing_months": 6,
            "optimal_temp_range": (15, 23),
            "min_rainfall_mm": 500,
            "max_rainfall_mm": 750,
            "soil_types": ["sandy loam", "loamy", "well-drained"],
            "climate_zones": ["temperate"],
            "water_requirement": "medium",
            "labor_intensity": "high",
            "risk_level": "medium"
        },
        "blueberries": {
            "name": "Blueberries",
            "category": "Fruit",
            "avg_yield_kg_per_hectare": 8000,
            "avg_market_price_inr_per_kg": 664,
            "investment_cost_inr_per_hectare": 996000,
            "growing_months": 24,
            "optimal_temp_range": (15, 25),
            "min_rainfall_mm": 600,
            "max_rainfall_mm": 1200,
            "soil_types": ["acidic", "sandy loam", "well-drained"],
            "climate_zones": ["temperate"],
            "water_requirement": "medium",
            "labor_intensity": "high",
            "risk_level": "medium"
        },
        
        # Legumes - nitrogen fixing
        "soybeans": {
            "name": "Soybeans",
            "category": "Legume",
            "avg_yield_kg_per_hectare": 2800,
            "avg_market_price_inr_per_kg": 41,
            "investment_cost_inr_per_hectare": 41500,
            "growing_months": 5,
            "optimal_temp_range": (20, 30),
            "min_rainfall_mm": 500,
            "max_rainfall_mm": 900,
            "soil_types": ["loamy", "fertile", "well-drained"],
            "climate_zones": ["temperate", "subtropical"],
            "water_requirement": "medium",
            "labor_intensity": "low",
            "risk_level": "low"
        },
        "chickpeas": {
            "name": "Chickpeas",
            "category": "Legume",
            "avg_yield_kg_per_hectare": 1500,
            "avg_market_price_inr_per_kg": 99,
            "investment_cost_inr_per_hectare": 33200,
            "growing_months": 5,
            "optimal_temp_range": (15, 30),
            "min_rainfall_mm": 300,
            "max_rainfall_mm": 600,
            "soil_types": ["loamy", "clay loam", "well-drained"],
            "climate_zones": ["temperate", "subtropical", "tropical"],
            "water_requirement": "low",
            "labor_intensity": "low",
            "risk_level": "low"
        },
        
        # Cash crops
        "cotton": {
            "name": "Cotton",
            "category": "Fiber",
            "avg_yield_kg_per_hectare": 1800,
            "avg_market_price_inr_per_kg": 149,
            "investment_cost_inr_per_hectare": 83000,
            "growing_months": 6,
            "optimal_temp_range": (21, 35),
            "min_rainfall_mm": 500,
            "max_rainfall_mm": 1000,
            "soil_types": ["loamy", "clay loam", "fertile"],
            "climate_zones": ["subtropical", "tropical"],
            "water_requirement": "medium",
            "labor_intensity": "medium",
            "risk_level": "medium"
        },
        "sugarcane": {
            "name": "Sugarcane",
            "category": "Industrial",
            "avg_yield_kg_per_hectare": 70000,
            "avg_market_price_inr_per_kg": 4,
            "investment_cost_inr_per_hectare": 124500,
            "growing_months": 12,
            "optimal_temp_range": (20, 35),
            "min_rainfall_mm": 1500,
            "max_rainfall_mm": 2500,
            "soil_types": ["loamy", "clay loam", "fertile"],
            "climate_zones": ["tropical", "subtropical"],
            "water_requirement": "high",
            "labor_intensity": "medium",
            "risk_level": "low"
        },
        
        # Herbs - niche high value
        "basil": {
            "name": "Basil (Fresh)",
            "category": "Herb",
            "avg_yield_kg_per_hectare": 12000,
            "avg_market_price_inr_per_kg": 498,
            "investment_cost_inr_per_hectare": 249000,
            "growing_months": 3,
            "optimal_temp_range": (18, 30),
            "min_rainfall_mm": 400,
            "max_rainfall_mm": 600,
            "soil_types": ["loamy", "well-drained", "fertile"],
            "climate_zones": ["temperate", "subtropical", "tropical"],
            "water_requirement": "medium",
            "labor_intensity": "high",
            "risk_level": "medium"
        },
        "lavender": {
            "name": "Lavender",
            "category": "Herb",
            "avg_yield_kg_per_hectare": 3000,
            "avg_market_price_inr_per_kg": 1245,
            "investment_cost_inr_per_hectare": 415000,
            "growing_months": 24,
            "optimal_temp_range": (15, 30),
            "min_rainfall_mm": 300,
            "max_rainfall_mm": 600,
            "soil_types": ["sandy loam", "well-drained", "alkaline"],
            "climate_zones": ["temperate", "subtropical"],
            "water_requirement": "low",
            "labor_intensity": "medium",
            "risk_level": "low"
        }
    }
    
    def __init__(self):
        """Initialize crop suggestion service"""
        self.timeout = 30.0
        logger.info("🌾 Crop Suggestion Service initialized with profit optimization")
    
    def _get_india_agro_climatic_zone(self, lat: float, lng: float) -> str:
        """
        Determine India's agro-climatic zone based on coordinates
        India has 15 major agro-climatic zones defined by ICAR
        """
        # Check if location is in India
        if not (6.0 <= lat <= 37.0 and 68.0 <= lng <= 97.0):
            return "non-india"
        
        # Simplified zone classification based on lat/lng
        # Western Himalayan Region
        if lat > 30 and lng < 78:
            return "western-himalayan"
        
        # Eastern Himalayan Region  
        elif lat > 26 and lng > 88:
            return "eastern-himalayan"
        
        # Lower Gangetic Plains (West Bengal, Bihar)
        elif 22 < lat < 27 and 85 < lng < 90:
            return "lower-gangetic-plains"
        
        # Middle Gangetic Plains (UP, Bihar)
        elif 24 < lat < 28 and 78 < lng < 86:
            return "middle-gangetic-plains"
        
        # Upper Gangetic Plains (Punjab, Haryana, W-UP)
        elif 28 < lat < 32 and 74 < lng < 80:
            return "upper-gangetic-plains"
        
        # Trans-Gangetic Plains (Punjab)
        elif lat > 30 and 74 < lng < 76:
            return "trans-gangetic-plains"
        
        # Eastern Plateau and Hills (Jharkhand, Odisha hills)
        elif 20 < lat < 25 and 84 < lng < 87:
            return "eastern-plateau-hills"
        
        # Central Plateau and Hills (MP, Chhattisgarh)
        elif 20 < lat < 26 and 75 < lng < 84:
            return "central-plateau-hills"
        
        # Western Plateau and Hills (Maharashtra plateau)
        elif 17 < lat < 22 and 73 < lng < 78:
            return "western-plateau-hills"
        
        # Southern Plateau and Hills (Karnataka, Telangana, AP)
        elif 13 < lat < 19 and 75 < lng < 80:
            return "southern-plateau-hills"
        
        # Eastern Coastal Plains (Odisha, AP, TN coast)
        elif lat < 20 and lng > 80:
            return "eastern-coastal-plains"
        
        # Western Coastal Plains and Ghats (Kerala, Karnataka coast, Goa)
        elif lat < 20 and 73 < lng < 77:
            return "western-coastal-plains-ghats"
        
        # Gujarat Plains and Hills
        elif 20 < lat < 24 and 69 < lng < 74:
            return "gujarat-plains-hills"
        
        # Western Dry Region (Rajasthan)
        elif 24 < lat < 30 and 70 < lng < 76:
            return "western-dry-region"
        
        # Island Region (Andaman, Lakshadweep - rare)
        elif lat < 15 and (lng < 74 or lng > 92):
            return "island-region"
        
        else:
            return "general-india"
    
    def _estimate_rainfall_by_zone(self, india_zone: str, lat: float, lng: float) -> float:
        """
        Estimate annual rainfall based on India's agro-climatic zones
        Returns rainfall in mm/year
        """
        zone_rainfall = {
            "western-himalayan": 1200,
            "eastern-himalayan": 2500,
            "lower-gangetic-plains": 1500,
            "middle-gangetic-plains": 1100,
            "upper-gangetic-plains": 800,
            "trans-gangetic-plains": 700,
            "eastern-plateau-hills": 1300,
            "central-plateau-hills": 1200,
            "western-plateau-hills": 800,
            "southern-plateau-hills": 900,
            "eastern-coastal-plains": 1200,
            "western-coastal-plains-ghats": 3000,  # Heavy monsoon
            "gujarat-plains-hills": 800,
            "western-dry-region": 400,  # Rajasthan desert
            "island-region": 3000,
            "general-india": 1000,
            "non-india": 800
        }
        
        return zone_rainfall.get(india_zone, 1000)
    
    async def get_crop_suggestions(
        self,
        lat: float,
        lng: float,
        land_class: str,
        weather_data: Optional[Dict] = None,
        crop_history: Optional[Dict] = None,
        farm_size_hectares: float = 1.0,
        risk_tolerance: str = "medium"  # low, medium, high
    ) -> Dict[str, Any]:
        """
        Generate profit-optimized crop suggestions
        
        Args:
            lat: Latitude
            lng: Longitude
            land_class: Current land classification
            weather_data: Current weather conditions
            crop_history: Historical crop data
            farm_size_hectares: Size of farm in hectares
            risk_tolerance: Farmer's risk tolerance (low/medium/high)
            
        Returns:
            Dict with ranked crop suggestions and profitability analysis
        """
        try:
            logger.info(f"🎯 Generating profit-optimized crop suggestions for ({lat}, {lng})")
            
            # Analyze conditions
            climate_analysis = self._analyze_climate(lat, lng, weather_data, crop_history)
            soil_suitability = self._analyze_soil_from_land_class(land_class)
            
            # Score all crops
            crop_scores = []
            for crop_id, crop_data in self.CROP_DATABASE.items():
                score_result = self._score_crop(
                    crop_data,
                    climate_analysis,
                    soil_suitability,
                    farm_size_hectares,
                    risk_tolerance
                )
                
                if score_result["suitability_score"] > 0.5:  # Raised threshold - only truly suitable crops
                    crop_scores.append({
                        "crop_id": crop_id,
                        "crop_data": crop_data,
                        **score_result
                    })
            
            # Sort by profit potential
            crop_scores.sort(key=lambda x: x["profit_score"], reverse=True)
            
            # Format top recommendations
            top_suggestions = crop_scores[:10]
            
            return {
                "location": {"lat": lat, "lng": lng},
                "farm_size_hectares": farm_size_hectares,
                "climate_zone": climate_analysis["climate_zone"],
                "soil_type": soil_suitability["primary_type"],
                "risk_tolerance": risk_tolerance,
                "top_suggestions": [
                    self._format_suggestion(suggestion, idx + 1)
                    for idx, suggestion in enumerate(top_suggestions)
                ],
                "crop_rotation_plan": self._generate_rotation_plan(top_suggestions[:3]),
                "seasonal_calendar": self._generate_seasonal_calendar(top_suggestions[:5]),
                "market_insights": self._get_market_insights(top_suggestions[:3]),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating crop suggestions: {e}")
            return self._get_fallback_suggestions(lat, lng)
    
    def _analyze_climate(
        self,
        lat: float,
        lng: float,
        weather_data: Optional[Dict],
        crop_history: Optional[Dict]
    ) -> Dict[str, Any]:
        """Analyze climate conditions for crop suitability"""
        
        # Determine climate zone
        abs_lat = abs(lat)
        if abs_lat < 23.5:
            climate_zone = "tropical"
        elif abs_lat < 35:
            climate_zone = "subtropical"
        elif abs_lat < 50:
            climate_zone = "temperate"
        else:
            climate_zone = "cold"
        
        # Determine India-specific agro-climatic zone
        india_zone = self._get_india_agro_climatic_zone(lat, lng)
        
        # Extract temperature from weather data
        avg_temp = 20  # Default fallback
        elevation = 0
        
        if weather_data:
            avg_temp = weather_data.get("avg_temp_c") or weather_data.get("temperature") or weather_data.get("current_temp_c") or 20
            elevation = weather_data.get("elevation_m", 0)
        
        # Extract rainfall - PRIORITIZE ACTUAL WEATHER DATA
        annual_rainfall = None
        
        # First try: Use actual weather API rainfall
        if weather_data and "avg_annual_rainfall_mm" in weather_data:
            annual_rainfall = weather_data.get("avg_annual_rainfall_mm")
            logger.info(f"Using weather API rainfall: {annual_rainfall}mm/year")
        
        # Second try: Use crop history rainfall
        elif crop_history and "ndvi_history" in crop_history:
            rainfall_values = [
                year.get("avg_precipitation_mm", 0) * 12
                for year in crop_history["ndvi_history"]
            ]
            if rainfall_values:
                annual_rainfall = sum(rainfall_values) / len(rainfall_values)
                logger.info(f"Using crop history rainfall: {annual_rainfall}mm/year")
        
        # Fallback: Estimate based on India agro-climatic zone
        if annual_rainfall is None or annual_rainfall < 100:
            annual_rainfall = self._estimate_rainfall_by_zone(india_zone, lat, lng)
            logger.info(f"Using estimated rainfall for zone {india_zone}: {annual_rainfall}mm/year")
        
        return {
            "climate_zone": climate_zone,
            "india_agro_zone": india_zone,
            "avg_temperature": avg_temp,
            "annual_rainfall_mm": annual_rainfall,
            "elevation_m": elevation,
            "hemisphere": "northern" if lat >= 0 else "southern",
            "growing_season_length": self._estimate_growing_season(climate_zone, abs_lat)
        }
    
    def _estimate_growing_season(self, climate_zone: str, abs_lat: float) -> int:
        """Estimate growing season length in months"""
        if climate_zone == "tropical":
            return 12  # Year-round
        elif climate_zone == "subtropical":
            return 10
        elif abs_lat < 45:
            return 6
        else:
            return 4
    
    def _analyze_soil_from_land_class(self, land_class: str) -> Dict[str, Any]:
        """Infer soil characteristics from land classification"""
        
        land_class_lower = land_class.lower()
        
        # Soil type mapping based on land class
        if "forest" in land_class_lower or "tree" in land_class_lower:
            primary_type = "loamy"
            fertility = "high"
            drainage = "good"
        elif "crop" in land_class_lower or "cultivated" in land_class_lower:
            primary_type = "fertile"
            fertility = "high"
            drainage = "good"
        elif "grass" in land_class_lower or "shrub" in land_class_lower:
            primary_type = "sandy loam"
            fertility = "medium"
            drainage = "good"
        elif "water" in land_class_lower or "flooded" in land_class_lower:
            primary_type = "clay"
            fertility = "medium"
            drainage = "poor"
        elif "urban" in land_class_lower or "built" in land_class_lower:
            primary_type = "disturbed"
            fertility = "low"
            drainage = "variable"
        elif "bare" in land_class_lower or "sparse" in land_class_lower:
            primary_type = "sandy"
            fertility = "low"
            drainage = "excellent"
        else:
            primary_type = "loamy"
            fertility = "medium"
            drainage = "good"
        
        return {
            "primary_type": primary_type,
            "fertility": fertility,
            "drainage": drainage,
            "suitable_for": self._get_suitable_soil_types(primary_type)
        }
    
    def _get_suitable_soil_types(self, primary_type: str) -> List[str]:
        """Get list of compatible soil types"""
        soil_families = {
            "loamy": ["loamy", "fertile", "well-drained", "sandy loam", "clay loam"],
            "sandy": ["sandy", "sandy loam", "well-drained"],
            "clay": ["clay", "clay loam", "waterlogged"],
            "fertile": ["fertile", "loamy", "rich organic", "well-drained"]
        }
        
        for family, types in soil_families.items():
            if primary_type in types:
                return types
        
        return [primary_type, "loamy", "well-drained"]
    
    def _score_crop(
        self,
        crop_data: Dict,
        climate: Dict,
        soil: Dict,
        farm_size: float,
        risk_tolerance: str
    ) -> Dict[str, Any]:
        """
        Score a crop based on multiple factors
        Returns suitability score, profit potential, and detailed breakdown
        """
        
        # 1. Climate suitability (0-1)
        climate_score = self._calculate_climate_score(crop_data, climate)
        
        # 2. Soil suitability (0-1)
        soil_score = self._calculate_soil_score(crop_data, soil)
        
        # 3. Profitability calculation
        gross_revenue = crop_data["avg_yield_kg_per_hectare"] * crop_data["avg_market_price_inr_per_kg"]
        net_profit = gross_revenue - crop_data["investment_cost_inr_per_hectare"]
        roi_percentage = (net_profit / crop_data["investment_cost_inr_per_hectare"]) * 100
        
        # 4. Risk adjustment
        risk_score = self._calculate_risk_score(crop_data, risk_tolerance)
        
        # 5. Combined suitability score - Climate matters MORE now
        # Increased climate weight from 0.35 to 0.50 (climate is CRITICAL)
        suitability_score = (climate_score * 0.50 + soil_score * 0.30 + risk_score * 0.20)
        
        # 6. AGGRESSIVE unsuitable crop filtering
        # If climate is wrong, crop simply won't grow well - REJECT IT
        if climate_score < 0.5:
            # Climate unsuitable - drastically reduce score
            suitability_score *= 0.1  # 90% penalty
        elif climate_score < 0.7:
            # Marginally suitable - still penalize heavily
            suitability_score *= 0.3  # 70% penalty
        
        # 7. Profit score (normalized) - Apply suitability multiplier first
        if suitability_score < 0.5:
            # Don't even bother showing low-suitability crops regardless of profit
            profit_score = suitability_score * 0.01  # Near-zero profit score
        else:
            profit_score = suitability_score * (roi_percentage / 100) * (net_profit / 10000)
        
        # 8. Scale to farm size
        total_investment = crop_data["investment_cost_inr_per_hectare"] * farm_size
        total_profit = net_profit * farm_size
        
        return {
            "suitability_score": round(suitability_score, 2),
            "profit_score": round(profit_score, 2),
            "climate_score": round(climate_score, 2),
            "soil_score": round(soil_score, 2),
            "risk_score": round(risk_score, 2),
            "gross_revenue_per_hectare": round(gross_revenue, 2),
            "net_profit_per_hectare": round(net_profit, 2),
            "roi_percentage": round(roi_percentage, 1),
            "total_investment_inr": round(total_investment, 2),
            "total_profit_inr": round(total_profit, 2),
            "payback_months": crop_data["growing_months"]
        }
    
    def _calculate_climate_score(self, crop_data: Dict, climate: Dict) -> float:
        """Calculate how well climate matches crop requirements - STRICT SCORING"""
        
        score = 0.0
        
        # Temperature match - STRICT
        temp = climate["avg_temperature"]
        min_temp, max_temp = crop_data["optimal_temp_range"]
        
        if min_temp <= temp <= max_temp:
            # Perfect match
            score += 0.4
        else:
            # Heavy penalty for being outside range
            if temp < min_temp:
                deviation = min_temp - temp
            else:
                deviation = temp - max_temp
            
            # Penalize heavily: -0.05 per degree deviation
            penalty = deviation * 0.05
            score += max(0, 0.4 - penalty)
        
        # Rainfall match - STRICT
        rainfall = climate["annual_rainfall_mm"]
        
        if crop_data["min_rainfall_mm"] <= rainfall <= crop_data["max_rainfall_mm"]:
            # Perfect match
            score += 0.4
        else:
            # Heavy penalty for being outside range
            if rainfall < crop_data["min_rainfall_mm"]:
                deficit = crop_data["min_rainfall_mm"] - rainfall
                # Severe penalty for drought-prone areas
                penalty = deficit / 200  # -0.005 per mm deficit
            else:
                excess = rainfall - crop_data["max_rainfall_mm"]
                # Moderate penalty for excess rain
                penalty = excess / 500  # -0.002 per mm excess
            
            score += max(0, 0.4 - penalty)
        
        # Climate zone match - MANDATORY for some crops
        if climate["climate_zone"] in crop_data["climate_zones"]:
            score += 0.2
        else:
            # SEVERE penalty for wrong climate zone
            score *= 0.3  # Reduce total score by 70%
        
        # Elevation penalty for high-altitude sensitive crops
        elevation = climate.get("elevation_m", 0)
        if elevation > 1000 and crop_data.get("name") in ["Vanilla", "Rice", "Banana", "Coconut"]:
            # These crops don't grow well at high altitude
            score *= 0.2
        elif elevation > 1500 and "tropical" in crop_data.get("climate_zones", []):
            # Most tropical crops struggle above 1500m
            score *= 0.4
        
        return min(1.0, max(0.0, score))
    
    def _calculate_soil_score(self, crop_data: Dict, soil: Dict) -> float:
        """Calculate soil suitability score"""
        
        score = 0.0
        
        # Check if soil type matches
        suitable_soils = soil["suitable_for"]
        crop_soil_requirements = crop_data["soil_types"]
        
        # Find overlap
        overlap = set(suitable_soils) & set(crop_soil_requirements)
        if overlap:
            score += 0.6
        else:
            # Partial match for similar types
            if any("loam" in s for s in suitable_soils) and any("loam" in s for s in crop_soil_requirements):
                score += 0.3
        
        # Fertility bonus
        if soil["fertility"] == "high":
            score += 0.3
        elif soil["fertility"] == "medium":
            score += 0.2
        
        # Drainage match
        if "well-drained" in crop_soil_requirements:
            if soil["drainage"] in ["good", "excellent"]:
                score += 0.1
        elif "waterlogged" in crop_soil_requirements:
            if soil["drainage"] == "poor":
                score += 0.1
        
        return min(1.0, score)
    
    def _calculate_risk_score(self, crop_data: Dict, risk_tolerance: str) -> float:
        """Calculate risk-adjusted score"""
        
        risk_level = crop_data["risk_level"]
        
        risk_matrix = {
            "low": {"low": 1.0, "medium": 0.9, "high": 0.6, "very high": 0.4},
            "medium": {"low": 0.9, "medium": 1.0, "high": 0.8, "very high": 0.6},
            "high": {"low": 0.7, "medium": 0.9, "high": 1.0, "very high": 0.9}
        }
        
        return risk_matrix.get(risk_tolerance, {}).get(risk_level, 0.7)
    
    def _format_suggestion(self, suggestion: Dict, rank: int) -> Dict[str, Any]:
        """Format crop suggestion for output"""
        
        crop = suggestion["crop_data"]
        
        return {
            "rank": rank,
            "crop_name": crop["name"],
            "category": crop["category"],
            "suitability_percentage": round(suggestion["suitability_score"] * 100, 1),
            "expected_profit_per_hectare_inr": suggestion["net_profit_per_hectare"],
            "roi_percentage": suggestion["roi_percentage"],
            "investment_required_inr": crop["investment_cost_inr_per_hectare"],
            "growing_period_months": crop["growing_months"],
            "harvest_cycles_per_year": max(1, 12 // crop["growing_months"]),
            "annual_profit_potential_inr": round(
                suggestion["net_profit_per_hectare"] * max(1, 12 // crop["growing_months"]),
                2
            ),
            "water_requirement": crop["water_requirement"],
            "labor_intensity": crop["labor_intensity"],
            "risk_level": crop["risk_level"],
            "key_advantages": self._get_crop_advantages(crop, suggestion),
            "success_tips": self._get_success_tips(crop)
        }
    
    def _get_crop_advantages(self, crop: Dict, scores: Dict) -> List[str]:
        """Generate key advantages for the crop"""
        advantages = []
        
        if scores["roi_percentage"] > 100:
            advantages.append(f"Excellent ROI of {scores['roi_percentage']:.0f}%")
        
        if crop["growing_months"] <= 4:
            advantages.append("Quick harvest - fast returns")
        
        if crop["risk_level"] == "low":
            advantages.append("Low risk - reliable income")
        
        if crop["water_requirement"] == "low":
            advantages.append("Water-efficient - lower costs")
        
        if crop["labor_intensity"] == "low":
            advantages.append("Low labor requirements")
        
        if scores["climate_score"] >= 0.8:
            advantages.append("Highly suitable for local climate")
        
        if crop["avg_market_price_inr_per_kg"] > 2:
            advantages.append("High market value")
        
        return advantages[:4]  # Top 4 advantages
    
    def _get_success_tips(self, crop: Dict) -> List[str]:
        """Generate success tips for growing the crop"""
        tips = []
        
        if crop["water_requirement"] == "high":
            tips.append("Ensure consistent irrigation system")
        
        if crop["labor_intensity"] in ["high", "very high"]:
            tips.append("Plan for adequate labor during harvest")
        
        if crop["risk_level"] in ["high", "very high"]:
            tips.append("Consider crop insurance")
        
        if "well-drained" in crop["soil_types"]:
            tips.append("Ensure proper drainage to prevent waterlogging")
        
        tips.append(f"Optimal temperature: {crop['optimal_temp_range'][0]}-{crop['optimal_temp_range'][1]}°C")
        
        return tips[:3]
    
    def _generate_rotation_plan(self, top_crops: List[Dict]) -> Dict[str, Any]:
        """Generate crop rotation plan for soil health and profit"""
        
        if len(top_crops) < 2:
            return {"recommendation": "Insufficient data for rotation plan"}
        
        # Prioritize diversity and soil health
        rotation_sequence = []
        
        for crop in top_crops:
            crop_data = crop["crop_data"]
            rotation_sequence.append({
                "crop": crop_data["name"],
                "duration_months": crop_data["growing_months"],
                "category": crop_data["category"],
                "benefit": self._get_rotation_benefit(crop_data["category"])
            })
        
        return {
            "rotation_type": "Sequential high-profit rotation",
            "sequence": rotation_sequence,
            "total_cycle_months": sum(c["duration_months"] for c in rotation_sequence),
            "benefits": [
                "Maximizes annual profit",
                "Reduces pest and disease pressure",
                "Maintains soil fertility",
                "Diversifies income streams"
            ]
        }
    
    def _get_rotation_benefit(self, category: str) -> str:
        """Get benefit of crop category in rotation"""
        benefits = {
            "Legume": "Fixes nitrogen in soil",
            "Grain": "Builds organic matter",
            "Vegetable": "Quick cash crop",
            "Oilseed": "Deep root system improves soil",
            "Herb": "Pest control through diversity",
            "Fruit": "Long-term investment"
        }
        return benefits.get(category, "Diversifies production")
    
    def _generate_seasonal_calendar(self, top_crops: List[Dict]) -> Dict[str, List[str]]:
        """Generate seasonal planting calendar"""
        
        calendar = {
            "Spring (Mar-May)": [],
            "Summer (Jun-Aug)": [],
            "Fall (Sep-Nov)": [],
            "Winter (Dec-Feb)": []
        }
        
        for crop in top_crops:
            crop_data = crop["crop_data"]
            temp_range = crop_data["optimal_temp_range"]
            
            # Assign to seasons based on temperature
            if 15 <= temp_range[0] <= 25:
                calendar["Spring (Mar-May)"].append(crop_data["name"])
                calendar["Fall (Sep-Nov)"].append(crop_data["name"])
            if 20 <= temp_range[1] <= 35:
                calendar["Summer (Jun-Aug)"].append(crop_data["name"])
            if temp_range[0] < 15:
                calendar["Winter (Dec-Feb)"].append(crop_data["name"])
        
        return calendar
    
    def _get_market_insights(self, top_crops: List[Dict]) -> Dict[str, Any]:
        """Generate market insights for top crops"""
        
        total_investment = sum(c["total_investment_inr"] for c in top_crops)
        total_profit = sum(c["total_profit_inr"] for c in top_crops)
        avg_roi = sum(c["roi_percentage"] for c in top_crops) / len(top_crops) if top_crops else 0
        
        return {
            "portfolio_strategy": "Diversified high-profit approach",
            "total_investment_required_inr": round(total_investment, 2),
            "expected_annual_profit_inr": round(total_profit, 2),
            "portfolio_roi_percentage": round(avg_roi, 1),
            "market_trends": [
                "Premium vegetables show strong demand",
                "Organic certification can increase prices by 20-30%",
                "Direct-to-consumer sales improve margins"
            ],
            "risk_mitigation": [
                "Diversify across multiple crops",
                "Stagger planting dates",
                "Build relationships with multiple buyers",
                "Consider value-added processing"
            ]
        }
    
    def _get_fallback_suggestions(self, lat: float, lng: float) -> Dict[str, Any]:
        """Provide basic suggestions when detailed analysis fails"""
        
        abs_lat = abs(lat)
        if abs_lat < 23.5:
            climate = "tropical"
            suggestions = ["rice", "sugarcane", "corn"]
        elif abs_lat < 35:
            climate = "subtropical"
            suggestions = ["corn", "soybeans", "cotton"]
        else:
            climate = "temperate"
            suggestions = ["wheat", "sunflower", "canola"]
        
        return {
            "location": {"lat": lat, "lng": lng},
            "climate_zone": climate,
            "top_suggestions": [
                {
                    "rank": i + 1,
                    "crop_name": self.CROP_DATABASE[crop]["name"],
                    "category": self.CROP_DATABASE[crop]["category"],
                    "note": "Basic recommendation - detailed analysis unavailable"
                }
                for i, crop in enumerate(suggestions)
            ],
            "status": "fallback_mode"
        }


# Global service instance
_crop_suggestion_service = None


def get_crop_suggestion_service() -> CropSuggestionService:
    """Get or create the global crop suggestion service instance"""
    global _crop_suggestion_service
    if _crop_suggestion_service is None:
        _crop_suggestion_service = CropSuggestionService()
    return _crop_suggestion_service
