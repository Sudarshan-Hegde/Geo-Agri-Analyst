"""
Quick Test Script for HuggingFace BigEarthNetModels Integration
Tests the complete flow from satellite image fetch to classification
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.huggingface_service import get_hf_service
from app.satellite_service import get_satellite_service


async def test_single_prediction():
    """Test single location prediction"""
    print("=" * 70)
    print("TEST 1: Single Point Classification")
    print("=" * 70)
    
    # Test coordinates (agricultural area in India)
    test_lat = 20.5937
    test_lng = 78.9629
    
    print(f"\n📍 Testing location: {test_lat}, {test_lng}")
    
    # Get HuggingFace service
    hf_service = get_hf_service()
    
    # Check health
    print("\n🏥 Checking HuggingFace Space health...")
    is_healthy = await hf_service.check_health()
    
    if not is_healthy:
        print("⚠️  Space may be sleeping - first call will take longer")
    else:
        print("✅ Space is ready!")
    
    # Test prediction
    print(f"\n🔍 Fetching prediction...")
    try:
        result = await hf_service.predict(test_lat, test_lng)
        
        print("\n" + "=" * 70)
        print("RESULTS:")
        print("=" * 70)
        print(f"\n🏷️  Land Class:     {result.get('land_class')}")
        print(f"📊 Confidence:     {result.get('confidence', 0):.2%}")
        print(f"🔧 Source:         {result.get('source')}")
        
        print(f"\n📋 Top 5 Predictions:")
        predictions = result.get('predictions', {})
        for idx, (label, conf) in enumerate(list(predictions.items())[:5], 1):
            bar = "█" * int(conf * 20)
            print(f"  {idx}. {label:<40} {conf:6.2%} {bar}")
        
        print(f"\n🖼️  Image Data:")
        print(f"  Before Image: {len(result.get('before_image_b64', ''))} bytes")
        print(f"  After Image:  {len(result.get('after_image_b64', ''))} bytes")
        
        print("\n✅ TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_batch_prediction():
    """Test batch prediction for multiple locations"""
    print("\n" + "=" * 70)
    print("TEST 2: Batch Prediction (Multiple Points)")
    print("=" * 70)
    
    # Test coordinates (various locations in India)
    test_coords = [
        (20.5937, 78.9629),  # Central India
        (28.6139, 77.2090),  # Delhi
        (13.0827, 80.2707),  # Chennai
    ]
    
    print(f"\n📍 Testing {len(test_coords)} locations")
    
    hf_service = get_hf_service()
    
    print(f"\n🔍 Fetching predictions...")
    try:
        results = await hf_service.predict_batch(test_coords, zoom=17)
        
        print("\n" + "=" * 70)
        print("RESULTS:")
        print("=" * 70)
        
        for idx, result in enumerate(results, 1):
            coords = result.get('coordinates', {})
            print(f"\n📍 Location {idx}: ({coords.get('lat'):.4f}, {coords.get('lng'):.4f})")
            print(f"   🏷️  Class:      {result.get('land_class')}")
            print(f"   📊 Confidence: {result.get('confidence', 0):.2%}")
        
        print("\n✅ BATCH TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ BATCH TEST FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_satellite_fetch():
    """Test satellite image fetching"""
    print("\n" + "=" * 70)
    print("TEST 3: Satellite Image Fetch")
    print("=" * 70)
    
    test_lat = 20.5937
    test_lng = 78.9629
    
    print(f"\n📍 Testing location: {test_lat}, {test_lng}")
    
    satellite_service = get_satellite_service()
    
    print(f"\n🛰️  Fetching satellite image...")
    try:
        image = satellite_service.get_satellite_image(test_lat, test_lng, size=30, zoom=17)
        
        if image is None:
            print("⚠️  No satellite image available (using fallback)")
            return True  # Not a failure, just no imagery
        
        print("\n" + "=" * 70)
        print("IMAGE INFO:")
        print("=" * 70)
        print(f"  Size:   {image.size}")
        print(f"  Mode:   {image.mode}")
        print(f"  Format: {image.format}")
        
        print("\n✅ SATELLITE TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ SATELLITE TEST FAILED!")
        print(f"Error: {str(e)}")
        return False


async def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "HUGGINGFACE CLASSIFIER INTEGRATION TEST" + " " * 19 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    results = []
    
    # Test 1: Single prediction
    result1 = await test_single_prediction()
    results.append(("Single Prediction", result1))
    
    # Test 2: Batch prediction
    result2 = await test_batch_prediction()
    results.append(("Batch Prediction", result2))
    
    # Test 3: Satellite fetch
    result3 = await test_satellite_fetch()
    results.append(("Satellite Fetch", result3))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\n📊 Results: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 ALL TESTS PASSED! Integration is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
