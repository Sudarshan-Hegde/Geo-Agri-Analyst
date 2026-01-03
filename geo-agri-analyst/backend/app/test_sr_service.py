"""
Test script for SR-Model integration
Tests the super resolution service directly
"""

import asyncio
from sr_service import get_sr_service
from PIL import Image
import io
import base64


async def test_sr_service():
    """Test the SR-Model service"""
    
    print("=" * 60)
    print("Testing SR-Model Service")
    print("=" * 60)
    
    # Get service instance
    sr_service = get_sr_service()
    
    # Test 1: Health check
    print("\n1. Health Check...")
    is_healthy = await sr_service.check_health()
    print(f"   SR-Model Status: {'✅ Available' if is_healthy else '❌ Unavailable'}")
    
    if not is_healthy:
        print("\n⚠️ SR-Model service is not available. Cannot proceed with tests.")
        return
    
    # Test 2: Create a simple test image
    print("\n2. Creating test image (30x30 RGB)...")
    test_image = Image.new('RGB', (30, 30), color=(100, 150, 200))
    
    # Add some pattern to make it more interesting
    pixels = test_image.load()
    for i in range(0, 30, 3):
        for j in range(0, 30, 3):
            if (i + j) % 6 == 0:
                pixels[i, j] = (50, 200, 50)  # Green dots
    
    print("   Test image created")
    
    # Test 3: Enhance the image
    print("\n3. Enhancing image with SR-Model...")
    enhanced_image = await sr_service.enhance_image(test_image)
    
    if enhanced_image:
        print(f"   ✅ Enhancement successful!")
        print(f"   Original size: {test_image.size}")
        print(f"   Enhanced size: {enhanced_image.size}")
        
        # Test 4: Convert to base64
        print("\n4. Converting to base64...")
        enhanced_b64 = await sr_service.enhance_image_to_base64(test_image)
        
        if enhanced_b64:
            print(f"   ✅ Base64 conversion successful")
            print(f"   Base64 length: {len(enhanced_b64)} characters")
        else:
            print("   ❌ Base64 conversion failed")
    else:
        print("   ❌ Enhancement failed")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_sr_service())
