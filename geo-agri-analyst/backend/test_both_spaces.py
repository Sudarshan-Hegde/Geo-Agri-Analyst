"""Quick test to verify the correct HuggingFace Space"""
from gradio_client import Client
import sys

# Test both possible Space names
spaces_to_test = [
    "HegdeSudarshan/Classifier",
    "HegdeSudarshan/BigEarthNetModels",
]

print("Testing HuggingFace Space connections...")
print("=" * 70)

for space_name in spaces_to_test:
    print(f"\n📡 Testing: {space_name}")
    print("-" * 70)
    
    try:
        client = Client(space_name)
        print(f"✅ Successfully connected to {space_name}!")
        print(f"\n📋 Available API endpoints:")
        api_info = client.view_api()
        print(api_info)
        print("\n" + "=" * 70)
        print(f"✅ WORKING SPACE: {space_name}")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Connection failed for {space_name}")
        print(f"   Error: {type(e).__name__}: {str(e)[:100]}")
        print()

print("\nTest complete!")
