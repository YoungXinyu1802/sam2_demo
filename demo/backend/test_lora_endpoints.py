#!/usr/bin/env python3
"""
Test script to verify LoRA endpoints are accessible
"""
import requests
import json

# Test configuration
BASE_URL = "http://localhost:5000"

def test_endpoint_exists(endpoint, method="POST"):
    """Test if an endpoint exists and accepts the specified method"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\nTesting {method} {url}...")
    
    try:
        if method == "POST":
            # Send a minimal request to see if endpoint exists
            response = requests.post(url, json={}, timeout=5)
        elif method == "GET":
            response = requests.get(url, timeout=5)
        
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 405:
            print(f"  ❌ Method {method} not allowed")
            return False
        elif response.status_code == 404:
            print(f"  ❌ Endpoint not found")
            return False
        else:
            print(f"  ✓ Endpoint exists (status: {response.status_code})")
            return True
            
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Connection failed - is the server running?")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Testing SAM2 Demo LoRA Endpoints")
    print("=" * 60)
    
    # Test existing endpoints
    print("\n[Existing Endpoints]")
    test_endpoint_exists("/healthy", "GET")
    test_endpoint_exists("/propagate_in_video", "POST")
    test_endpoint_exists("/propagate_to_frame", "POST")
    
    # Test new LoRA endpoints
    print("\n[New LoRA Endpoints]")
    train_exists = test_endpoint_exists("/train_lora", "POST")
    generate_exists = test_endpoint_exists("/generate_lora_candidates", "POST")
    
    print("\n" + "=" * 60)
    if train_exists and generate_exists:
        print("✓ All LoRA endpoints are accessible!")
    else:
        print("❌ Some LoRA endpoints are not accessible")
        print("\nTo fix this issue:")
        print("1. Make sure the backend server is running")
        print("2. Restart the backend server to load the new routes:")
        print("   cd demo/backend")
        print("   # Stop the current server (Ctrl+C)")
        print("   # Then restart it:")
        print("   python server/app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()

