import json
import time
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from statistics import median
from typing import List

import pytest
import requests

API_URL = "http://localhost:8000"

@pytest.fixture(scope="session", autouse=True)
def api_server():
    """Start API server for testing session."""
    # Start the server in background
    process = subprocess.Popen(
        ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    for _ in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get(f"{API_URL}/health", timeout=1)
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        process.kill()
        raise Exception("API server failed to start")
    
    yield
    
    # Cleanup
    process.terminate()
    process.wait()

# Valid test data for API requests
VALID_REQUEST = {
    "id": 1001,
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 2000,
    "sqft_lot": 8000,
    "floors": 2.0,
    "waterfront": 0,
    "view": 0,
    "condition": 4,
    "grade": 8,
    "sqft_above": 1800,
    "sqft_basement": 200,
    "yr_built": 1990,
    "yr_renovated": 0,
    "zipcode": "98103",
    "lat": 47.6694,
    "long": -122.346,
    "sqft_living15": 1850,
    "sqft_lot15": 7500
}


# Input Validation Tests

def test_valid_prediction_request():
    """Test valid request returns proper response."""
    response = requests.post(f"{API_URL}/predict", json=VALID_REQUEST)
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate response schema
    assert "predicted_price" in data
    assert "model_version" in data
    assert "timestamp" in data
    assert "features_used" in data
    
    # Validate data types and ranges
    assert isinstance(data["predicted_price"], (int, float))
    assert data["predicted_price"] > 0
    assert isinstance(data["model_version"], str)
    assert isinstance(data["timestamp"], str)


def test_missing_required_fields():
    """Test missing required fields return 422."""
    incomplete_request = VALID_REQUEST.copy()
    del incomplete_request["bedrooms"]
    
    response = requests.post(f"{API_URL}/predict", json=incomplete_request)
    assert response.status_code == 422


def test_invalid_field_types():
    """Test invalid data types return 422."""
    invalid_request = VALID_REQUEST.copy()
    invalid_request["bedrooms"] = "three"  # Should be int
    
    response = requests.post(f"{API_URL}/predict", json=invalid_request)
    assert response.status_code == 422


def test_out_of_range_values():
    """Test out-of-range values return 422."""
    # Test negative bedrooms
    invalid_request = VALID_REQUEST.copy()
    invalid_request["bedrooms"] = -1
    
    response = requests.post(f"{API_URL}/predict", json=invalid_request)
    assert response.status_code == 422
    
    # Test excessive sqft_living
    invalid_request = VALID_REQUEST.copy()
    invalid_request["sqft_living"] = 100000  # Too large
    
    response = requests.post(f"{API_URL}/predict", json=invalid_request)
    assert response.status_code == 422


def test_invalid_zipcode_format():
    """Test invalid zipcode format returns 422."""
    invalid_request = VALID_REQUEST.copy()
    invalid_request["zipcode"] = "1234"  # Too short
    
    response = requests.post(f"{API_URL}/predict", json=invalid_request)
    assert response.status_code == 422


def test_malformed_json():
    """Test malformed JSON returns 422."""
    response = requests.post(
        f"{API_URL}/predict",
        data="{'invalid': json}",
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


# API Functionality Tests

def test_health_endpoint():
    """Test health check endpoint."""
    response = requests.get(f"{API_URL}/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_unknown_zipcode_handling():
    """Test how API handles unknown zipcode."""
    unknown_request = VALID_REQUEST.copy()
    unknown_request["zipcode"] = "00000"  # Unknown zipcode
    
    response = requests.post(f"{API_URL}/predict", json=unknown_request)
    # Should still return 200 but handle gracefully
    assert response.status_code in [200, 500]  # Depends on implementation


def test_prediction_reasonableness():
    """Test prediction values are reasonable for Seattle area."""
    response = requests.post(f"{API_URL}/predict", json=VALID_REQUEST)
    
    assert response.status_code == 200
    data = response.json()
    predicted_price = data["predicted_price"]
    
    # Seattle area house prices should be in reasonable range
    assert 100_000 < predicted_price < 5_000_000


# Performance Tests

def test_single_request_latency():
    """Test single request latency is reasonable."""
    start_time = time.time()
    response = requests.post(f"{API_URL}/predict", json=VALID_REQUEST)
    end_time = time.time()
    
    assert response.status_code == 200
    latency_ms = (end_time - start_time) * 1000
    
    # Should respond within 100ms for single request
    assert latency_ms < 100


def test_concurrent_requests():
    """Test concurrent request handling and calculate P95 latency."""
    def make_request():
        start_time = time.time()
        response = requests.post(f"{API_URL}/predict", json=VALID_REQUEST)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        return response.status_code, latency_ms

    # Test with 20 concurrent requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(20)]
        results = [future.result() for future in futures]

    status_codes, latencies = zip(*results)
    
    # All requests should succeed
    assert all(status == 200 for status in status_codes)
    
    # Calculate performance metrics
    sorted_latencies = sorted(latencies)
    p95_latency = sorted_latencies[int(0.95 * len(sorted_latencies))]
    median_latency = median(latencies)
    
    # Performance assertions
    assert median_latency < 200  # Median should be under 200ms
    assert p95_latency < 500     # P95 should be under 500ms
    
    print(f"Concurrent test results:")
    print(f"  Median latency: {median_latency:.1f}ms")
    print(f"  P95 latency: {p95_latency:.1f}ms")
    print(f"  Max latency: {max(latencies):.1f}ms")


def test_sequential_requests_performance():
    """Test multiple sequential requests for consistency."""
    latencies = []
    
    for i in range(10):
        start_time = time.time()
        response = requests.post(f"{API_URL}/predict", json=VALID_REQUEST)
        end_time = time.time()
        
        assert response.status_code == 200
        latencies.append((end_time - start_time) * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    
    # Performance should be consistent
    assert avg_latency < 100  # Average under 100ms
    assert max_latency < 200  # No request over 200ms