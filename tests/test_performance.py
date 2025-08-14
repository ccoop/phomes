import time
import json
import statistics
from typing import List
import requests

API_URL = "http://localhost:8000"

# Sample test data for predictions (based on the actual features from the dataset)
test_data = [
    {
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
    },
    {
        "id": 1002,
        "bedrooms": 4,
        "bathrooms": 3.0,
        "sqft_living": 2500,
        "sqft_lot": 10000,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 4,
        "grade": 9,
        "sqft_above": 2500,
        "sqft_basement": 0,
        "yr_built": 2005,
        "yr_renovated": 0,
        "zipcode": "98115",
        "lat": 47.6974,
        "long": -122.313,
        "sqft_living15": 2200,
        "sqft_lot15": 9500
    },
    {
        "id": 1003,
        "bedrooms": 2,
        "bathrooms": 1.0,
        "sqft_living": 1200,
        "sqft_lot": 5000,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 6,
        "sqft_above": 1200,
        "sqft_basement": 0,
        "yr_built": 1955,
        "yr_renovated": 0,
        "zipcode": "98107",
        "lat": 47.6674,
        "long": -122.386,
        "sqft_living15": 1100,
        "sqft_lot15": 4800
    }
]

def test_single_prediction(data: dict) -> float:
    """Test a single prediction and return the time taken."""
    start_time = time.time()
    response = requests.post(f"{API_URL}/predict", json=data, timeout=30)
    end_time = time.time()
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    return end_time - start_time

def run_performance_test(num_requests: int = 10) -> List[float]:
    """Run multiple prediction requests and measure timing."""
    print(f"Running {num_requests} prediction requests...")
    
    # Test health endpoint first
    health_response = requests.get(f"{API_URL}/health", timeout=30)
    if health_response.status_code != 200:
        raise Exception("API health check failed")
    print("✓ API health check passed")
    
    times = []
    for i in range(num_requests):
        # Cycle through test data
        test_case = test_data[i % len(test_data)]
        
        try:
            duration = test_single_prediction(test_case)
            times.append(duration)
            print(f"Request {i+1}/{num_requests}: {duration:.4f}s")
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
            
    return times

def analyze_results(times: List[float]):
    """Analyze and print performance statistics."""
    if not times:
        print("No successful requests to analyze")
        return
    
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS")
    print("="*50)
    print(f"Total requests: {len(times)}")
    print(f"Mean response time: {statistics.mean(times):.4f}s")
    print(f"Median response time: {statistics.median(times):.4f}s")
    print(f"Min response time: {min(times):.4f}s")
    print(f"Max response time: {max(times):.4f}s")
    
    if len(times) > 1:
        print(f"Standard deviation: {statistics.stdev(times):.4f}s")
    
    # Percentiles
    times_sorted = sorted(times)
    p95_idx = int(0.95 * len(times_sorted))
    p99_idx = int(0.99 * len(times_sorted))
    
    print(f"95th percentile: {times_sorted[min(p95_idx, len(times_sorted)-1)]:.4f}s")
    print(f"99th percentile: {times_sorted[min(p99_idx, len(times_sorted)-1)]:.4f}s")

def main():
    print("Starting API performance test...")
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    try:
        times = run_performance_test(20)  # Test with 20 requests
        analyze_results(times)
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()