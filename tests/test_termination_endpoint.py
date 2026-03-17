#!/usr/bin/env python3
"""
Test the /termination_condition endpoint for CLI agent termination.

This test:
1. Starts the server with a test state
2. Calls the /termination_condition endpoint
3. Verifies it returns correct badge count from ROM memory
"""

import pytest
import requests
import time
import subprocess
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTerminationEndpoint:
    """Test class for /termination_condition endpoint"""
    
    @pytest.fixture(scope="class")
    def server(self):
        """Start server and yield, then cleanup"""
        port = 8011  # Use different port from other tests
        server_url = f"http://127.0.0.1:{port}"
        
        # Find a suitable state file
        state_file = None
        potential_states = [
            "Emerald-GBAdvance/quick_start_save.state",
            "Emerald-GBAdvance/start.state",
            "Emerald-GBAdvance/splits/01_tutorial/01_tutorial.state",
        ]
        
        for state in potential_states:
            if os.path.exists(state):
                state_file = state
                break
        
        if not state_file:
            pytest.skip("No suitable state file found")
        
        # Start server
        server_cmd = [
            sys.executable, "-m", "server.app",
            "--load-state", state_file,
            "--port", str(port),
        ]
        
        server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        started = False
        for i in range(30):
            try:
                response = requests.get(f"{server_url}/health", timeout=2)
                if response.status_code == 200:
                    started = True
                    break
            except requests.exceptions.RequestException:
                time.sleep(1)
        
        if not started:
            server_process.terminate()
            server_process.wait()
            pytest.fail("Server failed to start")
        
        yield {"url": server_url, "process": server_process}
        
        # Cleanup
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()

    def test_endpoint_exists(self, server):
        """Test that /termination_condition endpoint exists"""
        response = requests.get(f"{server['url']}/termination_condition", timeout=5)
        # Should not return 404
        assert response.status_code != 404, "Endpoint /termination_condition not found"
    
    def test_gym_badge_count_default(self, server):
        """Test default gym_badge_count condition (threshold=1)"""
        response = requests.get(f"{server['url']}/termination_condition", timeout=5)
        assert response.status_code == 200, f"Unexpected status: {response.status_code}"
        
        data = response.json()
        
        # Verify response structure
        assert "condition_type" in data, "Missing condition_type in response"
        assert "threshold" in data, "Missing threshold in response"
        assert "current_value" in data, "Missing current_value in response"
        assert "condition_met" in data, "Missing condition_met in response"
        
        # Verify values
        assert data["condition_type"] == "gym_badge_count"
        assert data["threshold"] == 1
        assert isinstance(data["current_value"], int)
        assert isinstance(data["condition_met"], bool)
        
        # Badge count should be >= 0
        assert data["current_value"] >= 0
        
        # condition_met should match threshold comparison
        assert data["condition_met"] == (data["current_value"] >= data["threshold"])
        
        print(f"Badge count: {data['current_value']}, condition_met: {data['condition_met']}")
        if data.get("badge_names"):
            print(f"Badges: {data['badge_names']}")
    
    def test_gym_badge_count_custom_threshold(self, server):
        """Test gym_badge_count with custom threshold"""
        response = requests.get(
            f"{server['url']}/termination_condition",
            params={"condition_type": "gym_badge_count", "threshold": 0},
            timeout=5
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["threshold"] == 0
        # With threshold 0, any badge count >= 0 should meet condition
        assert data["condition_met"] == True
    
    def test_gym_badge_count_high_threshold(self, server):
        """Test gym_badge_count with threshold higher than current badges"""
        response = requests.get(
            f"{server['url']}/termination_condition",
            params={"condition_type": "gym_badge_count", "threshold": 8},
            timeout=5
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["threshold"] == 8
        # Most test states won't have 8 badges
        # condition_met depends on actual badge count
        assert isinstance(data["condition_met"], bool)
        print(f"8-badge threshold test: current={data['current_value']}, met={data['condition_met']}")
    
    def test_unknown_condition_type(self, server):
        """Test that unknown condition types return error"""
        response = requests.get(
            f"{server['url']}/termination_condition",
            params={"condition_type": "unknown_condition"},
            timeout=5
        )
        assert response.status_code == 200  # Returns 200 with error info, not 4xx
        
        data = response.json()
        assert "error" in data or "supported_types" in data
    
    def test_badge_names_included(self, server):
        """Test that badge_names is included for gym_badge_count"""
        response = requests.get(f"{server['url']}/termination_condition", timeout=5)
        assert response.status_code == 200
        
        data = response.json()
        assert "badge_names" in data, "badge_names should be included in response"
        assert isinstance(data["badge_names"], (list, type(None)))


def run_standalone_test():
    """Run a standalone test without pytest (for quick manual testing)"""
    print("=" * 60)
    print("Standalone Termination Endpoint Test")
    print("=" * 60)
    
    # Check if server is running
    server_url = os.environ.get("SERVER_URL", "http://localhost:8000")
    
    try:
        print(f"\nTesting server at: {server_url}")
        
        # Test health
        health = requests.get(f"{server_url}/health", timeout=5)
        if health.status_code != 200:
            print(f"Server health check failed: {health.status_code}")
            return False
        print("Server health: OK")
        
        # Test termination_condition
        response = requests.get(f"{server_url}/termination_condition", timeout=5)
        if response.status_code != 200:
            print(f"Termination endpoint failed: {response.status_code}")
            return False
        
        data = response.json()
        print(f"\nTermination Condition Response:")
        print(f"  condition_type: {data.get('condition_type')}")
        print(f"  threshold: {data.get('threshold')}")
        print(f"  current_value: {data.get('current_value')}")
        print(f"  badge_names: {data.get('badge_names')}")
        print(f"  condition_met: {data.get('condition_met')}")
        
        # Verify structure
        required_fields = ["condition_type", "threshold", "current_value", "condition_met"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            print(f"\nMissing fields: {missing}")
            return False
        
        print("\nAll tests passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to server at {server_url}")
        print("Make sure the server is running or set SERVER_URL environment variable")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    # Run standalone test if called directly
    import sys
    success = run_standalone_test()
    sys.exit(0 if success else 1)
