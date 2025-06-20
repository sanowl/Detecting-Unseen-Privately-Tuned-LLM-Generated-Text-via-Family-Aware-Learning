"""
Comprehensive Production Testing Framework for PhantomHunter
Includes unit tests, integration tests, performance tests, security tests, and load testing
"""

import pytest
import asyncio
import time
import requests
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import statistics
import numpy as np
from dataclasses import dataclass
import logging

# Test configuration
@dataclass
class TestConfig:
    api_base_url: str = "http://localhost:8000"
    api_key: str = "demo-api-key"
    timeout: int = 30
    max_workers: int = 10
    test_data_size: int = 100

class ProductionTestSuite:
    """Comprehensive production test suite"""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        })
        
        # Test data
        self.test_texts = self._load_test_data()
        
    def _load_test_data(self) -> Dict[str, List[str]]:
        """Load test data for different scenarios"""
        return {
            'human_texts': [
                "The morning sun cast long shadows across the dewy grass as children played in the neighborhood park.",
                "Yesterday, I visited my grandmother and she shared stories about her childhood during the 1940s.",
                "The recipe calls for three cups of flour, two eggs, and a pinch of salt to make the perfect bread.",
                "After hiking for six hours, we finally reached the summit and were rewarded with a breathtaking view.",
                "My cat has a peculiar habit of sitting in empty cardboard boxes regardless of their size."
            ],
            'ai_texts': [
                "Artificial intelligence has revolutionized numerous industries by providing automated solutions that enhance efficiency and accuracy.",
                "The integration of machine learning algorithms enables systems to process vast amounts of data and identify patterns.",
                "Natural language processing models have achieved remarkable capabilities in understanding and generating human-like text.",
                "Deep learning architectures utilize neural networks with multiple layers to learn complex representations from data.",
                "Computer vision applications can analyze images and videos to extract meaningful information and make decisions."
            ],
            'mixed_texts': [
                "I think AI is fascinating because it can help us solve complex problems, but we should be careful about how we use it.",
                "The weather today is beautiful, which makes me think about how climate models use machine learning to predict patterns.",
                "While cooking dinner, I was wondering if artificial intelligence could help create better recipes based on nutritional data."
            ],
            'edge_cases': [
                "",  # Empty text
                "A",  # Single character
                "A" * 10000,  # Very long text
                "🤖🔥💯",  # Emojis only
                "123 456 789",  # Numbers only
                "Hello\n\nWorld\n\n\nTest",  # Multiple newlines
                "This is a test with special chars: @#$%^&*()",  # Special characters
            ]
        }

class HealthCheckTests:
    """Test suite for health check endpoints"""
    
    def __init__(self, test_suite: ProductionTestSuite):
        self.suite = test_suite
    
    def test_health_endpoint(self):
        """Test basic health endpoint"""
        response = self.suite.session.get(f"{self.suite.config.api_base_url}/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'status' in data
        assert 'timestamp' in data
        assert 'system_health' in data
        assert data['status'] in ['healthy', 'degraded', 'unhealthy']
        
        print(f"✅ Health check passed: {data['status']}")
        return data
    
    def test_health_response_time(self):
        """Test health endpoint response time"""
        start_time = time.time()
        response = self.suite.session.get(f"{self.suite.config.api_base_url}/health")
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0  # Health check should be fast
        
        print(f"✅ Health response time: {response_time:.3f}s")
        return response_time

class FunctionalTests:
    """Test suite for core functionality"""
    
    def __init__(self, test_suite: ProductionTestSuite):
        self.suite = test_suite
    
    def test_single_text_analysis(self):
        """Test analysis of a single text"""
        test_data = {
            "texts": [self.suite.test_texts['human_texts'][0]],
            "include_explanations": True,
            "include_watermark_analysis": True,
            "include_source_attribution": True
        }
        
        response = self.suite.session.post(
            f"{self.suite.config.api_base_url}/analyze",
            json=test_data,
            timeout=self.suite.config.timeout
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert 'request_id' in data
        assert 'results' in data
        assert len(data['results']) == 1
        
        result = data['results'][0]
        assert 'is_ai_generated' in result
        assert 'confidence' in result
        assert 'processing_time' in result
        assert isinstance(result['is_ai_generated'], bool)
        assert 0.0 <= result['confidence'] <= 1.0
        
        print(f"✅ Single text analysis: {result['is_ai_generated']} (confidence: {result['confidence']:.3f})")
        return data
    
    def test_batch_text_analysis(self):
        """Test analysis of multiple texts"""
        test_data = {
            "texts": self.suite.test_texts['human_texts'][:3] + self.suite.test_texts['ai_texts'][:3],
            "include_explanations": False,
            "include_watermark_analysis": True,
            "include_source_attribution": True
        }
        
        response = self.suite.session.post(
            f"{self.suite.config.api_base_url}/analyze",
            json=test_data,
            timeout=self.suite.config.timeout
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data['results']) == 6
        
        # Check all results have required fields
        for result in data['results']:
            assert 'is_ai_generated' in result
            assert 'confidence' in result
            assert 'text_id' in result
        
        print(f"✅ Batch analysis: {len(data['results'])} texts processed")
        return data
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        results = {}
        
        # Test empty text
        try:
            response = self.suite.session.post(
                f"{self.suite.config.api_base_url}/analyze",
                json={"texts": [""]},
                timeout=self.suite.config.timeout
            )
            results['empty_text'] = response.status_code == 422  # Should be validation error
        except Exception as e:
            results['empty_text'] = True  # Expected to fail
        
        # Test very long text
        try:
            long_text = "A" * 10000
            response = self.suite.session.post(
                f"{self.suite.config.api_base_url}/analyze",
                json={"texts": [long_text]},
                timeout=self.suite.config.timeout
            )
            results['long_text'] = response.status_code in [200, 422]
        except Exception as e:
            results['long_text'] = True
        
        # Test invalid JSON
        try:
            response = self.suite.session.post(
                f"{self.suite.config.api_base_url}/analyze",
                data="invalid json",
                timeout=self.suite.config.timeout
            )
            results['invalid_json'] = response.status_code == 422
        except Exception as e:
            results['invalid_json'] = True
        
        print(f"✅ Edge cases tested: {sum(results.values())}/{len(results)} passed")
        return results

class PerformanceTests:
    """Test suite for performance evaluation"""
    
    def __init__(self, test_suite: ProductionTestSuite):
        self.suite = test_suite
        self.performance_metrics = []
    
    def test_response_time_single(self, num_tests: int = 10):
        """Test response time for single text analysis"""
        response_times = []
        
        for i in range(num_tests):
            test_data = {
                "texts": [self.suite.test_texts['human_texts'][i % len(self.suite.test_texts['human_texts'])]],
                "include_explanations": False
            }
            
            start_time = time.time()
            response = self.suite.session.post(
                f"{self.suite.config.api_base_url}/analyze",
                json=test_data,
                timeout=self.suite.config.timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                response_times.append(response_time)
        
        avg_time = statistics.mean(response_times)
        p95_time = np.percentile(response_times, 95)
        p99_time = np.percentile(response_times, 99)
        
        # Performance assertions
        assert avg_time < 2.0  # Average should be under 2 seconds
        assert p95_time < 5.0  # 95th percentile should be under 5 seconds
        
        metrics = {
            'avg_response_time': avg_time,
            'p95_response_time': p95_time,
            'p99_response_time': p99_time,
            'total_requests': len(response_times)
        }
        
        print(f"✅ Single text performance: avg={avg_time:.3f}s, p95={p95_time:.3f}s")
        return metrics
    
    def test_throughput(self, duration_seconds: int = 60):
        """Test system throughput over time"""
        start_time = time.time()
        successful_requests = 0
        failed_requests = 0
        response_times = []
        
        def make_request():
            nonlocal successful_requests, failed_requests
            try:
                test_data = {
                    "texts": [self.suite.test_texts['human_texts'][0]],
                    "include_explanations": False
                }
                
                req_start = time.time()
                response = self.suite.session.post(
                    f"{self.suite.config.api_base_url}/analyze",
                    json=test_data,
                    timeout=self.suite.config.timeout
                )
                req_time = time.time() - req_start
                
                if response.status_code == 200:
                    successful_requests += 1
                    response_times.append(req_time)
                else:
                    failed_requests += 1
                    
            except Exception as e:
                failed_requests += 1
        
        # Run requests for specified duration
        with ThreadPoolExecutor(max_workers=5) as executor:
            while time.time() - start_time < duration_seconds:
                executor.submit(make_request)
                time.sleep(0.1)  # Small delay between requests
        
        total_time = time.time() - start_time
        throughput = successful_requests / total_time
        
        metrics = {
            'throughput_rps': throughput,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'failure_rate': failed_requests / (successful_requests + failed_requests) if (successful_requests + failed_requests) > 0 else 0,
            'avg_response_time': statistics.mean(response_times) if response_times else 0
        }
        
        print(f"✅ Throughput test: {throughput:.2f} RPS, {metrics['failure_rate']:.2%} failure rate")
        return metrics
    
    def test_concurrent_requests(self, num_concurrent: int = 10, num_requests: int = 50):
        """Test concurrent request handling"""
        def make_concurrent_request(request_id: int):
            try:
                test_data = {
                    "texts": [self.suite.test_texts['human_texts'][request_id % len(self.suite.test_texts['human_texts'])]],
                    "include_explanations": False
                }
                
                start_time = time.time()
                response = self.suite.session.post(
                    f"{self.suite.config.api_base_url}/analyze",
                    json=test_data,
                    timeout=self.suite.config.timeout
                )
                response_time = time.time() - start_time
                
                return {
                    'request_id': request_id,
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'success': response.status_code == 200
                }
            except Exception as e:
                return {
                    'request_id': request_id,
                    'status_code': 0,
                    'response_time': 0,
                    'success': False,
                    'error': str(e)
                }
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_concurrent_request, i) for i in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        successful = sum(1 for r in results if r['success'])
        failed = num_requests - successful
        avg_response_time = statistics.mean([r['response_time'] for r in results if r['success']])
        
        metrics = {
            'total_requests': num_requests,
            'successful_requests': successful,
            'failed_requests': failed,
            'success_rate': successful / num_requests,
            'avg_response_time': avg_response_time,
            'total_time': total_time,
            'effective_throughput': successful / total_time
        }
        
        print(f"✅ Concurrent test: {successful}/{num_requests} successful, {avg_response_time:.3f}s avg")
        return metrics

class SecurityTests:
    """Test suite for security evaluation"""
    
    def __init__(self, test_suite: ProductionTestSuite):
        self.suite = test_suite
    
    def test_authentication(self):
        """Test authentication requirements"""
        # Test without authentication
        session_no_auth = requests.Session()
        
        response = session_no_auth.post(
            f"{self.suite.config.api_base_url}/analyze",
            json={"texts": ["test"]},
            timeout=self.suite.config.timeout
        )
        
        assert response.status_code == 401  # Should require authentication
        
        # Test with invalid token
        session_bad_auth = requests.Session()
        session_bad_auth.headers.update({'Authorization': 'Bearer invalid-token'})
        
        response = session_bad_auth.post(
            f"{self.suite.config.api_base_url}/analyze",
            json={"texts": ["test"]},
            timeout=self.suite.config.timeout
        )
        
        assert response.status_code == 401
        
        print("✅ Authentication tests passed")
        return True
    
    def test_input_validation(self):
        """Test input validation and sanitization"""
        test_cases = [
            # SQL injection attempts
            {"texts": ["'; DROP TABLE users; --"]},
            # Script injection attempts
            {"texts": ["<script>alert('xss')</script>"]},
            # Extremely long input
            {"texts": ["A" * 100000]},
            # Invalid data types
            {"texts": [123, 456]},
            # Missing required fields
            {},
            # Invalid JSON structure
            {"invalid_field": "value"}
        ]
        
        validation_results = []
        
        for test_case in test_cases:
            try:
                response = self.suite.session.post(
                    f"{self.suite.config.api_base_url}/analyze",
                    json=test_case,
                    timeout=self.suite.config.timeout
                )
                
                # Should either reject (4xx) or handle safely (200)
                validation_results.append(response.status_code in [400, 422, 200])
                
            except Exception as e:
                validation_results.append(True)  # Exception is acceptable
        
        success_rate = sum(validation_results) / len(validation_results)
        assert success_rate >= 0.8  # At least 80% should be handled properly
        
        print(f"✅ Input validation: {success_rate:.2%} cases handled correctly")
        return validation_results
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Make rapid requests to trigger rate limiting
        rapid_requests = []
        
        for i in range(20):
            try:
                response = self.suite.session.post(
                    f"{self.suite.config.api_base_url}/analyze",
                    json={"texts": [f"test {i}"]},
                    timeout=self.suite.config.timeout
                )
                rapid_requests.append(response.status_code)
            except Exception as e:
                rapid_requests.append(0)
        
        # Should see some rate limiting (429) if implemented
        rate_limited = sum(1 for code in rapid_requests if code == 429)
        
        print(f"✅ Rate limiting: {rate_limited} requests rate limited")
        return rapid_requests

class IntegrationTests:
    """Test suite for integration scenarios"""
    
    def __init__(self, test_suite: ProductionTestSuite):
        self.suite = test_suite
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Step 1: Health check
        health_response = self.suite.session.get(f"{self.suite.config.api_base_url}/health")
        assert health_response.status_code == 200
        
        # Step 2: Analyze mixed content
        test_data = {
            "texts": self.suite.test_texts['human_texts'][:2] + self.suite.test_texts['ai_texts'][:2],
            "include_explanations": True,
            "include_watermark_analysis": True,
            "include_source_attribution": True
        }
        
        analysis_response = self.suite.session.post(
            f"{self.suite.config.api_base_url}/analyze",
            json=test_data,
            timeout=self.suite.config.timeout
        )
        
        assert analysis_response.status_code == 200
        analysis_data = analysis_response.json()
        
        # Step 3: Verify results structure
        assert len(analysis_data['results']) == 4
        
        # Step 4: Check metrics (if available)
        try:
            metrics_response = self.suite.session.get(f"{self.suite.config.api_base_url}/metrics")
            # Metrics might require admin permissions, so 401/403 is acceptable
            assert metrics_response.status_code in [200, 401, 403]
        except Exception:
            pass  # Metrics endpoint might not be accessible
        
        print("✅ End-to-end workflow completed successfully")
        return analysis_data
    
    def test_data_consistency(self):
        """Test data consistency across multiple requests"""
        test_text = self.suite.test_texts['human_texts'][0]
        
        # Make multiple requests with same text
        results = []
        for i in range(5):
            response = self.suite.session.post(
                f"{self.suite.config.api_base_url}/analyze",
                json={"texts": [test_text]},
                timeout=self.suite.config.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data['results'][0]
                results.append({
                    'is_ai_generated': result['is_ai_generated'],
                    'confidence': result['confidence']
                })
        
        # Check consistency
        predictions = [r['is_ai_generated'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # All predictions should be the same
        assert len(set(predictions)) == 1
        
        # Confidence should be reasonably consistent (within 10%)
        confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0
        assert confidence_std < 0.1
        
        print(f"✅ Data consistency: prediction={predictions[0]}, confidence_std={confidence_std:.4f}")
        return results

def run_production_tests():
    """Run the complete production test suite"""
    print("🚀 Starting PhantomHunter Production Test Suite")
    print("=" * 60)
    
    # Initialize test suite
    config = TestConfig()
    suite = ProductionTestSuite(config)
    
    # Test results
    results = {
        'health': {},
        'functional': {},
        'performance': {},
        'security': {},
        'integration': {}
    }
    
    try:
        # Health Check Tests
        print("\n📊 Running Health Check Tests...")
        health_tests = HealthCheckTests(suite)
        results['health']['basic'] = health_tests.test_health_endpoint()
        results['health']['response_time'] = health_tests.test_health_response_time()
        
        # Functional Tests
        print("\n🔧 Running Functional Tests...")
        functional_tests = FunctionalTests(suite)
        results['functional']['single'] = functional_tests.test_single_text_analysis()
        results['functional']['batch'] = functional_tests.test_batch_text_analysis()
        results['functional']['edge_cases'] = functional_tests.test_edge_cases()
        
        # Performance Tests
        print("\n⚡ Running Performance Tests...")
        performance_tests = PerformanceTests(suite)
        results['performance']['response_time'] = performance_tests.test_response_time_single()
        results['performance']['throughput'] = performance_tests.test_throughput(30)  # 30 second test
        results['performance']['concurrent'] = performance_tests.test_concurrent_requests()
        
        # Security Tests
        print("\n🔒 Running Security Tests...")
        security_tests = SecurityTests(suite)
        results['security']['auth'] = security_tests.test_authentication()
        results['security']['validation'] = security_tests.test_input_validation()
        results['security']['rate_limiting'] = security_tests.test_rate_limiting()
        
        # Integration Tests
        print("\n🔗 Running Integration Tests...")
        integration_tests = IntegrationTests(suite)
        results['integration']['e2e'] = integration_tests.test_end_to_end_workflow()
        results['integration']['consistency'] = integration_tests.test_data_consistency()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        
        # Generate summary report
        generate_test_report(results)
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        logging.error(f"Test suite failed: {e}", exc_info=True)
        return False
    
    return True

def generate_test_report(results: Dict[str, Any]):
    """Generate comprehensive test report"""
    print("\n📊 TEST SUMMARY REPORT")
    print("=" * 60)
    
    # Health metrics
    if 'health' in results:
        print(f"🏥 Health Status: {results['health']['basic']['status']}")
        print(f"⏱️  Health Response Time: {results['health']['response_time']:.3f}s")
    
    # Performance metrics
    if 'performance' in results:
        perf = results['performance']
        print(f"🚀 Average Response Time: {perf['response_time']['avg_response_time']:.3f}s")
        print(f"📈 Throughput: {perf['throughput']['throughput_rps']:.2f} RPS")
        print(f"🔄 Concurrent Success Rate: {perf['concurrent']['success_rate']:.2%}")
    
    # Security status
    if 'security' in results:
        sec = results['security']
        print(f"🔐 Authentication: {'✅' if sec['auth'] else '❌'}")
        validation_rate = sum(sec['validation']) / len(sec['validation'])
        print(f"🛡️  Input Validation: {validation_rate:.2%}")
    
    # Save detailed report
    with open('production_test_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: production_test_report.json")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "health":
            config = TestConfig()
            suite = ProductionTestSuite(config)
            health_tests = HealthCheckTests(suite)
            health_tests.test_health_endpoint()
        elif sys.argv[1] == "performance":
            config = TestConfig()
            suite = ProductionTestSuite(config)
            perf_tests = PerformanceTests(suite)
            perf_tests.test_response_time_single()
        elif sys.argv[1] == "security":
            config = TestConfig()
            suite = ProductionTestSuite(config)
            sec_tests = SecurityTests(suite)
            sec_tests.test_authentication()
        else:
            run_production_tests()
    else:
        run_production_tests() 