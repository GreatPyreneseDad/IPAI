"""
Performance Tests

Comprehensive performance testing for the IPAI system including
load testing, caching, and optimization validation.
"""

import pytest
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import statistics
from datetime import datetime, timedelta

from src.core.performance import PerformanceOptimizer, CacheManager, PerformanceMetrics
from src.coherence.gct_calculator import GCTCalculator
from src.models.coherence_profile import CoherenceProfile, GCTComponents, IndividualParameters
from tests.conftest import measure_async_performance, measure_sync_performance, assert_performance_metrics


class TestPerformanceMetrics:
    """Test performance metrics collection and analysis"""
    
    def test_performance_metrics_initialization(self):
        """Test performance metrics initialization"""
        metrics = PerformanceMetrics()
        
        assert metrics.request_count == 0
        assert metrics.total_response_time == 0.0
        assert metrics.avg_response_time == 0.0
        assert metrics.min_response_time == float('inf')
        assert metrics.max_response_time == 0.0
        assert metrics.error_count == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
    
    def test_metrics_update(self):
        """Test metrics update functionality"""
        metrics = PerformanceMetrics()
        
        # Update with response times
        response_times = [0.1, 0.2, 0.15, 0.3, 0.25]
        
        for i, response_time in enumerate(response_times):
            is_error = i == 2  # Make one request an error
            cache_hit = i % 2 == 0  # Alternate cache hits
            
            metrics.update(response_time, is_error, cache_hit)
        
        assert metrics.request_count == 5
        assert abs(metrics.avg_response_time - statistics.mean(response_times)) < 1e-6
        assert metrics.min_response_time == min(response_times)
        assert metrics.max_response_time == max(response_times)
        assert metrics.error_count == 1
        assert metrics.cache_hits == 3  # 0, 2, 4
        assert metrics.cache_misses == 2  # 1, 3
    
    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation"""
        metrics = PerformanceMetrics()
        
        # No cache requests
        assert metrics.cache_hit_rate == 0.0
        
        # Add cache hits and misses
        for i in range(10):
            cache_hit = i < 7  # 7 hits, 3 misses
            metrics.update(0.1, False, cache_hit)
        
        assert abs(metrics.cache_hit_rate - 0.7) < 1e-6
    
    def test_error_rate_calculation(self):
        """Test error rate calculation"""
        metrics = PerformanceMetrics()
        
        # No requests
        assert metrics.error_rate == 0.0
        
        # Add requests with some errors
        for i in range(10):
            is_error = i < 2  # 2 errors out of 10
            metrics.update(0.1, is_error)
        
        assert abs(metrics.error_rate - 0.2) < 1e-6


class TestPerformanceOptimizer:
    """Test performance optimizer functionality"""
    
    def test_performance_optimizer_initialization(self, performance_optimizer):
        """Test performance optimizer initialization"""
        assert performance_optimizer is not None
        assert hasattr(performance_optimizer, 'performance_monitor')
        assert hasattr(performance_optimizer, 'record_performance')
        assert hasattr(performance_optimizer, 'get_metrics')
    
    def test_performance_monitoring_decorator_sync(self, performance_optimizer):
        """Test performance monitoring decorator for sync functions"""
        @performance_optimizer.performance_monitor("test_sync_function")
        def slow_function(delay=0.1):
            time.sleep(delay)
            return "completed"
        
        # Execute function
        result = slow_function(0.05)
        
        assert result == "completed"
        
        # Check metrics were recorded
        metrics = performance_optimizer.get_metrics("test_sync_function")
        assert metrics['metrics'] is not None
        assert metrics['metrics']['request_count'] == 1
        assert metrics['metrics']['avg_response_time'] >= 0.05
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_decorator_async(self, performance_optimizer):
        """Test performance monitoring decorator for async functions"""
        @performance_optimizer.performance_monitor("test_async_function")
        async def async_slow_function(delay=0.1):
            await asyncio.sleep(delay)
            return "completed"
        
        # Execute function
        result = await async_slow_function(0.05)
        
        assert result == "completed"
        
        # Check metrics were recorded
        metrics = performance_optimizer.get_metrics("test_async_function")
        assert metrics['metrics'] is not None
        assert metrics['metrics']['request_count'] == 1
        assert metrics['metrics']['avg_response_time'] >= 0.05
    
    def test_slow_query_detection(self, performance_optimizer):
        """Test slow query detection and logging"""
        @performance_optimizer.performance_monitor("slow_operation")
        def slow_operation():
            time.sleep(0.2)  # Slower than threshold (0.1s)
            return "slow_result"
        
        result = slow_operation()
        assert result == "slow_result"
        
        # Check slow queries were recorded
        slow_queries = performance_optimizer.get_slow_queries()
        assert len(slow_queries) > 0
        
        slow_query = slow_queries[-1]
        assert slow_query['operation'] == "slow_operation"
        assert slow_query['response_time'] >= 0.2
    
    def test_performance_metrics_aggregation(self, performance_optimizer):
        """Test performance metrics aggregation"""
        @performance_optimizer.performance_monitor("test_operation")
        def test_operation(should_error=False):
            if should_error:
                raise ValueError("Test error")
            time.sleep(0.01)
            return "success"
        
        # Execute multiple operations
        for i in range(10):
            try:
                test_operation(should_error=(i == 5))  # One error
            except ValueError:
                pass
        
        # Check aggregated metrics
        summary = performance_optimizer.get_metrics()
        assert 'summary' in summary
        
        summary_data = summary['summary']
        assert summary_data['total_requests'] >= 10
        assert summary_data['total_errors'] >= 1
        assert summary_data['avg_response_time'] > 0
    
    def test_memory_optimization(self, performance_optimizer):
        """Test memory optimization functionality"""
        # This test simulates memory optimization
        # In a real scenario, this would check actual memory usage
        
        initial_metrics_count = len(performance_optimizer.performance_history)
        
        # Add many performance entries
        for i in range(1500):  # More than max history (1000)
            performance_optimizer.record_performance(
                f"operation_{i}",
                0.01,
                is_error=False
            )
        
        # Run optimization
        performance_optimizer.optimize_performance()
        
        # History should be trimmed
        assert len(performance_optimizer.performance_history) <= 1000
    
    @pytest.mark.asyncio
    async def test_thread_pool_execution(self, performance_optimizer):
        """Test thread pool execution for CPU-bound tasks"""
        def cpu_bound_task(n):
            # Simulate CPU-bound work
            total = 0
            for i in range(n):
                total += i * i
            return total
        
        # Execute in thread pool
        result = await performance_optimizer.run_in_thread(cpu_bound_task, 1000)
        
        assert result == sum(i * i for i in range(1000))
    
    def test_performance_cleanup(self, performance_optimizer):
        """Test performance data cleanup"""
        # Add old performance data
        old_time = datetime.utcnow() - timedelta(hours=25)
        
        performance_optimizer.performance_history.append({
            'timestamp': old_time.isoformat(),
            'operation': 'old_operation',
            'response_time': 0.1,
            'is_error': False
        })
        
        performance_optimizer.slow_queries.append({
            'timestamp': old_time.isoformat(),
            'operation': 'old_slow_operation',
            'response_time': 2.0
        })
        
        initial_history_count = len(performance_optimizer.performance_history)
        initial_slow_count = len(performance_optimizer.slow_queries)
        
        # Run cleanup
        performance_optimizer._cleanup_old_metrics()
        
        # Old data should be removed
        assert len(performance_optimizer.performance_history) < initial_history_count
        assert len(performance_optimizer.slow_queries) < initial_slow_count


class TestCacheManager:
    """Test cache manager functionality"""
    
    def test_cache_manager_initialization(self, cache_manager):
        """Test cache manager initialization"""
        assert cache_manager is not None
        assert hasattr(cache_manager, 'get')
        assert hasattr(cache_manager, 'set')
        assert hasattr(cache_manager, 'delete')
        assert hasattr(cache_manager, 'clear')
    
    def test_basic_cache_operations(self, cache_manager):
        """Test basic cache operations"""
        key = "test_key"
        value = {"data": "test_value", "number": 42}
        
        # Set value
        cache_manager.set(key, value)
        
        # Get value
        retrieved = cache_manager.get(key)
        assert retrieved == value
        
        # Check existence
        assert cache_manager.exists(key)
        
        # Delete value
        deleted = cache_manager.delete(key)
        assert deleted
        
        # Value should be gone
        assert cache_manager.get(key) is None
        assert not cache_manager.exists(key)
    
    def test_cache_ttl_expiration(self, cache_manager):
        """Test cache TTL expiration"""
        key = "ttl_test"
        value = "expires_soon"
        
        # Set with short TTL
        cache_manager.set(key, value, ttl=1)  # 1 second
        
        # Should be available immediately
        assert cache_manager.get(key) == value
        
        # Wait for expiration
        time.sleep(1.2)
        
        # Should be expired
        assert cache_manager.get(key) is None
    
    def test_cache_lru_eviction(self, cache_manager):
        """Test LRU eviction when cache is full"""
        # Fill cache to capacity
        for i in range(cache_manager.config['max_size']):
            cache_manager.set(f"key_{i}", f"value_{i}")
        
        # Add one more item (should evict oldest)
        cache_manager.set("new_key", "new_value")
        
        # First key should be evicted
        assert cache_manager.get("key_0") is None
        
        # New key should be present
        assert cache_manager.get("new_key") == "new_value"
        
        # Last key before eviction should still be present
        last_key = f"key_{cache_manager.config['max_size'] - 1}"
        assert cache_manager.get(last_key) == f"value_{cache_manager.config['max_size'] - 1}"
    
    def test_cache_key_generation(self, cache_manager):
        """Test cache key generation with parameters"""
        base_key = "function_result"
        
        # Test without parameters
        key1 = cache_manager._generate_key(base_key)
        assert key1 == base_key
        
        # Test with parameters
        key2 = cache_manager._generate_key(base_key, param1="value1", param2="value2")
        key3 = cache_manager._generate_key(base_key, param1="value1", param2="value2")
        key4 = cache_manager._generate_key(base_key, param1="different", param2="value2")
        
        # Same parameters should generate same key
        assert key2 == key3
        
        # Different parameters should generate different keys
        assert key2 != key4
        assert key1 != key2
    
    def test_cache_statistics(self, cache_manager):
        """Test cache statistics collection"""
        # Clear cache and reset stats
        cache_manager.clear()
        
        # Perform operations
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        
        # Cache miss
        cache_manager.get("nonexistent")
        
        # Cache hits
        cache_manager.get("key1")
        cache_manager.get("key2")
        
        # Delete
        cache_manager.delete("key1")
        
        # Get statistics
        stats = cache_manager.get_stats()
        
        assert stats['size'] == 1  # One item remaining
        assert stats['sets'] >= 2
        assert stats['hits'] >= 2
        assert stats['misses'] >= 1
        assert stats['deletes'] >= 1
        assert stats['hit_rate'] > 0
    
    def test_cache_compression(self):
        """Test cache compression functionality"""
        config = {
            'max_size': 100,
            'enable_compression': True,
            'compression_threshold': 10  # Very low threshold for testing
        }
        
        cache = CacheManager(config)
        
        # Large data that should be compressed
        large_data = "x" * 1000
        
        cache.set("large_key", large_data)
        retrieved = cache.get("large_key")
        
        assert retrieved == large_data
    
    def test_cache_function_decorator(self, cache_manager):
        """Test cache function decorator"""
        call_count = 0
        
        @cache_manager.cache_function(ttl=60, key_prefix="expensive_func")
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate expensive operation
            return x + y
        
        # First call should execute function
        result1 = expensive_function(2, 3)
        assert result1 == 5
        assert call_count == 1
        
        # Second call with same parameters should use cache
        result2 = expensive_function(2, 3)
        assert result2 == 5
        assert call_count == 1  # Function not called again
        
        # Different parameters should execute function
        result3 = expensive_function(3, 4)
        assert result3 == 7
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_async_cache_function_decorator(self, cache_manager):
        """Test cache function decorator with async functions"""
        call_count = 0
        
        @cache_manager.cache_function(ttl=60, key_prefix="async_func")
        async def async_expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async expensive operation
            return x * y
        
        # First call should execute function
        result1 = await async_expensive_function(3, 4)
        assert result1 == 12
        assert call_count == 1
        
        # Second call with same parameters should use cache
        result2 = await async_expensive_function(3, 4)
        assert result2 == 12
        assert call_count == 1  # Function not called again
    
    def test_concurrent_cache_access(self, cache_manager):
        """Test concurrent cache access safety"""
        def cache_worker(thread_id, operations_per_thread=100):
            for i in range(operations_per_thread):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                
                # Set, get, and delete
                cache_manager.set(key, value)
                retrieved = cache_manager.get(key)
                assert retrieved == value
                cache_manager.delete(key)
        
        # Run multiple threads concurrently
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=cache_worker, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Cache should be consistent (no crashes or data corruption)
        stats = cache_manager.get_stats()
        assert stats is not None


class TestLoadTesting:
    """Load testing for performance validation"""
    
    @pytest.mark.asyncio
    async def test_coherence_calculation_load(self, gct_calculator):
        """Test coherence calculation under load"""
        async def calculate_coherence_worker():
            components = GCTComponents(
                psi=0.7 + (0.1 * (hash(threading.current_thread().ident) % 10) / 10),
                rho=0.6 + (0.2 * (hash(threading.current_thread().ident) % 10) / 10),
                q=0.5 + (0.3 * (hash(threading.current_thread().ident) % 10) / 10),
                f=0.8 + (0.1 * (hash(threading.current_thread().ident) % 10) / 10)
            )
            params = IndividualParameters(k_m=0.5, k_i=2.0)
            
            start_time = time.time()
            result = gct_calculator.calculate_coherence(components, params)
            end_time = time.time()
            
            return {
                'result': result,
                'response_time': end_time - start_time,
                'success': True
            }
        
        # Run concurrent calculations
        tasks = [calculate_coherence_worker() for _ in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
        response_times = [r['response_time'] for r in successful_results]
        
        assert len(successful_results) == 50  # All should succeed
        assert max(response_times) < 1.0  # No calculation should take > 1 second
        assert statistics.mean(response_times) < 0.1  # Average should be fast
    
    def test_cache_performance_under_load(self, cache_manager):
        """Test cache performance under high load"""
        def cache_load_worker(worker_id, operations=1000):
            response_times = []
            
            for i in range(operations):
                key = f"load_test_{worker_id}_{i}"
                value = {"data": f"value_{i}", "worker": worker_id}
                
                # Set operation
                start = time.time()
                cache_manager.set(key, value)
                response_times.append(time.time() - start)
                
                # Get operation
                start = time.time()
                retrieved = cache_manager.get(key)
                response_times.append(time.time() - start)
                
                assert retrieved == value
            
            return response_times
        
        # Run concurrent cache operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(cache_load_worker, worker_id, 100)
                for worker_id in range(10)
            ]
            
            all_response_times = []
            for future in as_completed(futures):
                response_times = future.result()
                all_response_times.extend(response_times)
        
        # Analyze performance
        avg_response_time = statistics.mean(all_response_times)
        max_response_time = max(all_response_times)
        p95_response_time = statistics.quantiles(all_response_times, n=20)[18]  # 95th percentile
        
        assert avg_response_time < 0.001  # Average < 1ms
        assert max_response_time < 0.01   # Max < 10ms
        assert p95_response_time < 0.005  # 95th percentile < 5ms
    
    def test_memory_usage_under_load(self, performance_optimizer):
        """Test memory usage under load"""
        import psutil
        
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            @performance_optimizer.performance_monitor("memory_test")
            def memory_intensive_operation():
                # Create large data structures
                data = list(range(10000))
                processed = [x * x for x in data]
                return sum(processed)
            
            # Run many operations
            for _ in range(100):
                result = memory_intensive_operation()
                assert result > 0
            
            # Force garbage collection
            performance_optimizer._optimize_memory()
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (< 100MB)
            assert memory_increase < 100 * 1024 * 1024
            
        except ImportError:
            # psutil not available, skip memory test
            pytest.skip("psutil not available for memory testing")


class TestPerformanceOptimization:
    """Test performance optimization strategies"""
    
    def test_query_optimization_detection(self, performance_optimizer):
        """Test detection of optimization opportunities"""
        # Create operations with different performance characteristics
        @performance_optimizer.performance_monitor("fast_operation")
        def fast_operation():
            time.sleep(0.01)
            return "fast"
        
        @performance_optimizer.performance_monitor("slow_operation")
        def slow_operation():
            time.sleep(0.2)  # Slow operation
            return "slow"
        
        # Execute operations
        for _ in range(5):
            fast_operation()
            slow_operation()
        
        # Get metrics
        metrics = performance_optimizer.get_metrics()
        
        # Slow operation should be identified
        slow_metrics = metrics['operations']['slow_operation']
        fast_metrics = metrics['operations']['fast_operation']
        
        assert slow_metrics['avg_response_time'] > fast_metrics['avg_response_time']
        
        # Check slow queries
        slow_queries = performance_optimizer.get_slow_queries()
        slow_operations = [q for q in slow_queries if q['operation'] == 'slow_operation']
        assert len(slow_operations) >= 5
    
    def test_caching_effectiveness(self, cache_manager):
        """Test caching effectiveness for performance improvement"""
        computation_count = 0
        
        def expensive_computation(n):
            nonlocal computation_count
            computation_count += 1
            # Simulate expensive computation
            time.sleep(0.1)
            return sum(range(n))
        
        # Without caching
        start_time = time.time()
        result1 = expensive_computation(100)
        result2 = expensive_computation(100)  # Same computation
        no_cache_time = time.time() - start_time
        
        assert computation_count == 2
        assert result1 == result2
        
        # Reset counter
        computation_count = 0
        
        # With caching
        @cache_manager.cache_function(ttl=300)
        def cached_expensive_computation(n):
            nonlocal computation_count
            computation_count += 1
            time.sleep(0.1)
            return sum(range(n))
        
        start_time = time.time()
        result3 = cached_expensive_computation(100)
        result4 = cached_expensive_computation(100)  # Should use cache
        cached_time = time.time() - start_time
        
        assert computation_count == 1  # Only computed once
        assert result3 == result4
        assert cached_time < no_cache_time  # Should be faster
    
    def test_performance_monitoring_overhead(self, performance_optimizer):
        """Test that performance monitoring has minimal overhead"""
        def simple_operation():
            return sum(range(100))
        
        # Measure without monitoring
        start_time = time.time()
        for _ in range(1000):
            simple_operation()
        baseline_time = time.time() - start_time
        
        # Measure with monitoring
        @performance_optimizer.performance_monitor("monitored_operation")
        def monitored_operation():
            return sum(range(100))
        
        start_time = time.time()
        for _ in range(1000):
            monitored_operation()
        monitored_time = time.time() - start_time
        
        # Overhead should be minimal (< 50% increase)
        overhead_ratio = monitored_time / baseline_time
        assert overhead_ratio < 1.5, f"Performance monitoring overhead too high: {overhead_ratio}"


class TestResourceUtilization:
    """Test resource utilization and limits"""
    
    def test_connection_pool_efficiency(self):
        """Test database connection pool efficiency"""
        # This would test database connection pooling
        # For now, we'll simulate connection usage
        
        from src.core.database import Database
        
        config = {
            'url': 'postgresql://localhost/test',
            'pool_size': 5,
            'max_overflow': 10
        }
        
        db = Database(config)
        
        # Configuration should be set correctly
        assert db.config['pool_size'] == 5
        assert db.config['max_overflow'] == 10
    
    def test_thread_pool_resource_management(self, performance_optimizer):
        """Test thread pool resource management"""
        def cpu_task(n):
            return sum(i * i for i in range(n))
        
        async def run_concurrent_tasks():
            tasks = []
            for i in range(20):  # More tasks than thread pool size
                task = performance_optimizer.run_in_thread(cpu_task, 1000)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        # Run tasks
        start_time = time.time()
        results = asyncio.run(run_concurrent_tasks())
        execution_time = time.time() - start_time
        
        # All tasks should complete successfully
        assert len(results) == 20
        assert all(isinstance(r, int) for r in results)
        
        # Should complete in reasonable time despite thread pool limits
        assert execution_time < 10.0  # Should complete within 10 seconds
    
    def test_memory_leak_prevention(self, cache_manager):
        """Test memory leak prevention in cache"""
        initial_size = len(cache_manager._cache)
        
        # Add many items to trigger cleanup
        for i in range(cache_manager.config['max_size'] * 2):
            cache_manager.set(f"temp_key_{i}", f"temp_value_{i}", ttl=1)
        
        # Wait for TTL expiration
        time.sleep(1.5)
        
        # Force cleanup
        cache_manager._cleanup_expired()
        
        # Cache should not grow unbounded
        final_size = len(cache_manager._cache)
        assert final_size <= cache_manager.config['max_size']


class TestBenchmarking:
    """Benchmarking tests for performance baselines"""
    
    def test_coherence_calculation_benchmark(self, gct_calculator):
        """Benchmark coherence calculation performance"""
        components = GCTComponents(psi=0.8, rho=0.7, q=0.6, f=0.9)
        params = IndividualParameters(k_m=0.5, k_i=2.0)
        
        # Warmup
        for _ in range(10):
            gct_calculator.calculate_coherence(components, params)
        
        # Benchmark
        iterations = 1000
        start_time = time.time()
        
        for _ in range(iterations):
            result = gct_calculator.calculate_coherence(components, params)
            assert result['coherence_score'] >= 0
        
        total_time = time.time() - start_time
        avg_time = total_time / iterations
        
        # Performance baseline: < 1ms per calculation
        assert avg_time < 0.001, f"Coherence calculation too slow: {avg_time:.4f}s"
        
        # Throughput baseline: > 1000 calculations/second
        throughput = iterations / total_time
        assert throughput > 1000, f"Throughput too low: {throughput:.0f} calc/s"
    
    def test_cache_operation_benchmark(self, cache_manager):
        """Benchmark cache operation performance"""
        # Warmup
        for i in range(100):
            cache_manager.set(f"warmup_{i}", f"value_{i}")
            cache_manager.get(f"warmup_{i}")
        
        # Benchmark set operations
        iterations = 10000
        start_time = time.time()
        
        for i in range(iterations):
            cache_manager.set(f"bench_key_{i}", f"bench_value_{i}")
        
        set_time = time.time() - start_time
        set_throughput = iterations / set_time
        
        # Benchmark get operations
        start_time = time.time()
        
        for i in range(iterations):
            value = cache_manager.get(f"bench_key_{i}")
            assert value == f"bench_value_{i}"
        
        get_time = time.time() - start_time
        get_throughput = iterations / get_time
        
        # Performance baselines
        assert set_throughput > 10000, f"Cache set throughput too low: {set_throughput:.0f} ops/s"
        assert get_throughput > 50000, f"Cache get throughput too low: {get_throughput:.0f} ops/s"