"""
Performance Optimization and Caching

This module provides performance optimization and caching capabilities
for the IPAI system with defensive practices.
"""

import time
import asyncio
import logging
from typing import Any, Dict, Optional, List, Callable, Union
from functools import wraps, lru_cache
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
import gc

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    
    request_count: int = 0
    total_response_time: float = 0.0
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def update(self, response_time: float, is_error: bool = False, cache_hit: bool = False):
        """Update metrics with new data"""
        self.request_count += 1
        self.total_response_time += response_time
        self.avg_response_time = self.total_response_time / self.request_count
        self.min_response_time = min(self.min_response_time, response_time)
        self.max_response_time = max(self.max_response_time, response_time)
        
        if is_error:
            self.error_count += 1
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests == 0:
            return 0.0
        return self.cache_hits / total_cache_requests
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count


class PerformanceOptimizer:
    """Performance optimization manager"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.metrics = defaultdict(PerformanceMetrics)
        self.slow_queries = []
        self.performance_history = []
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config['max_threads'])
        self._lock = threading.Lock()
        
    def _default_config(self) -> Dict:
        """Default performance configuration"""
        return {
            'slow_query_threshold': 1.0,  # seconds
            'max_slow_queries_tracked': 100,
            'max_performance_history': 1000,
            'cache_cleanup_interval': 3600,  # seconds
            'max_threads': 10,
            'enable_profiling': False,
            'enable_memory_monitoring': True,
            'gc_threshold': 0.8,  # Memory usage threshold for GC
            'optimization_interval': 300  # seconds
        }
    
    def performance_monitor(self, operation_name: str = None):
        """Decorator for monitoring performance"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    response_time = time.time() - start_time
                    self.record_performance(op_name, response_time, is_error=False)
                    
                    # Check for slow operations
                    if response_time > self.config['slow_query_threshold']:
                        self._record_slow_query(op_name, response_time, args, kwargs)
                    
                    return result
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    self.record_performance(op_name, response_time, is_error=True)
                    logger.error(f"Performance monitor caught error in {op_name}: {e}")
                    raise
                    
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                try:
                    result = func(*args, **kwargs)
                    response_time = time.time() - start_time
                    self.record_performance(op_name, response_time, is_error=False)
                    
                    if response_time > self.config['slow_query_threshold']:
                        self._record_slow_query(op_name, response_time, args, kwargs)
                    
                    return result
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    self.record_performance(op_name, response_time, is_error=True)
                    logger.error(f"Performance monitor caught error in {op_name}: {e}")
                    raise
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def record_performance(self, operation: str, response_time: float, is_error: bool = False, cache_hit: bool = False):
        """Record performance metrics"""
        with self._lock:
            self.metrics[operation].update(response_time, is_error, cache_hit)
            
            # Add to history
            self.performance_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'operation': operation,
                'response_time': response_time,
                'is_error': is_error,
                'cache_hit': cache_hit
            })
            
            # Trim history
            if len(self.performance_history) > self.config['max_performance_history']:
                self.performance_history = self.performance_history[-self.config['max_performance_history']:]
    
    def _record_slow_query(self, operation: str, response_time: float, args: tuple, kwargs: dict):
        """Record slow query for analysis"""
        with self._lock:
            slow_query = {
                'timestamp': datetime.utcnow().isoformat(),
                'operation': operation,
                'response_time': response_time,
                'args_hash': hashlib.md5(str(args).encode()).hexdigest(),
                'kwargs_hash': hashlib.md5(str(kwargs).encode()).hexdigest()
            }
            
            self.slow_queries.append(slow_query)
            
            # Trim slow queries
            if len(self.slow_queries) > self.config['max_slow_queries_tracked']:
                self.slow_queries = self.slow_queries[-self.config['max_slow_queries_tracked']:]
            
            logger.warning(f"Slow operation detected: {operation} took {response_time:.2f}s")
    
    def get_metrics(self, operation: str = None) -> Dict:
        """Get performance metrics"""
        with self._lock:
            if operation:
                if operation in self.metrics:
                    return {
                        'operation': operation,
                        'metrics': self.metrics[operation].__dict__
                    }
                return {'operation': operation, 'metrics': None}
            
            return {
                'operations': {
                    op: metrics.__dict__ 
                    for op, metrics in self.metrics.items()
                },
                'summary': {
                    'total_operations': len(self.metrics),
                    'avg_response_time': sum(m.avg_response_time for m in self.metrics.values()) / len(self.metrics) if self.metrics else 0,
                    'total_requests': sum(m.request_count for m in self.metrics.values()),
                    'total_errors': sum(m.error_count for m in self.metrics.values()),
                    'avg_cache_hit_rate': sum(m.cache_hit_rate for m in self.metrics.values()) / len(self.metrics) if self.metrics else 0
                }
            }
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict]:
        """Get recent slow queries"""
        with self._lock:
            return self.slow_queries[-limit:]
    
    def optimize_performance(self):
        """Run performance optimizations"""
        logger.info("Running performance optimizations...")
        
        # Garbage collection if memory usage is high
        if self.config['enable_memory_monitoring']:
            self._optimize_memory()
        
        # Clear old metrics
        self._cleanup_old_metrics()
        
        # Log performance summary
        metrics = self.get_metrics()
        logger.info(f"Performance summary: {metrics.get('summary', {})}")
    
    def _optimize_memory(self):
        """Optimize memory usage"""
        import psutil
        
        try:
            # Get memory usage
            memory_percent = psutil.virtual_memory().percent / 100
            
            if memory_percent > self.config['gc_threshold']:
                logger.info(f"Memory usage {memory_percent:.2%} above threshold, running garbage collection")
                collected = gc.collect()
                logger.info(f"Garbage collection freed {collected} objects")
                
        except ImportError:
            # psutil not available, use basic GC
            collected = gc.collect()
            if collected > 0:
                logger.info(f"Garbage collection freed {collected} objects")
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        with self._lock:
            # Clean performance history
            self.performance_history = [
                entry for entry in self.performance_history
                if datetime.fromisoformat(entry['timestamp']) > cutoff_time
            ]
            
            # Clean slow queries
            self.slow_queries = [
                query for query in self.slow_queries
                if datetime.fromisoformat(query['timestamp']) > cutoff_time
            ]
    
    async def run_in_thread(self, func: Callable, *args, **kwargs):
        """Run CPU-bound operation in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
    
    def shutdown(self):
        """Cleanup resources"""
        self.thread_pool.shutdown(wait=True)


class CacheManager:
    """Advanced caching manager with TTL and LRU eviction"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self._cache = OrderedDict()
        self._ttl_cache = {}
        self._access_times = {}
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0,
            'deletes': 0
        }
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _default_config(self) -> Dict:
        """Default cache configuration"""
        return {
            'max_size': 1000,
            'default_ttl': 300,  # 5 minutes
            'cleanup_interval': 60,  # seconds
            'enable_compression': False,
            'compression_threshold': 1024,  # bytes
            'enable_stats': True
        }
    
    def _start_cleanup_task(self):
        """Start periodic cleanup task"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.config['cleanup_interval'])
                    self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        import threading
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _generate_key(self, key: str, **kwargs) -> str:
        """Generate cache key with optional parameters"""
        if kwargs:
            key_parts = [key] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
            return hashlib.md5(":".join(key_parts).encode()).hexdigest()
        return key
    
    def _compress_value(self, value: Any) -> bytes:
        """Compress value if enabled and worthwhile"""
        import pickle
        
        serialized = pickle.dumps(value)
        
        if (self.config['enable_compression'] and 
            len(serialized) > self.config['compression_threshold']):
            import zlib
            return zlib.compress(serialized)
        
        return serialized
    
    def _decompress_value(self, data: bytes) -> Any:
        """Decompress value"""
        import pickle
        
        if self.config['enable_compression']:
            try:
                import zlib
                decompressed = zlib.decompress(data)
                return pickle.loads(decompressed)
            except:
                # Fallback to uncompressed
                return pickle.loads(data)
        
        return pickle.loads(data)
    
    def _cleanup_expired(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, expiry_time in self._ttl_cache.items():
                if current_time > expiry_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                if key in self._cache:
                    del self._cache[key]
                del self._ttl_cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                
                if self.config['enable_stats']:
                    self._stats['evictions'] += 1
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _evict_lru(self):
        """Evict least recently used item"""
        with self._lock:
            if self._cache:
                # Remove oldest item (LRU)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                
                # Clean up related data
                if oldest_key in self._ttl_cache:
                    del self._ttl_cache[oldest_key]
                if oldest_key in self._access_times:
                    del self._access_times[oldest_key]
                
                if self.config['enable_stats']:
                    self._stats['evictions'] += 1
    
    def get(self, key: str, default: Any = None, **kwargs) -> Any:
        """Get value from cache"""
        cache_key = self._generate_key(key, **kwargs)
        current_time = time.time()
        
        with self._lock:
            # Check if key exists and not expired
            if cache_key in self._cache:
                if cache_key in self._ttl_cache:
                    if current_time > self._ttl_cache[cache_key]:
                        # Expired
                        del self._cache[cache_key]
                        del self._ttl_cache[cache_key]
                        if cache_key in self._access_times:
                            del self._access_times[cache_key]
                        
                        if self.config['enable_stats']:
                            self._stats['misses'] += 1
                        return default
                
                # Move to end (mark as recently used)
                value = self._cache.pop(cache_key)
                self._cache[cache_key] = value
                self._access_times[cache_key] = current_time
                
                if self.config['enable_stats']:
                    self._stats['hits'] += 1
                
                return self._decompress_value(value)
            
            if self.config['enable_stats']:
                self._stats['misses'] += 1
            
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, **kwargs):
        """Set value in cache"""
        cache_key = self._generate_key(key, **kwargs)
        current_time = time.time()
        ttl = ttl or self.config['default_ttl']
        
        with self._lock:
            # Check if we need to evict
            if len(self._cache) >= self.config['max_size']:
                self._evict_lru()
            
            # Store compressed value
            compressed_value = self._compress_value(value)
            self._cache[cache_key] = compressed_value
            self._ttl_cache[cache_key] = current_time + ttl
            self._access_times[cache_key] = current_time
            
            if self.config['enable_stats']:
                self._stats['sets'] += 1
    
    def delete(self, key: str, **kwargs):
        """Delete value from cache"""
        cache_key = self._generate_key(key, **kwargs)
        
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                
                if cache_key in self._ttl_cache:
                    del self._ttl_cache[cache_key]
                if cache_key in self._access_times:
                    del self._access_times[cache_key]
                
                if self.config['enable_stats']:
                    self._stats['deletes'] += 1
                
                return True
            
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._ttl_cache.clear()
            self._access_times.clear()
            
            if self.config['enable_stats']:
                self._stats = {
                    'hits': 0,
                    'misses': 0,
                    'evictions': 0,
                    'sets': 0,
                    'deletes': 0
                }
    
    def exists(self, key: str, **kwargs) -> bool:
        """Check if key exists in cache"""
        cache_key = self._generate_key(key, **kwargs)
        current_time = time.time()
        
        with self._lock:
            if cache_key in self._cache:
                if cache_key in self._ttl_cache:
                    if current_time > self._ttl_cache[cache_key]:
                        # Expired
                        del self._cache[cache_key]
                        del self._ttl_cache[cache_key]
                        if cache_key in self._access_times:
                            del self._access_times[cache_key]
                        return False
                
                return True
            
            return False
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.config['max_size'],
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'sets': self._stats['sets'],
                'deletes': self._stats['deletes']
            }
    
    def get_info(self) -> Dict:
        """Get detailed cache information"""
        with self._lock:
            return {
                'config': self.config,
                'stats': self.get_stats(),
                'cache_keys': list(self._cache.keys())[:10],  # First 10 keys
                'memory_usage': len(str(self._cache).encode()),  # Rough estimate
            }
    
    def cache_function(self, ttl: Optional[int] = None, key_prefix: str = "func"):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                func_name = f"{func.__module__}.{func.__name__}"
                cache_key = f"{key_prefix}:{func_name}:{hashlib.md5(str(args + tuple(kwargs.items())).encode()).hexdigest()}"
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                self.set(cache_key, result, ttl)
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Generate cache key
                func_name = f"{func.__module__}.{func.__name__}"
                cache_key = f"{key_prefix}:{func_name}:{hashlib.md5(str(args + tuple(kwargs.items())).encode()).hexdigest()}"
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                self.set(cache_key, result, ttl)
                
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator


# Global instances
performance_optimizer = PerformanceOptimizer()
cache_manager = CacheManager()


def cached(ttl: Optional[int] = None, key_prefix: str = "func"):
    """Simple caching decorator"""
    return cache_manager.cache_function(ttl, key_prefix)


def monitored(operation_name: str = None):
    """Simple performance monitoring decorator"""
    return performance_optimizer.performance_monitor(operation_name)