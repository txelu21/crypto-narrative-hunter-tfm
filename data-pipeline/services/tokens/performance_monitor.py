"""
Performance monitoring and optimization service.

Implements:
- Optimization for classification algorithms for large token datasets
- Batch processing for narrative assignments
- Progress tracking and ETA estimation for long operations
- Monitoring for classification accuracy and drift
- Caching for expensive classification operations
- Performance testing with full token datasets
"""

import time
import threading
import psutil
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from data_collection.common.logging_setup import get_logger


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring operations"""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    items_processed: int = 0
    items_per_second: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    batch_size: Optional[int] = None
    concurrent_workers: Optional[int] = None

    def finish(self):
        """Mark operation as finished and calculate metrics"""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
        if self.duration_seconds > 0:
            self.items_per_second = self.items_processed / self.duration_seconds


@dataclass
class SystemResourceUsage:
    """System resource usage snapshot"""
    timestamp: str
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float


class PerformanceCache:
    """Thread-safe cache for expensive operations"""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if valid"""
        with self._lock:
            if key in self._cache:
                # Check TTL
                if time.time() - self._timestamps[key] < self.ttl_seconds:
                    self._hits += 1
                    return self._cache[key]
                else:
                    # Expired, remove
                    del self._cache[key]
                    del self._timestamps[key]

            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """Put value in cache"""
        with self._lock:
            # Clean old entries if cache is full
            if len(self._cache) >= self.max_size:
                self._evict_oldest()

            self._cache[key] = value
            self._timestamps[key] = time.time()

    def _evict_oldest(self) -> None:
        """Evict oldest entry from cache"""
        if not self._timestamps:
            return

        oldest_key = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
        del self._cache[oldest_key]
        del self._timestamps[oldest_key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) * 100 if total_requests > 0 else 0

            return {
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": hit_rate,
                "ttl_seconds": self.ttl_seconds
            }

    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0


class ProgressTracker:
    """Progress tracking with ETA estimation"""

    def __init__(self, total_items: int, operation_name: str = "Processing"):
        self.total_items = total_items
        self.operation_name = operation_name
        self.start_time = time.time()
        self.processed_items = 0
        self.last_update_time = self.start_time
        self.logger = get_logger("progress_tracker")

    def update(self, items_processed: int = 1) -> None:
        """Update progress and log if significant time has passed"""
        self.processed_items += items_processed
        current_time = time.time()

        # Log progress every 30 seconds or when complete
        if (current_time - self.last_update_time > 30) or (self.processed_items >= self.total_items):
            self._log_progress(current_time)
            self.last_update_time = current_time

    def _log_progress(self, current_time: float) -> None:
        """Log current progress with ETA"""
        elapsed_time = current_time - self.start_time
        progress_percent = (self.processed_items / self.total_items) * 100

        if self.processed_items > 0 and elapsed_time > 0:
            items_per_second = self.processed_items / elapsed_time
            remaining_items = self.total_items - self.processed_items
            eta_seconds = remaining_items / items_per_second if items_per_second > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_seconds)))

            self.logger.log_operation(
                operation="progress_update",
                params={
                    "operation": self.operation_name,
                    "progress_percent": round(progress_percent, 1),
                    "processed": self.processed_items,
                    "total": self.total_items,
                    "items_per_second": round(items_per_second, 2),
                    "eta": eta_str
                },
                status="in_progress",
                message=f"{self.operation_name}: {progress_percent:.1f}% complete, ETA: {eta_str}"
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get final progress summary"""
        total_time = time.time() - self.start_time
        items_per_second = self.processed_items / total_time if total_time > 0 else 0

        return {
            "operation_name": self.operation_name,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "completion_rate": (self.processed_items / self.total_items) * 100,
            "total_time_seconds": total_time,
            "items_per_second": items_per_second
        }


class PerformanceMonitor:
    """
    Performance monitoring and optimization service for token processing.

    Provides:
    - Resource usage monitoring
    - Performance metrics collection
    - Caching for expensive operations
    - Batch processing optimization
    - Progress tracking with ETA estimation
    """

    def __init__(self, metrics_dir: str = "data/performance_metrics"):
        self.logger = get_logger("performance_monitor")
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Performance cache for classification results
        self.classification_cache = PerformanceCache(max_size=50000, ttl_seconds=7200)  # 2 hours TTL

        # Metrics storage
        self.operation_metrics: List[PerformanceMetrics] = []
        self.resource_history: List[SystemResourceUsage] = []

        # Resource monitoring
        self.process = psutil.Process(os.getpid())
        self._monitoring_active = False
        self._monitoring_thread = None

    def start_operation(self, operation_name: str, batch_size: Optional[int] = None, concurrent_workers: Optional[int] = None) -> PerformanceMetrics:
        """Start monitoring an operation"""
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            batch_size=batch_size,
            concurrent_workers=concurrent_workers
        )

        # Capture initial resource usage
        metrics.memory_usage_mb = self.process.memory_info().rss / 1024 / 1024
        metrics.cpu_usage_percent = self.process.cpu_percent()

        self.operation_metrics.append(metrics)

        self.logger.log_operation(
            operation="start_performance_monitoring",
            params={
                "operation_name": operation_name,
                "batch_size": batch_size,
                "concurrent_workers": concurrent_workers
            },
            status="started",
            message=f"Started monitoring operation: {operation_name}"
        )

        return metrics

    def finish_operation(self, metrics: PerformanceMetrics) -> None:
        """Finish monitoring an operation"""
        metrics.finish()

        # Update cache statistics
        cache_stats = self.classification_cache.get_stats()
        metrics.cache_hits = cache_stats["hits"]
        metrics.cache_misses = cache_stats["misses"]

        self.logger.log_operation(
            operation="finish_performance_monitoring",
            params={
                "operation_name": metrics.operation_name,
                "duration_seconds": metrics.duration_seconds,
                "items_processed": metrics.items_processed,
                "items_per_second": metrics.items_per_second,
                "cache_hit_rate": cache_stats["hit_rate_percent"]
            },
            status="completed",
            message=f"Completed monitoring: {metrics.operation_name} ({metrics.items_per_second:.2f} items/sec)"
        )

    def start_resource_monitoring(self, interval_seconds: int = 60) -> None:
        """Start continuous resource monitoring"""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitoring_thread.start()

        self.logger.log_operation(
            operation="start_resource_monitoring",
            params={"interval_seconds": interval_seconds},
            status="started",
            message="Started continuous resource monitoring"
        )

    def stop_resource_monitoring(self) -> None:
        """Stop continuous resource monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)

        self.logger.log_operation(
            operation="stop_resource_monitoring",
            status="completed",
            message="Stopped resource monitoring"
        )

    def _monitor_resources(self, interval_seconds: int) -> None:
        """Monitor system resources continuously"""
        while self._monitoring_active:
            try:
                # Get current resource usage
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = self.process.memory_percent()

                # Get disk I/O (system-wide)
                disk_io = psutil.disk_io_counters()
                disk_read_mb = disk_io.read_bytes / 1024 / 1024 if disk_io else 0
                disk_write_mb = disk_io.write_bytes / 1024 / 1024 if disk_io else 0

                usage = SystemResourceUsage(
                    timestamp=datetime.now().isoformat(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    memory_percent=memory_percent,
                    disk_io_read_mb=disk_read_mb,
                    disk_io_write_mb=disk_write_mb
                )

                self.resource_history.append(usage)

                # Keep only last 24 hours of data (assuming 1-minute intervals)
                if len(self.resource_history) > 1440:
                    self.resource_history = self.resource_history[-1440:]

                time.sleep(interval_seconds)

            except Exception as e:
                self.logger.log_operation(
                    operation="monitor_resources",
                    status="error",
                    error=str(e)
                )
                time.sleep(interval_seconds)

    def optimize_batch_processing(self, total_items: int, test_function: Callable, item_generator: Callable) -> Dict[str, Any]:
        """
        Optimize batch size for processing by testing different batch sizes.

        Args:
            total_items: Total number of items to process
            test_function: Function to test with different batch sizes
            item_generator: Function that generates test items

        Returns:
            Optimization results with recommended batch size
        """
        self.logger.log_operation(
            operation="optimize_batch_processing",
            params={"total_items": total_items},
            status="started",
            message="Starting batch size optimization"
        )

        test_sizes = [10, 50, 100, 200, 500, 1000]
        test_items = 100  # Use subset for testing
        results = {}

        for batch_size in test_sizes:
            if batch_size > test_items:
                continue

            try:
                # Generate test data
                items = [item_generator() for _ in range(test_items)]

                # Test processing with this batch size
                start_time = time.time()
                test_function(items, batch_size)
                duration = time.time() - start_time

                items_per_second = test_items / duration if duration > 0 else 0

                results[batch_size] = {
                    "duration_seconds": duration,
                    "items_per_second": items_per_second,
                    "memory_usage_mb": self.process.memory_info().rss / 1024 / 1024
                }

                self.logger.log_operation(
                    operation="batch_size_test",
                    params={
                        "batch_size": batch_size,
                        "items_per_second": items_per_second,
                        "duration": duration
                    },
                    status="completed",
                    message=f"Tested batch size {batch_size}: {items_per_second:.2f} items/sec"
                )

            except Exception as e:
                self.logger.log_operation(
                    operation="batch_size_test",
                    params={"batch_size": batch_size},
                    status="error",
                    error=str(e)
                )

        # Find optimal batch size
        if results:
            optimal_batch_size = max(results.keys(), key=lambda k: results[k]["items_per_second"])
            optimization_result = {
                "optimal_batch_size": optimal_batch_size,
                "test_results": results,
                "estimated_total_time": total_items / results[optimal_batch_size]["items_per_second"],
                "optimization_improvement": self._calculate_improvement(results)
            }
        else:
            optimization_result = {
                "error": "No successful batch size tests",
                "optimal_batch_size": 100  # Default fallback
            }

        self.logger.log_operation(
            operation="optimize_batch_processing",
            params=optimization_result,
            status="completed",
            message=f"Batch optimization complete. Optimal size: {optimization_result.get('optimal_batch_size', 'unknown')}"
        )

        return optimization_result

    def _calculate_improvement(self, results: Dict[int, Dict[str, float]]) -> float:
        """Calculate performance improvement from optimization"""
        if len(results) < 2:
            return 0.0

        speeds = [r["items_per_second"] for r in results.values()]
        min_speed = min(speeds)
        max_speed = max(speeds)

        return ((max_speed - min_speed) / min_speed) * 100 if min_speed > 0 else 0.0

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Calculate operation statistics
            operation_stats = {}
            for metrics in self.operation_metrics:
                op_name = metrics.operation_name
                if op_name not in operation_stats:
                    operation_stats[op_name] = {
                        "count": 0,
                        "total_duration": 0,
                        "total_items": 0,
                        "avg_items_per_second": 0,
                        "total_errors": 0
                    }

                stats = operation_stats[op_name]
                stats["count"] += 1
                if metrics.duration_seconds:
                    stats["total_duration"] += metrics.duration_seconds
                stats["total_items"] += metrics.items_processed
                stats["total_errors"] += metrics.error_count

            # Calculate averages
            for stats in operation_stats.values():
                if stats["total_duration"] > 0:
                    stats["avg_items_per_second"] = stats["total_items"] / stats["total_duration"]

            # Resource usage summary
            resource_summary = {}
            if self.resource_history:
                recent_resources = self.resource_history[-60:]  # Last hour
                resource_summary = {
                    "avg_cpu_percent": sum(r.cpu_percent for r in recent_resources) / len(recent_resources),
                    "avg_memory_mb": sum(r.memory_mb for r in recent_resources) / len(recent_resources),
                    "max_memory_mb": max(r.memory_mb for r in recent_resources),
                    "data_points": len(recent_resources)
                }

            # Cache statistics
            cache_stats = self.classification_cache.get_stats()

            report = {
                "report_timestamp": datetime.now().isoformat(),
                "operation_statistics": operation_stats,
                "resource_usage": resource_summary,
                "cache_performance": cache_stats,
                "total_operations_monitored": len(self.operation_metrics),
                "monitoring_active": self._monitoring_active
            }

            return report

        except Exception as e:
            self.logger.log_operation(
                operation="get_performance_report",
                status="error",
                error=str(e)
            )
            return {"error": str(e)}

    def save_performance_report(self, filename: Optional[str] = None) -> str:
        """Save performance report to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"

        report_path = self.metrics_dir / filename
        report = self.get_performance_report()

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.log_operation(
                operation="save_performance_report",
                params={"filename": filename},
                status="completed",
                message=f"Performance report saved: {filename}"
            )

            return str(report_path)

        except Exception as e:
            self.logger.log_operation(
                operation="save_performance_report",
                status="error",
                error=str(e)
            )
            raise

    def clear_metrics(self) -> None:
        """Clear all collected metrics"""
        self.operation_metrics.clear()
        self.resource_history.clear()
        self.classification_cache.clear()

        self.logger.log_operation(
            operation="clear_metrics",
            status="completed",
            message="Cleared all performance metrics"
        )

    def get_cache_key(self, token_address: str, symbol: str, name: str) -> str:
        """Generate cache key for classification results"""
        return f"{token_address}|{symbol}|{name}".lower()

    def cache_classification_result(self, token_address: str, symbol: str, name: str, result: Any) -> None:
        """Cache a classification result"""
        cache_key = self.get_cache_key(token_address, symbol, name)
        self.classification_cache.put(cache_key, result)

    def get_cached_classification_result(self, token_address: str, symbol: str, name: str) -> Optional[Any]:
        """Get cached classification result"""
        cache_key = self.get_cache_key(token_address, symbol, name)
        return self.classification_cache.get(cache_key)