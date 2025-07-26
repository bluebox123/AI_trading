"""
System Monitor for AI Trading System

This module provides basic system monitoring, performance tracking,
and health checks for the institutional-grade trading system.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import threading
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class SystemMonitor:
    """
    Basic system monitoring with alerting and performance tracking.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the system monitor"""
        self.config = config or {}
        self.monitoring_interval = self.config.get('monitoring_interval', 30)
        self.enable_alerts = self.config.get('enable_alerts', True)
        
        # Performance counters
        self.api_request_counter = defaultdict(int)
        self.error_counter = defaultdict(int)
        self.response_times = deque(maxlen=1000)
        
        # System info
        self.start_time = time.time()
        self.monitoring_active = False
        self.monitor_thread = None
        
        logger.info("System Monitor initialized")
    
    def start_monitoring(self):
        """Start the background monitoring thread"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop the background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Basic monitoring (simplified for now)
                self._cleanup_counters()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def record_api_request(self, response_time_ms: float = None, is_error: bool = False):
        """Record an API request for metrics"""
        current_minute = int(time.time() / 60)
        self.api_request_counter[current_minute] += 1
        
        if is_error:
            self.error_counter[current_minute] += 1
        
        if response_time_ms is not None:
            self.response_times.append(response_time_ms)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health summary"""
        current_minute = int(time.time() / 60)
        
        # API requests per minute
        api_requests = sum(self.api_request_counter[minute] 
                         for minute in range(current_minute - 1, current_minute + 1))
        
        # Error rate calculation
        total_requests = sum(self.api_request_counter.values())
        total_errors = sum(self.error_counter.values())
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
        
        # Average response time
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        status = 'HEALTHY'
        issues = []
        
        if error_rate > 15:
            status = 'CRITICAL'
            issues.append(f"High error rate: {error_rate:.1f}%")
        elif error_rate > 5:
            status = 'WARNING'
            issues.append(f"Elevated error rate: {error_rate:.1f}%")
        
        return {
            'status': status,
            'issues': issues,
            'timestamp': datetime.now().isoformat(),
            'uptime_hours': round((time.time() - self.start_time) / 3600, 2),
            'monitoring_active': self.monitoring_active,
            'api_requests_per_minute': api_requests,
            'error_rate_percent': round(error_rate, 2),
            'avg_response_time_ms': round(avg_response_time, 2)
        }
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the specified time period"""
        current_minute = int(time.time() / 60)
        cutoff_minute = current_minute - (hours * 60)
        
        # Filter recent data
        recent_requests = sum(self.api_request_counter[minute] 
                            for minute in self.api_request_counter 
                            if minute > cutoff_minute)
        
        recent_errors = sum(self.error_counter[minute] 
                          for minute in self.error_counter 
                          if minute > cutoff_minute)
        
        error_rate = (recent_errors / recent_requests * 100) if recent_requests > 0 else 0
        
        # Mock system performance metrics
        return {
            'time_period_hours': hours,
            'total_requests': recent_requests,
            'total_errors': recent_errors,
            'error_rate_percent': round(error_rate, 2),
            'system_performance': {
                'cpu': {
                    'avg': 15.0,  # Mock CPU usage
                    'max': 25.0,
                    'min': 5.0
                },
                'memory': {
                    'avg': 45.0,  # Mock memory usage
                    'max': 55.0,
                    'min': 35.0
                }
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _cleanup_counters(self):
        """Clean up old counter data"""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep 1 hour of data
        
        # Clean up minute-based counters
        cutoff_minute = int(cutoff_time / 60)
        for minute in list(self.api_request_counter.keys()):
            if minute < cutoff_minute:
                del self.api_request_counter[minute]
                
        for minute in list(self.error_counter.keys()):
            if minute < cutoff_minute:
                del self.error_counter[minute]


# Global system monitor instance
system_monitor = SystemMonitor()

# Example usage
if __name__ == "__main__":
    # Initialize and start monitoring
    monitor = SystemMonitor()
    
    print("🚀 Starting System Monitor...")
    monitor.start_monitoring()
    
    # Simulate some activity
    for i in range(5):
        monitor.record_api_request(response_time_ms=50 + i * 10)
        time.sleep(1)
    
    # Get health status
    health = monitor.get_system_health()
    print(f"\n📊 System Health: {health['status']}")
    print(f"   Uptime: {health['uptime_hours']} hours")
    print(f"   API Requests/min: {health['api_requests_per_minute']}")
    print(f"   Error Rate: {health['error_rate_percent']}%")
    
    monitor.stop_monitoring()
    print("\n✅ System Monitor testing completed!") 