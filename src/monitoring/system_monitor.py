#!/usr/bin/env python3
"""
System Monitor - Monitors the health and performance of the trading system.

This module provides tools for monitoring the health and performance
of the trading system, including component status, resource usage,
error tracking, and performance metrics.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import threading
import time
import psutil
import socket
import platform
from collections import deque
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemMonitor:
    def __init__(self):
        """Initialize the System Monitor"""
        self.data_dir = Path('data')
        self.monitoring_dir = self.data_dir / "monitoring"
        self._create_directories()
        logger.info("System Monitor initialized")
    
    def _create_directories(self):
        """Create necessary data directories"""
        self.data_dir.mkdir(exist_ok=True)
        self.monitoring_dir.mkdir(exist_ok=True)
        logger.debug("Created monitoring directories")
    
    def get_system_status(self):
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "resources": {
                    "cpu": psutil.cpu_percent(),
                    "memory": psutil.virtual_memory().percent,
                    "disk": psutil.disk_usage('/').percent
                }
            }
        except Exception as e:
            print(f"Error getting system status: {str(e)}")
            return {"status": "error"} 