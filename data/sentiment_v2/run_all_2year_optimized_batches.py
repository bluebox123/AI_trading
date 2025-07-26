#!/usr/bin/env python3
"""
Master Coordinator for Aggressive 2-Year Optimized Parallel Sentiment Analysis
============================================================================

This script coordinates all 4 aggressive 2-year optimized parallel sentiment analysis batches
with comprehensive monitoring, reporting, and error handling.
"""

import sys
import os
from pathlib import Path
from datetime import date, datetime, timedelta
import time
import logging
from typing import List, Dict, Any
import json
import pandas as pd
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import psutil

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

class Aggressive2YearMasterCoordinator:
    """Master coordinator for aggressive 2-year optimized parallel sentiment analysis"""
    
    def __init__(self):
        """Initialize the master coordinator"""
        self.setup_logging()
        self.logger.info("Initializing Aggressive 2-Year Master Coordinator")
        
        # Configuration
        self.batch_scripts = [
            "run_2year_optimized_1.py",
            "run_2year_optimized_2.py", 
            "run_2year_optimized_3.py",
            "run_2year_optimized_4.py"
        ]
        
        self.batch_names = [
            "Batch 1 (Stocks 1-28)",
            "Batch 2 (Stocks 29-56)", 
            "Batch 3 (Stocks 57-84)",
            "Batch 4 (Stocks 85-112)"
        ]
        
        # Date range: January 1st, 2024 to December 31st, 2025 (2 years)
        self.start_date = date(2024, 1, 1)
        self.end_date = date(2025, 12, 31)
        
        # Create output directory
        self.output_dir = Path("2year_master_coordinator_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.batch_results = {}
        self.batch_processes = {}
        self.start_time = None
        self.completed_batches = 0
        self.failed_batches = 0
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info(f"Date range: {self.start_date} to {self.end_date} (2 years)")
        self.logger.info(f"Total batches: {len(self.batch_scripts)}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = f"2year_master_coordinator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("aggressive_2year_master_coordinator")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_all_batches()
        sys.exit(0)
    
    def shutdown_all_batches(self):
        """Shutdown all running batch processes"""
        self.logger.info("Shutting down all batch processes...")
        for batch_name, process in self.batch_processes.items():
            if process and process.poll() is None:
                self.logger.info(f"Terminating {batch_name}...")
                try:
                    process.terminate()
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Force killing {batch_name}...")
                    process.kill()
                except Exception as e:
                    self.logger.error(f"Error terminating {batch_name}: {e}")
    
    def calculate_total_workload(self) -> Dict[str, Any]:
        """Calculate the total workload for all 4 batches"""
        total_days = (self.end_date - self.start_date).days + 1
        total_stocks = 112  # 28 stocks per batch * 4 batches
        total_requests = total_days * total_stocks
        
        # Estimate processing time (with aggressive rate limiting)
        base_delay = 0.8  # Aggressive base delay
        estimated_hours = (total_requests * base_delay) / 3600
        
        # Account for parallel processing and efficiency gains
        efficiency_factor = 0.6  # 40% efficiency gain from parallel processing
        estimated_hours *= efficiency_factor
        
        return {
            "total_days": total_days,
            "total_stocks": total_stocks,
            "total_requests": total_requests,
            "estimated_hours": estimated_hours,
            "estimated_hours_per_batch": estimated_hours / 4,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "parallel_batches": len(self.batch_scripts)
        }
    
    def run_batch_script(self, script_path: str, batch_name: str) -> Dict[str, Any]:
        """Run a single batch script and return results"""
        self.logger.info(f"Starting {batch_name} with script: {script_path}")
        
        try:
            # Run the batch script
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path(__file__).parent
            )
            
            # Store process for potential shutdown
            self.batch_processes[batch_name] = process
            
            # Monitor the process
            stdout, stderr = process.communicate()
            return_code = process.returncode
            
            # Remove from active processes
            if batch_name in self.batch_processes:
                del self.batch_processes[batch_name]
            
            if return_code == 0:
                self.logger.info(f"✅ {batch_name} completed successfully")
                self.completed_batches += 1
                return {
                    "batch_name": batch_name,
                    "status": "completed",
                    "return_code": return_code,
                    "stdout": stdout,
                    "stderr": stderr
                }
            else:
                self.logger.error(f"❌ {batch_name} failed with return code {return_code}")
                self.failed_batches += 1
                return {
                    "batch_name": batch_name,
                    "status": "failed",
                    "return_code": return_code,
                    "stdout": stdout,
                    "stderr": stderr
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error running {batch_name}: {e}")
            self.failed_batches += 1
            return {
                "batch_name": batch_name,
                "status": "error",
                "error": str(e)
            }
    
    def run_parallel_analysis(self):
        """Run all 4 batches in parallel"""
        workload = self.calculate_total_workload()
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING AGGRESSIVE 2-YEAR MASTER COORDINATOR")
        self.logger.info("=" * 80)
        self.logger.info(f"Total days: {workload['total_days']}")
        self.logger.info(f"Total stocks: {workload['total_stocks']}")
        self.logger.info(f"Total requests: {workload['total_requests']:,}")
        self.logger.info(f"Estimated time: {workload['estimated_hours']:.1f} hours total")
        self.logger.info(f"Estimated time per batch: {workload['estimated_hours_per_batch']:.1f} hours")
        self.logger.info(f"Parallel batches: {workload['parallel_batches']}")
        self.logger.info("=" * 80)
        
        self.start_time = datetime.now()
        
        try:
            # Run all batches in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all batch jobs
                future_to_batch = {
                    executor.submit(self.run_batch_script, script, name): name
                    for script, name in zip(self.batch_scripts, self.batch_names)
                }
                
                # Monitor and collect results
                for future in as_completed(future_to_batch):
                    batch_name = future_to_batch[future]
                    try:
                        result = future.result()
                        self.batch_results[batch_name] = result
                        
                        # Log progress
                        progress = (self.completed_batches + self.failed_batches) / len(self.batch_scripts) * 100
                        self.logger.info(f"Progress: {progress:.1f}% ({self.completed_batches} completed, {self.failed_batches} failed)")
                        
                    except Exception as e:
                        self.logger.error(f"Error in {batch_name}: {e}")
                        self.batch_results[batch_name] = {
                            "batch_name": batch_name,
                            "status": "error",
                            "error": str(e)
                        }
                        self.failed_batches += 1
            
            # Generate final report
            self.generate_final_report(workload)
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
            self.shutdown_all_batches()
            raise
        except Exception as e:
            self.logger.error(f"Error in parallel analysis: {e}")
            self.shutdown_all_batches()
            raise
    
    def generate_final_report(self, workload: Dict[str, Any]):
        """Generate comprehensive final report"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Calculate success rate
        total_batches = len(self.batch_scripts)
        success_rate = (self.completed_batches / total_batches) * 100 if total_batches > 0 else 0
        
        # System performance metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        final_report = {
            "master_coordinator_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_hours": duration.total_seconds() / 3600,
                "duration_minutes": duration.total_seconds() / 60,
                "total_batches": total_batches,
                "completed_batches": self.completed_batches,
                "failed_batches": self.failed_batches,
                "success_rate": success_rate
            },
            "workload_analysis": workload,
            "batch_results": self.batch_results,
            "system_performance": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3)
            },
            "date_range": {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "total_days": workload['total_days']
            }
        }
        
        # Save final report
        report_file = self.output_dir / "2year_master_coordinator_final_report.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        # Print summary
        self.print_final_summary(final_report)
        
        return final_report
    
    def print_final_summary(self, report: Dict[str, Any]):
        """Print comprehensive final summary"""
        summary = report['master_coordinator_summary']
        workload = report['workload_analysis']
        
        self.logger.info("")
        self.logger.info("🎯 AGGRESSIVE 2-YEAR MASTER COORDINATOR FINAL SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"⏱️  Total Duration: {summary['duration_hours']:.2f} hours ({summary['duration_minutes']:.1f} minutes)")
        self.logger.info(f"📊 Success Rate: {summary['success_rate']:.1f}% ({summary['completed_batches']}/{summary['total_batches']} batches)")
        self.logger.info(f"📈 Total Stocks: {workload['total_stocks']}")
        self.logger.info(f"📅 Date Range: {self.start_date} to {self.end_date}")
        self.logger.info(f"📊 Total Days: {workload['total_days']}")
        self.logger.info(f"🔄 Total Requests: {workload['total_requests']:,}")
        self.logger.info(f"⚡ Estimated vs Actual: {workload['estimated_hours']:.1f}h vs {summary['duration_hours']:.1f}h")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
        self.logger.info("")
        
        # Batch-specific results
        self.logger.info("📋 BATCH RESULTS:")
        for batch_name, result in self.batch_results.items():
            status_emoji = "✅" if result['status'] == 'completed' else "❌"
            self.logger.info(f"   {status_emoji} {batch_name}: {result['status']}")
        
        self.logger.info("")
        self.logger.info("🖥️  SYSTEM PERFORMANCE:")
        sys_perf = report['system_performance']
        self.logger.info(f"   • CPU Usage: {sys_perf['cpu_percent']:.1f}%")
        self.logger.info(f"   • Memory Usage: {sys_perf['memory_percent']:.1f}%")
        self.logger.info(f"   • Available Memory: {sys_perf['memory_available_gb']:.1f} GB")
        self.logger.info("=" * 80)

def main():
    """Main function to run the aggressive 2-year master coordinator"""
    try:
        coordinator = Aggressive2YearMasterCoordinator()
        report = coordinator.run_parallel_analysis()
        
        print("\n" + "="*80)
        print("🎯 AGGRESSIVE 2-YEAR MASTER COORDINATOR COMPLETED!")
        print("="*80)
        print(f"📊 Duration: {report['master_coordinator_summary']['duration_hours']:.2f} hours")
        print(f"📈 Success Rate: {report['master_coordinator_summary']['success_rate']:.1f}%")
        print(f"📁 Output directory: {coordinator.output_dir}")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error in master coordinator: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 