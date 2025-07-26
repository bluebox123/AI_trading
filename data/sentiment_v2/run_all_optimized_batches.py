#!/usr/bin/env python3
"""
Master Script to Run All 4 Optimized Parallel Sentiment Analysis Batches
=======================================================================

This script coordinates the execution of all 4 optimized parallel sentiment analysis batches
with improved rate limiting to avoid API throttling and slowdowns.

Batch Distribution:
- Batch 1: Stocks 1-28 (28 stocks)
- Batch 2: Stocks 29-56 (28 stocks) 
- Batch 3: Stocks 57-84 (28 stocks)
- Batch 4: Stocks 85-110 (26 stocks)

Total: 110 stocks across 4 batches
Expected Duration: ~18-20 hours per batch (instead of 96 hours total)
"""

import sys
import os
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import json

class OptimizedBatchCoordinator:
    """Coordinates the execution of all 4 optimized parallel sentiment analysis batches"""
    
    def __init__(self):
        """Initialize the batch coordinator"""
        self.setup_logging()
        self.logger.info("Initializing Optimized Batch Coordinator")
        
        # Batch configuration
        self.batches = [
            {
                "name": "Batch 1",
                "script": "run_parallel_optimized_1.py",
                "stocks": "1-28",
                "count": 28
            },
            {
                "name": "Batch 2", 
                "script": "run_parallel_optimized_2.py",
                "stocks": "29-56",
                "count": 28
            },
            {
                "name": "Batch 3",
                "script": "run_parallel_optimized_3.py", 
                "stocks": "57-84",
                "count": 28
            },
            {
                "name": "Batch 4",
                "script": "run_parallel_optimized_4.py",
                "stocks": "85-110", 
                "count": 26
            }
        ]
        
        self.total_stocks = sum(batch["count"] for batch in self.batches)
        self.expected_duration_per_batch = 20  # hours
        self.total_expected_duration = self.expected_duration_per_batch * len(self.batches)
        
        self.logger.info(f"Total stocks: {self.total_stocks}")
        self.logger.info(f"Expected duration per batch: {self.expected_duration_per_batch} hours")
        self.logger.info(f"Total expected duration: {self.total_expected_duration} hours")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = f"optimized_batch_coordinator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("optimized_batch_coordinator")
    
    def print_batch_summary(self):
        """Print a summary of all batches"""
        print("\n" + "=" * 80)
        print("OPTIMIZED PARALLEL SENTIMENT ANALYSIS - BATCH SUMMARY")
        print("=" * 80)
        
        print(f"📊 Batch Distribution:")
        for i, batch in enumerate(self.batches, 1):
            print(f"   • {batch['name']}: Stocks {batch['stocks']} ({batch['count']} stocks)")
        
        print(f"\n⏱️ Performance Expectations:")
        print(f"   • Total Stocks: {self.total_stocks}")
        print(f"   • Duration per Batch: ~{self.expected_duration_per_batch} hours")
        print(f"   • Total Duration: ~{self.total_expected_duration} hours")
        print(f"   • Improvement: From 96 hours to ~{self.total_expected_duration} hours")
        
        print(f"\n🚀 Optimization Features:")
        print(f"   • Rate Limiting: 3s base delay, 10s max delay")
        print(f"   • Exponential Backoff: Smart retry logic")
        print(f"   • Jitter: Random delays to avoid thundering herd")
        print(f"   • Reduced Competition: 4 batches instead of 7")
        
        print(f"\n📁 Output Directories:")
        for batch in self.batches:
            batch_num = batch['name'].split()[-1]
            print(f"   • {batch['name']}: parallel_sentiment_batch_{batch_num}/")
        
        print("=" * 80)
    
    def run_batch(self, batch: Dict[str, Any]) -> bool:
        """Run a single batch"""
        script_path = Path(__file__).parent / batch["script"]
        
        if not script_path.exists():
            self.logger.error(f"Script not found: {script_path}")
            return False
        
        self.logger.info(f"Starting {batch['name']}...")
        self.logger.info(f"Script: {script_path}")
        self.logger.info(f"Stocks: {batch['stocks']} ({batch['count']} stocks)")
        
        try:
            # Run the batch script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=None  # No timeout - let it run as long as needed
            )
            
            if result.returncode == 0:
                self.logger.info(f"{batch['name']} completed successfully")
                self.logger.info(f"Output: {result.stdout}")
                return True
            else:
                self.logger.error(f"{batch['name']} failed with return code {result.returncode}")
                self.logger.error(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"{batch['name']} timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error running {batch['name']}: {e}")
            return False
    
    def run_all_batches_sequential(self):
        """Run all batches sequentially (recommended to avoid rate limiting)"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING SEQUENTIAL BATCH EXECUTION")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        results = []
        
        for i, batch in enumerate(self.batches, 1):
            self.logger.info(f"\n{'='*20} BATCH {i}/{len(self.batches)} {'='*20}")
            self.logger.info(f"Running {batch['name']}...")
            
            batch_start = datetime.now()
            success = self.run_batch(batch)
            batch_duration = datetime.now() - batch_start
            
            results.append({
                "batch": batch["name"],
                "success": success,
                "duration_hours": batch_duration.total_seconds() / 3600,
                "stocks": batch["stocks"],
                "count": batch["count"]
            })
            
            if success:
                self.logger.info(f"{batch['name']} completed in {batch_duration.total_seconds() / 3600:.2f} hours")
            else:
                self.logger.error(f"{batch['name']} failed after {batch_duration.total_seconds() / 3600:.2f} hours")
            
            # Add delay between batches to avoid overwhelming the system
            if i < len(self.batches):
                delay = 300  # 5 minutes
                self.logger.info(f"Waiting {delay} seconds before next batch...")
                time.sleep(delay)
        
        # Generate final summary
        total_duration = datetime.now() - start_time
        successful_batches = sum(1 for r in results if r["success"])
        total_stocks_processed = sum(r["count"] for r in results if r["success"])
        
        self.logger.info("=" * 80)
        self.logger.info("BATCH EXECUTION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Duration: {total_duration.total_seconds() / 3600:.2f} hours")
        self.logger.info(f"Successful Batches: {successful_batches}/{len(self.batches)}")
        self.logger.info(f"Stocks Processed: {total_stocks_processed}/{self.total_stocks}")
        
        for result in results:
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            self.logger.info(f"{result['batch']}: {status} ({result['duration_hours']:.2f}h)")
        
        # Save results
        summary = {
            "execution_summary": {
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration_hours": total_duration.total_seconds() / 3600,
                "successful_batches": successful_batches,
                "total_batches": len(self.batches),
                "stocks_processed": total_stocks_processed,
                "total_stocks": self.total_stocks
            },
            "batch_results": results
        }
        
        with open("optimized_batch_execution_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info("Summary saved to: optimized_batch_execution_summary.json")
        self.logger.info("=" * 80)
        
        return successful_batches == len(self.batches)
    
    def run_all_batches_parallel(self):
        """Run all batches in parallel (use with caution)"""
        self.logger.warning("PARALLEL EXECUTION MAY CAUSE RATE LIMITING ISSUES")
        self.logger.warning("Consider using sequential execution instead")
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING PARALLEL BATCH EXECUTION")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        processes = []
        
        # Start all batches
        for batch in self.batches:
            script_path = Path(__file__).parent / batch["script"]
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append((batch, process))
            self.logger.info(f"Started {batch['name']} (PID: {process.pid})")
        
        # Wait for all processes to complete
        results = []
        for batch, process in processes:
            try:
                stdout, stderr = process.communicate(timeout=None)
                success = process.returncode == 0
                
                results.append({
                    "batch": batch["name"],
                    "success": success,
                    "stocks": batch["stocks"],
                    "count": batch["count"]
                })
                
                if success:
                    self.logger.info(f"{batch['name']} completed successfully")
                else:
                    self.logger.error(f"{batch['name']} failed: {stderr}")
                    
            except subprocess.TimeoutExpired:
                process.kill()
                self.logger.error(f"{batch['name']} timed out and was killed")
                results.append({
                    "batch": batch["name"],
                    "success": False,
                    "stocks": batch["stocks"],
                    "count": batch["count"]
                })
        
        # Generate summary
        total_duration = datetime.now() - start_time
        successful_batches = sum(1 for r in results if r["success"])
        
        self.logger.info("=" * 80)
        self.logger.info("PARALLEL EXECUTION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Duration: {total_duration.total_seconds() / 3600:.2f} hours")
        self.logger.info(f"Successful Batches: {successful_batches}/{len(self.batches)}")
        
        for result in results:
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            self.logger.info(f"{result['batch']}: {status}")
        
        return successful_batches == len(self.batches)

def main():
    """Main function to coordinate all optimized batches"""
    print("🚀 Optimized Parallel Sentiment Analysis Coordinator")
    print("This will run 4 optimized batches with improved rate limiting")
    print("Total: 110 stocks across 4 batches")
    print("Expected duration: ~80 hours total (instead of 96 hours)")
    
    coordinator = OptimizedBatchCoordinator()
    coordinator.print_batch_summary()
    
    print("\nExecution Options:")
    print("1. Sequential (Recommended) - Run batches one by one")
    print("2. Parallel (Use with caution) - Run all batches simultaneously")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                print("\n🚀 Starting sequential execution...")
                success = coordinator.run_all_batches_sequential()
                if success:
                    print("\n✅ All batches completed successfully!")
                else:
                    print("\n❌ Some batches failed. Check logs for details.")
                break
                
            elif choice == "2":
                print("\n⚠️  WARNING: Parallel execution may cause rate limiting issues!")
                confirm = input("Are you sure you want to continue? (y/N): ").strip().lower()
                if confirm == 'y':
                    print("\n🚀 Starting parallel execution...")
                    success = coordinator.run_all_batches_parallel()
                    if success:
                        print("\n✅ All batches completed successfully!")
                    else:
                        print("\n❌ Some batches failed. Check logs for details.")
                break
                
            elif choice == "3":
                print("\n❌ Execution cancelled")
                break
                
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n❌ Execution cancelled by user")
            break

if __name__ == "__main__":
    main() 