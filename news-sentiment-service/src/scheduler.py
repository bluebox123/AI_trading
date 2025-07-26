"""
Scheduler module for the news sentiment pipeline
Uses APScheduler to run the pipeline at regular intervals
"""
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Optional
import threading

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

from pipeline import NewsSentimentPipeline
from config.config import (
    POLLING_INTERVAL_MINUTES, TIMEZONE, SCHEDULER_COALESCE,
    SCHEDULER_MAX_INSTANCES, CLEANUP_INTERVAL_HOURS,
    MAX_CONSECUTIVE_FAILURES, FAILURE_COOLDOWN_MINUTES
)

class NewsSentimentScheduler:
    """
    Scheduler for the news sentiment pipeline
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scheduler = None
        self.pipeline = None
        self.is_running = False
        self.shutdown_event = threading.Event()
        self.last_successful_run = None
        self.failure_count = 0
        self.in_cooldown = False
        
        # Initialize pipeline
        self._initialize_pipeline()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_pipeline(self):
        """Initialize the news sentiment pipeline"""
        try:
            self.logger.info("Initializing news sentiment pipeline...")
            self.pipeline = NewsSentimentPipeline()
            self.logger.info("Pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)
    
    def _job_listener(self, event):
        """Listen to job execution events"""
        if event.exception:
            self.logger.error(f"Job {event.job_id} crashed: {event.exception}")
            self.failure_count += 1
            
            # Check if we need to enter cooldown mode
            if self.failure_count >= MAX_CONSECUTIVE_FAILURES:
                self._enter_cooldown()
        else:
            self.logger.info(f"Job {event.job_id} executed successfully")
            self.last_successful_run = datetime.now()
            self.failure_count = 0  # Reset failure count on success
            self.in_cooldown = False  # Exit cooldown on success
    
    def _enter_cooldown(self):
        """Enter cooldown mode after too many failures"""
        if not self.in_cooldown:
            self.logger.warning(f"Entering cooldown mode for {FAILURE_COOLDOWN_MINUTES} minutes due to {self.failure_count} consecutive failures")
            self.in_cooldown = True
            
            # Schedule a job to exit cooldown
            cooldown_time = datetime.now() + timedelta(minutes=FAILURE_COOLDOWN_MINUTES)
            self.scheduler.add_job(
                self._exit_cooldown,
                'date',
                run_date=cooldown_time,
                id='cooldown_exit',
                replace_existing=True
            )
    
    def _exit_cooldown(self):
        """Exit cooldown mode"""
        self.logger.info("Exiting cooldown mode, resuming normal operations")
        self.in_cooldown = False
        self.failure_count = 0
    
    def _run_pipeline_job(self):
        """Job function that runs the pipeline iteration"""
        if self.in_cooldown:
            self.logger.info("Skipping pipeline run - in cooldown mode")
            return
        
        try:
            self.logger.info("Starting scheduled pipeline iteration...")
            start_time = time.time()
            
            # Run the pipeline
            result = self.pipeline.run_single_iteration()
            
            duration = time.time() - start_time
            
            if result['success']:
                self.logger.info(f"Pipeline iteration completed successfully in {duration:.2f} seconds. "
                               f"Processed {result['articles_saved']} articles.")
            else:
                self.logger.error(f"Pipeline iteration failed: {result.get('error', 'Unknown error')}")
                raise Exception(result.get('error', 'Pipeline iteration failed'))
                
        except Exception as e:
            self.logger.error(f"Error in scheduled pipeline job: {e}")
            raise
    
    def _run_cleanup_job(self):
        """Job function that runs periodic cleanup"""
        try:
            self.logger.info("Starting scheduled cleanup job...")
            
            # Clean up old data (keep last 30 days)
            cleanup_result = self.pipeline.cleanup_old_data(days=30)
            
            if 'error' in cleanup_result:
                self.logger.error(f"Cleanup job failed: {cleanup_result['error']}")
            else:
                self.logger.info(f"Cleanup job completed: {cleanup_result['deleted_records']} records deleted")
                
        except Exception as e:
            self.logger.error(f"Error in cleanup job: {e}")
    
    def _run_health_check_job(self):
        """Job function that performs health checks"""
        try:
            health_status = self.pipeline.health_check()
            
            if health_status['overall_status'] == 'healthy':
                self.logger.info("Health check passed - all systems operational")
            else:
                self.logger.warning(f"Health check failed: {health_status}")
                
                # Log component-specific issues
                for component, status in health_status.get('components', {}).items():
                    if status.get('status') != 'healthy':
                        self.logger.warning(f"Component {component} is unhealthy: {status}")
                        
        except Exception as e:
            self.logger.error(f"Error in health check job: {e}")
    
    def setup_scheduler(self):
        """Set up the APScheduler with all jobs"""
        try:
            self.scheduler = BackgroundScheduler(
                timezone=TIMEZONE,
                job_defaults={
                    'coalesce': SCHEDULER_COALESCE,
                    'max_instances': SCHEDULER_MAX_INSTANCES
                }
            )
            
            # Add event listener
            self.scheduler.add_listener(self._job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
            
            # Main pipeline job - runs every POLLING_INTERVAL_MINUTES
            self.scheduler.add_job(
                self._run_pipeline_job,
                IntervalTrigger(minutes=POLLING_INTERVAL_MINUTES),
                id='pipeline_job',
                name='News Sentiment Pipeline',
                next_run_time=datetime.now() + timedelta(seconds=30)  # Start after 30 seconds
            )
            
            # Cleanup job - runs daily at 2 AM
            self.scheduler.add_job(
                self._run_cleanup_job,
                CronTrigger(hour=2, minute=0),
                id='cleanup_job',
                name='Data Cleanup'
            )
            
            # Health check job - runs every hour
            self.scheduler.add_job(
                self._run_health_check_job,
                IntervalTrigger(hours=1),
                id='health_check_job',
                name='Health Check',
                next_run_time=datetime.now() + timedelta(minutes=5)  # Start after 5 minutes
            )
            
            self.logger.info("Scheduler configured successfully")
            self.logger.info(f"Pipeline will run every {POLLING_INTERVAL_MINUTES} minutes")
            self.logger.info(f"Next run scheduled for: {datetime.now() + timedelta(seconds=30)}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup scheduler: {e}")
            raise
    
    def start(self):
        """Start the scheduler"""
        try:
            if self.is_running:
                self.logger.warning("Scheduler is already running")
                return
            
            self.setup_scheduler()
            self.scheduler.start()
            self.is_running = True
            
            self.logger.info("News sentiment scheduler started successfully")
            self.logger.info(f"Monitoring {len(self.scheduler.get_jobs())} scheduled jobs")
            
            # Log scheduled jobs
            for job in self.scheduler.get_jobs():
                self.logger.info(f"Job: {job.name} (ID: {job.id}) - Next run: {job.next_run_time}")
            
        except Exception as e:
            self.logger.error(f"Failed to start scheduler: {e}")
            raise
    
    def stop(self):
        """Stop the scheduler"""
        try:
            if not self.is_running:
                self.logger.warning("Scheduler is not running")
                return
            
            self.logger.info("Stopping scheduler...")
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            self.logger.info("Scheduler stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping scheduler: {e}")
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Initiating graceful shutdown...")
        self.shutdown_event.set()
        self.stop()
    
    def run_forever(self):
        """Run the scheduler indefinitely until shutdown signal"""
        try:
            self.start()
            
            self.logger.info("Scheduler is running. Press Ctrl+C to stop.")
            
            # Keep the main thread alive
            while not self.shutdown_event.is_set():
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
        finally:
            self.shutdown()
    
    def run_once(self):
        """Run the pipeline once without scheduling"""
        try:
            self.logger.info("Running pipeline once...")
            result = self.pipeline.run_single_iteration()
            
            if result['success']:
                self.logger.info(f"Single run completed successfully. Processed {result['articles_saved']} articles.")
            else:
                self.logger.error(f"Single run failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in single run: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_status(self) -> dict:
        """Get current status of the scheduler and pipeline"""
        status = {
            'scheduler_running': self.is_running,
            'in_cooldown': self.in_cooldown,
            'failure_count': self.failure_count,
            'last_successful_run': self.last_successful_run.isoformat() if self.last_successful_run else None,
            'scheduled_jobs': [],
            'pipeline_health': None
        }
        
        if self.scheduler:
            for job in self.scheduler.get_jobs():
                status['scheduled_jobs'].append({
                    'id': job.id,
                    'name': job.name,
                    'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None
                })
        
        if self.pipeline:
            status['pipeline_health'] = self.pipeline.health_check()
        
        return status


def main():
    """Main function to run the scheduler"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/scheduler.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create and run scheduler
        scheduler = NewsSentimentScheduler()
        scheduler.run_forever()
        
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 