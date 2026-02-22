import schedule
import time
import pytz
from datetime import datetime
from main import run_pipeline
from config import UPLOAD_TIME, TIMEZONE

def check_time_and_run():
    ist_now = datetime.now(pytz.timezone(TIMEZONE))
    # E.g. UPLOAD_TIME is "08:00" from config.py
    if ist_now.strftime("%H:%M") == UPLOAD_TIME:
        print(f"[{ist_now}] Triggering Top Tech News Pipeline...")
        run_pipeline()
        time.sleep(61) # Sleep to avoid double triggering

def start_scheduler():
    print(f"Tech News Scheduler Started. Target time: {UPLOAD_TIME} {TIMEZONE}")
    # We check every 30 seconds to ensure we hit the 1 minute window precisely
    schedule.every(30).seconds.do(check_time_and_run)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    start_scheduler()
