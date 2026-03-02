import schedule
import time
import pytz
from datetime import datetime
from main import run_pipeline
from config import UPLOAD_TIMES, TIMEZONE

def check_time_and_run():
    ist_now = datetime.now(pytz.timezone(TIMEZONE))
    current_hhmm = ist_now.strftime("%H:%M")
    
    if current_hhmm in UPLOAD_TIMES:
        if current_hhmm == "11:00":
            topic_type = "research"
        elif current_hhmm == "17:00":
            topic_type = "tools"
        else:
            topic_type = "tools"
            
        print(f"[{ist_now}] Triggering AI Pipeline ({topic_type.upper()}) for {current_hhmm} slot...")
        run_pipeline(topic_type=topic_type)
        time.sleep(61) # Sleep to avoid double triggering

def start_scheduler():
    print(f"AI Research Scheduler Started. Target times: {', '.join(UPLOAD_TIMES)} {TIMEZONE}")
    # We check every 30 seconds to ensure we hit the 1 minute window precisely
    schedule.every(30).seconds.do(check_time_and_run)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    start_scheduler()
