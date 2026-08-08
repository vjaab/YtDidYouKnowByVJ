import schedule
import time
import pytz
from datetime import datetime
from main import run_pipeline
from community_post import create_community_post, schedule_community_posts
from config import UPLOAD_TIMES, TIMEZONE

COMMUNITY_SCHEDULE = schedule_community_posts()

def check_time_and_run():
    ist_now = datetime.now(pytz.timezone(TIMEZONE))
    current_hhmm = ist_now.strftime("%H:%M")
    current_day = ist_now.strftime("%A")
    
    # Video pipeline at scheduled times
    if current_hhmm in UPLOAD_TIMES:
        topic_type = "auto"
        print(f"[{ist_now}] Triggering AI Pipeline (AUTO-BALANCED) for {current_hhmm} slot...")
        run_pipeline(topic_type=topic_type)
        time.sleep(61) # Sleep to avoid double triggering
    
    # Community posts twice a week (Monday & Thursday at 10:00 and 16:00)
    if current_day in COMMUNITY_SCHEDULE["days"] and current_hhmm in COMMUNITY_SCHEDULE["times"]:
        print(f"[{ist_now}] Triggering Community Post for {current_day} {current_hhmm}...")
        create_community_post()
        time.sleep(61)

def start_scheduler():
    print(f"AI Research Scheduler Started.")
    print(f"  Video pipeline: {', '.join(UPLOAD_TIMES)} {TIMEZONE}")
    print(f"  Community posts: {', '.join(COMMUNITY_SCHEDULE['days'])} at {', '.join(COMMUNITY_SCHEDULE['times'])} {COMMUNITY_SCHEDULE['timezone']}")
    # We check every 30 seconds to ensure we hit the 1 minute window precisely
    schedule.every(30).seconds.do(check_time_and_run)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    start_scheduler()
