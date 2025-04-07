import os
from dotenv import load_dotenv
from runwayml import RunwayML
from datetime import datetime, timedelta

def check_runway_usage():
    # Load environment variables
    load_dotenv()
    RUNWAYML_API_SECRET = os.getenv("RUNWAYML_API_SECRET")
    
    if not RUNWAYML_API_SECRET:
        print("Error: RUNWAYML_API_SECRET not found in .env file")
        return
    
    try:
        # Initialize Runway client
        client = RunwayML(api_key=RUNWAYML_API_SECRET)
        
        # Get tasks from the last 24 hours
        yesterday = datetime.now() - timedelta(days=1)
        
        # Get all tasks
        tasks = client.tasks.list()
        
        # Filter tasks from last 24 hours
        recent_tasks = [task for task in tasks if task.created_at > yesterday]
        
        # Count completed tasks
        completed_tasks = [task for task in recent_tasks if task.status in ["COMPLETED", "SUCCEEDED"]]
        
        print("\nRunway API Usage Statistics (Last 24 hours):")
        print(f"Total tasks: {len(recent_tasks)}")
        print(f"Completed tasks: {len(completed_tasks)}")
        print(f"Remaining daily quota: {50 - len(completed_tasks)}")
        
        # Print task details
        print("\nRecent Task Details:")
        for task in recent_tasks:
            print(f"\nTask ID: {task.id}")
            print(f"Status: {task.status}")
            print(f"Created at: {task.created_at}")
            if hasattr(task, 'error') and task.error:
                print(f"Error: {task.error}")
                
    except Exception as e:
        print(f"Error checking Runway usage: {e}")
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")

if __name__ == "__main__":
    check_runway_usage() 