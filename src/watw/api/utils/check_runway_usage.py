"""
Check RunwayML API usage.

This script checks the usage of the RunwayML API for the Women Around The World project.
"""

from watw.api.clients.runway import RunwayClient


def check_runway_usage(days: int = 1) -> None:
    """
    Check RunwayML API usage for the specified number of days.

    Args:
        days: Number of days to check
    """
    try:
        # Initialize Runway client
        client = RunwayClient()

        # Get usage info
        usage = client.check_usage(days)

        # Print usage statistics
        print("\nRunway API Usage Statistics:")
        print(f"Period: Last {days} {'day' if days == 1 else 'days'}")
        print(f"Total tasks: {usage['total_tasks']}")
        print(f"Completed tasks: {usage['completed_tasks']}")
        print(f"Failed tasks: {usage['failed_tasks']}")
        print(f"Daily quota: {usage['daily_quota']}")
        print(f"Remaining quota: {usage['remaining_quota']}")

        # Print task details
        print("\nRecent Task Details:")
        for task in usage["tasks"]:
            print(f"\nTask ID: {task['id']}")
            print(f"Status: {task['status']}")
            print(f"Created at: {task['created_at']}")
            if task["error"]:
                print(f"Error: {task['error']}")

    except Exception as e:
        print(f"Error checking Runway usage: {e}")


if __name__ == "__main__":
    check_runway_usage()
