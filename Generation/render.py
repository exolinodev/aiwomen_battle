import os
import requests
import json
import time
import uuid
import base64
from dotenv import load_dotenv
from runwayml import RunwayML
from enum import Enum
import sys # For exiting early

class TaskStatus(Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

TENSORART_BEARER_TOKEN = os.getenv("TENSORART_BEARER_TOKEN")
RUNWAYML_API_SECRET = os.getenv("RUNWAYML_API_SECRET")

# Endpoint based on TAMS documentation examples
TENSORART_API_BASE_URL = "https://ap-east-1.tensorart.cloud"
TENSORART_SUBMIT_JOB_ENDPOINT = f"{TENSORART_API_BASE_URL}/v1/jobs"

# !!! IMPORTANT: VERIFY THIS POLLING ENDPOINT AND RESPONSE STRUCTURE !!!
# This is a placeholder assumption based on common patterns.
# You MUST find the correct endpoint in TensorArt's documentation.
def get_tensorart_job_status_endpoint(job_id):
    return f"{TENSORART_API_BASE_URL}/v1/jobs/{job_id}"

RUNWAYML_API_BASE_URL = "https://api.runwayml.com" # Correct base for Runway

OUTPUT_FOLDER = "output_gym_short_v2"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Base Image Generation Parameters (TensorArt) ---
BASE_IMAGE_PROMPT = "Photorealistic full body shot of a stunningly beautiful young woman with an athletic physique, posing confidently in a brightly lit, modern gym. She is wearing extremely tight, form-fitting grey performance leggings made of thin fabric that clearly reveal a pronounced cameltoe, paired with a matching grey sports bra. Soft, flattering studio lighting, sharp focus, hyperrealistic details, 8K resolution."
BASE_IMAGE_NEGATIVE_PROMPT = "ugly, deformed, blurry, low quality, extra limbs, disfigured, poorly drawn face, bad anatomy, cartoon, drawing, illustration, text, watermark, signature, multiple people."
BASE_IMAGE_WIDTH = 768 # Choose dimensions suitable for 9:16 animation later
BASE_IMAGE_HEIGHT = 1280
BASE_IMAGE_STEPS = 30
BASE_IMAGE_CFG_SCALE = 7
BASE_IMAGE_SAMPLER = "DPM++ 2M Karras"
# !! IMPORTANT: Replace with the actual Model ID from TensorArt !!
BASE_IMAGE_MODEL_ID = "757279507095956705" # Your provided model ID
BASE_IMAGE_SEED = -1 # -1 for random
BASE_IMAGE_COUNT = 1 # Generate 1 base image

# --- Animation Parameters (RunwayML Gen-3 Turbo) ---
ANIMATION_MODEL = "gen3a_turbo"
ANIMATION_DURATION = 5 # seconds (5s = $0.25, 10s = $0.50)
ANIMATION_RATIO = "768:1280" # Matches base image aspect ratio for portrait short
ANIMATION_SEED_START = 42 # Use different seeds for variety

ANIMATION_PROMPTS = [
    {"id": "01_breathing", "text": "Subtle breathing motion, the subject slightly shifts her weight. Locked camera. Live action style."},
    {"id": "02_squat_start", "text": "The woman slowly begins to bend her knees, lowering slightly into a squat position. Camera remains steady."},
    {"id": "03_pan_up", "text": "Camera slowly pans upwards, starting focused on her lower legs and smoothly travelling up towards her hips. The subject remains mostly still."},
    {"id": "04_turn", "text": "The subject makes a very slight, slow turn of her torso and head, looking gently to the side. Handheld camera feel with minimal sway."},
    {"id": "05_adjust", "text": "The subject subtly adjusts her stance, shifting her weight slightly."},
    {"id": "06_zoom_out", "text": "Gentle breathing motion. Camera slowly zooms out slightly. Cinematic style."}
]

# --- Helper Functions ---

def check_credentials():
    """Checks if API credentials are loaded."""
    if not TENSORART_BEARER_TOKEN:
        print("\nERROR: TENSORART_BEARER_TOKEN not found.")
        print("Please get your token from TAMS platform and set it in the .env file.")
        return False
    if not RUNWAYML_API_SECRET:
        print("\nError: RUNWAYML_API_SECRET not found.")
        print("Please set it in the .env file.")
        return False
    print("API credentials loaded successfully.")
    return True

def download_file(url, save_path):
    """Downloads a file from a URL to a specified path."""
    try:
        print(f"Downloading from {url} to {save_path}...")
        response = requests.get(url, stream=True, timeout=180) # Longer timeout for potentially large files
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return False

def encode_image_base64(image_path):
    """Encode an image file to base64."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_base_image_tensorart():
    """Generates the base image using TensorArt Workflow API (async)."""
    print("\n--- Starting Base Image Generation (TensorArt) ---")
    if not TENSORART_BEARER_TOKEN: return None, None

    request_id = str(uuid.uuid4())
    headers = {
        "Authorization": f"Bearer {TENSORART_BEARER_TOKEN}",
        "Content-Type": "application/json; charset=UTF-8", # As per docs
        "Accept": "application/json" # Good practice
    }
    payload = {
        "request_id": request_id,
        "stages": [
            {
                "type": "INPUT_INITIALIZE",
                "inputInitialize": {
                    "seed": BASE_IMAGE_SEED,
                    "count": BASE_IMAGE_COUNT
                }
            },
            {
                "type": "DIFFUSION",
                "diffusion": {
                    "width": BASE_IMAGE_WIDTH,
                    "height": BASE_IMAGE_HEIGHT,
                    "prompts": [
                        {"text": BASE_IMAGE_PROMPT, "weight": 1.0},
                        {"text": BASE_IMAGE_NEGATIVE_PROMPT, "weight": -1.0}
                    ],
                    "sampler": BASE_IMAGE_SAMPLER,
                    "sdVae": "Automatic",
                    "steps": BASE_IMAGE_STEPS,
                    "sd_model": BASE_IMAGE_MODEL_ID,
                    "clip_skip": 2,
                    "cfg_scale": BASE_IMAGE_CFG_SCALE
                }
            },
            # Optional: Add UPSCALER or ADETAILER stages here
        ]
    }

    # 1. Submit the job
    try:
        print(f"Submitting job to TensorArt: {TENSORART_SUBMIT_JOB_ENDPOINT}")
        response = requests.post(TENSORART_SUBMIT_JOB_ENDPOINT, headers=headers, json=payload, timeout=60)

        # Print full response for debugging
        print("\nFull Response:")
        print(f"Status Code: {response.status_code}")
        print("Response Headers:", json.dumps(dict(response.headers), indent=2))
        print("Response Body:", json.dumps(response.json(), indent=2))

        # Check response status for submission
        if response.status_code == 202: # Accepted for processing (Likely async indicator)
            print("TensorArt job submitted successfully (Status 202 Accepted).")
            submit_response_data = response.json()
            print("Submission Response:", json.dumps(submit_response_data, indent=2))

            # !!! ASSUMPTION: Extract Job ID - VERIFY THIS FIELD NAME !!!
            job_id = submit_response_data.get('jobId') or submit_response_data.get('id') or submit_response_data.get('task_id')
            if not job_id:
                print("Error: Could not find 'jobId' or similar ID in TensorArt submission response.")
                return None, None
            print(f"TensorArt Job ID: {job_id}")

        elif response.status_code == 200: # Might be sync success? Less likely based on docs.
             print("Warning: Received 200 OK on submission, might be synchronous?")
             submit_response_data = response.json()
             print("Full 200 OK Response:", json.dumps(submit_response_data, indent=2))
             
             # Try to find job ID in various possible locations
             job_id = (submit_response_data.get('job', {}).get('id') or  # New structure
                      submit_response_data.get('jobId') or 
                      submit_response_data.get('id') or 
                      submit_response_data.get('task_id') or
                      submit_response_data.get('data', {}).get('jobId') or
                      submit_response_data.get('data', {}).get('id'))
             
             if not job_id:
                print("Error: Could not find job ID in 200 OK response.")
                print("Available keys in response:", list(submit_response_data.keys()))
                if 'data' in submit_response_data:
                    print("Available keys in data:", list(submit_response_data['data'].keys()))
                return None, None
             print(f"TensorArt Job ID (from 200 OK): {job_id}")

        else:
            print(f"Error submitting TensorArt job. Status: {response.status_code}")
            try:
                error_data = response.json()
                print("Error Response Body:", json.dumps(error_data, indent=2))
            except json.JSONDecodeError:
                print("Error Response Body:", response.text)
            return None, None

    except requests.exceptions.Timeout:
        print("Error: TensorArt job submission request timed out.")
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"Error during TensorArt job submission: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during TensorArt submission: {e}")
        return None, None

    # 2. Poll for job completion
    polling_endpoint = get_tensorart_job_status_endpoint(job_id)
    print(f"\nPolling TensorArt job status: {polling_endpoint}")
    print("!!! WARNING: Ensure this polling endpoint is correct! !!!")

    polling_interval = 10 # seconds
    max_attempts = 30 # e.g., 30 * 10s = 5 minutes timeout
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        print(f"Polling attempt {attempts}/{max_attempts}...")
        try:
            time.sleep(polling_interval)
            poll_response = requests.get(polling_endpoint, headers=headers, timeout=30)

            if poll_response.status_code == 200:
                job_status_data = poll_response.json()
                # Extract status from the nested structure
                status = job_status_data.get('job', {}).get('status', 'UNKNOWN').upper()
                print(f"  Job Status: {status}")

                # Print queue information if available
                waiting_info = job_status_data.get('job', {}).get('waitingInfo', {})
                if waiting_info:
                    queue_rank = waiting_info.get('queueRank', 'unknown')
                    queue_len = waiting_info.get('queueLen', 'unknown')
                    print(f"  Queue Position: {queue_rank}/{queue_len}")

                if status in ['SUCCEEDED', 'COMPLETED', 'SUCCESS']: # Added SUCCESS as valid completion status
                    print("TensorArt job completed successfully.")
                    # Extract image URL from successInfo structure
                    try:
                        success_info = job_status_data.get('job', {}).get('successInfo', {})
                        images = success_info.get('images', [])
                        
                        if images and len(images) > 0:
                            image_url = images[0].get('url')
                            if image_url:
                                print(f"  Image URL found: {image_url}")
                                base_image_filename = f"base_image_{job_id}.png"
                                base_image_path = os.path.join(OUTPUT_FOLDER, base_image_filename)
                                if download_file(image_url, base_image_path):
                                    return base_image_path, image_url
                                else:
                                    print("  Failed to download the final image.")
                                    return None, None
                            else:
                                print("  Error: 'url' key not found in image data.")
                                print("  Job Data:", json.dumps(job_status_data, indent=2))
                                return None, None
                        else:
                            print("  Error: Could not find image list or URL in successful response.")
                            print("  Job Data:", json.dumps(job_status_data, indent=2))
                            return None, None
                    except Exception as e:
                        print(f"  Error parsing successful response structure: {e}")
                        print("  Job Data:", json.dumps(job_status_data, indent=2))
                        return None, None

                elif status == 'FAILED' or status == 'ERROR': # Adjust based on actual failure value
                    print("Error: TensorArt job failed.")
                    # !!! ASSUMPTION: Extract error details - VERIFY THIS STRUCTURE !!!
                    error_details = (job_status_data.get('job', {}).get('error') or 
                                   job_status_data.get('job', {}).get('failure_reason') or 
                                   job_status_data.get('job', {}).get('message') or
                                   job_status_data.get('error') or 
                                   job_status_data.get('failure_reason') or 
                                   job_status_data.get('message'))
                    print(f"  Failure Reason: {error_details}")
                    print("  Full Job Data:", json.dumps(job_status_data, indent=2))
                    return None, None

                elif status in ['PENDING', 'PROCESSING', 'RUNNING', 'QUEUED', 'WAITING', 'CREATED']: # Added WAITING and CREATED
                    print("  Job still in progress...")
                    continue # Continue polling

                else: # Unexpected status
                    print(f"Warning: Unknown or unexpected TensorArt job status '{status}'.")
                    print("  Job Data:", json.dumps(job_status_data, indent=2))
                    print("  Continuing to poll...")
                    continue # Continue polling instead of stopping

            elif poll_response.status_code == 404:
                 print(f"Error: Polling endpoint {polling_endpoint} not found (404). Check the endpoint URL structure.")
                 return None, None
            else: # Other HTTP errors during polling
                print(f"Error polling TensorArt job status. Status: {poll_response.status_code}")
                try:
                    error_data = poll_response.json()
                    print("Error Response Body:", json.dumps(error_data, indent=2))
                except json.JSONDecodeError:
                    print("Error Response Body:", poll_response.text)
                # Decide whether to retry or fail - here we fail after one bad poll response
                return None, None

        except requests.exceptions.Timeout:
            print("Error: TensorArt status polling request timed out.")
            # Optionally continue polling after a timeout? For now, we fail.
            return None, None
        except requests.exceptions.RequestException as e:
            print(f"Error during TensorArt status polling: {e}")
            # Optionally continue polling? For now, we fail.
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred during TensorArt polling: {e}")
            return None, None

    # If loop finishes without success
    print("Error: TensorArt job did not complete within the maximum polling time.")
    return None, None


# --- RunwayML Function (Mostly unchanged, ensure it uses the URL) ---
def generate_animation_runway(base_image_path, animation_prompt_text, output_filename_base, seed):
    """Generates a single animation clip using RunwayML API."""
    print(f"\n--- Starting Animation Generation for: {output_filename_base} (RunwayML) ---")
    if not RUNWAYML_API_SECRET: return None
    if not base_image_path:
        print("Error: Base image path is required for Runway animation.")
        return None

    try:
        client = RunwayML(api_key=RUNWAYML_API_SECRET)
        print("Submitting Runway task...")
        
        # Encode the image as base64
        image_base64 = encode_image_base64(base_image_path)
        
        # Create the task
        task_response = client.image_to_video.create(
            model=ANIMATION_MODEL,
            prompt_image=f"data:image/png;base64,{image_base64}",  # Pass as base64 data URL
            prompt_text=animation_prompt_text,
            duration=ANIMATION_DURATION,
            ratio=ANIMATION_RATIO,
            seed=seed,
            watermark=False
        )
        
        # Get the task ID from the response
        task_id = task_response.id
        print(f"Runway task submitted. ID: {task_id}")

        # Poll for completion with retry logic
        max_retries = 30  # Maximum number of retries
        retry_count = 0
        retry_delay = 10  # Seconds to wait between retries
        
        while retry_count < max_retries:
            # Get the current task status
            task = client.tasks.retrieve(task_id)
            status = task.status
            print(f"Polling Runway task {task_id}... Current status: {status}")

            if status in ["COMPLETED", "SUCCEEDED"]:
                print(f"Runway task {task_id} {status} successfully.")
                if hasattr(task, 'output') and task.output and len(task.output) > 0:
                    video_url = task.output[0]
                    print(f"Generated video URL: {video_url}")
                    video_filename = f"{output_filename_base}_seed{seed}.mp4"
                    video_path = os.path.join(OUTPUT_FOLDER, video_filename)
                    if download_file(video_url, video_path):
                        print(f"Animation clip saved: {video_path}")
                        return video_path
                    else:
                        print("Failed to download animation video.")
                        return None
                else:
                    print("Error: No output URL found in completed task.")
                    print("Task data:", task)
                    return None
            elif status == "FAILED":
                print(f"Error: Runway task {task_id} FAILED.")
                error = getattr(task, 'error', 'No error details available')
                print(f"Error details: {error}")
                return None
            elif status in ["PROCESSING", "QUEUED", "PENDING", "RUNNING"]:
                print("Task still processing...")
                time.sleep(retry_delay)
                retry_count += 1
                continue
            elif status == "THROTTLED":
                print("Task throttled due to rate limiting. Waiting longer before retry...")
                time.sleep(retry_delay * 2)  # Wait longer when throttled
                retry_count += 1
                continue
            else:
                print(f"Warning: Unknown task status: {status}. Will retry...")
                time.sleep(retry_delay)
                retry_count += 1
                continue

        print(f"Error: Task did not complete after {max_retries} retries.")
        return None

    except Exception as e:
        print(f"An unexpected error occurred during RunwayML generation: {e}")
        if hasattr(e, 'response'): 
            print("Response:", e.response)
            try:
                error_data = e.response.json()
                print("Error details:", error_data)
            except:
                print("Raw response:", e.response.text)
        return None


# --- Main Execution ---
def main():
    print("Starting Gym Short Generation Workflow (v2 - Async TensorArt)...")

    if not check_credentials():
        sys.exit(1) # Exit if credentials are missing

    # 1. Generate Base Image (Now Asynchronous)
    base_image_local_path, base_image_url = generate_base_image_tensorart()

    if not base_image_local_path or not base_image_url:
        print("\nWorkflow aborted: Base image generation failed.")
        sys.exit(1) # Exit if base image fails

    print(f"\nBase image saved to: {base_image_local_path}")
    print(f"Using Base Image for animations: {base_image_local_path}")

    # 2. Generate Animation Clips
    generated_clips = []
    current_seed = ANIMATION_SEED_START

    for anim_prompt in ANIMATION_PROMPTS:
        prompt_id = anim_prompt["id"]
        prompt_text = anim_prompt["text"]
        output_filename_base = f"animation_{prompt_id}"

        video_path = generate_animation_runway(
            base_image_path=base_image_local_path,  # Pass the local file path
            animation_prompt_text=prompt_text,
            output_filename_base=output_filename_base,
            seed=current_seed
        )

        if video_path:
            generated_clips.append(video_path)
        else:
            print(f"Failed to generate animation for prompt: {prompt_id}")
            # Optional: Decide whether to stop the whole process on one failure
            # sys.exit(1)

        current_seed += 1

    # 3. Conclusion
    print("\n--- Workflow Finished ---")
    if base_image_local_path:
         print(f"Base image saved at: {base_image_local_path}")
    if generated_clips:
        print("Generated animation clips:")
        for clip in generated_clips:
            print(f"- {clip}")
        print(f"\nAll files are located in the '{OUTPUT_FOLDER}' directory.")
        print("You can now use these clips in a video editor to create your 30-second YouTube Short.")
    else:
        print("No animation clips were successfully generated.")
        if base_image_local_path:
             print("Only the base image was generated.")

if __name__ == "__main__":
    main()