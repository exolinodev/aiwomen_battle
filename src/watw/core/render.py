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
import random # For random seeds
from watw.utils.common.exceptions import FFmpegError, APIError
from watw.utils.common.media_utils import concatenate_videos, combine_video_with_audio
from watw.utils.common.api_utils import APIConfiguration
from pathlib import Path
from watw.utils.countries import get_country_info

# Animation prompts for different types of videos
ANIMATION_PROMPTS = [
    {
        "id": "dynamic_motion",
        "text": "Create a dynamic and cinematic animation of an athlete in motion. The movement should be smooth and fluid, with a gentle camera motion that follows the athlete's graceful movements. The lighting should be dramatic and cinematic, enhancing the sense of motion and energy."
    },
    {
        "id": "pose_transition",
        "text": "Create a smooth transition between athletic poses, with elegant and graceful movement. The camera should slowly rotate around the subject, capturing the beauty of the motion. Use dramatic lighting to enhance the visual impact."
    },
    {
        "id": "dance_flow",
        "text": "Create a flowing dance-like animation with smooth, continuous movement. The camera should follow the motion in a cinematic way, with dynamic lighting that emphasizes the grace and fluidity of the movement."
    }
]

class TaskStatus(Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

# Load configuration
api_config = APIConfiguration('config.json')

# Get API keys from config
TENSORART_BEARER_TOKEN = api_config.get_api_key('tensorart')
RUNWAYML_API_SECRET = api_config.get_api_key('runwayml')

# Get API endpoints from config
TENSORART_API_BASE_URL = api_config.get_base_url('tensorart')
TENSORART_SUBMIT_JOB_ENDPOINT = api_config.get_endpoint('tensorart', '/v1/jobs')
RUNWAYML_API_BASE_URL = api_config.get_base_url('runwayml')

# !!! IMPORTANT: VERIFY THIS POLLING ENDPOINT AND RESPONSE STRUCTURE !!!
# This is a placeholder assumption based on common patterns.
# You MUST find the correct endpoint in TensorArt's documentation.
def get_tensorart_job_status_endpoint(job_id):
    return f"{TENSORART_API_BASE_URL}/v1/jobs/{job_id}"

# Remove the hardcoded OUTPUT_FOLDER
# OUTPUT_FOLDER = "output_gym_short_v2"
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Base Image Generation Parameters (TensorArt) ---
# Base image prompts are now imported from base_image_prompts.py
BASE_IMAGE_WIDTH = 768 # Choose dimensions suitable for 9:16 animation later
BASE_IMAGE_HEIGHT = 1280
BASE_IMAGE_STEPS = 30
BASE_IMAGE_CFG_SCALE = 7
BASE_IMAGE_SAMPLER = "Euler"
# !! IMPORTANT: Replace with the actual Model ID from TensorArt !!
BASE_IMAGE_MODEL_ID = api_config.get_model_id('tensorart')
BASE_IMAGE_SEED = -1 # -1 for random
BASE_IMAGE_COUNT = 1 # Generate 1 base image

# --- Animation Parameters (RunwayML Gen-3 Turbo) ---
ANIMATION_MODEL = "gen3a_turbo"  # Correct model name
ANIMATION_DURATION = 5  # Must be either 5 or 10 seconds
ANIMATION_RATIO = "768:1280"  # Must be either 768:1280 or 1280:768
ANIMATION_SEED_START = random.randint(1, 1000000)  # Random seed for variety

# Animation prompts are now imported from animation_prompts.py

# Common negative prompts
BASE_IMAGE_NEGATIVE_PROMPT = (
    "ugly, deformed, blurry, low quality, extra limbs, disfigured, poorly drawn face, "
    "bad anatomy, cartoon, drawing, illustration, text, watermark, signature, multiple people, "
    "nudity, inappropriate content."
)
ANIMATION_NEGATIVE_PROMPT = BASE_IMAGE_NEGATIVE_PROMPT

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
        return save_path  # Return the path on success
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return None

def encode_image_base64(image_path):
    """Encode an image file to base64."""
    try:
        from PIL import Image
        import io
        
        # Open and validate the image
        with Image.open(image_path) as img:
            # Check dimensions
            width, height = img.size
            if width != 768 or height != 1280:
                print(f"Warning: Image dimensions {width}x{height} do not match required 768x1280")
                # Resize image to match requirements
                img = img.resize((768, 1280), Image.Resampling.LANCZOS)
                print("Image has been resized to 768x1280")
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return base64.b64encode(img_byte_arr).decode('utf-8')
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def generate_base_image_tensorart(output_directory, prompt_id=None, prompt_text=None):
    """Generates a base image using TensorArt Workflow API (async)."""
    print("\n--- Starting Base Image Generation (TensorArt) ---")
    if not TENSORART_BEARER_TOKEN: return None, None
    
    # Load country data from config
    with open('config.json', 'r') as f:
        config_data = json.load(f)
    
    # Use the provided prompt or default to the first country
    if prompt_id is None or prompt_text is None:
        # Get the first country from config
        for key, value in config_data.items():
            if isinstance(value, dict) and 'base_image_prompt' in value:
                prompt_text = value['base_image_prompt']
                prompt_id = value['id']
                break
    
    print(f"Generating base image for: {prompt_id}")
    
    request_id = str(uuid.uuid4())
    headers = {
        "Authorization": f"Bearer {TENSORART_BEARER_TOKEN}",
        "Content-Type": "application/json; charset=UTF-8",
        "Accept": "application/json"
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
                        {"text": prompt_text, "weight": 1.0},
                        {"text": BASE_IMAGE_NEGATIVE_PROMPT, "weight": -1.0}
                    ],
                    "sampler": BASE_IMAGE_SAMPLER,
                    "sdVae": "Automatic",
                    "steps": BASE_IMAGE_STEPS,
                    "sd_model": BASE_IMAGE_MODEL_ID,
                    "clip_skip": 2,
                    "cfg_scale": BASE_IMAGE_CFG_SCALE
                }
            }
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
                                base_image_filename = f"tensorart_base_image_{job_id}.png"
                                base_image_path = os.path.join(output_directory, base_image_filename)
                                download_result = download_file(image_url, base_image_path)
                                if download_result:
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
def generate_animation_runway(base_image_path, animation_prompt_text, output_directory, output_filename_base, seed):
    """Generates a single animation clip using RunwayML API."""
    print(f"\n--- Starting Animation Generation for: {output_filename_base} (RunwayML) ---")
    if not RUNWAYML_API_SECRET: 
        print("Error: RunwayML API secret not found")
        return None
    if not base_image_path:
        print("Error: Base image path is required for Runway animation.")
        return None
    if not animation_prompt_text:
        print("Error: Animation prompt text is required.")
        return None

    try:
        client = RunwayML(api_key=RUNWAYML_API_SECRET)
        print("Submitting Runway task...")
        
        # Encode and validate the image
        image_base64 = encode_image_base64(base_image_path)
        if not image_base64:
            print("Error: Failed to encode image")
            return None
        
        # Create the task with retry logic for rate limits
        max_retries = 5
        base_delay = 60  # Start with 1 minute delay
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Enhance the animation prompt
                enhanced_prompt = f"Create a smooth, cinematic animation. {animation_prompt_text} Maintain consistent motion and high quality throughout the 5-second duration."
                
                # Create the task
                task_response = client.image_to_video.create(
                    model=ANIMATION_MODEL,
                    prompt_image=f"data:image/png;base64,{image_base64}",
                    prompt_text=enhanced_prompt,
                    duration=ANIMATION_DURATION,  # 5 seconds
                    ratio=ANIMATION_RATIO,  # "768:1280"
                    seed=seed,
                    watermark=False
                )
                break  # If successful, break the retry loop
                
            except Exception as e:
                if hasattr(e, 'response') and e.response and e.response.status_code == 429:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print("Error: Maximum retries reached for rate limit. Please try again tomorrow.")
                        return None
                    
                    # Calculate exponential backoff delay
                    delay = base_delay * (2 ** (retry_count - 1))
                    print(f"Rate limit reached. Waiting {delay} seconds before retry {retry_count}/{max_retries}...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Error creating task: {str(e)}")
                    if hasattr(e, 'response'):
                        print(f"Response status: {e.response.status_code}")
                        try:
                            print(f"Response body: {e.response.json()}")
                        except:
                            print(f"Response text: {e.response.text}")
                    return None
        
        # Get the task ID from the response
        task_id = task_response.id
        print(f"Runway task submitted. ID: {task_id}")

        # Poll for completion with retry logic
        max_poll_retries = 30  # Maximum number of retries
        poll_retry_count = 0
        poll_retry_delay = 10
        
        while poll_retry_count < max_poll_retries:
            try:
                # Get the current task status
                task = client.tasks.retrieve(task_id)
                status = task.status
                print(f"Polling Runway task {task_id}... Current status: {status}")

                if status in ["COMPLETED", "SUCCEEDED"]:
                    print(f"Runway task {task_id} {status} successfully.")
                    if hasattr(task, 'output') and task.output and len(task.output) > 0:
                        video_url = task.output[0]
                        print(f"Generated video URL: {video_url}")
                        
                        # Download the video
                        video_filename = f"{output_filename_base}_seed{seed}.mp4"
                        video_path = os.path.join(output_directory, video_filename)
                        
                        print(f"Downloading video to: {video_path}")
                        success = download_file(video_url, video_path)
                        
                        if success and os.path.exists(video_path):
                            print(f"Successfully downloaded video to: {video_path}")
                            return video_path
                        else:
                            print("Error: Failed to download video file")
                            return None
                    else:
                        print("Error: No video URL in task output")
                        return None
                elif status == "FAILED":
                    print(f"Error: Runway task {task_id} failed.")
                    if hasattr(task, 'error'):
                        print(f"Error details: {task.error}")
                    return None
                else:
                    print(f"Task still processing...")
                    time.sleep(poll_retry_delay)
                    poll_retry_count += 1
                    
            except Exception as e:
                print(f"Error polling Runway task: {str(e)}")
                poll_retry_count += 1
                time.sleep(poll_retry_delay)
                
        print("Error: Maximum polling retries reached")
        return None
        
    except Exception as e:
        print(f"Error in Runway animation generation: {str(e)}")
        return None


# --- Main Execution ---
def main():
    print("Starting Gym Short Generation Workflow (v2 - Async TensorArt)...")

    if not check_credentials():
        sys.exit(1) # Exit if credentials are missing

    # 1. Generate Base Image (Now Asynchronous)
    base_image_local_path, base_image_url = generate_base_image_tensorart(OUTPUT_FOLDER)

    if not base_image_local_path or not base_image_url:
        print("\nWorkflow aborted: Base image generation failed.")
        sys.exit(1) # Exit if base image fails

    print(f"\nBase image saved to: {base_image_local_path}")
    print(f"Using Base Image for animations: {base_image_local_path}")

    # Add confirmation prompt before starting animations
    print("\nReady to start generating animations.")
    response = input("Do you want to continue with animation generation? (yes/no): ").lower().strip()
    if response != 'yes':
        print("Animation generation cancelled by user.")
        sys.exit(0)

    # 2. Generate Animation Clips
    generated_clips = []
    current_seed = ANIMATION_SEED_START

    for country_key, country_data in get_country_info().items():
        prompt_id = country_data["id"]
        prompt_text = country_data["animation_prompt"]  # Use animation_prompt instead of base_image_prompt
        output_filename_base = f"animation_{prompt_id}"

        video_path = generate_animation_runway(
            base_image_path=base_image_local_path,
            animation_prompt_text=prompt_text,
            output_directory=OUTPUT_FOLDER,
            output_filename_base=output_filename_base,
            seed=current_seed
        )

        if video_path:
            generated_clips.append(video_path)
        else:
            print(f"Failed to generate animation for prompt: {prompt_id}")

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

def generate_all_base_images(output_directory):
    """Generates all base images for the different countries."""
    print("\n--- Generating All Base Images ---")
    base_images = []
    
    for country_key, country_data in get_country_info().items():
        prompt_id = country_data["id"]
        prompt_text = country_data["base_image_prompt"]
        
        print(f"\nGenerating base image for: {prompt_id}")
        base_image_path, _ = generate_base_image_tensorart(
            output_directory=output_directory,
            prompt_id=prompt_id,
            prompt_text=prompt_text
        )
        
        if base_image_path:
            base_images.append({
                "id": prompt_id,
                "path": base_image_path
            })
            print(f"Base image for {prompt_id} saved to: {base_image_path}")
        else:
            print(f"Failed to generate base image for: {prompt_id}")
    
    return base_images

class VideoRenderer:
    """Class for rendering videos using TensorArt and RunwayML."""
    
    def __init__(self, config=None):
        """Initialize the VideoRenderer with optional configuration.
        
        Args:
            config (dict, optional): Configuration parameters for video rendering.
        """
        self.config = config or {}
        self.logger = setup_logger("watw.video_renderer")
        
        # Load API configuration
        self.api_config = APIConfiguration('config.json')
        
        # Initialize API clients
        self.tensorart_token = self.api_config.get_api_key('tensorart')
        self.runway_token = self.api_config.get_api_key('runwayml')
    
    def combine_video_audio(self, video_path, audio_path, output_path):
        """Combine video with audio.
        
        Args:
            video_path: Path to the video file
            audio_path: Path to the audio file
            output_path: Path to save the combined video
            
        Returns:
            Path to the combined video
        """
        return combine_video_with_audio(video_path, audio_path, output_path)
    
    def create_video(self, country_code, output_dir):
        """Create a video for a country.
        
        Args:
            country_code: Country code to generate video for
            output_dir: Directory to save the output
            
        Returns:
            Path to the final video
        """
        # Load country data from config
        with open('config.json', 'r') as f:
            config_data = json.load(f)
        
        # Find country data
        country_data = None
        for key, value in config_data.items():
            if isinstance(value, dict) and value.get('id') == country_code:
                country_data = value
                break
        
        if not country_data:
            raise ValueError(f"Country code '{country_code}' not found in configuration")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate base image
        base_image_path, _ = generate_base_image_tensorart(
            output_dir,
            prompt_id=country_data['id'],
            prompt_text=country_data['base_image_prompt']
        )
        
        if not base_image_path:
            raise RuntimeError("Failed to generate base image")
        
        # Generate animation
        animation_path = generate_animation_runway(
            base_image_path,
            country_data['animation_prompt'],
            output_dir,
            f"animation_{country_data['id']}",
            ANIMATION_SEED_START
        )
        
        if not animation_path:
            raise RuntimeError("Failed to generate animation")
        
        return animation_path

# Remove or comment out the main execution block if it's only used for imports
# if __name__ == "__main__":
#     main()