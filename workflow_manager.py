#!/usr/bin/env python3
import os
import sys
import re
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Union, Tuple

# Import from your refactored scripts
try:
    sys.path.append(os.path.dirname(__file__))
    from src.watw.core.render import (
        check_credentials,
        generate_all_base_images,
        generate_animation_runway,
        ANIMATION_PROMPTS,
        ANIMATION_SEED_START
    )
    from src.watw.core.voiceover import generate_voiceover_for_video
    from src.watw.core.video_editor import create_final_video
    from src.watw.utils.common.logging_utils import setup_logger
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure render.py, voiceover.py, and video_editor.py are in the correct directory.")
    sys.exit(1)

# Set up logging
logger = setup_logger(
    name="workflow_manager",
    level=logging.INFO,
    log_file="workflow_manager.log"
)

# Define workflow stages
WORKFLOW_STAGES = {
    "base_images": "Generate base images",
    "animations": "Generate animations",
    "voiceover": "Generate voice-over",
    "final_video": "Create final video"
}

class WorkflowManager:
    def __init__(self, workflow_dir: str):
        # Generate a unique workflow ID first
        self.run_history_file = "workflow_history.json"
        self.run_history = self._load_run_history()
        self.workflow_id = self._generate_workflow_id()
        
        # Create workflow directory with ID
        self.workflow_dir = f"{workflow_dir}{self.workflow_id}"
        self.checklist_file = os.path.join(self.workflow_dir, "workflow_checklist.md")
        
        # Create necessary directories
        self.dirs = {
            "generated_clips": os.path.join(self.workflow_dir, "generated_clips"),
            "temp_files": os.path.join(self.workflow_dir, "temp_files"),
            "final_video": os.path.join(self.workflow_dir, "final_video"),
            "voiceovers": os.path.join(self.workflow_dir, "voiceovers")
        }
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        # Parse checklist after directories are created
        self.tasks = self._parse_checklist()
        
        # Add current run to history if it doesn't exist
        if self.workflow_dir not in self.run_history:
            self.run_history[self.workflow_dir] = {
                "id": self.workflow_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "in_progress",
                "failed_tasks": [],
                "completed_tasks": []
            }
            self._save_run_history()
        else:
            # If workflow exists but doesn't have an ID, add it
            if "id" not in self.run_history[self.workflow_dir]:
                self.run_history[self.workflow_dir]["id"] = self.workflow_id
                self._save_run_history()
            else:
                self.workflow_id = self.run_history[self.workflow_dir]["id"]
    
    def _generate_workflow_id(self) -> int:
        """Generate a unique workflow ID by finding the highest existing ID and incrementing it."""
        max_id = 0
        
        # Find the highest existing ID
        for workflow_info in self.run_history.values():
            if "id" in workflow_info and workflow_info["id"] > max_id:
                max_id = workflow_info["id"]
        
        # Return the next ID
        return max_id + 1
    
    def _load_run_history(self) -> Dict[str, Any]:
        """Load run history from file."""
        if os.path.exists(self.run_history_file):
            with open(self.run_history_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_run_history(self) -> None:
        """Save run history to file."""
        with open(self.run_history_file, "w") as f:
            json.dump(self.run_history, f, indent=2)
    
    def _parse_checklist(self) -> List[Dict[str, Any]]:
        """Parse the markdown checklist into a list of tasks."""
        tasks = []
        
        # Check if the checklist file exists
        if not os.path.exists(self.checklist_file):
            # Create a default checklist if it doesn't exist
            self._create_default_checklist()
        
        current_section = None
        
        with open(self.checklist_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("## "):
                    current_section = line[3:]
                elif line.startswith("- ["):
                    checked = "[x]" in line
                    task_text = line[line.find("]")+2:].strip()
                    tasks.append({
                        "section": current_section,
                        "text": task_text,
                        "checked": checked
                    })
        
        return tasks
    
    def _create_default_checklist(self) -> None:
        """Create a default checklist file with standard tasks."""
        os.makedirs(os.path.dirname(self.checklist_file), exist_ok=True)
        
        with open(self.checklist_file, "w") as f:
            f.write("# Video Generation Workflow Checklist\n\n")
            f.write(f"## Workflow Information\n")
            f.write(f"- Workflow ID: {self.workflow_id}\n")
            f.write(f"- Workflow Directory: {self.workflow_dir}\n")
            f.write(f"- Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- Status: in_progress\n\n")
            
            # Base Images Generation
            f.write("## Base Images Generation\n")
            f.write("- [ ] Check API credentials\n")
            f.write("- [ ] Generate base image for segment 1\n")
            f.write("- [ ] Generate base image for segment 2\n")
            f.write("- [ ] Generate base image for segment 3\n")
            f.write("- [ ] Generate base image for segment 4\n")
            f.write("- [ ] Generate base image for segment 5\n\n")
            
            # Animation Generation
            f.write("## Animation Generation\n")
            f.write("- [ ] Generate animation for segment 1\n")
            f.write("- [ ] Generate animation for segment 2\n")
            f.write("- [ ] Generate animation for segment 3\n")
            f.write("- [ ] Generate animation for segment 4\n")
            f.write("- [ ] Generate animation for segment 5\n\n")
            
            # Voice-over Generation
            f.write("## Voice-over Generation\n")
            f.write("- [ ] Generate intro voice-over\n")
            f.write("- [ ] Generate voice-over for segment 1\n")
            f.write("- [ ] Generate voice-over for segment 2\n")
            f.write("- [ ] Generate voice-over for segment 3\n")
            f.write("- [ ] Generate voice-over for segment 4\n")
            f.write("- [ ] Generate voice-over for segment 5\n\n")
            
            # Final Video Creation
            f.write("## Final Video Creation\n")
            f.write("- [ ] Combine all animations\n")
            f.write("- [ ] Add voice-overs\n")
            f.write("- [ ] Add background music\n")
            f.write("- [ ] Export final video\n\n")
            
            # Output Information
            f.write("## Output Information\n")
            for name, path in self.dirs.items():
                f.write(f"- {name.replace('_', ' ').title()}: {path}\n")
    
    def _update_checklist(self) -> None:
        """Update the markdown checklist with current task status."""
        sections = {}
        for task in self.tasks:
            if task["section"] not in sections:
                sections[task["section"]] = []
            sections[task["section"]].append(task)
        
        with open(self.checklist_file, "w") as f:
            f.write("# Video Generation Workflow Checklist\n\n")
            f.write(f"## Workflow Information\n")
            f.write(f"- Workflow ID: {self.workflow_id}\n")
            f.write(f"- Workflow Directory: {self.workflow_dir}\n")
            f.write(f"- Created: {self.run_history[self.workflow_dir]['timestamp']}\n")
            f.write(f"- Status: {self.run_history[self.workflow_dir]['status']}\n\n")
            
            for section, tasks in sections.items():
                f.write(f"## {section}\n")
                for task in tasks:
                    status = "[x]" if task["checked"] else "[ ]"
                    f.write(f"- {status} {task['text']}\n")
                f.write("\n")
            
            # Write output information
            f.write("## Output Information\n")
            for name, path in self.dirs.items():
                f.write(f"- {name.replace('_', ' ').title()}: {path}\n")
    
    def _get_task_index(self, task_text: str) -> int:
        """Get the index of a task by its text."""
        for i, task in enumerate(self.tasks):
            if task["text"] == task_text:
                return i
        return -1
    
    def _mark_task_complete(self, task_text: str) -> bool:
        """Mark a task as complete and update run history."""
        for task in self.tasks:
            if task["text"] == task_text:
                task["checked"] = True
                # Update run history
                if task_text not in self.run_history[self.workflow_dir]["completed_tasks"]:
                    self.run_history[self.workflow_dir]["completed_tasks"].append(task_text)
                if task_text in self.run_history[self.workflow_dir]["failed_tasks"]:
                    self.run_history[self.workflow_dir]["failed_tasks"].remove(task_text)
                
                # Update workflow status
                if len(self.run_history[self.workflow_dir]["failed_tasks"]) == 0:
                    # Check if all tasks are completed
                    all_tasks = [t["text"] for t in self.tasks]
                    completed_tasks = self.run_history[self.workflow_dir]["completed_tasks"]
                    if all(task in completed_tasks for task in all_tasks):
                        self.run_history[self.workflow_dir]["status"] = "completed"
                    else:
                        self.run_history[self.workflow_dir]["status"] = "in_progress"
                
                self._update_checklist()
                self._save_run_history()
                return True
        return False
    
    def _mark_task_failed(self, task_text: str) -> None:
        """Mark a task as failed and update run history."""
        # Update run history
        if task_text not in self.run_history[self.workflow_dir]["failed_tasks"]:
            self.run_history[self.workflow_dir]["failed_tasks"].append(task_text)
        
        # Update workflow status
        self.run_history[self.workflow_dir]["status"] = "failed"
        
        self._save_run_history()
    
    def _get_next_unchecked_task(self) -> Optional[Dict[str, Any]]:
        """Get the first unchecked task."""
        for task in self.tasks:
            if not task["checked"]:
                return task
        return None
    
    def _get_failed_tasks(self) -> List[Dict[str, Any]]:
        """Get all failed tasks from the current run."""
        failed_tasks = []
        for task_text in self.run_history[self.workflow_dir]["failed_tasks"]:
            task = next((t for t in self.tasks if t["text"] == task_text), None)
            if task:
                failed_tasks.append(task)
        return failed_tasks
    
    def run_task(self, task_text: str) -> bool:
        """Run a specific task."""
        task = next((t for t in self.tasks if t["text"] == task_text), None)
        if not task:
            logger.error(f"Task not found: {task_text}")
            return False
        
        # Skip if task is already completed
        if task["checked"]:
            logger.info(f"Task already completed: {task['text']}")
            return True
            
        logger.info(f"Running task: {task['text']}")
        
        try:
            if task["section"] == "Base Images Generation":
                if task["text"] == "Check API credentials":
                    if not check_credentials():
                        self._mark_task_failed(task["text"])
                        return False
                else:
                    # Extract segment ID from task text
                    match = re.search(r"segment (\d+)", task["text"])
                    if match:
                        segment_id = match.group(1)
                        # Check if base image already exists
                        base_image_path = os.path.join(self.dirs["temp_files"], f"base_image_{segment_id}.png")
                        if os.path.exists(base_image_path):
                            logger.info(f"Base image for segment {segment_id} already exists")
                            self._mark_task_complete(task["text"])
                            return True
                        # Generate base image for the segment
                        base_images = generate_all_base_images(output_directory=self.dirs["temp_files"])
                        if not base_images:
                            self._mark_task_failed(task["text"])
                            return False
            
            elif task["section"] == "Animation Generation":
                # Extract segment ID from task text
                match = re.search(r"segment (\d+)", task["text"])
                if match:
                    segment_id = match.group(1)
                    # Check if animation already exists
                    animation_path = os.path.join(self.dirs["generated_clips"], f"animation_{segment_id}.mp4")
                    if os.path.exists(animation_path):
                        logger.info(f"Animation for segment {segment_id} already exists")
                        self._mark_task_complete(task["text"])
                        return True
                    # Find base image for the segment
                    base_images = [img for img in os.listdir(self.dirs["temp_files"]) 
                                 if img.startswith(f"base_image_{segment_id}")]
                    if base_images:
                        base_image_path = os.path.join(self.dirs["temp_files"], base_images[0])
                        video_path = generate_animation_runway(
                            base_image_path=base_image_path,
                            animation_prompt_text=ANIMATION_PROMPTS[0]["text"],
                            output_directory=self.dirs["generated_clips"],
                            output_filename_base=f"animation_{segment_id}",
                            seed=ANIMATION_SEED_START + int(segment_id) - 1
                        )
                        if not video_path:
                            self._mark_task_failed(task["text"])
                            return False
            
            elif task["section"] == "Voice-over Generation":
                if task["text"] == "Generate intro voice-over":
                    # Check if voiceover already exists
                    voiceover_path = os.path.join(self.dirs["voiceovers"], "voiceover_1.mp3")
                    if os.path.exists(voiceover_path):
                        logger.info("Intro voice-over already exists")
                        self._mark_task_complete(task["text"])
                        return True
                    voiceover_path = generate_voiceover_for_video(
                        video_number=1,
                        country1="Segment 1",
                        country2="Segment 2",
                        output_dir=self.dirs["voiceovers"]
                    )
                    if not voiceover_path:
                        self._mark_task_failed(task["text"])
                        return False
                else:
                    # Extract segment ID from task text
                    match = re.search(r"segment (\d+)", task["text"])
                    if match:
                        segment_id = match.group(1)
                        # Check if voiceover already exists
                        voiceover_path = os.path.join(self.dirs["voiceovers"], f"voiceover_{segment_id}.mp3")
                        if os.path.exists(voiceover_path):
                            logger.info(f"Voice-over for segment {segment_id} already exists")
                            self._mark_task_complete(task["text"])
                            return True
                        voiceover_path = generate_voiceover_for_video(
                            video_number=1,
                            country1=f"Segment {segment_id}",
                            country2=None,
                            output_dir=self.dirs["voiceovers"]
                        )
                        if not voiceover_path:
                            self._mark_task_failed(task["text"])
                            return False
            
            elif task["section"] == "Final Video Creation":
                if task["text"] == "Combine all animations":
                    # Check if all animations exist
                    all_animations_exist = all(
                        os.path.exists(os.path.join(self.dirs["generated_clips"], f"animation_{i}.mp4"))
                        for i in range(1, 6)
                    )
                    if not all_animations_exist:
                        logger.error("Not all animations exist yet")
                        return False
                    # This step is handled by the video editor
                    pass
                elif task["text"] == "Add voice-overs":
                    # Check if all voiceovers exist
                    all_voiceovers_exist = all(
                        os.path.exists(os.path.join(self.dirs["voiceovers"], f"voiceover_{i}.mp3"))
                        for i in range(1, 6)
                    )
                    if not all_voiceovers_exist:
                        logger.error("Not all voice-overs exist yet")
                        return False
                elif task["text"] == "Add background music":
                    # Check if background music exists
                    background_music_path = os.path.join("music", "background_music.mp3")
                    if not os.path.exists(background_music_path):
                        self._mark_task_failed(task["text"])
                        return False
                elif task["text"] == "Export final video":
                    # Check if all required files exist
                    final_video_path = os.path.join(self.dirs["final_video"], "final_video.mp4")
                    if os.path.exists(final_video_path):
                        logger.info("Final video already exists")
                        self._mark_task_complete(task["text"])
                        return True
                    # Create final video
                    final_video_path = create_final_video(
                        video_clips_dir=self.dirs["generated_clips"],
                        voiceover_path=os.path.join(self.dirs["voiceovers"], "voiceover_1.mp3"),
                        output_dir=self.dirs["final_video"],
                        background_music_path=os.path.join("music", "background_music.mp3"),
                        music_volume=0.1
                    )
                    if not final_video_path:
                        self._mark_task_failed(task["text"])
                        return False
            
            self._mark_task_complete(task["text"])
            return True
            
        except Exception as e:
            logger.error(f"Error running task {task['text']}: {e}")
            self._mark_task_failed(task["text"])
            return False
    
    def run_next_task(self) -> bool:
        """Run the next unchecked task."""
        task = self._get_next_unchecked_task()
        if task:
            return self.run_task(task["text"])
        return True
    
    def run_tasks(self, start_task: str, end_task: Optional[str] = None) -> bool:
        """Run a range of tasks."""
        start_index = self._get_task_index(start_task)
        if start_index == -1:
            logger.error(f"Start task not found: {start_task}")
            return False
        
        if end_task:
            end_index = self._get_task_index(end_task)
            if end_index == -1:
                logger.error(f"End task not found: {end_task}")
                return False
        else:
            end_index = len(self.tasks)
        
        for i in range(start_index, end_index):
            if not self.run_task(self.tasks[i]["text"]):
                return False
        
        return True
    
    def retry_failed_tasks(self) -> bool:
        """Retry all failed tasks from the current run."""
        failed_tasks = self._get_failed_tasks()
        if not failed_tasks:
            logger.info("No failed tasks to retry")
            return True
        
        logger.info(f"Retrying {len(failed_tasks)} failed tasks")
        for task in failed_tasks:
            if not self.run_task(task["text"]):
                return False
        
        return True
    
    def list_previous_runs(self) -> None:
        """List all previous workflow runs."""
        if not self.run_history:
            print("No previous runs found")
            return
        
        print("\nPrevious Workflow Runs:")
        print("=======================\n")
        
        for i, (workflow_dir, run_info) in enumerate(self.run_history.items(), 1):
            workflow_id = run_info.get("id", "N/A")
            status = run_info["status"]
            timestamp = run_info["timestamp"]
            failed_count = len(run_info["failed_tasks"])
            completed_count = len(run_info["completed_tasks"])
            
            print(f"{i}. Workflow #{workflow_id}: {workflow_dir}")
            print(f"   Time: {timestamp}")
            print(f"   Status: {status}")
            print(f"   Completed: {completed_count} tasks")
            print(f"   Failed: {failed_count} tasks")
            
            if failed_count > 0:
                print("   Failed tasks:")
                for task in run_info["failed_tasks"]:
                    print(f"   - {task}")
            
            print()
    
    def select_previous_run(self, run_index: int) -> bool:
        """Select a previous run to continue from."""
        if not self.run_history:
            logger.error("No previous runs found")
            return False
        
        if run_index < 1 or run_index > len(self.run_history):
            logger.error(f"Invalid run index: {run_index}")
            logger.info(f"Available run indices: 1-{len(self.run_history)}")
            return False
        
        # Get the selected run
        workflow_dir = list(self.run_history.keys())[run_index - 1]
        run_info = self.run_history[workflow_dir]
        
        # Update current workflow directory
        self.workflow_dir = workflow_dir
        self.workflow_id = run_info.get("id", self.workflow_id)
        self.checklist_file = os.path.join(workflow_dir, "workflow_checklist.md")
        
        # Update directories
        self.dirs = {
            "generated_clips": os.path.join(workflow_dir, "generated_clips"),
            "temp_files": os.path.join(workflow_dir, "temp_files"),
            "final_video": os.path.join(workflow_dir, "final_video"),
            "voiceovers": os.path.join(workflow_dir, "voiceovers")
        }
        
        # Update task status based on completed tasks
        self.tasks = self._parse_checklist()
        for task in self.tasks:
            if task["text"] in run_info["completed_tasks"]:
                task["checked"] = True
        
        # Update checklist
        self._update_checklist()
        
        logger.info(f"Selected previous run: {workflow_dir} (ID: {self.workflow_id})")
        logger.info(f"Completed tasks: {len(run_info['completed_tasks'])}")
        logger.info(f"Failed tasks: {len(run_info['failed_tasks'])}")
        
        return True
    
    def select_workflow_by_id(self, workflow_id: int) -> bool:
        """Select a workflow by its ID."""
        if not self.run_history:
            logger.error("No previous runs found")
            return False
        
        # Find the workflow with the given ID
        for workflow_dir, run_info in self.run_history.items():
            if "id" in run_info and run_info["id"] == workflow_id:
                # Update current workflow directory
                self.workflow_dir = workflow_dir
                self.workflow_id = workflow_id
                self.checklist_file = os.path.join(workflow_dir, "workflow_checklist.md")
                
                # Update directories
                self.dirs = {
                    "generated_clips": os.path.join(workflow_dir, "generated_clips"),
                    "temp_files": os.path.join(workflow_dir, "temp_files"),
                    "final_video": os.path.join(workflow_dir, "final_video"),
                    "voiceovers": os.path.join(workflow_dir, "voiceovers")
                }
                
                # Update task status based on completed tasks
                self.tasks = self._parse_checklist()
                for task in self.tasks:
                    if task["text"] in run_info["completed_tasks"]:
                        task["checked"] = True
                
                # Update checklist
                self._update_checklist()
                
                logger.info(f"Selected workflow by ID: {workflow_id} ({workflow_dir})")
                logger.info(f"Completed tasks: {len(run_info['completed_tasks'])}")
                logger.info(f"Failed tasks: {len(run_info['failed_tasks'])}")
                
                return True
        
        logger.error(f"No workflow found with ID: {workflow_id}")
        return False

def analyze_workflow_state(workflow_dir: str) -> Dict[str, Any]:
    """
    Analyze the state of a workflow by examining the files in the workflow directory.
    Returns a dictionary with the state information.
    """
    state = {
        "base_images": [],
        "animations": [],
        "voiceovers": [],
        "final_video": None,
        "completed_tasks": [],
        "failed_tasks": [],
        "next_tasks": []
    }
    
    # Check for base images
    temp_files_dir = os.path.join(workflow_dir, "temp_files")
    if os.path.exists(temp_files_dir):
        for file in os.listdir(temp_files_dir):
            if file.startswith("base_image_") and file.endswith((".png", ".jpg", ".jpeg")):
                segment_id = file.split("_")[-1].split(".")[0]
                state["base_images"].append(segment_id)
    
    # Check for animations
    generated_clips_dir = os.path.join(workflow_dir, "generated_clips")
    if os.path.exists(generated_clips_dir):
        for file in os.listdir(generated_clips_dir):
            if file.startswith("animation_") and file.endswith((".mp4", ".mov")):
                segment_id = file.split("_")[-1].split(".")[0]
                state["animations"].append(segment_id)
    
    # Check for voiceovers
    voiceovers_dir = os.path.join(workflow_dir, "voiceovers")
    if os.path.exists(voiceovers_dir):
        for file in os.listdir(voiceovers_dir):
            if file.startswith("voiceover_") and file.endswith(".mp3"):
                segment_id = file.split("_")[-1].split(".")[0]
                state["voiceovers"].append(segment_id)
    
    # Check for final video
    final_video_dir = os.path.join(workflow_dir, "final_video")
    if os.path.exists(final_video_dir):
        for file in os.listdir(final_video_dir):
            if file.endswith((".mp4", ".mov")):
                state["final_video"] = file
                break
    
    # Determine completed and failed tasks based on files
    # Base Images Generation
    for segment_id in state["base_images"]:
        state["completed_tasks"].append(f"Generate base image for segment {segment_id}")
    
    # Animation Generation
    for segment_id in state["animations"]:
        state["completed_tasks"].append(f"Generate animation for segment {segment_id}")
    
    # Voice-over Generation
    if "1" in state["voiceovers"]:
        state["completed_tasks"].append("Generate intro voice-over")
    
    for segment_id in state["voiceovers"]:
        if segment_id != "1":  # Skip intro voiceover
            state["completed_tasks"].append(f"Generate voice-over for segment {segment_id}")
    
    # Final Video Creation
    if state["final_video"]:
        state["completed_tasks"].extend([
            "Combine all animations",
            "Add voice-overs",
            "Add background music",
            "Export final video"
        ])
    
    # Determine next tasks
    all_tasks = [
        "Check API credentials",
        "Generate base image for segment 1",
        "Generate base image for segment 2",
        "Generate base image for segment 3",
        "Generate base image for segment 4",
        "Generate base image for segment 5",
        "Generate animation for segment 1",
        "Generate animation for segment 2",
        "Generate animation for segment 3",
        "Generate animation for segment 4",
        "Generate animation for segment 5",
        "Generate intro voice-over",
        "Generate voice-over for segment 1",
        "Generate voice-over for segment 2",
        "Generate voice-over for segment 3",
        "Generate voice-over for segment 4",
        "Generate voice-over for segment 5",
        "Combine all animations",
        "Add voice-overs",
        "Add background music",
        "Export final video"
    ]
    
    for task in all_tasks:
        if task not in state["completed_tasks"]:
            state["next_tasks"].append(task)
    
    return state

def resume_workflow(workflow_id: int, base_dir: str = "/Users/dev/womanareoundtheworld/workflow_output_id_") -> bool:
    """
    Resume a workflow by ID, analyzing its current state and continuing from where it left off.
    """
    # Construct the workflow directory path
    workflow_dir = f"{base_dir}{workflow_id}"
    
    # Check if the workflow directory exists
    if not os.path.exists(workflow_dir):
        logger.error(f"Workflow directory not found: {workflow_dir}")
        return False
    
    logger.info(f"Analyzing workflow {workflow_id} at {workflow_dir}")
    
    # Analyze the current state of the workflow
    state = analyze_workflow_state(workflow_dir)
    
    # Print the current state
    logger.info("Current workflow state:")
    logger.info(f"  Base images: {', '.join(state['base_images']) if state['base_images'] else 'None'}")
    logger.info(f"  Animations: {', '.join(state['animations']) if state['animations'] else 'None'}")
    logger.info(f"  Voiceovers: {', '.join(state['voiceovers']) if state['voiceovers'] else 'None'}")
    logger.info(f"  Final video: {state['final_video'] if state['final_video'] else 'None'}")
    logger.info(f"  Completed tasks: {len(state['completed_tasks'])}")
    logger.info(f"  Next tasks: {len(state['next_tasks'])}")
    
    # Create a WorkflowManager instance for this workflow
    manager = WorkflowManager(base_dir)
    
    # Select the workflow by ID
    if not manager.select_workflow_by_id(workflow_id):
        logger.error(f"Failed to select workflow {workflow_id}")
        return False
    
    logger.info(f"Selected workflow {workflow_id}")
    
    # Update the run history with the completed tasks from our analysis
    for task in state["completed_tasks"]:
        if task not in manager.run_history[manager.workflow_dir]["completed_tasks"]:
            manager.run_history[manager.workflow_dir]["completed_tasks"].append(task)
    
    # Update the checklist
    manager._update_checklist()
    
    # Run the next tasks
    if not state["next_tasks"]:
        logger.info("All tasks are already completed!")
        return True
    
    logger.info(f"Resuming workflow with {len(state['next_tasks'])} remaining tasks")
    
    # Run each next task
    for task in state["next_tasks"]:
        logger.info(f"Running task: {task}")
        if not manager.run_task(task):
            logger.error(f"Failed to run task: {task}")
            return False
    
    logger.info("Workflow resumed successfully")
    return True

def main():
    parser = argparse.ArgumentParser(description="Unified Workflow Manager for Women Around The World")
    
    # Add arguments
    parser.add_argument("--workflow-dir", type=str, default="/Users/dev/womanareoundtheworld/workflow_output_id_",
                        help="Directory to store workflow outputs (will append workflow ID)")
    parser.add_argument("--task", type=str, help="Run a specific task")
    parser.add_argument("--start", type=str, help="Start task for range (by name)")
    parser.add_argument("--end", type=str, help="End task for range (by name)")
    parser.add_argument("--start-id", type=int, help="Start task for range (by ID number)")
    parser.add_argument("--end-id", type=int, help="End task for range (by ID number)")
    parser.add_argument("--next", action="store_true", help="Run next unchecked task")
    parser.add_argument("--list", action="store_true", help="List all tasks and their status")
    parser.add_argument("--task-number", type=int, help="Run task by number")
    parser.add_argument("--retry-failed", action="store_true", help="Retry all failed tasks")
    parser.add_argument("--list-runs", action="store_true", help="List all previous workflow runs")
    parser.add_argument("--select-run", type=int, help="Select a previous run to continue from")
    parser.add_argument("--select-by-id", type=int, help="Select a workflow by its ID")
    parser.add_argument("--resume", type=int, help="Resume a workflow by ID, analyzing its current state")
    
    args = parser.parse_args()
    
    # Resume workflow if requested
    if args.resume:
        success = resume_workflow(args.resume, args.workflow_dir)
        if not success:
            logger.error("Failed to resume workflow")
            sys.exit(1)
        logger.info("Workflow resumed successfully")
        return
    
    # Create workflow manager
    manager = WorkflowManager(args.workflow_dir)
    
    # List previous runs if requested
    if args.list_runs:
        manager.list_previous_runs()
        return
    
    # Select previous run if requested
    if args.select_run:
        if manager.select_previous_run(args.select_run):
            print(f"Selected run: {manager.workflow_dir} (ID: {manager.workflow_id})")
            print("Use --list to see tasks, --next to continue, or --retry-failed to retry failed tasks")
        return
    
    # Select run by ID if requested
    if args.select_by_id:
        if manager.select_workflow_by_id(args.select_by_id):
            print(f"Selected run: {manager.workflow_dir} (ID: {manager.workflow_id})")
            print("Use --list to see tasks, --next to continue, or --retry-failed to retry failed tasks")
        return
    
    # List tasks if requested
    if args.list:
        print("\nVideo Generation Workflow Tasks:")
        print("================================\n")
        for i, task in enumerate(manager.tasks, 1):
            status = "✓" if task["checked"] else " "
            print(f"{i:2d}. [{status}] {task['section']}: {task['text']}")
        print("\nUse --task-number, --task, --start/--end, --start-id/--end-id, or --next to run tasks")
        return
    
    # Retry failed tasks if requested
    if args.retry_failed:
        success = manager.retry_failed_tasks()
        if not success:
            logger.error("Failed to retry tasks")
            sys.exit(1)
        logger.info("Successfully retried failed tasks")
        return
    
    # Run task by number if specified
    if args.task_number:
        if 1 <= args.task_number <= len(manager.tasks):
            task = manager.tasks[args.task_number - 1]
            success = manager.run_task(task["text"])
        else:
            logger.error(f"Invalid task number: {args.task_number}")
            logger.info(f"Available task numbers: 1-{len(manager.tasks)}")
            return
    
    # Run tasks based on arguments
    elif args.task:
        success = manager.run_task(args.task)
    elif args.start_id or args.end_id:
        start_task = manager.tasks[args.start_id - 1]["text"] if args.start_id else manager.tasks[0]["text"]
        end_task = manager.tasks[args.end_id - 1]["text"] if args.end_id else None
        success = manager.run_tasks(start_task, end_task)
    elif args.start:
        success = manager.run_tasks(args.start, args.end)
    elif args.next:
        success = manager.run_next_task()
    else:
        success = manager.run_next_task()
    
    if not success:
        logger.error("Workflow failed")
        sys.exit(1)
    
    logger.info("Workflow completed successfully")

if __name__ == "__main__":
    main() 