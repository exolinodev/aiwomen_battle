#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, cast, Set
from pathlib import Path

# Import from your refactored scripts
try:
    sys.path.append(os.path.dirname(__file__))
    from watw.core.render import (
        ANIMATION_PROMPTS,
        ANIMATION_SEED_START,
        check_credentials,
        generate_all_base_images,
        generate_animation_runway,
        generate_base_image_tensorart,
    )
    # from watw.utils.common.media_utils import create_final_video  # Removed import
    from watw.core.voiceover import VoiceoverGenerator
    from watw.utils.common.logging_utils import setup_logger
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(
        "Ensure render.py, voiceover.py, and media_utils.py are in the correct directory."
    )
    sys.exit(1)

# Set up logging
logger = setup_logger(
    name="workflow_manager", level=logging.INFO, log_file="workflow_manager.log"
)

# Define workflow stages
WORKFLOW_STAGES = {
    "base_images": "Generate base images",
    "animations": "Generate animations",
    "voiceover": "Generate voice-over",
    "final_video": "Create final video",
}


class WorkflowManager:
    def __init__(self, workflow_dir: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the WorkflowManager with a workflow directory and optional configuration.

        Args:
            workflow_dir (str): Base directory for workflow output
            config (Dict[str, Any], optional): Configuration dictionary containing:
                - countries: List of countries to process
                - background_music_path: Path to background music file
                - music_volume: Volume level for background music (0.0-1.0)
        """
        self.workflow_dir = Path(workflow_dir)
        self.config = config or {}
        self.run_history = {}
        self.workflow_id = self._generate_workflow_id()
        self.checklist: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {
            "base_images": [],
            "animations": [],
            "voiceover": None,
            "final_video": None,
            "current_stage": None,
            "completed_tasks": [],
            "failed_tasks": [],
        }
        self.voiceover_gen = VoiceoverGenerator()
        
        # Initialize directory structure
        self.dirs: Dict[str, Path] = {
            "base": self.workflow_dir,
            "temp_files": self.workflow_dir / "temp_files",
            "generated_clips": self.workflow_dir / "generated_clips",
            "voiceovers": self.workflow_dir / "voiceovers",
            "final_video": self.workflow_dir / "final_video",
        }
        
        # Ensure directories exist
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.initialize_components()

    def initialize_components(self) -> bool:
        """
        Initialize all necessary components for the workflow.

        Returns:
            bool: True if all components initialized successfully, False otherwise
        """
        try:
            # Check API credentials
            if not check_credentials():
                logger.error("Failed to initialize: API credentials check failed")
                return False

            # Verify directories
            for dir_name, dir_path in self.dirs.items():
                if not os.path.exists(dir_path):
                    logger.error(
                        f"Failed to initialize: Directory {dir_name} does not exist"
                    )
                    return False

            # Verify configuration
            if "countries" not in self.config:
                logger.warning("No countries specified in configuration")

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error during component initialization: {e}")
            return False

    def execute_workflow(self) -> bool:
        """
        Execute the complete workflow.

        Returns:
            bool: True if workflow completed successfully, False otherwise
        """
        try:
            # Initialize components
            if not self.initialize_components():
                return False

            # Stage 1: Generate Base Images
            self.state["current_stage"] = "base_images"
            logger.info("=" * 20 + " Stage 1: Generating Base Images " + "=" * 20)

            base_images = generate_all_base_images(
                output_directory=self.dirs["temp_files"]
            )
            if not base_images:
                logger.error("Failed to generate base images")
                return False

            self.state["base_images"] = base_images
            self._mark_task_complete("Generate base image for segment 1")
            logger.info(f"Successfully generated {len(base_images)} base images")

            # Stage 2: Generate Animations
            self.state["current_stage"] = "animations"
            logger.info("=" * 20 + " Stage 2: Generating Animations " + "=" * 20)

            current_seed = ANIMATION_SEED_START
            for base_image in base_images:
                base_image_id = base_image["id"]
                base_image_path = base_image["path"]

                logger.info(f"Generating animation for base image: {base_image_id}")

                anim_prompt = ANIMATION_PROMPTS[0]
                output_filename_base = f"animation_{base_image_id}"

                video_path = generate_animation_runway(
                    base_image_path=base_image_path,
                    animation_prompt_text=anim_prompt["text"],
                    output_directory=self.dirs["generated_clips"],
                    output_filename_base=output_filename_base,
                    seed=current_seed,
                )

                if video_path:
                    self.state["animations"].append(video_path)
                    self._mark_task_complete(
                        f"Generate animation for segment {len(self.state['animations'])}"
                    )
                    logger.info(f"Animation for {base_image_id} saved to: {video_path}")
                else:
                    logger.warning(
                        f"Failed to generate animation for base image: {base_image_id}"
                    )

                current_seed += 1

            if not self.state["animations"]:
                logger.error("No animations were successfully generated")
                return False

            # Stage 3: Generate Voice-over
            self.state["current_stage"] = "voiceover"
            logger.info("=" * 20 + " Stage 3: Generating Voice-over " + "=" * 20)

            countries = self.config.get(
                "countries", ["Japan", "France"]
            )  # Default countries if not specified
            for i, (country1, country2) in enumerate(
                zip(countries[::2], countries[1::2]), 1
            ):
                voiceover_filename = f"voiceover_{i}.mp3"
                output_path = self.dirs["voiceovers"] / voiceover_filename
                
                voiceover_path = self.voiceover_gen.generate_voiceover(
                    video_number=i,
                    country1=country1,
                    country2=country2,
                    output_path=output_path,
                )

                if voiceover_path:
                    self.state["voiceover"] = str(voiceover_path)
                    self._mark_task_complete(f"Generate voice-over for segment {i}")
                    logger.info(f"Voice-over for video {i} saved to: {voiceover_path}")
                else:
                    logger.warning(f"Failed to generate voice-over for video {i}")

            # Stage 4: Create Final Video
            self.state["current_stage"] = "final_video"
            logger.info("=" * 20 + " Stage 4: Creating Final Video " + "=" * 20)

            background_music_path = self.config.get("background_music_path")
            music_volume = self.config.get("music_volume", 0.1)

            # Handle background music path
            effective_music_path: Optional[Path] = None
            if background_music_path:
                bg_path = Path(background_music_path)
                if bg_path.is_file():
                    effective_music_path = bg_path
                else:
                    logger.warning(f"Configured background music path not found: {bg_path}")
                    # Try default location
                    default_path = bg_path.parent / "background_music.mp3"
                    if default_path.is_file():
                        effective_music_path = default_path

            # Handle voiceover path
            effective_voiceover_path: Optional[Path] = None
            if self.state["voiceover"]:
                vo_path = Path(self.state["voiceover"])
                if vo_path.is_file():
                    effective_voiceover_path = vo_path
                else:
                    logger.warning(f"Voiceover path from state not found: {vo_path}")
                    # Try default location
                    default_vo = self.dirs["voiceovers"] / "voiceover_1.mp3"
                    if default_vo.is_file():
                        effective_voiceover_path = default_vo

            if not effective_voiceover_path:
                logger.error("Cannot find a valid voiceover file")
                return False

            # Placeholder for final video creation - needs refactoring to use VideoEditor
            final_video_path = None
            logger.warning("Final video creation step needs refactoring to use a VideoEditor instance.")

            if final_video_path:
                self.state["final_video"] = str(final_video_path)
                self._mark_task_complete("Create final video")
                logger.info(f"Final video saved to: {final_video_path}")
            else:
                logger.error("Failed to create final video (Refactoring needed)")
                return False

            # Update workflow status
            self.run_history[str(self.workflow_dir)]["status"] = "completed"
            self._save_run_history()

            logger.info("Workflow completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Error during workflow execution: {e}")
            self.state["failed_tasks"].append(str(e))
            self.run_history[str(self.workflow_dir)]["status"] = "failed"
            self._save_run_history()
            return False

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
        """Load the run history from the history file."""
        history_file = self.workflow_dir / "run_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    return cast(Dict[str, Any], json.load(f))
            except Exception as e:
                logger.error(f"Error loading run history: {e}")
        return {}

    def _save_run_history(self) -> None:
        """Save workflow run history."""
        history_file = self.workflow_dir / "workflow_history.json"
        try:
            with open(history_file, "w") as f:
                json.dump(self.run_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving run history: {e}")

    def _parse_checklist(self) -> List[Dict[str, Any]]:
        """Parse the workflow checklist."""
        self.checklist_file: Path = self.workflow_dir / "workflow_checklist.md"
        if not self.checklist_file.exists():
            self._create_default_checklist()
            logger.warning(f"Checklist file not found: {self.checklist_file}. Created default checklist.")
            return []
            
        try:
            with open(self.checklist_file, "r") as f:
                content = f.read()
                # Parse markdown content into checklist items
                checklist = []
                current_section = ""
                for line in content.split("\n"):
                    if line.startswith("## "):
                        current_section = line[3:].strip()
                    elif line.startswith("- [ ] ") or line.startswith("- [x] "):
                        checked = line.startswith("- [x] ")
                        text = line[6:].strip()
                        checklist.append({
                            "section": current_section,
                            "text": text,
                            "checked": checked
                        })
                return checklist
        except Exception as e:
            logger.error(f"Error parsing checklist: {e}")
            return []

    def _create_default_checklist(self) -> None:
        """Create a default checklist file if it doesn't exist."""
        checklist_path = self.checklist_file
        checklist_path.parent.mkdir(parents=True, exist_ok=True)

        default_tasks = [
            {"section": "Setup", "text": "Check API credentials", "checked": False},
            {"section": "Base Images Generation", "text": "Generate base image for segment 1", "checked": False},
            {"section": "Base Images Generation", "text": "Generate base image for segment 2", "checked": False},
            {"section": "Animation Generation", "text": "Generate animation for segment 1", "checked": False},
            {"section": "Animation Generation", "text": "Generate animation for segment 2", "checked": False},
            {"section": "Voice-over Generation", "text": "Generate voice-over for segment 1", "checked": False},
            {"section": "Voice-over Generation", "text": "Generate voice-over for segment 2", "checked": False},
            {"section": "Final Video", "text": "Export final video", "checked": False},
        ]

        try:
            with open(checklist_path, "w") as f:
                f.write("# Video Generation Workflow Checklist\n\n")
                current_section = ""
                for task in default_tasks:
                    if task["section"] != current_section:
                        f.write(f"## {task['section']}\n")
                        current_section = task["section"]
                    status = "x" if task["checked"] else " "
                    f.write(f"- [{status}] {task['text']}\n")
                f.write("\n## Output Information\n")
                f.write(f"- Workflow Directory: {str(self.dirs['base'].name)}\n")
                f.write(f"- Generated Clips: {str(self.dirs['generated_clips'].name)}\n")
                f.write(f"- Voiceovers: {str(self.dirs['voiceovers'].name)}\n")
                f.write(f"- Final Video: {self.dirs['final_video'].name}\n")
            logger.info(f"Created default checklist file: {checklist_path}")
        except Exception as e:
            logger.error(f"Failed to create default checklist: {e}")

    def _update_checklist(self) -> None:
        """Update the markdown checklist with current task status."""
        sections: Dict[str, List[Dict[str, Any]]] = {}
        for task in self.checklist:
            if task["section"] not in sections:
                sections[task["section"]] = []
            sections[task["section"]].append(task)

        checklist_file = self.workflow_dir / "workflow_checklist.md"
        try:
            with open(checklist_file, "w") as f:
                for section, tasks in sections.items():
                    f.write(f"## {section}\n\n")
                    for task in tasks:
                        status = "x" if task["checked"] else " "
                        f.write(f"- [{status}] {task['text']}\n")
                    f.write("\n")
        except Exception as e:
            logger.error(f"Error updating checklist: {e}")

    def _get_task_index(self, task_text: str) -> int:
        """Get the index of a task by its text."""
        for i, task in enumerate(self.checklist):
            if task["text"] == task_text:
                return i
        return -1

    def _mark_task_complete(self, task_text: str) -> bool:
        """Mark a task as complete and update run history."""
        for task in self.checklist:
            if task["text"] == task_text:
                task["checked"] = True
                # Update run history
                if (
                    task_text
                    not in self.run_history[str(self.workflow_dir)]["completed_tasks"]
                ):
                    self.run_history[str(self.workflow_dir)]["completed_tasks"].append(
                        task_text
                    )
                if (
                    task_text
                    in self.run_history[str(self.workflow_dir)]["failed_tasks"]
                ):
                    self.run_history[str(self.workflow_dir)]["failed_tasks"].remove(
                        task_text
                    )

                # Update workflow status
                if len(self.run_history[str(self.workflow_dir)]["failed_tasks"]) == 0:
                    # Check if all tasks are completed
                    all_tasks = [t["text"] for t in self.checklist]
                    completed_tasks = self.run_history[str(self.workflow_dir)][
                        "completed_tasks"
                    ]
                    if all(task in completed_tasks for task in all_tasks):
                        self.run_history[str(self.workflow_dir)]["status"] = "completed"
                    else:
                        self.run_history[str(self.workflow_dir)]["status"] = (
                            "in_progress"
                        )

                self._update_checklist()
                self._save_run_history()
                return True
        return False

    def _mark_task_failed(self, task_text: str) -> None:
        """Mark a task as failed and update run history."""
        # Update run history
        if task_text not in self.run_history[str(self.workflow_dir)]["failed_tasks"]:
            self.run_history[str(self.workflow_dir)]["failed_tasks"].append(task_text)

        # Update workflow status
        self.run_history[str(self.workflow_dir)]["status"] = "failed"

        self._save_run_history()

    def _get_next_unchecked_task(self) -> Optional[Dict[str, Any]]:
        """Get the first unchecked task."""
        for task in self.checklist:
            if not task["checked"]:
                return task
        return None

    def _get_failed_tasks(self) -> List[Dict[str, Any]]:
        """Get all failed tasks from the current run."""
        failed_tasks = []
        for task_text in self.run_history[str(self.workflow_dir)]["failed_tasks"]:
            task = next((t for t in self.checklist if t["text"] == task_text), None)
            if task:
                failed_tasks.append(task)
        return failed_tasks

    def run_task(self, task_text: str) -> bool:
        """
        Execute a specific task in the workflow.

        Args:
            task_text (str): Description of the task to execute

        Returns:
            bool: True if task completed successfully, False otherwise
        """
        try:
            # Parse task details
            task_parts = task_text.split()
            if not task_parts:
                logger.error("Invalid task format")
                return False

            task_type = task_parts[0].lower()
            segment_id = None
            if len(task_parts) > 1:
                segment_id = task_parts[-1]

            # Handle different task types
            if task_type == "generate" and "base image" in task_text:
                # Base image generation
                try:
                    segment_id_str = str(segment_id) if segment_id else "1"
                    base_image_filename = f"base_image_{segment_id_str}.png"
                    base_image_path = self.dirs["temp_files"] / base_image_filename

                    prompt_text = f"Prompt for segment {segment_id_str}"  # Replace with actual prompt logic
                    
                    # Generate base image
                    img_path_obj: Optional[Path]
                    img_url: Optional[str]
                    img_path_obj, img_url = generate_base_image_tensorart(
                        output_directory=self.dirs["temp_files"],
                        prompt_text=prompt_text,
                    )

                    # Check if path object is not None
                    if img_path_obj:
                        # Convert Path to string for state dictionary
                        img_path_str = str(img_path_obj)

                        # Rename logic using the Path object
                        if img_path_obj.name != base_image_filename:
                            try:
                                img_path_obj.rename(base_image_path)
                                logger.info(f"Renamed generated image to {base_image_filename}")
                                img_path_str = str(base_image_path)  # Update str path after rename
                            except OSError as rename_e:
                                logger.error(f"Failed to rename {img_path_obj} to {base_image_path}: {rename_e}")
                                self._mark_task_failed(task_text)
                                return False

                        # Store the string path in the state
                        self.state["base_images"].append({"id": segment_id_str, "path": img_path_str})
                        self._mark_task_complete(task_text)
                        return True
                    else:
                        # Handle the case where image generation failed
                        logger.error(f"generate_base_image_tensorart returned None for segment {segment_id_str}")
                        self._mark_task_failed(task_text)
                        return False

                except Exception as gen_e:
                    logger.error(f"Error generating base image: {str(gen_e)}")
                    self._mark_task_failed(task_text)
                    return False

            elif task_type == "generate" and "animation" in task_text:
                # Animation generation
                try:
                    segment_id_str = str(segment_id) if segment_id else "1"
                    animation_filename = f"animation_{segment_id_str}.mp4"
                    animation_path = self.dirs["generated_clips"] / animation_filename

                    if animation_path.exists():
                        logger.info(f"Animation for segment {segment_id_str} already exists at {animation_path}")
                        self._mark_task_complete(task_text)
                        return True

                    base_image_info = next(
                        (img for img in self.state.get("base_images", []) if img.get("id") == segment_id_str),
                        None
                    )

                    if base_image_info and Path(base_image_info["path"]).exists():
                        base_image_path = Path(base_image_info["path"])
                        try:
                            video_path = generate_animation_runway(
                                base_image_path=base_image_path,
                                animation_prompt_text=ANIMATION_PROMPTS[0]["text"],
                                output_directory=self.dirs["generated_clips"],
                                output_filename_base=f"animation_{segment_id_str}",
                                seed=ANIMATION_SEED_START + int(segment_id_str) - 1,
                            )
                            if video_path:
                                self.state["animations"].append({"id": segment_id_str, "path": str(video_path)})
                                self._mark_task_complete(task_text)
                                return True
                            else:
                                self._mark_task_failed(task_text)
                                return False
                        except Exception as gen_e:
                            logger.error(f"Error generating animation for segment {segment_id_str}: {gen_e}")
                            self._mark_task_failed(task_text)
                            return False
                    else:
                        logger.error(f"Base image for segment {segment_id_str} not found or generation failed previously.")
                        self._mark_task_failed(task_text)
                        return False

                except Exception as gen_e:
                    logger.error(f"Error generating animation: {str(gen_e)}")
                    self._mark_task_failed(task_text)
                    return False

            elif task_type == "generate" and "voice-over" in task_text:
                # Voice-over generation
                try:
                    segment_id_str = str(segment_id) if segment_id else "1"
                    voiceover_filename = f"voiceover_{segment_id_str}.mp3"
                    output_path = self.dirs["voiceovers"] / voiceover_filename

                    if output_path.exists():
                        logger.info(f"Voiceover for segment {segment_id_str} already exists at {output_path}")
                        self._mark_task_complete(task_text)
                        return True

                    voiceover_path = self.voiceover_gen.generate_voiceover(
                        video_number=int(segment_id_str),
                        country1=f"Segment {segment_id_str}",
                        country2="Placeholder",
                        output_path=output_path,
                    )
                    if voiceover_path:
                        self.state["voiceover"] = str(voiceover_path)
                        self._mark_task_complete(task_text)
                        return True
                    else:
                        self._mark_task_failed(task_text)
                        return False

                except Exception as gen_e:
                    logger.error(f"Error generating voiceover: {str(gen_e)}")
                    self._mark_task_failed(task_text)
                    return False

            elif task_type == "export" and "final video" in task_text:
                # Final video export
                try:
                    # ... existing final video generation code ...
                    pass

                except Exception as gen_e:
                    logger.error(f"Error exporting final video: {str(gen_e)}")
                    self._mark_task_failed(task_text)
                    return False

            return False

        except Exception as e:
            logger.error(f"Error running task {task_text}: {e}")
            self._mark_task_failed(task_text)
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
            end_index = len(self.checklist)

        for i in range(start_index, end_index):
            if not self.run_task(self.checklist[i]["text"]):
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
        """Select a previous run by index."""
        if not self.run_history:
            logger.error("No previous runs found")
            return False

        if run_index < 1 or run_index > len(self.run_history):
            logger.error(f"Invalid run index: {run_index}")
            return False

        workflow_dir_str = list(self.run_history.keys())[run_index - 1]
        run_info = self.run_history[workflow_dir_str]
        self.workflow_dir = Path(workflow_dir_str)
        self.workflow_id = run_info.get("id", 0)
        self.state = run_info.get("state", {})
        self.checklist = run_info.get("checklist", [])
        return True

    def select_workflow_by_id(self, workflow_id: int) -> bool:
        """Select a workflow by its ID."""
        for workflow_dir_str, run_info in self.run_history.items():
            if "id" in run_info and run_info["id"] == workflow_id:
                self.workflow_dir = Path(workflow_dir_str)
                self.workflow_id = workflow_id
                self.state = run_info.get("state", {})
                self.checklist = run_info.get("checklist", [])
                return True
        return False


def analyze_workflow_state(workflow_dir: str) -> Dict[str, Any]:
    """Analyze the current state of a workflow directory."""
    workflow_dir_path = Path(workflow_dir)
    state: Dict[str, Any] = {
        "base_images": [],
        "animations": [],
        "voiceover": None,
        "final_video": None,
        "completed_tasks": [],
        "next_tasks": [],
    }

    # Define directory paths
    temp_files_dir = workflow_dir_path / "temp_files"
    generated_clips_dir = workflow_dir_path / "generated_clips"
    voiceovers_dir = workflow_dir_path / "voiceovers"
    final_video_dir = workflow_dir_path / "final_video"

    # Analyze base images
    for i in range(1, 6):  # Assuming 5 segments
        task_text = f"Generate base image for segment {i}"
        base_image_path = temp_files_dir / f"base_image_{i}.png"
        if base_image_path.exists() and base_image_path.stat().st_size > 0:
            state["base_images"].append({"id": str(i), "path": str(base_image_path)})
            state["completed_tasks"].append(task_text)
        else:
            state["next_tasks"].append(task_text)

    # Analyze animations
    for i in range(1, 6):
        task_text = f"Generate animation for segment {i}"
        animation_path = generated_clips_dir / f"animation_{i}.mp4"
        if animation_path.exists() and animation_path.stat().st_size > 0:
            state["animations"].append({"id": str(i), "path": str(animation_path)})
            state["completed_tasks"].append(task_text)
        else:
            state["next_tasks"].append(task_text)

    # Analyze voiceovers
    for i in range(1, 6):
        task_text = f"Generate voice-over for segment {i}"
        voiceover_path = voiceovers_dir / f"voiceover_{i}.mp3"
        if voiceover_path.exists() and voiceover_path.stat().st_size > 0:
            if i == 1:  # Use first voiceover as the main one
                state["voiceover"] = str(voiceover_path)
            state["completed_tasks"].append(task_text)
        else:
            state["next_tasks"].append(task_text)

    # Analyze final video
    final_video_task = "Export final video"
    final_video_path = final_video_dir / "final_video.mp4"
    if final_video_path.exists() and final_video_path.stat().st_size > 0:
        state["final_video"] = str(final_video_path)
        state["completed_tasks"].append(final_video_task)
    else:
        state["next_tasks"].append(final_video_task)

    # Remove duplicates from next_tasks while preserving order
    unique_next_tasks: List[str] = []
    seen: Set[str] = set()
    for task in state["next_tasks"]:
        if task not in seen:
            unique_next_tasks.append(task)
            seen.add(task)
    state["next_tasks"] = unique_next_tasks

    return state


def resume_workflow(
    workflow_id: int,
    base_dir: str = "/Users/dev/womanareoundtheworld/workflow_output_id_",
) -> bool:
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
    logger.info(
        f"  Base images: {', '.join(state['base_images']) if state['base_images'] else 'None'}"
    )
    logger.info(
        f"  Animations: {', '.join(state['animations']) if state['animations'] else 'None'}"
    )
    logger.info(
        f"  Voiceovers: {', '.join(state['voiceovers']) if state['voiceovers'] else 'None'}"
    )
    logger.info(
        f"  Final video: {state['final_video'] if state['final_video'] else 'None'}"
    )
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
        if (
            task
            not in manager.run_history[str(manager.workflow_dir)]["completed_tasks"]
        ):
            manager.run_history[str(manager.workflow_dir)]["completed_tasks"].append(
                task
            )

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


def main() -> None:
    """Main entry point for the workflow manager."""
    parser = argparse.ArgumentParser(description="Workflow Manager for Video Generation")
    parser.add_argument("--workflow-dir", required=True, help="Workflow directory")
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)

    manager = WorkflowManager(args.workflow_dir, config)
    manager.execute_workflow()


if __name__ == "__main__":
    main()
