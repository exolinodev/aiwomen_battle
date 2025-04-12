"""
Command-line interface for Women Around The World.

This module provides a unified CLI interface with improved user experience features
including progress bars, status reporting, configuration support, and interactive mode.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast, Tuple

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table

from watw.config.config import Configuration, ConfigurationError
from watw.utils.common.logging_utils import setup_logger
from watw.utils.common.media_utils import (
    TransitionConfig,
    TransitionType,
    combine_video_with_audio as combine_video_audio_util,
    concatenate_videos as concat_videos_util,
    concatenate_videos_with_transitions as concat_videos_transitions_util,
    detect_beats as detect_beats_util,
    trim_video as trim_video_util,
    FFmpegError,
    VideoValidationError
)

# Set up rich console
console = Console()

# Set up logger
logger = setup_logger("watw.cli")

# Default configuration paths
DEFAULT_CONFIG_PATH = Path.home() / ".watw" / "config.yaml"
CONFIG_PATH = (
    Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    / "config"
    / "config.json"
)


class ProgressManager:
    """Manager for progress bars and status reporting."""

    def __init__(self) -> None:
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        self.tasks: Dict[str, TaskID] = {}

    def start(self) -> None:
        """Start progress tracking."""
        self.progress.start()

    def stop(self) -> None:
        """Stop progress tracking."""
        self.progress.stop()

    def add_task(self, description: str, total: Optional[float] = None) -> str:
        """Add a new task to track."""
        task_id = self.progress.add_task(description, total=total)
        self.tasks[description] = task_id
        return description

    def update(self, description: str, advance: float = 1.0, **kwargs: Any) -> None:
        """Update task progress."""
        if description in self.tasks:
            task_id = self.tasks[description]
            self.progress.update(task_id, advance=advance, **kwargs)

    def complete(self, description: str) -> None:
        """Mark a task as complete."""
        if description in self.tasks:
            task_id = self.tasks[description]
            task = self.progress.tasks[self.progress.task_ids.index(task_id)]
            if task.total is not None:
                self.progress.update(task_id, completed=task.total)
            else:
                self.progress.stop_task(task_id)
                self.progress.update(task_id, visible=False)


class WorkflowManager:
    """Manager for multi-stage workflows."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.stages: List[Dict[str, Any]] = []
        self.current_stage = 0
        self.progress = ProgressManager()

    def add_stage(
        self, name: str, description: str, total: Optional[float] = None
    ) -> None:
        """Add a workflow stage."""
        self.stages.append(
            {
                "name": name,
                "description": description,
                "total": total,
                "status": "pending",
            }
        )

    def start(self) -> None:
        """Start workflow execution."""
        console.print(Panel(f"Starting workflow: {self.name}", style="bold blue"))
        self.progress.start()
        for stage in self.stages:
            self.progress.add_task(stage["description"], total=stage["total"])

    def update_stage(self, stage_name: str, advance: float = 1) -> None:
        """Update current stage progress."""
        for stage in self.stages:
            if stage["name"] == stage_name:
                self.progress.update(stage["description"], advance=advance)
                break

    def complete_stage(self, stage_name: str, status: str = "completed") -> None:
        """Mark a stage as complete."""
        for stage in self.stages:
            if stage["name"] == stage_name:
                stage["status"] = status
                self.progress.complete(stage["description"])
                console.print(f"✓ {stage['name']}: {status}", style="green")
                break

    def show_summary(self) -> None:
        """Show workflow summary."""
        table = Table(title=f"Workflow Summary: {self.name}")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="green")

        for stage in self.stages:
            table.add_row(stage["name"], stage["status"])

        console.print(table)
        self.progress.stop()


def load_config(config_path: Optional[str] = None) -> Configuration:
    """Load configuration from default or specified path."""
    if config_path:
        return Configuration(config_path)
    return Configuration(CONFIG_PATH)


@click.group()
@click.option("--config", "-c", help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def cli(config: Optional[str], verbose: bool, quiet: bool) -> None:
    """Women Around The World - Video Generation Tool"""
    load_config(config)

    # Set up logging
    log_level = "DEBUG" if verbose else "INFO"
    if quiet:
        log_level = "WARNING"

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--start", "-s", type=float, help="Start time in seconds")
@click.option("--duration", "-d", type=float, help="Duration in seconds")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
def trim(
    video_path: str,
    output_path: str,
    start: Optional[float],
    duration: Optional[float],
    interactive: bool,
) -> None:
    """Trim a video file."""
    load_config()

    if interactive:
        if start is None:
            start = float(Prompt.ask("Enter start time (seconds)"))
        if duration is None:
            duration = float(Prompt.ask("Enter duration (seconds)"))

    if start is None or duration is None:
        console.print("[red]Error: Start time and duration are required[/red]")
        return

    workflow = WorkflowManager("Trim Video")
    workflow.add_stage("validate", "Validating input video")
    workflow.add_stage("trim", "Trimming video")

    try:
        workflow.start()
        workflow.complete_stage("validate")

        trim_video_util(
            input_path=video_path,
            output_path=output_path,
            start_time=start,
            duration=duration,
        )

        workflow.complete_stage("trim")
        workflow.show_summary()
        console.print(f"[green]Video trimmed successfully: {output_path}[/green]")
    except Exception as e:
        workflow.complete_stage("trim", "failed")
        workflow.show_summary()
        console.print(f"[red]Error trimming video: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.argument("audio_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--volume", "-v", type=float, help="Background music volume (0.0-1.0)")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
def combine(
    video_path: str,
    audio_path: str,
    output_path: str,
    volume: Optional[float],
    interactive: bool,
) -> None:
    """Combine video with audio."""
    load_config()

    if interactive:
        if volume is None:
            volume = float(Prompt.ask("Enter volume (0.0-1.0)", default="1.0"))

    workflow = WorkflowManager("Combine Video with Audio")
    workflow.add_stage("validate", "Validating input files")
    workflow.add_stage("combine", "Combining video with audio")

    try:
        workflow.start()
        workflow.complete_stage("validate")

        combine_video_audio_util(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            volume=volume,
        )

        workflow.complete_stage("combine")
        workflow.show_summary()
        console.print(f"[green]Video combined successfully: {output_path}[/green]")
    except Exception as e:
        workflow.complete_stage("combine", "failed")
        workflow.show_summary()
        console.print(f"[red]Error combining video: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("video_paths", nargs=-1, type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--transition", "-t", type=str, help="Transition type")
@click.option("--duration", "-d", type=float, help="Transition duration")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
def concat(
    video_paths: Tuple[str, ...],
    output_path: str,
    transition: Optional[str],
    duration: Optional[float],
    interactive: bool,
) -> None:
    """Concatenate multiple videos."""
    load_config()

    if interactive:
        if transition is None:
            transition = Prompt.ask(
                "Enter transition type",
                choices=list(TransitionType.__members__.keys()),
                default="none",
            )
        if duration is None:
            duration = float(
                Prompt.ask("Enter transition duration (seconds)", default="0.5")
            )

    workflow = WorkflowManager("Concatenate Videos")
    workflow.add_stage("validate", "Validating input videos")
    workflow.add_stage("concat", "Concatenating videos")

    try:
        workflow.start()
        workflow.complete_stage("validate")

        # Create transition config
        transition_config: Optional[TransitionConfig] = None
        if transition:
            try:
                transition_type_enum = TransitionType[transition.upper()]
                transition_config = TransitionConfig(type=transition_type_enum, duration=duration or 0.5)
            except KeyError:
                console.print(f"[yellow]Warning: Invalid transition type '{transition}'. Using CUT.[/yellow]")
                transition_config = TransitionConfig(type=TransitionType.CUT, duration=0)

        # Create list of transitions
        num_videos = len(video_paths)
        transitions_list: Optional[List[TransitionConfig]] = None
        if num_videos > 1 and transition_config:
            transitions_list = [transition_config] * (num_videos - 1)

        concat_videos_transitions_util(
            video_paths=list(video_paths),
            output_path=output_path,
            transitions=transitions_list
        )

        workflow.complete_stage("concat")
        workflow.show_summary()
        console.print(f"[green]Videos concatenated successfully: {output_path}[/green]")
    except Exception as e:
        workflow.complete_stage("concat", "failed")
        workflow.show_summary()
        console.print(f"[red]Error concatenating videos: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output path for visualization")
@click.option("--min-bpm", type=float, help="Minimum BPM")
@click.option("--max-bpm", type=float, help="Maximum BPM")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
def detect_beats_command(
    audio_path: str,
    output: Optional[str],
    min_bpm: Optional[float],
    max_bpm: Optional[float],
    interactive: bool,
) -> None:
    """Detect beats in an audio file."""
    load_config()

    if interactive:
        if min_bpm is None:
            min_bpm = float(Prompt.ask("Enter minimum BPM", default="60"))
        if max_bpm is None:
            max_bpm = float(Prompt.ask("Enter maximum BPM", default="180"))

    workflow = WorkflowManager("Detect Beats")
    workflow.add_stage("analyze", "Analyzing audio file")
    workflow.add_stage("detect", "Detecting beats")

    try:
        workflow.start()
        workflow.complete_stage("analyze")

        # Provide defaults if arguments are None
        default_min_bpm = 60.0
        default_max_bpm = 180.0

        beats = detect_beats_util(
            audio_path=audio_path,
            output_path=output,
            min_bpm=min_bpm if min_bpm is not None else default_min_bpm,
            max_bpm=max_bpm if max_bpm is not None else default_max_bpm,
        )

        workflow.complete_stage("detect")
        workflow.show_summary()

        if output:
            console.print(
                f"[green]Beat detection visualization saved to: {output}[/green]"
            )
        else:
            console.print(f"[green]Detected {len(beats)} beats[/green]")
    except Exception as e:
        workflow.complete_stage("detect", "failed")
        workflow.show_summary()
        console.print(f"[red]Error detecting beats: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode for configuration")
@click.pass_context
def config(ctx: click.Context, interactive: bool) -> None:
    """View or interactively update configuration settings."""
    config_obj: Configuration = ctx.obj['config']
    config_path = config_obj.config_file

    if interactive:
        console.print(Panel("Interactive Configuration Mode", style="bold blue"))
        
        elevenlabs_key = Prompt.ask(
            "ElevenLabs API Key",
            default=config_obj.get("elevenlabs_api_key", ""),
            password=True,
        )
        
        # Add other configuration options as needed
        
        if Confirm.ask("Save configuration changes?"):
            try:
                if elevenlabs_key:
                    config_obj.set("elevenlabs_api_key", elevenlabs_key)
                # Add other settings as needed
                config_obj.save()
                console.print("[green]Configuration saved successfully![/green]")
            except ConfigurationError as e:
                console.print(f"[red]Configuration error: {str(e)}[/red]")
            except Exception as e:
                console.print(f"[red]Error saving configuration: {str(e)}[/red]")
    else:
        # Display current configuration
        console.print(Panel(f"Current Configuration ({config_path})", style="bold yellow"))
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in config_obj.config_data.items():
            # Mask sensitive values
            display_value = "********" if "key" in key.lower() or "secret" in key.lower() else str(value)
            table.add_row(key, display_value)
        
        console.print(table)


if __name__ == "__main__":
    cli()
