"""
Command-line interface for Women Around The World.

This module provides a unified CLI interface with improved user experience features
including progress bars, status reporting, configuration support, and interactive mode.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

import click
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.logging import RichHandler

from watw.utils.common.media_utils import (
    trim_video,
    combine_video_with_audio,
    add_background_music,
    concatenate_videos,
    concatenate_videos_with_transitions,
    TransitionType,
    TransitionConfig,
    detect_beats
)
from watw.utils.common.validation_utils import validate_file_exists, validate_directory_exists
from watw.utils.common.logging_utils import setup_logger

# Set up rich console
console = Console()

# Set up logger
logger = setup_logger("watw.cli")

class Config:
    """Configuration manager for WATW."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else Path.home() / ".watw" / "config.yaml"
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                self.config = {}
        else:
            self.config = self._default_config()
            self.save()
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'output_dir': str(Path.home() / "Videos" / "WATW"),
            'temp_dir': str(Path.home() / ".watw" / "temp"),
            'default_transition': 'crossfade',
            'transition_duration': 1.0,
            'background_music_volume': 0.1,
            'log_level': 'INFO',
            'interactive_mode': True
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
        self.save()

class ProgressManager:
    """Manager for progress bars and status reporting."""
    
    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console
        )
        self.tasks: Dict[str, int] = {}
    
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
    
    def update(self, description: str, advance: float = 1) -> None:
        """Update task progress."""
        if description in self.tasks:
            self.progress.update(self.tasks[description], advance=advance)
    
    def complete(self, description: str) -> None:
        """Mark a task as complete."""
        if description in self.tasks:
            self.progress.update(self.tasks[description], completed=True)

class WorkflowManager:
    """Manager for multi-stage workflows."""
    
    def __init__(self, name: str):
        self.name = name
        self.stages: List[Dict[str, Any]] = []
        self.current_stage = 0
        self.progress = ProgressManager()
    
    def add_stage(self, name: str, description: str, total: Optional[float] = None) -> None:
        """Add a workflow stage."""
        self.stages.append({
            'name': name,
            'description': description,
            'total': total,
            'status': 'pending'
        })
    
    def start(self) -> None:
        """Start workflow execution."""
        console.print(Panel(f"Starting workflow: {self.name}", style="bold blue"))
        self.progress.start()
        for stage in self.stages:
            self.progress.add_task(stage['description'], total=stage['total'])
    
    def update_stage(self, stage_name: str, advance: float = 1) -> None:
        """Update current stage progress."""
        for stage in self.stages:
            if stage['name'] == stage_name:
                self.progress.update(stage['description'], advance=advance)
                break
    
    def complete_stage(self, stage_name: str, status: str = 'completed') -> None:
        """Mark a stage as complete."""
        for stage in self.stages:
            if stage['name'] == stage_name:
                stage['status'] = status
                self.progress.complete(stage['description'])
                console.print(f"✓ {stage['name']}: {status}", style="green")
                break
    
    def show_summary(self) -> None:
        """Show workflow summary."""
        table = Table(title=f"Workflow Summary: {self.name}")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="green")
        
        for stage in self.stages:
            table.add_row(stage['name'], stage['status'])
        
        console.print(table)
        self.progress.stop()

def load_config() -> Config:
    """Load configuration from default or specified path."""
    config_path = os.environ.get('WATW_CONFIG')
    return Config(config_path)

@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def cli(config: Optional[str], verbose: bool, quiet: bool):
    """Women Around The World - Video Generation Tool"""
    cfg = load_config()
    
    # Set up logging
    log_level = 'DEBUG' if verbose else 'INFO'
    if quiet:
        log_level = 'WARNING'
    
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--start', '-s', type=float, help='Start time in seconds')
@click.option('--duration', '-d', type=float, help='Duration in seconds')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
def trim(video_path: str, output_path: str, start: Optional[float], duration: Optional[float], interactive: bool):
    """Trim a video file."""
    cfg = load_config()
    
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
        
        trim_video(
            input_path=video_path,
            output_path=output_path,
            start_time=start,
            duration=duration
        )
        
        workflow.complete_stage("trim")
        workflow.show_summary()
        
        console.print(f"[green]Successfully trimmed video: {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        workflow.complete_stage("trim", "failed")
        workflow.show_summary()

@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.argument('audio_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--volume', '-v', type=float, help='Background music volume (0.0-1.0)')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
def combine(video_path: str, audio_path: str, output_path: str, volume: Optional[float], interactive: bool):
    """Combine video with audio."""
    cfg = load_config()
    
    if volume is None:
        volume = cfg.get('background_music_volume', 0.1)
        if interactive:
            volume = float(Prompt.ask("Enter background music volume (0.0-1.0)", default=str(volume)))
    
    workflow = WorkflowManager("Combine Video and Audio")
    workflow.add_stage("validate", "Validating input files")
    workflow.add_stage("combine", "Combining video and audio")
    
    try:
        workflow.start()
        workflow.complete_stage("validate")
        
        combine_video_with_audio(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path
        )
        
        workflow.complete_stage("combine")
        workflow.show_summary()
        
        console.print(f"[green]Successfully combined video and audio: {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        workflow.complete_stage("combine", "failed")
        workflow.show_summary()

@cli.command()
@click.argument('video_paths', nargs=-1, type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--transition', '-t', type=str, help='Transition type')
@click.option('--duration', '-d', type=float, help='Transition duration')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
def concat(video_paths: tuple, output_path: str, transition: Optional[str], duration: Optional[float], interactive: bool):
    """Concatenate multiple videos with transitions."""
    cfg = load_config()
    
    if not video_paths:
        console.print("[red]Error: At least one video path is required[/red]")
        return
    
    if transition is None:
        transition = cfg.get('default_transition', 'crossfade')
        if interactive:
            transition = Prompt.ask(
                "Select transition type",
                choices=[t.value for t in TransitionType],
                default=transition
            )
    
    if duration is None:
        duration = cfg.get('transition_duration', 1.0)
        if interactive:
            duration = float(Prompt.ask("Enter transition duration (seconds)", default=str(duration)))
    
    transitions = [
        TransitionConfig(TransitionType(transition), duration=duration)
        for _ in range(len(video_paths) - 1)
    ]
    
    workflow = WorkflowManager("Concatenate Videos")
    workflow.add_stage("validate", "Validating input videos")
    workflow.add_stage("concat", "Concatenating videos")
    
    try:
        workflow.start()
        workflow.complete_stage("validate")
        
        concatenate_videos_with_transitions(
            video_paths=list(video_paths),
            output_path=output_path,
            transitions=transitions
        )
        
        workflow.complete_stage("concat")
        workflow.show_summary()
        
        console.print(f"[green]Successfully concatenated videos: {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        workflow.complete_stage("concat", "failed")
        workflow.show_summary()

@cli.command()
@click.argument('audio_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output path for visualization')
@click.option('--min-bpm', type=float, help='Minimum BPM')
@click.option('--max-bpm', type=float, help='Maximum BPM')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
def detect_beats(audio_path: str, output: Optional[str], min_bpm: Optional[float], max_bpm: Optional[float], interactive: bool):
    """Detect beats in an audio file."""
    if min_bpm is None:
        min_bpm = 60.0
        if interactive:
            min_bpm = float(Prompt.ask("Enter minimum BPM", default=str(min_bpm)))
    
    if max_bpm is None:
        max_bpm = 180.0
        if interactive:
            max_bpm = float(Prompt.ask("Enter maximum BPM", default=str(max_bpm)))
    
    workflow = WorkflowManager("Detect Beats")
    workflow.add_stage("analyze", "Analyzing audio")
    workflow.add_stage("detect", "Detecting beats")
    
    try:
        workflow.start()
        workflow.complete_stage("analyze")
        
        result = detect_beats(
            audio_path=audio_path,
            output_path=output,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            save_plot=output is not None
        )
        
        workflow.complete_stage("detect")
        workflow.show_summary()
        
        # Display results
        table = Table(title="Beat Detection Results")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Tempo", f"{result['tempo']:.1f} BPM")
        table.add_row("Beats", str(len(result['beats'])))
        table.add_row("Average Interval", f"{np.mean(result['beat_intervals']):.3f}s")
        
        console.print(table)
        
        if output:
            console.print(f"[green]Saved visualization to: {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        workflow.complete_stage("detect", "failed")
        workflow.show_summary()

@cli.command()
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
def config(interactive: bool):
    """Configure WATW settings."""
    cfg = load_config()
    
    if interactive:
        console.print(Panel("WATW Configuration", style="bold blue"))
        
        output_dir = Prompt.ask(
            "Output directory",
            default=cfg.get('output_dir')
        )
        cfg.set('output_dir', output_dir)
        
        temp_dir = Prompt.ask(
            "Temporary directory",
            default=cfg.get('temp_dir')
        )
        cfg.set('temp_dir', temp_dir)
        
        default_transition = Prompt.ask(
            "Default transition",
            choices=[t.value for t in TransitionType],
            default=cfg.get('default_transition')
        )
        cfg.set('default_transition', default_transition)
        
        transition_duration = float(Prompt.ask(
            "Transition duration (seconds)",
            default=str(cfg.get('transition_duration'))
        ))
        cfg.set('transition_duration', transition_duration)
        
        background_music_volume = float(Prompt.ask(
            "Background music volume (0.0-1.0)",
            default=str(cfg.get('background_music_volume'))
        ))
        cfg.set('background_music_volume', background_music_volume)
        
        log_level = Prompt.ask(
            "Log level",
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default=cfg.get('log_level')
        )
        cfg.set('log_level', log_level)
        
        interactive_mode = Confirm.ask(
            "Enable interactive mode by default?",
            default=cfg.get('interactive_mode')
        )
        cfg.set('interactive_mode', interactive_mode)
        
        console.print("[green]Configuration saved successfully![/green]")
    
    else:
        # Display current configuration
        table = Table(title="WATW Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in cfg.config.items():
            table.add_row(key, str(value))
        
        console.print(table)

if __name__ == '__main__':
    cli() 