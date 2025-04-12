#!/usr/bin/env python3

import json
import os
from pathlib import Path

from watw.core.workflow.workflow_manager import WorkflowManager

# Load configuration
config_path = Path("config/config.json")
with open(config_path) as f:
    config = json.load(f)

# Create workflow directory
workflow_dir = Path("workflow_output")
workflow_dir.mkdir(parents=True, exist_ok=True)

# Initialize workflow manager
workflow_manager = WorkflowManager(str(workflow_dir), config=config)

# Execute workflow
success = workflow_manager.execute_workflow()

if success:
    print("Workflow completed successfully!")
    print(f"Output files can be found in: {workflow_dir}")
else:
    print("Workflow failed. Check the logs for details.") 