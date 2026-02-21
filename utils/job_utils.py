"""Utilities for job scheduling and execution environment detection."""

import os


def get_job_info():
    """
    Detect execution environment and return job information.
    Returns: (job_id: str, task_id: int, is_slurm: bool)
    """
    slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    
    if slurm_job_id is not None and slurm_task_id is not None:
        # SLURM environment
        return f"{slurm_job_id}_{slurm_task_id}", int(slurm_task_id), True
    else:
        # Rewrite this function for other job schedulers if needed
        # Default to local execution
        return "local_0", 0, False
