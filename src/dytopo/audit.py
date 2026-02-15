"""
SwarmAuditLog: JSONL-based audit logging for swarm executions.

Writes one JSON object per line to ~/dytopo-logs/{task_id}/audit.jsonl
Every event includes: timestamp, event_type, round, task_id, and event-specific data.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SwarmAuditLog:
    """
    Audit logger for swarm execution events.

    Writes JSONL format to ~/dytopo-logs/{task_id}/audit.jsonl
    Each event contains: timestamp, event_type, round, task_id, and event-specific data.
    """

    def __init__(self, task_id: str, base_dir: Optional[Path] = None):
        """
        Initialize the audit log for a specific task.

        Args:
            task_id: Unique identifier for the task
            base_dir: Base directory for logs (defaults to ~/dytopo-logs)
        """
        self.task_id = task_id
        self.base_dir = Path(base_dir).expanduser() if base_dir else Path.home() / "dytopo-logs"
        self.log_dir = self.base_dir / task_id

        # Create directories if they don't exist
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create log directory {self.log_dir}: {e}")
            raise

        # Open the audit log file
        self.log_path = self.log_dir / "audit.jsonl"
        try:
            self.file_handle = open(self.log_path, 'a', encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to open audit log file {self.log_path}: {e}")
            raise

    def _write_event(self, event_type: str, round: int, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Write an event to the audit log.

        Args:
            event_type: Type of event
            round: Current round number
            data: Additional event-specific data
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "round": round,
            "task_id": self.task_id,
        }

        # Merge in event-specific data
        if data:
            record.update(data)

        try:
            # Write JSON with default=str to handle non-serializable types
            json_line = json.dumps(record, default=str)
            self.file_handle.write(json_line + '\n')
            self.file_handle.flush()  # Flush after each write for streaming
        except Exception as e:
            logger.error(f"Failed to write audit event {event_type}: {e}")

    def swarm_started(self, query: str, max_rounds: int, agents: list) -> None:
        """
        Log swarm execution start.

        Args:
            query: The input query
            max_rounds: Maximum number of rounds allowed
            agents: List of agent names or configurations
        """
        self._write_event(
            event_type="swarm_started",
            round=0,
            data={
                "query": query,
                "max_rounds": max_rounds,
                "agents": agents,
            }
        )

    def round_started(self, round: int) -> None:
        """
        Log the start of a new round.

        Args:
            round: Round number
        """
        self._write_event(
            event_type="round_started",
            round=round,
        )

    def routing_computed(self, round: int, router_output: Any, selected_agents: list) -> None:
        """
        Log routing computation results.

        Args:
            round: Current round number
            router_output: Raw output from the router
            selected_agents: List of selected agent names
        """
        self._write_event(
            event_type="routing_computed",
            round=round,
            data={
                "router_output": router_output,
                "selected_agents": selected_agents,
            }
        )

    def agent_executed(self, round: int, agent_name: str, output: Any, execution_time: float) -> None:
        """
        Log successful agent execution.

        Args:
            round: Current round number
            agent_name: Name of the executed agent
            output: Agent's output
            execution_time: Time taken to execute (in seconds)
        """
        self._write_event(
            event_type="agent_executed",
            round=round,
            data={
                "agent_name": agent_name,
                "output": output,
                "execution_time": execution_time,
            }
        )

    def agent_failed(self, round: int, agent_name: str, error: str) -> None:
        """
        Log agent execution failure.

        Args:
            round: Current round number
            agent_name: Name of the failed agent
            error: Error message or exception details
        """
        self._write_event(
            event_type="agent_failed",
            round=round,
            data={
                "agent_name": agent_name,
                "error": error,
            }
        )

    def convergence_detected(self, round: int, reason: str, confidence: Optional[float] = None) -> None:
        """
        Log convergence detection.

        Args:
            round: Current round number
            reason: Reason for convergence detection
            confidence: Optional confidence score
        """
        data = {"reason": reason}
        if confidence is not None:
            data["confidence"] = confidence

        self._write_event(
            event_type="convergence_detected",
            round=round,
            data=data,
        )

    def redelegation(self, round: int, from_agent: str, to_agents: list, reason: str) -> None:
        """
        Log task redelegation.

        Args:
            round: Current round number
            from_agent: Agent that delegated the task
            to_agents: List of agents receiving the delegated task
            reason: Reason for redelegation
        """
        self._write_event(
            event_type="redelegation",
            round=round,
            data={
                "from_agent": from_agent,
                "to_agents": to_agents,
                "reason": reason,
            }
        )

    def swarm_completed(self, round: int, final_output: Any, total_rounds: int) -> None:
        """
        Log successful swarm completion.

        Args:
            round: Final round number
            final_output: Final aggregated output
            total_rounds: Total number of rounds executed
        """
        self._write_event(
            event_type="swarm_completed",
            round=round,
            data={
                "final_output": final_output,
                "total_rounds": total_rounds,
            }
        )

    def swarm_failed(self, round: int, reason: str, error: Optional[str] = None) -> None:
        """
        Log swarm execution failure.

        Args:
            round: Round number where failure occurred
            reason: Reason for failure
            error: Optional error message or exception details
        """
        data = {"reason": reason}
        if error is not None:
            data["error"] = error

        self._write_event(
            event_type="swarm_failed",
            round=round,
            data=data,
        )

    def close(self) -> None:
        """
        Close the audit log file handle.
        """
        try:
            if hasattr(self, 'file_handle') and self.file_handle:
                self.file_handle.close()
        except Exception as e:
            logger.error(f"Failed to close audit log file: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
