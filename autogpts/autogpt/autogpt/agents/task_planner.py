"""Pre-execution task planner: decomposes agent goals into a tracked subtask list.

Called once before the main agent loop. The resulting TaskPlan is injected into
every prompt cycle so the agent always has a visible, up-to-date roadmap.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic.v1 import BaseModel

logger = logging.getLogger(__name__)

PLAN_FILE = "task_plan.json"

PLANNING_PROMPT = """\
You are a task-planning assistant. Decompose the following goals into a numbered \
list of concrete, sequential, verifiable subtasks.

Goals:
{goals}

Return ONLY a JSON object with this exact structure (no markdown, no explanation):
{{
  "goal_summary": "one-sentence summary of the overall objective",
  "subtasks": [
    {{"id": 1, "description": "specific concrete action", "status": "pending", "notes": ""}},
    ...
  ]
}}

Rules:
- Each subtask must be a single, verifiable action.
- Order them so each depends only on previously completed steps.
- Be specific: include filenames, expected outputs, verification steps.
- Maximum 12 subtasks; combine closely related work.
- The final subtask must always be: verify all goals are met, then call finish.
"""


class SubTask(BaseModel):
    id: int
    description: str
    status: Literal["pending", "in_progress", "complete", "failed"] = "pending"
    notes: str = ""


class TaskPlan(BaseModel):
    goal_summary: str
    subtasks: list[SubTask]

    def save(self, workspace_path: Path) -> None:
        (workspace_path / PLAN_FILE).write_text(self.json(indent=2))

    @classmethod
    def load(cls, workspace_path: Path) -> Optional["TaskPlan"]:
        p = workspace_path / PLAN_FILE
        if p.exists():
            try:
                return cls.parse_raw(p.read_text())
            except Exception as e:
                logger.warning(f"Could not load task plan: {e}")
        return None

    @property
    def next_pending(self) -> Optional[SubTask]:
        for t in self.subtasks:
            if t.status in ("pending", "in_progress"):
                return t
        return None

    @property
    def done_count(self) -> int:
        return sum(1 for t in self.subtasks if t.status == "complete")

    @property
    def is_complete(self) -> bool:
        return all(t.status in ("complete", "failed") for t in self.subtasks)

    def advance(self) -> None:
        """Mark the current in_progress task complete and start the next pending one."""
        for t in self.subtasks:
            if t.status == "in_progress":
                t.status = "complete"
                break
        for t in self.subtasks:
            if t.status == "pending":
                t.status = "in_progress"
                break

    def start_next(self) -> None:
        """Set the first pending subtask to in_progress."""
        for t in self.subtasks:
            if t.status == "pending":
                t.status = "in_progress"
                return

    def format_status(self) -> str:
        icons = {
            "pending": "⬜",
            "in_progress": "🔄",
            "complete": "✅",
            "failed": "❌",
        }
        lines = [f"### Current Task Plan: {self.goal_summary}"]
        for t in self.subtasks:
            lines.append(f"{icons[t.status]} {t.id}. {t.description}")
            if t.notes:
                lines.append(f"   ↳ {t.notes}")
        lines.append(
            f"\nProgress: {self.done_count}/{len(self.subtasks)} subtasks complete."
        )
        current = self.next_pending
        if current:
            lines.append(f"**Currently working on: subtask {current.id}**")
        return "\n".join(lines)


async def create_task_plan(
    goals: list[str],
    llm_provider,
    model_name: str,
) -> TaskPlan:
    """Call the LLM once to decompose goals into a structured subtask plan."""
    from autogpt.core.resource.model_providers.schema import ChatMessage
    from autogpt.json_utils.utilities import extract_dict_from_response

    goals_text = "\n".join(f"{i + 1}. {g}" for i, g in enumerate(goals))
    messages = [
        ChatMessage.system(
            "You are a precise task-planning assistant. Return only valid JSON."
        ),
        ChatMessage.user(PLANNING_PROMPT.format(goals=goals_text)),
    ]

    try:
        response = await llm_provider.create_chat_completion(
            model_prompt=messages,
            model_name=model_name,
        )
        content = response.response.get("content", "")
        data = extract_dict_from_response(content)
        if data and "subtasks" in data:
            plan = TaskPlan(**data)
            plan.start_next()
            logger.info(
                f"Task plan created: {len(plan.subtasks)} subtasks for '{plan.goal_summary}'"
            )
            return plan
    except Exception as e:
        logger.warning(f"Planning LLM call failed ({e}); falling back to goal list.")

    # Fallback: turn the raw goal strings directly into subtasks
    subtasks = [SubTask(id=i + 1, description=g) for i, g in enumerate(goals)]
    subtasks.append(
        SubTask(
            id=len(goals) + 1,
            description="Verify all goals are fully complete and call finish",
        )
    )
    plan = TaskPlan(
        goal_summary=" / ".join(goals[:2]) + (" / ..." if len(goals) > 2 else ""),
        subtasks=subtasks,
    )
    plan.start_next()
    return plan
