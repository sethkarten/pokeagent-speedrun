"""Shared runtime helpers for global-step ownership and publication."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


HistoryPublisher = Callable[..., None]
FunctionResultPublisher = Callable[[str, str], None]
StepChangePublisher = Callable[[int], None]


@dataclass
class StepRecord:
    step_number: int
    owner: str
    interaction_name: str
    published_to_history: bool = False
    published_function_results: List[str] = field(default_factory=list)


class PokeAgentRuntime:
    """Owns global-step allocation and orchestrator publication boundaries."""

    def __init__(
        self,
        *,
        initial_step: int = 0,
        publish_history: HistoryPublisher,
        publish_function_result: FunctionResultPublisher,
        on_step_change: Optional[StepChangePublisher] = None,
    ):
        self.current_step = max(0, int(initial_step or 0))
        self._publish_history = publish_history
        self._publish_function_result = publish_function_result
        self._on_step_change = on_step_change
        self._records: List[StepRecord] = []

    def peek_next_step(self) -> int:
        return self.current_step + 1

    def claim_step(self, *, owner: str, interaction_name: str) -> int:
        self.current_step += 1
        os.environ["LLM_STEP_NUMBER"] = str(self.current_step)
        if self._on_step_change:
            self._on_step_change(self.current_step)
        self._records.append(
            StepRecord(
                step_number=self.current_step,
                owner=owner,
                interaction_name=interaction_name,
            )
        )
        return self.current_step

    def sync_step(self, step_number: int) -> None:
        self.current_step = max(self.current_step, int(step_number or 0))
        os.environ["LLM_STEP_NUMBER"] = str(self.current_step)
        if self._on_step_change:
            self._on_step_change(self.current_step)

    def publish_history(
        self,
        *,
        step_number: int,
        prompt: str,
        response: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        action_details: Optional[str] = None,
        player_coords: Optional[tuple[int, int]] = None,
        start_coords: Optional[tuple[int, int]] = None,
        end_coords: Optional[tuple[int, int]] = None,
    ) -> None:
        self._publish_history(
            prompt,
            response,
            tool_calls=tool_calls,
            action_details=action_details,
            player_coords=player_coords,
            start_coords=start_coords,
            end_coords=end_coords,
            step_number=step_number,
        )
        self._mark_record(step_number, published_to_history=True)

    def publish_function_result(self, *, step_number: int, function_name: str, result_json: str) -> None:
        self._publish_function_result(function_name, result_json)
        self._mark_record(step_number, function_name=function_name)

    def get_step_record(self, step_number: int) -> Optional[StepRecord]:
        for record in reversed(self._records):
            if record.step_number == step_number:
                return record
        return None

    def _mark_record(
        self,
        step_number: int,
        *,
        published_to_history: Optional[bool] = None,
        function_name: Optional[str] = None,
    ) -> None:
        record = self.get_step_record(step_number)
        if not record:
            return
        if published_to_history is not None:
            record.published_to_history = published_to_history
        if function_name:
            record.published_function_results.append(function_name)
