"""
Pipeline Auto-Capture
======================

Middleware that automatically captures agent interactions, tool executions,
error recoveries, and workflow results into the RealtimeDataPipeline.

This is the glue between the Orchestrator / LangGraph agents and the
dual-route pipeline (Helox training + Cyrex runtime).

Integration:
    from app.core.pipeline_auto_capture import PipelineAutoCapture

    auto_capture = PipelineAutoCapture()
    await auto_capture.initialize()

    # After an orchestrator request completes:
    await auto_capture.capture_interaction(user_input, response, metadata)

    # After a tool executes:
    await auto_capture.capture_tool_execution(tool_name, input_params, result, ...)

    # After an error is recovered:
    await auto_capture.capture_error_recovery(error_msg, recovery_response, ...)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import json

from ..logging_config import get_logger

logger = get_logger("cyrex.auto_capture")


class PipelineAutoCapture:
    """
    Automatically captures data from the Cyrex runtime and feeds it
    into the RealtimeDataPipeline for both Helox training and Cyrex
    agent self-improvement.

    Captures:
    - Agent interactions (user input → model response)
    - Tool executions (tool name, input, output, timing)
    - Error recoveries (error → recovery attempt)
    - Workflow results (multi-step workflow completions)
    - User feedback (explicit ratings or corrections)
    - Document processing results (lease/contract extraction)
    """

    def __init__(self):
        self._pipeline = None
        self._initialized = False
        self._enabled = True  # Can be toggled off for testing
        self.logger = logger

    async def initialize(self):
        """Connect to the pipeline"""
        if self._initialized:
            return

        try:
            from .realtime_data_pipeline import get_realtime_pipeline
            self._pipeline = await get_realtime_pipeline()
            self._initialized = True
            self.logger.info("PipelineAutoCapture initialized")
        except Exception as e:
            self.logger.warning(f"Auto-capture could not connect to pipeline: {e}")

    def set_enabled(self, enabled: bool):
        """Enable/disable auto-capture"""
        self._enabled = enabled
        self.logger.info(f"Auto-capture {'enabled' if enabled else 'disabled'}")

    # ------------------------------------------------------------------
    # Capture: Agent Interactions
    # ------------------------------------------------------------------

    async def capture_interaction(
        self,
        user_input: str,
        response: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        model_name: Optional[str] = None,
        duration_ms: Optional[float] = None,
        context_sources: int = 0,
        safety_scores: Optional[Dict[str, float]] = None,
        intermediate_steps: Optional[List[Dict]] = None,
        quality_score: Optional[float] = None,
    ):
        """
        Capture a complete agent interaction (user → model).
        Called after WorkflowOrchestrator.process_request completes.
        """
        if not self._enabled or not self._initialized:
            return

        # Auto-estimate quality if not provided
        if quality_score is None:
            quality_score = self._estimate_interaction_quality(
                response, duration_ms, safety_scores,
            )

        try:
            await self._pipeline.ingest_raw(
                input_text=user_input,
                output_text=response,
                instruction="Respond to the following user request:",
                category="agent_interaction",
                route="both",
                data_format="raw",
                user_id=user_id,
                session_id=session_id,
                model_name=model_name,
                quality_score=quality_score,
                execution_time_ms=duration_ms,
                tags=self._build_interaction_tags(context_sources, intermediate_steps),
                metadata={
                    "context_sources": context_sources,
                    "safety_scores": safety_scores or {},
                    "tool_calls_count": len(intermediate_steps) if intermediate_steps else 0,
                },
            )

            # Also capture tool calls from intermediate steps as structured data
            if intermediate_steps:
                for step in intermediate_steps:
                    await self._capture_intermediate_step(
                        step, user_id=user_id, session_id=session_id,
                        model_name=model_name,
                    )

        except Exception as e:
            self.logger.warning(f"Auto-capture interaction failed: {e}")

    # ------------------------------------------------------------------
    # Capture: Tool Executions
    # ------------------------------------------------------------------

    async def capture_tool_execution(
        self,
        tool_name: str,
        input_params: Any,
        result: Any,
        *,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        execution_time_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """Capture a tool execution for training better tool selection"""
        if not self._enabled or not self._initialized:
            return

        input_str = json.dumps(input_params, default=str) if not isinstance(input_params, str) else input_params
        result_str = json.dumps(result, default=str) if not isinstance(result, str) else result
        quality = 0.8 if success else 0.3

        try:
            await self._pipeline.ingest_raw(
                input_text=input_str,
                output_text=result_str,
                instruction=f"Execute tool '{tool_name}' with the given parameters",
                category="tool_execution",
                route="both",
                data_format="structured",
                agent_id=agent_id,
                session_id=session_id,
                tool_name=tool_name,
                quality_score=quality,
                execution_time_ms=execution_time_ms,
                tags=["tool_execution", tool_name, "success" if success else "failure"],
                metadata={
                    "tool_name": tool_name,
                    "success": success,
                    "error": error,
                },
                structured_payload={
                    "tool_name": tool_name,
                    "input_params": input_params if isinstance(input_params, dict) else {"raw": input_str},
                    "result": result if isinstance(result, dict) else {"raw": result_str},
                    "success": success,
                    "error": error,
                    "execution_time_ms": execution_time_ms,
                },
            )
        except Exception as e:
            self.logger.warning(f"Auto-capture tool execution failed: {e}")

    # ------------------------------------------------------------------
    # Capture: Error Recovery
    # ------------------------------------------------------------------

    async def capture_error_recovery(
        self,
        error_message: str,
        recovery_response: str,
        *,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        original_input: Optional[str] = None,
    ):
        """Capture error recovery scenarios for training resilience"""
        if not self._enabled or not self._initialized:
            return

        try:
            context = f"Original input: {original_input}" if original_input else ""
            await self._pipeline.ingest_raw(
                input_text=f"Error: {error_message}",
                output_text=recovery_response,
                instruction="Recover from the following error scenario:",
                context=context,
                category="error_recovery",
                route="both",
                data_format="raw",
                agent_id=agent_id,
                session_id=session_id,
                user_id=user_id,
                quality_score=0.6,  # error recovery is valuable but uncertain quality
                tags=["error_recovery"],
                metadata={"original_input": original_input},
            )
        except Exception as e:
            self.logger.warning(f"Auto-capture error recovery failed: {e}")

    # ------------------------------------------------------------------
    # Capture: Workflow Results
    # ------------------------------------------------------------------

    async def capture_workflow_result(
        self,
        workflow_id: str,
        workflow_type: str,
        steps_completed: int,
        total_steps: int,
        final_result: Any,
        *,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ):
        """Capture workflow completion results"""
        if not self._enabled or not self._initialized:
            return

        result_str = json.dumps(final_result, default=str) if not isinstance(final_result, str) else final_result
        completion_rate = steps_completed / max(total_steps, 1)
        quality = 0.9 if completion_rate >= 1.0 else 0.5 + (completion_rate * 0.4)

        try:
            await self._pipeline.ingest_structured(
                payload={
                    "workflow_id": workflow_id,
                    "workflow_type": workflow_type,
                    "steps_completed": steps_completed,
                    "total_steps": total_steps,
                    "completion_rate": completion_rate,
                    "result": final_result if isinstance(final_result, dict) else {"raw": result_str},
                    "duration_ms": duration_ms,
                },
                category="workflow_result",
                route="both",
                agent_id=agent_id,
                session_id=session_id,
                quality_score=quality,
                tags=["workflow", workflow_type],
            )
        except Exception as e:
            self.logger.warning(f"Auto-capture workflow result failed: {e}")

    # ------------------------------------------------------------------
    # Capture: User Feedback
    # ------------------------------------------------------------------

    async def capture_user_feedback(
        self,
        original_input: str,
        original_response: str,
        feedback: str,
        rating: Optional[float] = None,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Capture explicit user feedback on agent responses"""
        if not self._enabled or not self._initialized:
            return

        quality = rating if rating is not None else 0.7

        try:
            await self._pipeline.ingest_raw(
                input_text=original_input,
                output_text=original_response,
                instruction="Incorporate the following user feedback:",
                context=f"User feedback: {feedback}",
                category="user_feedback",
                route="both",
                data_format="raw",
                user_id=user_id,
                session_id=session_id,
                quality_score=quality,
                tags=["user_feedback", "correction" if rating and rating < 0.5 else "positive"],
                metadata={
                    "feedback_text": feedback,
                    "rating": rating,
                },
            )
        except Exception as e:
            self.logger.warning(f"Auto-capture user feedback failed: {e}")

    # ------------------------------------------------------------------
    # Capture: Document Processing
    # ------------------------------------------------------------------

    async def capture_document_processing(
        self,
        document_type: str,
        document_text: str,
        extracted_data: Dict[str, Any],
        *,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        quality_score: float = 0.7,
    ):
        """Capture document processing results (leases, contracts, etc.)"""
        if not self._enabled or not self._initialized:
            return

        try:
            await self._pipeline.ingest_structured(
                payload={
                    "document_type": document_type,
                    "extracted_fields": extracted_data,
                },
                category="document_processing",
                route="both",
                agent_id=agent_id,
                session_id=session_id,
                user_id=user_id,
                quality_score=quality_score,
                tags=["document_processing", document_type],
            )

            # Also send raw text pair for training
            await self._pipeline.ingest_raw(
                input_text=document_text[:10000],  # truncate very large docs
                output_text=json.dumps(extracted_data, default=str),
                instruction=f"Extract structured data from the following {document_type}:",
                category="document_processing",
                route="helox",  # raw extraction goes only to training
                data_format="raw",
                agent_id=agent_id,
                quality_score=quality_score,
                tags=["document_processing", document_type, "extraction"],
            )
        except Exception as e:
            self.logger.warning(f"Auto-capture document processing failed: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _capture_intermediate_step(
        self,
        step: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """Capture an individual intermediate step (tool call from LangGraph)"""
        tool_name = step.get("tool", "unknown")
        tool_input = step.get("input", "")
        tool_output = step.get("output", "")

        if not tool_input and not tool_output:
            return

        input_str = json.dumps(tool_input, default=str) if not isinstance(tool_input, str) else tool_input
        output_str = json.dumps(tool_output, default=str) if not isinstance(tool_output, str) else tool_output

        try:
            await self._pipeline.ingest_raw(
                input_text=input_str,
                output_text=output_str,
                instruction=f"Execute tool '{tool_name}' with the given parameters",
                category="tool_execution",
                route="both",
                data_format="raw",
                session_id=session_id,
                user_id=user_id,
                tool_name=tool_name,
                model_name=model_name,
                quality_score=0.7,
                tags=["tool_execution", tool_name, "intermediate_step"],
            )
        except Exception as e:
            self.logger.debug(f"Failed to capture intermediate step: {e}")

    @staticmethod
    def _estimate_interaction_quality(
        response: str,
        duration_ms: Optional[float],
        safety_scores: Optional[Dict[str, float]],
    ) -> float:
        """Heuristic quality estimate for an interaction"""
        score = 0.6  # base

        # Response length heuristic
        if response:
            length = len(response)
            if length > 100:
                score += 0.1
            if length > 500:
                score += 0.05

        # Penalize very slow responses
        if duration_ms and duration_ms > 30000:
            score -= 0.1

        # Safety scores
        if safety_scores:
            input_score = safety_scores.get("input_score", 0)
            output_score = safety_scores.get("output_score", 0)
            # Higher safety scores = worse content → lower quality
            if input_score > 0.5 or output_score > 0.5:
                score -= 0.2

        return max(0.0, min(1.0, score))

    @staticmethod
    def _build_interaction_tags(
        context_sources: int,
        intermediate_steps: Optional[List[Dict]],
    ) -> List[str]:
        """Build tags for an interaction record"""
        tags = ["agent_interaction"]
        if context_sources > 0:
            tags.append("rag_augmented")
        if intermediate_steps:
            tags.append("tool_augmented")
            tools_used = set()
            for step in intermediate_steps:
                tool = step.get("tool")
                if tool:
                    tools_used.add(tool)
            for tool in tools_used:
                tags.append(f"tool:{tool}")
        return tags


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_auto_capture: Optional[PipelineAutoCapture] = None


async def get_auto_capture() -> PipelineAutoCapture:
    """Get or create the auto-capture singleton"""
    global _auto_capture
    if _auto_capture is None:
        _auto_capture = PipelineAutoCapture()
        await _auto_capture.initialize()
    return _auto_capture

