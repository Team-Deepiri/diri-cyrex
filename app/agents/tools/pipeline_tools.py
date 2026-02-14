"""
Pipeline Tools for Agents
==========================

LangChain-compatible tools that allow agents to submit data
to the RealtimeDataPipeline via tool calls.

Agents use these to:
  - Record high-quality interaction examples for Helox training
  - Store learned knowledge into Cyrex runtime for self-improvement
  - Log tool execution results for future optimisation
  - Submit structured data (documents, extractions) to the pipeline
  - Submit raw unprocessed data for pre-training
  - Submit user feedback for both training and context updates
  - Query pipeline statistics
"""

from typing import Optional, List, Dict, Any
from ...logging_config import get_logger

logger = get_logger("cyrex.agent.tools.pipeline")


async def register_pipeline_tools(agent):
    """Register pipeline tools with an agent"""

    # ------------------------------------------------------------------
    # submit_training_data  (both routes, raw data)
    # ------------------------------------------------------------------

    async def submit_training_data(
        input_text: str,
        output_text: str,
        instruction: str = "",
        category: str = "agent_interaction",
        quality_score: float = 0.7,
        tags: str = "",
    ) -> str:
        """
        Submit an input/output pair to the training pipeline.
        This data flows to both Helox (for model training) and
        Cyrex runtime (for agent context improvement).

        Args:
            input_text: The user input or task description
            output_text: The generated response or result
            instruction: Optional instruction describing what to do
            category: One of: agent_interaction, tool_execution,
                      user_feedback, conversation, error_recovery,
                      workflow_result, knowledge_update, performance_metric,
                      document_processing, compliance_check, fraud_detection
            quality_score: Quality rating from 0.0 to 1.0
            tags: Comma-separated tags (e.g. "finance,invoice,high-quality")

        Returns:
            Confirmation with record ID
        """
        try:
            from ...core.realtime_data_pipeline import get_realtime_pipeline
            pipeline = await get_realtime_pipeline()
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            record_id = await pipeline.ingest_raw(
                input_text=input_text,
                output_text=output_text,
                instruction=instruction,
                category=category,
                route="both",
                data_format="raw",
                agent_id=getattr(agent, "agent_id", None),
                session_id=getattr(agent, "session_id", None),
                quality_score=quality_score,
                tags=tag_list,
            )
            return f"Data submitted to pipeline (record: {record_id}). Routed to Helox training + Cyrex runtime."
        except Exception as e:
            logger.error(f"Pipeline submission failed: {e}")
            return f"Error submitting to pipeline: {str(e)}"

    # ------------------------------------------------------------------
    # submit_structured_data  (both routes, structured data)
    # ------------------------------------------------------------------

    async def submit_structured_data(
        payload_json: str,
        category: str = "agent_interaction",
        quality_score: float = 0.7,
        tags: str = "",
    ) -> str:
        """
        Submit structured data (JSON) to the pipeline.
        Use this for typed payloads like document extractions,
        compliance results, or fraud detection findings.

        Args:
            payload_json: JSON string of the structured payload
            category: Data category
            quality_score: Quality rating from 0.0 to 1.0
            tags: Comma-separated tags

        Returns:
            Confirmation with record ID
        """
        try:
            import json
            payload = json.loads(payload_json)
            from ...core.realtime_data_pipeline import get_realtime_pipeline
            pipeline = await get_realtime_pipeline()
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            record_id = await pipeline.ingest_structured(
                payload=payload,
                category=category,
                route="both",
                agent_id=getattr(agent, "agent_id", None),
                session_id=getattr(agent, "session_id", None),
                quality_score=quality_score,
                tags=tag_list,
            )
            return f"Structured data submitted (record: {record_id}). Routed to Helox + Cyrex."
        except Exception as e:
            logger.error(f"Structured data submission failed: {e}")
            return f"Error submitting structured data: {str(e)}"

    # ------------------------------------------------------------------
    # submit_to_helox  (Helox-only, raw data)
    # ------------------------------------------------------------------

    async def submit_to_helox(
        input_text: str,
        output_text: str,
        instruction: str = "",
        category: str = "agent_interaction",
        quality_score: float = 0.7,
        tags: str = "",
    ) -> str:
        """
        Submit data exclusively to Helox for model training.
        Use this when data should only be used for training and
        NOT stored in agent runtime memory.

        Args:
            input_text: The user input or task description
            output_text: The generated response or result
            instruction: Optional instruction describing what to do
            category: Data category
            quality_score: Quality rating from 0.0 to 1.0
            tags: Comma-separated tags

        Returns:
            Confirmation with record ID
        """
        try:
            from ...core.realtime_data_pipeline import get_realtime_pipeline
            pipeline = await get_realtime_pipeline()
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            record_id = await pipeline.ingest_raw(
                input_text=input_text,
                output_text=output_text,
                instruction=instruction,
                category=category,
                route="helox",
                data_format="raw",
                agent_id=getattr(agent, "agent_id", None),
                session_id=getattr(agent, "session_id", None),
                quality_score=quality_score,
                tags=tag_list,
            )
            return f"Data submitted to Helox training (record: {record_id})."
        except Exception as e:
            logger.error(f"Helox submission failed: {e}")
            return f"Error submitting to Helox: {str(e)}"

    # ------------------------------------------------------------------
    # submit_raw_to_helox  (Helox-only, raw pre-training text)
    # ------------------------------------------------------------------

    async def submit_raw_to_helox(
        text: str,
        source: str = "agent",
        quality_score: float = 0.6,
    ) -> str:
        """
        Submit raw unprocessed text to Helox for pre-training.
        This bypasses instruction formatting and sends plain text.

        Args:
            text: Raw text content
            source: Source identifier
            quality_score: Quality rating

        Returns:
            Confirmation with record ID
        """
        try:
            from ...core.realtime_data_pipeline import get_realtime_pipeline
            pipeline = await get_realtime_pipeline()
            record_id = await pipeline.ingest_raw(
                input_text=text,
                output_text="",
                category="knowledge_update",
                route="helox",
                data_format="raw",
                agent_id=getattr(agent, "agent_id", None),
                quality_score=quality_score,
                tags=["raw_text", source],
                metadata={"source": source},
            )
            return f"Raw text submitted to Helox (record: {record_id})."
        except Exception as e:
            logger.error(f"Raw Helox submission failed: {e}")
            return f"Error submitting raw text: {str(e)}"

    # ------------------------------------------------------------------
    # submit_to_cyrex_runtime  (Cyrex-only)
    # ------------------------------------------------------------------

    async def submit_to_cyrex_runtime(
        content: str,
        category: str = "knowledge_update",
        quality_score: float = 0.8,
        tags: str = "",
    ) -> str:
        """
        Submit knowledge or context exclusively to Cyrex runtime.
        Use this to store learned information that agents can
        immediately access via memory search. NOT sent to Helox.

        Args:
            content: The knowledge or context to store
            category: Data category
            quality_score: Importance rating from 0.0 to 1.0
            tags: Comma-separated tags

        Returns:
            Confirmation with record ID
        """
        try:
            from ...core.realtime_data_pipeline import get_realtime_pipeline
            pipeline = await get_realtime_pipeline()
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            record_id = await pipeline.ingest_raw(
                input_text=content,
                output_text="",
                category=category,
                route="cyrex",
                data_format="raw",
                agent_id=getattr(agent, "agent_id", None),
                session_id=getattr(agent, "session_id", None),
                quality_score=quality_score,
                tags=tag_list,
            )
            return (
                f"Knowledge stored in Cyrex runtime (record: {record_id}). "
                "Agents can access this via memory search."
            )
        except Exception as e:
            logger.error(f"Cyrex runtime submission failed: {e}")
            return f"Error submitting to Cyrex runtime: {str(e)}"

    # ------------------------------------------------------------------
    # log_tool_result
    # ------------------------------------------------------------------

    async def log_tool_result(
        tool_name: str,
        input_params: str,
        result: str,
        execution_time_ms: float = 0.0,
        quality_score: float = 0.7,
    ) -> str:
        """
        Log a tool execution result to the pipeline.
        Helps train better tool selection and improve tool usage.

        Args:
            tool_name: Name of the tool executed
            input_params: Input parameters (as string/JSON)
            result: Tool execution result
            execution_time_ms: Execution time in milliseconds
            quality_score: Quality of the result (0.0 to 1.0)

        Returns:
            Confirmation with record ID
        """
        try:
            from ...core.realtime_data_pipeline import get_realtime_pipeline
            pipeline = await get_realtime_pipeline()
            record_id = await pipeline.ingest_raw(
                input_text=input_params,
                output_text=result,
                instruction=f"Execute tool '{tool_name}' with the given parameters",
                category="tool_execution",
                route="both",
                data_format="structured",
                agent_id=getattr(agent, "agent_id", None),
                session_id=getattr(agent, "session_id", None),
                tool_name=tool_name,
                quality_score=quality_score,
                execution_time_ms=execution_time_ms,
                tags=["tool_execution", tool_name],
                metadata={"execution_time_ms": execution_time_ms},
                structured_payload={
                    "tool_name": tool_name,
                    "input": input_params,
                    "output": result,
                    "execution_time_ms": execution_time_ms,
                },
            )
            return (
                f"Tool result logged (record: {record_id}). "
                f"Tool: {tool_name}, Time: {execution_time_ms:.0f}ms"
            )
        except Exception as e:
            logger.error(f"Tool result logging failed: {e}")
            return f"Error logging tool result: {str(e)}"

    # ------------------------------------------------------------------
    # submit_feedback
    # ------------------------------------------------------------------

    async def submit_feedback(
        original_input: str,
        original_response: str,
        feedback: str,
        rating: float = 0.5,
    ) -> str:
        """
        Submit user feedback on an agent response.
        This helps improve both training data quality and agent context.

        Args:
            original_input: The original user input
            original_response: The agent's original response
            feedback: The user's feedback text
            rating: User satisfaction rating (0.0 = bad, 1.0 = great)

        Returns:
            Confirmation with record ID
        """
        try:
            from ...core.pipeline_auto_capture import get_auto_capture
            auto_capture = await get_auto_capture()
            await auto_capture.capture_user_feedback(
                original_input=original_input,
                original_response=original_response,
                feedback=feedback,
                rating=rating,
                user_id=getattr(agent, "user_id", None),
                session_id=getattr(agent, "session_id", None),
            )
            return f"Feedback submitted (rating: {rating:.1f}). Thank you for the feedback."
        except Exception as e:
            logger.error(f"Feedback submission failed: {e}")
            return f"Error submitting feedback: {str(e)}"

    # ------------------------------------------------------------------
    # get_pipeline_stats
    # ------------------------------------------------------------------

    async def get_pipeline_stats() -> str:
        """
        Get current pipeline processing statistics.

        Returns:
            Pipeline stats including total processed, sent to Helox,
            stored in Cyrex, error count, and buffer size.
        """
        try:
            from ...core.realtime_data_pipeline import get_realtime_pipeline
            import json
            pipeline = await get_realtime_pipeline()
            stats = pipeline.get_stats()
            return json.dumps(stats, indent=2)
        except Exception as e:
            return f"Error getting pipeline stats: {str(e)}"

    # ------------------------------------------------------------------
    # Register all tools
    # ------------------------------------------------------------------

    agent.register_tool(
        "submit_training_data",
        submit_training_data,
        "Submit input/output pairs to the dual-route pipeline (Helox training + Cyrex runtime)",
    )
    agent.register_tool(
        "submit_structured_data",
        submit_structured_data,
        "Submit structured JSON data to the pipeline for both training and runtime",
    )
    agent.register_tool(
        "submit_to_helox",
        submit_to_helox,
        "Submit data exclusively to Helox for model training",
    )
    agent.register_tool(
        "submit_raw_to_helox",
        submit_raw_to_helox,
        "Submit raw unprocessed text to Helox for pre-training",
    )
    agent.register_tool(
        "submit_to_cyrex_runtime",
        submit_to_cyrex_runtime,
        "Store knowledge/context in Cyrex runtime for agent self-improvement",
    )
    agent.register_tool(
        "log_tool_result",
        log_tool_result,
        "Log a tool execution result to the pipeline for training optimisation",
    )
    agent.register_tool(
        "submit_feedback",
        submit_feedback,
        "Submit user feedback on a response for training and context improvement",
    )
    agent.register_tool(
        "get_pipeline_stats",
        get_pipeline_stats,
        "Get current pipeline processing statistics",
    )
