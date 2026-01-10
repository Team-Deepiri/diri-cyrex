
import sys
import os
from typing import Dict, Any, Optional

# Add parent directory to path to import Cyrex modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    RunnablePassthrough = None
    StrOutputParser = None

# Import Cyrex modules
try:
    from app.integrations.local_llm import get_local_llm, LLMBackend
    from app.settings import settings
    from app.logging_config import get_logger
    HAS_CYREX = True
except ImportError:
    HAS_CYREX = False

# Import our local modules
from .prompts import LEARNING_PROMPT
from .tools import fake_tool
from .guardrails import validate_input, validate_output

logger = get_logger("cyrex.luke.chain") if HAS_CYREX else None


def create_learning_chain() -> Optional[Any]:
    """
    Create a minimal LangChain chain using Ollama via Cyrex.
    
    Returns:
        A LangChain chain that can be invoked, or None if setup fails
    """
    if not HAS_LANGCHAIN:
        if logger:
            logger.error("LangChain not available - cannot create chain")
        return None
    
    if not HAS_CYREX:
        if logger:
            logger.error("Cyrex modules not available - cannot create chain")
        return None
    
    try:
        # Get Ollama LLM through Cyrex's routing
        # This uses settings.LOCAL_LLM_BACKEND and settings.LOCAL_LLM_MODEL
        local_llm = get_local_llm(
            backend=settings.LOCAL_LLM_BACKEND,
            model_name=settings.LOCAL_LLM_MODEL,
            base_url=getattr(settings, 'OLLAMA_BASE_URL', None),
        )
        
        if not local_llm or not local_llm.is_available():
            if logger:
                logger.error("Ollama LLM not available - check Ollama is running")
            return None
        
        # Get the LangChain LLM object
        llm = local_llm.get_llm()
        
        if logger:
            logger.info(f"Created learning chain with Ollama model: {settings.LOCAL_LLM_MODEL}")
        
        # Create a simple chain: Prompt → LLM → Output Parser
        # This is the simplest possible LangChain chain using LCEL (LangChain Expression Language)
        chain = (
            {"user_input": RunnablePassthrough()}
            | LEARNING_PROMPT
            | llm
            | StrOutputParser()
        )
        
        return chain
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to create learning chain: {e}", exc_info=True)
        return None


def run_chain(chain: Any, user_input: str) -> Dict[str, Any]:
    """
    Run the chain with user input, including validation.
    
    Args:
        chain: The LangChain chain to run
        user_input: The user's input string
    
    Returns:
        Dict with 'success', 'output', and optional 'error' keys
    """
    # Validate input
    input_validation = validate_input(user_input)
    if not input_validation["valid"]:
        return {
            "success": False,
            "error": input_validation.get("error", "Invalid input"),
            "output": None
        }
    
    try:
        # Invoke the chain
        output = chain.invoke({"user_input": user_input})
        
        # Validate output
        output_validation = validate_output(output)
        if not output_validation["valid"]:
            return {
                "success": False,
                "error": output_validation.get("error", "Invalid output"),
                "output": output
            }
        
        return {
            "success": True,
            "output": output,
            "error": None
        }
        
    except Exception as e:
        error_msg = str(e)
        if logger:
            logger.error(f"Chain execution failed: {error_msg}", exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "output": None
        }


# Create the chain at module level (will be None if setup fails)
learning_chain = create_learning_chain()

