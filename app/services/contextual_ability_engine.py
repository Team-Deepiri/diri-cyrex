"""
Contextual Ability Engine
LLM + RAG-based dynamic ability generation
Tier 2: High Creativity & Flexibility
Uses LangChain for orchestration
"""
from typing import Dict, List, Optional
import json
import os
from datetime import datetime
from pydantic import BaseModel, Field
from ..logging_config import get_logger
from ..settings import settings
from .embedding_service import get_embedding_service

logger = get_logger("cyrex.contextual_ability_engine")

# LangChain imports with graceful fallbacks
HAS_LANGCHAIN = False
HAS_OPENAI = False

try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.documents import Document
    HAS_LANGCHAIN = True
except ImportError as e:
    logger.warning(f"LangChain core not available: {e}")
    ChatPromptTemplate = None
    MessagesPlaceholder = None
    JsonOutputParser = None
    PydanticOutputParser = None
    RunnablePassthrough = None
    Document = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        logger.warning("Text splitter not available")
        RecursiveCharacterTextSplitter = None

try:
    from langchain_community.vectorstores import Chroma, Milvus
except ImportError:
    logger.warning("LangChain community vectorstores not available")
    Chroma = None
    Milvus = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    logger.warning("HuggingFaceEmbeddings not available")
    HuggingFaceEmbeddings = None

try:
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import OpenAIEmbeddings
    HAS_OPENAI = True
except ImportError:
    logger.info("OpenAI LangChain packages not available, will use local LLM")
    ChatOpenAI = None
    OpenAIEmbeddings = None


class AbilityDefinition(BaseModel):
    """Structured output for generated abilities"""
    ability_name: str = Field(description="Creative, descriptive name for the ability")
    description: str = Field(description="Clear explanation of what the ability does")
    category: str = Field(description="One of: productivity, automation, boost, skill, gamification")
    steps: List[str] = Field(description="List of actionable steps to execute the ability")
    parameters: Dict = Field(description="Action parameters including target and options")
    momentum_cost: int = Field(description="Momentum cost (0-100, based on complexity)", ge=0, le=100)
    estimated_duration: int = Field(description="Estimated duration in minutes", ge=1, le=480)
    success_criteria: str = Field(description="How to measure success")
    prerequisites: List[str] = Field(default_factory=list, description="List of requirements")
    confidence: float = Field(description="Confidence in this ability (0-1)", ge=0, le=1)


class ContextualAbilityEngine:
    """
    Dynamic ability generation engine using LLM + RAG.
    Generates unique, contextual abilities on-the-fly based on user requests.
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4-turbo-preview",
        vector_store_type: str = "chroma",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.llm_model_name = llm_model
        self.vector_store_type = vector_store_type
        
        # Initialize LLM - prefer OpenAI/external API, fallback to local LLM
        if HAS_OPENAI and settings.OPENAI_API_KEY:
            # Try OpenAI first
            try:
                self.llm = ChatOpenAI(
                    model=llm_model,
                    temperature=0.7,
                    api_key=settings.OPENAI_API_KEY
                )
                logger.info(f"Using OpenAI: {llm_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}, falling back to local LLM")
                # Fall through to local LLM
                self.llm = None
        else:
            self.llm = None
        
        # Fallback to local LLM if OpenAI not available or failed
        if not self.llm:
            try:
                from ..integrations.local_llm import get_local_llm
                local_llm = get_local_llm()
                
                if local_llm and local_llm.is_available():
                    self.llm = local_llm.get_langchain_llm()
                    logger.info(f"Using local LLM: {local_llm.config.backend}")
                else:
                    raise ValueError("Local LLM not available")
            except Exception as e:
                logger.error(f"Failed to initialize local LLM: {e}")
                raise ValueError("No LLM available. Configure OpenAI API key or local LLM.")
        
        # Initialize embeddings
        if embedding_model.startswith("sentence-transformers"):
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        elif OpenAIEmbeddings and settings.OPENAI_API_KEY:
            self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        else:
            # Default to HuggingFace
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize vector store
        self.vectorstore = self._initialize_vectorstore()
        
        # Initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=AbilityDefinition)
        
        # Build prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Build RAG chain
        self.rag_chain = self._build_rag_chain()
        
        logger.info(f"Contextual ability engine initialized with {llm_model} and {vector_store_type}")
    
    def _initialize_vectorstore(self):
        """Initialize vector store for RAG"""
        if self.vector_store_type == "chroma":
            persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db/deepiri_abilities")
            os.makedirs(persist_directory, exist_ok=True)
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name="deepiri_abilities"
            )
        elif self.vector_store_type == "milvus":
            milvus_host = os.getenv("MILVUS_HOST", "localhost")
            milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
            return Milvus(
                embedding_function=self.embeddings,
                connection_args={"host": milvus_host, "port": milvus_port},
                collection_name="deepiri_abilities"
            )
        else:
            raise ValueError(f"Unknown vector store type: {self.vector_store_type}")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create structured prompt for ability generation"""
        template = """You are an AI assistant for Deepiri, a gamified productivity platform.

RETRIEVED CONTEXT FROM KNOWLEDGE BASE:
{context}

USER PROFILE:
- User ID: {user_id}
- Role: {user_role}
- Current Momentum: {momentum}
- Level: {level}
- Active Boosts: {active_boosts}
- Recent Activities: {recent_activities}

USER REQUEST:
{user_command}

PROJECT CONTEXT:
{project_context}

TASK:
Generate a dynamic, contextual ability that helps the user accomplish their request.
The ability must be actionable, fit within the gamification system, and respect user constraints.

{format_instructions}

RULES:
1. Abilities must align with user's role ({user_role}) and skill level (Level {level})
2. Momentum cost should be proportional to complexity:
   - Simple tasks: 5-15 momentum
   - Medium tasks: 20-40 momentum
   - Complex tasks: 50-100 momentum
3. Include clear, actionable steps that can be executed
4. Respect momentum balance (user has {momentum} momentum)
5. Be creative but practical
6. Consider active boosts: {active_boosts}
7. Estimated duration should be realistic (1-480 minutes)

Generate the ability now:"""
        
        return ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{user_command}")
        ])
    
    def _build_rag_chain(self):
        """Build LangChain RAG chain"""
        # Retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Format documents
        def format_docs(docs):
            return "\n\n".join([f"Document {i+1}:\n{doc.page_content}\nMetadata: {doc.metadata}" 
                              for i, doc in enumerate(docs)])
        
        # Chain: retrieve -> format -> prompt -> llm -> parse
        chain = (
            {
                "context": retriever | format_docs,
                "user_command": RunnablePassthrough(),
                "user_id": RunnablePassthrough(),
                "user_role": RunnablePassthrough(),
                "momentum": RunnablePassthrough(),
                "level": RunnablePassthrough(),
                "active_boosts": RunnablePassthrough(),
                "recent_activities": RunnablePassthrough(),
                "project_context": RunnablePassthrough(),
                "format_instructions": lambda _: self.output_parser.get_format_instructions(),
                "chat_history": lambda _: []
            }
            | self.prompt_template
            | self.llm
            | self.output_parser
        )
        
        return chain
    
    def generate_ability(
        self,
        user_id: str,
        user_command: str,
        user_profile: Dict,
        project_context: Optional[Dict] = None,
        chat_history: Optional[List] = None
    ) -> Dict:
        """
        Generate dynamic ability using LLM + RAG
        
        Args:
            user_id: User identifier
            user_command: User's request
            user_profile: User profile (role, momentum, level, etc.)
            project_context: Optional project context
            chat_history: Optional conversation history
        
        Returns:
            Generated ability with metadata
        """
        try:
            # Prepare inputs
            inputs = {
                "user_command": user_command,
                "user_id": user_id,
                "user_role": user_profile.get("role", "general"),
                "momentum": str(user_profile.get("momentum", 0)),
                "level": str(user_profile.get("level", 1)),
                "active_boosts": json.dumps(user_profile.get("active_boosts", [])),
                "recent_activities": json.dumps(user_profile.get("recent_activities", [])),
                "project_context": json.dumps(project_context or {})
            }
            
            # Invoke chain
            ability = self.rag_chain.invoke(inputs)
            
            # Validate and post-process
            ability_dict = ability.dict()
            ability_dict = self._validate_ability(ability_dict, user_profile)
            
            # Generate alternatives
            alternatives = self._generate_alternatives(user_command, ability_dict, user_profile)
            
            return {
                "success": True,
                "ability": ability_dict,
                "alternatives": alternatives,
                "generated_at": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Ability generation failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "ability": None
            }
    
    def _validate_ability(self, ability: Dict, user_profile: Dict) -> Dict:
        """Validate and adjust generated ability"""
        user_momentum = user_profile.get("momentum", 0)
        
        # Ensure momentum cost doesn't exceed user's balance
        if ability["momentum_cost"] > user_momentum:
            ability["momentum_cost"] = min(
                ability["momentum_cost"],
                int(user_momentum * 0.5)  # Max 50% of current momentum
            )
        
        # Ensure duration is reasonable
        if ability["estimated_duration"] > 480:  # 8 hours
            ability["estimated_duration"] = 480
        
        # Add metadata
        ability["user_level"] = user_profile.get("level", 1)
        ability["generated_at"] = datetime.utcnow().isoformat()
        
        return ability
    
    def _generate_alternatives(
        self,
        user_command: str,
        primary_ability: Dict,
        user_profile: Dict
    ) -> List[Dict]:
        """Generate alternative approaches"""
        alternatives = []
        
        # Lower cost alternative
        if primary_ability["momentum_cost"] > 20:
            alt = primary_ability.copy()
            alt["ability_name"] = f"{alt['ability_name']} (Lite Version)"
            alt["momentum_cost"] = int(alt["momentum_cost"] * 0.6)
            alt["estimated_duration"] = int(alt["estimated_duration"] * 1.3)
            alt["description"] = f"Simplified version: {alt['description']}"
            alternatives.append(alt)
        
        # Faster alternative
        if primary_ability["estimated_duration"] > 60:
            alt = primary_ability.copy()
            alt["ability_name"] = f"{alt['ability_name']} (Quick Mode)"
            alt["estimated_duration"] = int(alt["estimated_duration"] * 0.7)
            alt["momentum_cost"] = int(alt["momentum_cost"] * 1.2)
            alt["description"] = f"Faster execution: {alt['description']}"
            alternatives.append(alt)
        
        return alternatives
    
    def add_to_knowledge_base(
        self,
        ability: Dict,
        user_id: str,
        feedback: Optional[Dict] = None
    ):
        """Add generated ability to knowledge base for future RAG retrieval"""
        try:
            # Create document
            content = f"""
Ability: {ability['ability_name']}
Description: {ability['description']}
Category: {ability['category']}
Steps: {', '.join(ability['steps'])}
Success Criteria: {ability['success_criteria']}
"""
            
            metadata = {
                "user_id": user_id,
                "ability_name": ability["ability_name"],
                "category": ability["category"],
                "momentum_cost": ability["momentum_cost"],
                "success_rate": feedback.get("success", 0) if feedback else 0,
                "user_rating": feedback.get("rating", 0) if feedback else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Split text if needed
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = text_splitter.create_documents([content], [metadata])
            
            # Add to vector store
            self.vectorstore.add_documents(docs)
            self.vectorstore.persist()
            
            logger.info(f"Added ability to knowledge base: {ability['ability_name']}")
        
        except Exception as e:
            logger.error(f"Failed to add to knowledge base: {str(e)}")


# Singleton instance
_contextual_ability_engine = None

def get_contextual_ability_engine() -> ContextualAbilityEngine:
    """Get singleton ContextualAbilityEngine instance"""
    global _contextual_ability_engine
    if _contextual_ability_engine is None:
        _contextual_ability_engine = ContextualAbilityEngine()
    return _contextual_ability_engine

