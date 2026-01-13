"""
Predefined Prompt Templates and Data Structures
Standardized prompt system for agent interactions
"""
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import json
import uuid


class PromptCategory(str, Enum):
    """Prompt template categories"""
    TASK_DECOMPOSITION = "task_decomposition"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    CONVERSATION = "conversation"
    TOOL_USE = "tool_use"
    RAG_QUERY = "rag_query"
    CREATIVE = "creative"
    VENDOR_FRAUD = "vendor_fraud"
    DOCUMENT_PROCESSING = "document_processing"
    AUTOMATION = "automation"


class PromptRole(str, Enum):
    """Prompt roles for chat-style interactions"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


# ============================================================================
# Pydantic Models for Prompt Data Structures
# ============================================================================

class PromptMessage(BaseModel):
    """Single message in a prompt chain"""
    role: PromptRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PromptVariable(BaseModel):
    """Variable definition for prompt templates"""
    name: str
    description: str
    type: str = "string"  # string, number, boolean, json, list
    required: bool = True
    default: Optional[Any] = None
    examples: List[str] = Field(default_factory=list)
    validation: Optional[str] = None  # regex pattern


class ToolDefinition(BaseModel):
    """Tool definition for function calling"""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_params: List[str] = Field(default_factory=list)
    returns: Optional[str] = None


class PromptTemplate(BaseModel):
    """Complete prompt template with metadata"""
    template_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    category: PromptCategory
    version: str = "1.0.0"
    
    # System prompt template
    system_template: str
    
    # User prompt template (with variables)
    user_template: str
    
    # Variables that can be injected
    variables: List[PromptVariable] = Field(default_factory=list)
    
    # Available tools for this prompt
    tools: List[ToolDefinition] = Field(default_factory=list)
    
    # Example messages
    examples: List[PromptMessage] = Field(default_factory=list)
    
    # Configuration
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def render(self, variables: Dict[str, Any]) -> Dict[str, str]:
        """Render template with variables"""
        system_prompt = self.system_template
        user_prompt = self.user_template
        
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            str_value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            system_prompt = system_prompt.replace(placeholder, str_value)
            user_prompt = user_prompt.replace(placeholder, str_value)
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def to_messages(self, variables: Dict[str, Any]) -> List[Dict[str, str]]:
        """Convert to message format for LLM"""
        rendered = self.render(variables)
        messages = [
            {"role": "system", "content": rendered["system"]},
            {"role": "user", "content": rendered["user"]}
        ]
        return messages


class PromptContext(BaseModel):
    """Context for prompt execution"""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    memories: List[str] = Field(default_factory=list)
    history: List[PromptMessage] = Field(default_factory=list)
    tools_available: List[str] = Field(default_factory=list)
    custom_data: Dict[str, Any] = Field(default_factory=dict)


class PromptRequest(BaseModel):
    """Request to execute a prompt"""
    template_id: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[PromptContext] = None
    override_config: Optional[Dict[str, Any]] = None


class PromptResponse(BaseModel):
    """Response from prompt execution"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    template_id: str
    content: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tokens_used: int = 0
    execution_time_ms: float = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Predefined Prompt Templates
# ============================================================================

PREDEFINED_TEMPLATES: Dict[str, PromptTemplate] = {}

# Task Decomposition Template
PREDEFINED_TEMPLATES["task_decomposition"] = PromptTemplate(
    template_id="task_decomposition_v1",
    name="Task Decomposition Agent",
    description="Break down complex tasks into manageable subtasks",
    category=PromptCategory.TASK_DECOMPOSITION,
    system_template="""You are a Task Decomposition Agent specialized in breaking down complex tasks into clear, actionable subtasks.

Your capabilities:
- Analyze task complexity and dependencies
- Create hierarchical task breakdowns
- Estimate time and effort for each subtask
- Identify potential blockers and risks

Guidelines:
- Be specific and actionable
- Consider dependencies between subtasks
- Provide time estimates when possible
- Flag any tasks that require clarification

Current context:
{context}""",
    user_template="""Please decompose the following task:

Task: {task_description}

Requirements:
{requirements}

Constraints:
{constraints}

Provide a structured breakdown with:
1. Main subtasks (numbered)
2. Dependencies between tasks
3. Time estimates
4. Priority levels (high/medium/low)
5. Potential risks or blockers""",
    variables=[
        PromptVariable(name="task_description", description="The main task to decompose", required=True),
        PromptVariable(name="requirements", description="Specific requirements for the task", default="None specified"),
        PromptVariable(name="constraints", description="Any constraints or limitations", default="None specified"),
        PromptVariable(name="context", description="Additional context", default="{}")
    ],
    temperature=0.5,
    max_tokens=2000,
    tags=["productivity", "planning", "task-management"]
)

# Code Generation Template
PREDEFINED_TEMPLATES["code_generation"] = PromptTemplate(
    template_id="code_generation_v1",
    name="Code Generation Agent",
    description="Generate code based on requirements",
    category=PromptCategory.CODE_GENERATION,
    system_template="""You are a Code Generation Agent specialized in writing clean, efficient, and well-documented code.

Your capabilities:
- Generate code in multiple programming languages
- Follow best practices and design patterns
- Write comprehensive documentation
- Include error handling and edge cases

Guidelines:
- Write clean, readable code
- Include comments explaining complex logic
- Follow language-specific conventions
- Consider security best practices

Language preference: {language}
Framework/Libraries: {frameworks}""",
    user_template="""Please generate code for the following:

Description: {description}

Functionality:
{functionality}

Input/Output:
- Input: {input_spec}
- Output: {output_spec}

Additional requirements:
{additional_requirements}

Please provide:
1. Complete, working code
2. Usage examples
3. Any necessary dependencies""",
    variables=[
        PromptVariable(name="language", description="Programming language", default="Python"),
        PromptVariable(name="frameworks", description="Frameworks or libraries to use", default="None"),
        PromptVariable(name="description", description="What the code should do", required=True),
        PromptVariable(name="functionality", description="Detailed functionality requirements", required=True),
        PromptVariable(name="input_spec", description="Expected input specification", default="Not specified"),
        PromptVariable(name="output_spec", description="Expected output specification", default="Not specified"),
        PromptVariable(name="additional_requirements", description="Any additional requirements", default="None")
    ],
    temperature=0.3,
    max_tokens=4000,
    tags=["development", "code", "automation"]
)

# Data Analysis Template
PREDEFINED_TEMPLATES["data_analysis"] = PromptTemplate(
    template_id="data_analysis_v1",
    name="Data Analysis Agent",
    description="Analyze data and provide insights",
    category=PromptCategory.DATA_ANALYSIS,
    system_template="""You are a Data Analysis Agent specialized in extracting insights from data.

Your capabilities:
- Statistical analysis and interpretation
- Pattern recognition and trend identification
- Data visualization recommendations
- Predictive insights

Guidelines:
- Be precise with numbers and statistics
- Explain findings in clear, accessible language
- Highlight key insights and actionable recommendations
- Consider data quality and limitations

Analysis type: {analysis_type}""",
    user_template="""Please analyze the following data:

Data Description: {data_description}

Data:
{data}

Analysis Goals:
{analysis_goals}

Please provide:
1. Summary statistics
2. Key patterns and trends
3. Notable outliers or anomalies
4. Actionable insights
5. Visualization recommendations""",
    variables=[
        PromptVariable(name="analysis_type", description="Type of analysis", default="General"),
        PromptVariable(name="data_description", description="Description of the data", required=True),
        PromptVariable(name="data", description="The data to analyze", required=True, type="json"),
        PromptVariable(name="analysis_goals", description="What you want to learn", default="General insights")
    ],
    temperature=0.4,
    max_tokens=3000,
    tags=["analytics", "data", "insights"]
)

# RAG Query Template
PREDEFINED_TEMPLATES["rag_query"] = PromptTemplate(
    template_id="rag_query_v1",
    name="RAG Query Agent",
    description="Answer questions using retrieved context",
    category=PromptCategory.RAG_QUERY,
    system_template="""You are a Knowledge Assistant that answers questions based on retrieved context.

Guidelines:
- Only use information from the provided context
- If the context doesn't contain the answer, say so clearly
- Cite specific parts of the context when possible
- Be concise but thorough

Context documents:
{retrieved_context}""",
    user_template="""Question: {question}

Please answer based on the provided context. If the context doesn't contain enough information to fully answer, indicate what's missing.""",
    variables=[
        PromptVariable(name="retrieved_context", description="Context retrieved from vector store", required=True),
        PromptVariable(name="question", description="User's question", required=True)
    ],
    temperature=0.3,
    max_tokens=1500,
    tags=["rag", "knowledge", "qa"]
)

# Tool Use Template
PREDEFINED_TEMPLATES["tool_use"] = PromptTemplate(
    template_id="tool_use_v1",
    name="Tool Use Agent",
    description="Agent that can use tools to accomplish tasks",
    category=PromptCategory.TOOL_USE,
    system_template="""You are a Tool-Using Agent capable of calling external tools to accomplish tasks.

Available Tools:
{available_tools}

Guidelines:
- Only use tools when necessary
- Provide tool calls in the format: [TOOL:tool_name:{"param": "value"}]
- Explain why you're using each tool
- Handle tool errors gracefully

Current session: {session_id}""",
    user_template="""Task: {task}

Additional context:
{context}

Complete this task using the available tools. Explain your reasoning and provide the final result.""",
    variables=[
        PromptVariable(name="available_tools", description="List of available tools", required=True),
        PromptVariable(name="task", description="Task to complete", required=True),
        PromptVariable(name="context", description="Additional context", default="{}"),
        PromptVariable(name="session_id", description="Current session ID", default="default")
    ],
    temperature=0.5,
    max_tokens=2000,
    tools=[
        ToolDefinition(
            name="search_memory",
            description="Search agent memory for relevant information",
            parameters={"query": {"type": "string"}, "limit": {"type": "integer"}},
            required_params=["query"]
        ),
        ToolDefinition(
            name="store_memory",
            description="Store information in agent memory",
            parameters={"content": {"type": "string"}, "importance": {"type": "number"}},
            required_params=["content"]
        ),
        ToolDefinition(
            name="call_api",
            description="Call an external API",
            parameters={"endpoint": {"type": "string"}, "method": {"type": "string"}, "data": {"type": "object"}},
            required_params=["endpoint"]
        )
    ],
    tags=["tools", "automation", "agentic"]
)

# Vendor Fraud Detection Template
PREDEFINED_TEMPLATES["vendor_fraud"] = PromptTemplate(
    template_id="vendor_fraud_v1",
    name="Vendor Fraud Detection Agent",
    description="Analyze invoices and detect potential fraud",
    category=PromptCategory.VENDOR_FRAUD,
    system_template="""You are a Vendor Fraud Detection Agent specialized in identifying fraudulent invoices and vendor behavior.

Your capabilities:
- Invoice analysis and anomaly detection
- Price benchmarking against market rates
- Vendor behavior pattern analysis
- Document verification
- Risk scoring

Industry: {industry}

Guidelines:
- Be thorough in your analysis
- Provide confidence scores for findings
- Flag high-risk items clearly
- Consider context and industry norms
- Recommend specific actions

Risk thresholds:
- High Risk: Score > 0.7
- Medium Risk: Score 0.4-0.7
- Low Risk: Score < 0.4""",
    user_template="""Please analyze the following for potential fraud:

Document Type: {document_type}
Vendor: {vendor_name}
Amount: {amount}

Document Content:
{document_content}

Historical Context:
{historical_context}

Please provide:
1. Risk assessment (score 0-1)
2. Identified anomalies
3. Comparison to benchmarks
4. Recommended actions
5. Supporting evidence""",
    variables=[
        PromptVariable(name="industry", description="Industry niche", default="general"),
        PromptVariable(name="document_type", description="Type of document (invoice, contract, etc.)", required=True),
        PromptVariable(name="vendor_name", description="Vendor name", required=True),
        PromptVariable(name="amount", description="Total amount", required=True),
        PromptVariable(name="document_content", description="Document content to analyze", required=True),
        PromptVariable(name="historical_context", description="Historical data for comparison", default="{}")
    ],
    temperature=0.2,
    max_tokens=2500,
    tags=["fraud", "vendor", "finance", "risk"]
)

# Document Processing Template
PREDEFINED_TEMPLATES["document_processing"] = PromptTemplate(
    template_id="document_processing_v1",
    name="Document Processing Agent",
    description="Extract and structure information from documents",
    category=PromptCategory.DOCUMENT_PROCESSING,
    system_template="""You are a Document Processing Agent specialized in extracting structured information from documents.

Your capabilities:
- Text extraction and OCR interpretation
- Entity recognition (names, dates, amounts, etc.)
- Table and list extraction
- Document classification
- Data normalization

Output format: {output_format}

Guidelines:
- Extract all relevant information
- Maintain data relationships
- Handle ambiguous or unclear text carefully
- Validate extracted data when possible""",
    user_template="""Process the following document:

Document Type: {document_type}

Content:
{document_content}

Please extract:
{extraction_fields}

Provide the extracted data in the specified format.""",
    variables=[
        PromptVariable(name="output_format", description="Output format (JSON, CSV, etc.)", default="JSON"),
        PromptVariable(name="document_type", description="Type of document", required=True),
        PromptVariable(name="document_content", description="Document content", required=True),
        PromptVariable(name="extraction_fields", description="Fields to extract", default="All relevant fields")
    ],
    temperature=0.2,
    max_tokens=2000,
    tags=["document", "extraction", "processing"]
)

# Automation Template
PREDEFINED_TEMPLATES["automation"] = PromptTemplate(
    template_id="automation_v1",
    name="Automation Agent",
    description="Plan and execute automation workflows",
    category=PromptCategory.AUTOMATION,
    system_template="""You are an Automation Agent specialized in planning and executing automated workflows.

Your capabilities:
- Workflow design and optimization
- Task scheduling and orchestration
- Error handling and recovery
- Integration with external systems

Available integrations: {available_integrations}

Guidelines:
- Design robust, fault-tolerant workflows
- Consider edge cases and error scenarios
- Optimize for efficiency and reliability
- Provide clear step-by-step plans""",
    user_template="""Automate the following:

Goal: {automation_goal}

Current manual process:
{manual_process}

Constraints:
{constraints}

Please provide:
1. Automation workflow design
2. Required integrations
3. Error handling strategy
4. Testing approach
5. Rollback plan""",
    variables=[
        PromptVariable(name="available_integrations", description="Available system integrations", default="None specified"),
        PromptVariable(name="automation_goal", description="What to automate", required=True),
        PromptVariable(name="manual_process", description="Current manual process", required=True),
        PromptVariable(name="constraints", description="Constraints or limitations", default="None")
    ],
    temperature=0.4,
    max_tokens=2500,
    tags=["automation", "workflow", "integration"]
)

# Conversation Template
PREDEFINED_TEMPLATES["conversation"] = PromptTemplate(
    template_id="conversation_v1",
    name="Conversational Agent",
    description="General conversation and assistance",
    category=PromptCategory.CONVERSATION,
    system_template="""You are a helpful AI assistant named {agent_name}.

Personality: {personality}

Guidelines:
- Be helpful, accurate, and friendly
- Ask clarifying questions when needed
- Admit when you don't know something
- Maintain context from the conversation

User preferences:
{user_preferences}

Conversation history:
{conversation_history}""",
    user_template="""{user_message}""",
    variables=[
        PromptVariable(name="agent_name", description="Name of the agent", default="Assistant"),
        PromptVariable(name="personality", description="Agent personality traits", default="Professional and friendly"),
        PromptVariable(name="user_preferences", description="User preferences", default="{}"),
        PromptVariable(name="conversation_history", description="Previous conversation", default="No previous context"),
        PromptVariable(name="user_message", description="User's message", required=True)
    ],
    temperature=0.7,
    max_tokens=1500,
    tags=["conversation", "chat", "assistant"]
)


# ============================================================================
# Prompt Template Manager
# ============================================================================

class PromptTemplateManager:
    """Manager for prompt templates"""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = PREDEFINED_TEMPLATES.copy()
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)
    
    def list_templates(self, category: Optional[PromptCategory] = None) -> List[PromptTemplate]:
        """List all templates, optionally filtered by category"""
        templates = list(self.templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates
    
    def add_template(self, template: PromptTemplate) -> str:
        """Add a new template"""
        self.templates[template.template_id] = template
        return template.template_id
    
    def update_template(self, template_id: str, updates: Dict[str, Any]) -> Optional[PromptTemplate]:
        """Update an existing template"""
        if template_id not in self.templates:
            return None
        
        template = self.templates[template_id]
        for key, value in updates.items():
            if hasattr(template, key):
                setattr(template, key, value)
        template.updated_at = datetime.utcnow()
        return template
    
    def delete_template(self, template_id: str) -> bool:
        """Delete a template"""
        if template_id in self.templates:
            del self.templates[template_id]
            return True
        return False
    
    def render_template(self, template_id: str, variables: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Render a template with variables"""
        template = self.get_template(template_id)
        if template:
            return template.render(variables)
        return None
    
    def validate_variables(self, template_id: str, variables: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate variables against template requirements"""
        template = self.get_template(template_id)
        if not template:
            return {"errors": ["Template not found"]}
        
        errors = []
        warnings = []
        
        for var in template.variables:
            if var.required and var.name not in variables:
                if var.default is None:
                    errors.append(f"Required variable '{var.name}' is missing")
            
            if var.name in variables and var.validation:
                import re
                if not re.match(var.validation, str(variables[var.name])):
                    errors.append(f"Variable '{var.name}' does not match validation pattern")
        
        return {"errors": errors, "warnings": warnings}


# Singleton instance
_template_manager: Optional[PromptTemplateManager] = None


def get_prompt_template_manager() -> PromptTemplateManager:
    """Get or create prompt template manager singleton"""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    return _template_manager

