"""
DEPRECATED: Use contextual_ability_engine.py instead
This file is kept for backward compatibility but should not be used in new code.
"""

class AbilityGenerator:
    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize RAG components
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )
        
        # Load prompt template
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create structured prompt for ability generation"""
        template = """
You are an AI assistant for Deepiri, a gamified productivity platform.

RETRIEVED CONTEXT:
{context}

USER PROFILE:
- User ID: {user_id}
- Role: {user_role}
- Current Momentum: {momentum}
- Level: {level}
- Active Boosts: {active_boosts}

USER REQUEST:
{user_command}

PROJECT CONTEXT:
{project_context}

TASK:
Generate a dynamic, contextual ability that helps the user accomplish their request.
The ability should be actionable, fit within the gamification system, and respect user constraints.

OUTPUT (JSON format):
{{
  "ability_name": "string (creative, descriptive name)",
  "description": "string (clear explanation of what it does)",
  "category": "string (productivity|automation|boost|skill)",
  "steps": ["step1", "step2", "step3"],
  "parameters": {{
    "action": "string",
    "target": "string",
    "options": {{}}
  }},
  "momentum_cost": number (0-100, based on complexity),
  "estimated_duration": number (minutes),
  "success_criteria": "string (how to measure success)",
  "prerequisites": ["list of requirements"],
  "confidence": number (0-1, your confidence in this ability)
}}

RULES:
1. Abilities must align with user's role and skill level
2. Cost should be proportional to complexity (simple: 5-15, medium: 20-40, complex: 50-100)
3. Include clear, actionable steps
4. Respect momentum balance (user current: {momentum})
5. Be creative but practical
6. Consider active boosts and suggest synergies

Generate the ability now:
"""
        return PromptTemplate(
            template=template,
            input_variables=[
                "context", "user_id", "user_role", "momentum", "level",
                "active_boosts", "user_command", "project_context"
            ]
        )
    
    def generate_ability(
        self,
        user_id: str,
        user_command: str,
        user_profile: Dict,
        project_context: Dict = None
    ) -> Dict:
        """
        Generate dynamic ability using LLM + RAG
        
        Args:
            user_id: User identifier
            user_command: User's request
            user_profile: User profile (role, momentum, level, etc.)
            project_context: Optional project context
        
        Returns:
            Generated ability as dict
        """
        # Retrieve relevant context from vector store
        retrieved_docs = self.vectorstore.similarity_search(
            user_command,
            k=5,
            filter={"user_id": user_id}
        )
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # Prepare inputs
        inputs = {
            "context": context,
            "user_id": user_id,
            "user_command": user_command,
            "user_role": user_profile.get("role", "general"),
            "momentum": user_profile.get("momentum", 0),
            "level": user_profile.get("level", 1),
            "active_boosts": json.dumps(user_profile.get("active_boosts", [])),
            "project_context": json.dumps(project_context or {})
        }
        
        # Generate ability
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        result = chain.run(**inputs)
        
        # Parse JSON response
        try:
            ability = json.loads(result)
            
            # Validate and post-process
            ability = self._validate_ability(ability, user_profile)
            
            return {
                "success": True,
                "ability": ability,
                "alternatives": self._generate_alternatives(user_command, ability)
            }
        
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Failed to parse LLM response: {str(e)}",
                "raw_response": result
            }
    
    def _validate_ability(self, ability: Dict, user_profile: Dict) -> Dict:
        """Validate and adjust generated ability"""
        # Ensure momentum cost doesn't exceed user's balance
        if ability["momentum_cost"] > user_profile.get("momentum", 0):
            ability["momentum_cost"] = min(
                ability["momentum_cost"],
                user_profile["momentum"] * 0.5  # Max 50% of current momentum
            )
        
        # Ensure duration is reasonable
        if ability["estimated_duration"] > 480:  # 8 hours
            ability["estimated_duration"] = 480
        
        # Add metadata
        ability["generated_at"] = self._get_timestamp()
        ability["user_level"] = user_profile.get("level", 1)
        
        return ability
    
    def _generate_alternatives(self, user_command: str, primary_ability: Dict) -> List[Dict]:
        """Generate alternative approaches"""
        # Simple implementation - can be expanded
        alternatives = []
        
        # Lower cost alternative
        if primary_ability["momentum_cost"] > 20:
            alt = primary_ability.copy()
            alt["ability_name"] = f"{alt['ability_name']} (Lite)"
            alt["momentum_cost"] = int(alt["momentum_cost"] * 0.6)
            alt["estimated_duration"] = int(alt["estimated_duration"] * 1.3)
            alternatives.append(alt)
        
        return alternatives
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def add_to_knowledge_base(self, user_id: str, ability: Dict, feedback: Dict):
        """Add generated ability to knowledge base for future RAG retrieval"""
        document = {
            "content": f"{ability['ability_name']}: {ability['description']}",
            "metadata": {
                "user_id": user_id,
                "ability_name": ability["ability_name"],
                "category": ability["category"],
                "success_rate": feedback.get("success", 0),
                "user_rating": feedback.get("rating", 0),
                "timestamp": self._get_timestamp()
            }
        }
        
        self.vectorstore.add_texts(
            texts=[document["content"]],
            metadatas=[document["metadata"]]
        )

# Example usage
if __name__ == "__main__":
    generator = AbilityGenerator()
    
    user_profile = {
        "user_id": "user123",
        "role": "software_engineer",
        "momentum": 450,
        "level": 15,
        "active_boosts": ["focus"]
    }
    
    command = "I need to refactor this codebase to use TypeScript"
    
    result = generator.generate_ability(
        user_id="user123",
        user_command=command,
        user_profile=user_profile,
        project_context={"language": "JavaScript", "files": 50}
    )
    
    print(json.dumps(result, indent=2))

