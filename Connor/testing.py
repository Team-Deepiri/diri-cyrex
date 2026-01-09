import asyncio
from .agent_init import create_agent
from app.integrations.local_llm import get_local_llm  # or your LLM constructor

async def main():
    # 1. Initialize LLM
    llm = get_local_llm()

    # 2. Create agent
    agent = create_agent(llm)

    # 3. Provide input
    user_input = "Write a step-by-step plan to set up a PostgreSQL database with replication"

    # 4. Run agent
    output = await agent.run(user_input)

    # 5. Print results
    print(output)

if __name__ == "__main__":
    asyncio.run(main())