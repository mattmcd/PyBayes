import asyncio
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.tools import ComponentTool


async def create_agent():
    generator = OllamaChatGenerator(model="qwen3:8b")
    # search_tool = ComponentTool(component=SerperDevWebSearch())

    agent = Agent(
        chat_generator=generator,
        system_prompt="You are a helpful assistant.  Give one sentence answer to the user's question.",
        tools=[],
    )
    return agent

# generator = OllamaChatGenerator(
#     model="qwen3:8b",
#     url="http://localhost:11434",
#     generation_kwargs={
#         "num_predict": 100,
#         "temperature": 0.9,
#     }
# )


async def main():
    agent_task = asyncio.create_task(create_agent())
    question = input('Please enter your question: ')
    agent = await agent_task
    result = agent.run([ChatMessage.from_user(question)])
    print(result['last_message'].text)


if __name__ == "__main__":
    asyncio.run(main())
