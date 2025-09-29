from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types
import asyncio

import os
from dotenv import load_dotenv
load_dotenv(override=True)


DS_API_KEY = os.getenv("DS_API_KEY")
DS_BASE_URL = os.getenv("DS_BASE_URL")

model=LiteLlm(
    model="deepseek/deepseek-chat",  
    api_base=DS_BASE_URL,
    api_key=DS_API_KEY
)

init_agent = Agent(
    name="chabot",
    model=model,
    instruction="你是一位乐于助人的智能助手，请根据用户的问题给出最合适的回答。"
)

# Set up session service
session_service = InMemorySessionService()
APP_NAME = "streaming_app"
USER_ID = "user_123"
SESSION_ID = "session_456"

async def stream_response(query: str, runner: Runner):
    """Streams the agent's response token by token."""
    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    print(f"User: {query}")
    print("Agent: ", end="", flush=True)
    
    # 配置流式模式
    run_config = RunConfig(streaming_mode=StreamingMode.SSE)
    
    # 用于累积文本，避免重复输出
    accumulated_text = ""
        
    # 执行代理运行的流式输出
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
        run_config=run_config
    ):

        # um = event.usage_metadata
        # prompt_count = getattr(um, "prompt_token_count", None)
        # candidates_count = getattr(um, "candidates_token_count", None)
        # total_count = getattr(um, "total_token_count", None)
        # print(f"prompt_count: {prompt_count}")
        # print(f"candidates_count: {candidates_count}")
        # print(f"total_count: {total_count}")
        
        if event.content and event.content.parts:
            # 如果事件中存在工具调用，则打印执行的工具和参数
            if event.get_function_calls():
                print(f"需要执行外部工具:")
                for call in event.get_function_calls():
                    tool_name = call.name
                    arguments = call.args 
                    print(f"  执行的工具是: {tool_name}, 传递的参数是: {arguments}")

            # 如果存在工具调用且工具调用成功，则打印工具调用的结果
            if event.get_function_responses():
                print(f"工具调用成功:")
                for response in event.get_function_responses():
                    print(f"  工具：{response.name} 调用的结果是: {response.response}")
            
            # 输出文本内容
            if event.content.parts[0].text:
                current_text = event.content.parts[0].text
                print(current_text, end="", flush=True)  # 直接输出增量
    
    print("\n")  # 在每次回复结束后换行

# 运行交互式对话
async def main():
    # 创建会话
    await session_service.create_session(
        app_name=APP_NAME, 
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    # 创建runner
    runner = Runner(
        agent=init_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    print("=== 智能助手聊天程序 ===")
    print("输入 'quit' 或 'exit' 退出程序")
    print("=" * 25)
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入你的问题: ").strip()
            
            # 检查退出条件
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("再见!")
                break
            
            # 检查空输入
            if not user_input:
                print("请输入有效的问题")
                continue
            
            # 流式输出回复
            await stream_response(user_input, runner)
            
        except KeyboardInterrupt:
            print("\n\n程序被中断，再见!")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")

if __name__ == "__main__":
    # 运行
    asyncio.run(main())
