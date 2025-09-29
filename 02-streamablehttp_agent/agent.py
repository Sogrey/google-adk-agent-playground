
import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
load_dotenv(override=True)

# 配置模型
model = LiteLlm(
    model="deepseek/deepseek-chat",  
    api_base=os.getenv("DS_BASE_URL"),
    api_key=os.getenv("DS_API_KEY")
)


# 构建 Firecrawl MCP Server
firecrawl_mcp_server = MCPToolset(
    connection_params=StreamableHTTPServerParams(
        url="https://mcp.api-inference.modelscope.net/2fae18c219bd42/mcp",  # Firecrawl MCP 服务器
        timeout=30.0,  # 增加超时时间
        sse_read_timeout=600.0,  # 增加 SSE 读取超时时间
        terminate_on_close=True  # 设客户端关闭连接时，请求体里带 terminate=true，服务器立即回收资源，避免僵尸会话
    )
)

# 创建Agent
root_agent = LlmAgent(
    name="firecrawl_web_agent",  # 网页抓取助手
    model=model,
    instruction="""
    你是一个专业的网页内容抓取和分析助手，专门帮助用户抓取、解析和分析网页内容。
    当用户提出网页抓取需求时，你需要根据用户的需求，灵活调用`firecrawl_mcp_server`工具来提供专业的网页数据提取服务。
    
    你的主要功能包括但不限于：
    1. 网页抓取：抓取指定URL的完整网页内容
    2. 内容提取：从网页中提取结构化数据
    3. 批量抓取：对多个URL进行批量处理
    4. 内容清理：清理和格式化抓取的内容
    5. 数据解析：解析HTML、提取文本、链接等信息
    6. 内容分析：分析网页结构、提取关键信息
    7. 格式转换：将抓取的内容转换为不同格式（JSON、Markdown等）
    
    工作流程：
    1. 理解用户的抓取需求（目标URL、提取内容类型等）
    2. 使用合适的Firecrawl工具进行网页抓取
    3. 处理和清理抓取的数据
    4. 根据用户需求进行内容分析和提取
    5. 以用户友好的格式返回结果
    6. 如有需要，提供进一步的数据处理建议
    
    注意事项：
    - 遵守网站的robots.txt和使用条款
    - 合理控制抓取频率，避免对目标网站造成压力
    - 处理可能出现的网络错误和异常情况
    - 保护用户隐私和数据安全
    """,
    tools=[firecrawl_mcp_server], # 接入 Firecrawl MCP 服务器
)