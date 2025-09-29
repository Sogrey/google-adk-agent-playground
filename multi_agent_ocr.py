"""
高精度多智能体图像分析系统
"""

import asyncio
import os
import mimetypes
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from dotenv import load_dotenv
load_dotenv(override=True)

QWEN_MODEL = os.getenv("QWEN_MODEL_NAME")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL_NAME")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")


APP_NAME = "multi_agent_analyzer"
USER_ID = "analyst"
SESSION_ID = "session_001"

# ================ 工具函数 ================
def load_local_image(image_path: str) -> tuple[bytes, str]:
    """加载本地图片"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image/'):
        mime_type = 'image/jpeg'
    
    print(f"加载图片: {image_path}")
    print(f"文件大小: {len(image_data)} bytes")
    
    return image_data, mime_type

# ================ 智能体定义 ================

# 1. 文本提取专家 - 负责OCR和文本识别
text_extractor = Agent(
    name="text_extractor",
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY,
    ),
    description="文本提取专家，专门从图像中提取文字内容",
    instruction="""
你是专业的OCR文本提取专家。请仔细观察图片并：

1. 提取图片中的所有可见文字
2. 按照在图片中的位置顺序整理文字
3. 区分不同类型的文字（标题、正文、标签等）
4. 如果是图表，重点关注数据标签和数值

输出格式：
- 直接输出提取到的文字内容
- 用换行分隔不同区域的文字
- 在文字前标注位置（如：标题、左上角、图表标签等）
"""
)

# 2. 布局分析专家 - 负责分析图像的空间结构
layout_analyzer = Agent(
    name="layout_analyzer", 
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY,
    ),
    description="布局分析专家，专门分析图像的空间结构和视觉设计",
    instruction="""
你是专业的视觉布局分析专家。请分析图片的空间结构：

1. 描述图片的整体布局（网格、列表、卡片等）
2. 识别主要的视觉区域和元素分组
3. 分析视觉层次和重要性排序
4. 描述元素之间的空间关系

输出格式：
- 整体布局类型
- 主要区域划分
- 视觉重点区域
- 元素排列规律
"""
)

# 3. 内容理解专家 - 负责理解图像的含义和上下文
content_analyzer = Agent(
    name="content_analyzer",
    model=LiteLlm(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    ),
    description="内容理解专家，负责深度分析图像内容的含义",
    instruction="""
你是资深的内容分析专家。基于提供的文本和布局信息，请深度分析：

1. 判断图像的领域类型（商业图表、学术研究、界面截图、课程教程、论文等）
2. 识别关键信息和重要数据点
3. 分析数据趋势和模式（如有）
4. 理解图像要传达的核心信息

输出格式：
- 图像类型和用途
- 关键信息总结
- 重要发现和洞察
- 数据趋势分析（如适用）
"""
)

# 4. 总结专家 - 负责生成最终的综合报告
summary_generator = Agent(
    name="summary_generator",
    model=LiteLlm(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    ),
    description="总结专家，生成易懂的综合分析报告",
    instruction="""
你是专业的报告撰写专家。基于前面专家的分析，生成一份清晰易懂的综合报告：

1. 用通俗语言描述图像的主要内容
2. 突出最重要的3-5个关键发现
3. 如果是数据图表，解释趋势和含义
4. 提供实用的结论或建议

输出格式：
# 图像概览：[一句话概括]

# 关键发现：
1. [发现1]
2. [发现2] 
3. [发现3]

# 深度分析：[详细解读]

# 结论建议：[实用建议]
"""
)

# ================ 分析函数 ================

async def analyze_image_step_by_step(image_path: str):
    """逐步分析图像（顺序模式）"""
    print("开始逐步分析...")
    
    # 加载图像
    image_data, mime_type = load_local_image(image_path)
    
    # 创建图像消息
    image_message = types.Content(
        role="user",
        parts=[
            types.Part(text="请分析这张图片"),
            types.Part(
                inline_data=types.Blob(
                    data=image_data,
                    mime_type=mime_type
                )
            )
        ]
    )
    
    # 初始化session
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    
    # 存储各步骤结果
    results = {}
    
    # 步骤1: 文本提取
    print("\n步骤1: 文本提取")
    print("-" * 30)
    
    text_runner = Runner(
        agent=text_extractor,
        session_service=session_service,
        app_name=APP_NAME,
    )
    
    async for event in text_runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID, 
        new_message=image_message,
    ):
        if event.content and event.content.parts and event.content.parts[0].text:
            print(f"文本提取结果:")
            print(event.content.parts[0].text)
            results['text_extraction'] = event.content.parts[0].text
            break
    
    # 步骤2: 布局分析
    print("\n步骤2: 布局分析")
    print("-" * 30)
    
    layout_runner = Runner(
        agent=layout_analyzer,
        session_service=session_service,
        app_name=APP_NAME,
    )
    
    # 布局分析也需要看图片，所以发送图像消息
    async for event in layout_runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=image_message,
    ):
        if event.content and event.content.parts and event.content.parts[0].text:
            print(f"布局分析结果:")
            print(event.content.parts[0].text)
            results['layout_analysis'] = event.content.parts[0].text
            break
    
    # 步骤3: 内容理解（基于session中前面的分析结果）
    print("\n步骤3: 内容理解")
    print("-" * 30)
    
    # 基于session上下文进行分析，不需要重复提供之前的结果
    context_message = types.Content(
        role="user",
        parts=[
            types.Part(text="请基于前面专家的文本提取和布局分析结果，进行深度的内容理解和分析。")
        ]
    )
    
    content_runner = Runner(
        agent=content_analyzer,
        session_service=session_service,
        app_name=APP_NAME,
    )
    
    async for event in content_runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,  # 使用相同session，可以读取之前的对话历史
        new_message=context_message,
    ):
        if event.content and event.content.parts and event.content.parts[0].text:
            print(f"内容理解结果:")
            print(event.content.parts[0].text)
            results['content_analysis'] = event.content.parts[0].text
            break
    
    # 步骤4: 综合总结（基于完整的session对话历史）
    print("\n步骤4: 综合总结")
    print("-" * 30)
    
    # 基于完整的session对话历史生成最终报告
    summary_message = types.Content(
        role="user",
        parts=[
            types.Part(text="请基于前面所有专家的分析结果，生成一份完整的综合分析报告。")
        ]
    )
    
    summary_runner = Runner(
        agent=summary_generator,
        session_service=session_service,
        app_name=APP_NAME,
    )
    
    async for event in summary_runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,  # 使用相同session，可以看到完整的对话历史
        new_message=summary_message,
    ):
        if event.content and event.content.parts and event.content.parts[0].text:
            print(f"最终报告:")
            print("=" * 50)
            print(event.content.parts[0].text)
            results['final_summary'] = event.content.parts[0].text
            break
    
    return results

async def analyze_image_parallel(image_path: str):
    """并行分析图像（快速模式）"""
    print("开始并行分析...")
    
    # 加载图像
    image_data, mime_type = load_local_image(image_path)
    
    # 创建图像消息
    image_message = types.Content(
        role="user",
        parts=[
            types.Part(text="请分析这张图片"),
            types.Part(
                inline_data=types.Blob(
                    data=image_data,
                    mime_type=mime_type
                )
            )
        ]
    )
    
    # 初始化session
    session_service = InMemorySessionService()
    
    # 为并行任务创建独立的session
    text_session_id = SESSION_ID + "_text_parallel"
    layout_session_id = SESSION_ID + "_layout_parallel" 
    final_session_id = SESSION_ID + "_final"
    
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=text_session_id)
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=layout_session_id)
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=final_session_id)
    
    # 并行运行文本提取和布局分析
    async def run_text_extraction():
        runner = Runner(
            agent=text_extractor,
            session_service=session_service,
            app_name=APP_NAME,
        )
        
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=text_session_id,
            new_message=image_message,
        ):
            if event.content and event.content.parts and event.content.parts[0].text:
                return event.content.parts[0].text
        return ""
    
    async def run_layout_analysis():
        runner = Runner(
            agent=layout_analyzer,
            session_service=session_service,
            app_name=APP_NAME,
        )
        
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=layout_session_id,
            new_message=image_message,
        ):
            if event.content and event.content.parts and event.content.parts[0].text:
                return event.content.parts[0].text
        return ""
    
    # 同时运行两个任务
    print("同时进行文本提取和布局分析...")
    text_result, layout_result = await asyncio.gather(
        run_text_extraction(),
        run_layout_analysis()
    )
    
    print(f"\n文本提取结果:")
    print(text_result)
    print(f"\n布局分析结果:")
    print(layout_result)
    
    # 基于并行结果生成最终总结
    summary_message = types.Content(
        role="user",
        parts=[
            types.Part(text=f"""
基于并行分析的结果，生成一份完整的综合分析报告：

【文本内容】
{text_result}

【布局结构】
{layout_result}
""")
        ]
    )
    
    summary_runner = Runner(
        agent=summary_generator,
        session_service=session_service,
        app_name=APP_NAME,
    )
    
    print(f"\n生成最终总结...")
    async for event in summary_runner.run_async(
        user_id=USER_ID,
        session_id=final_session_id,
        new_message=summary_message,
    ):
        if event.content and event.content.parts and event.content.parts[0].text:
            print("=" * 50)
            print("并行分析总结报告")
            print("=" * 50)
            print(event.content.parts[0].text)
            break
    
    return {
        'text_extraction': text_result,
        'layout_analysis': layout_result,
        'mode': 'parallel'
    }

# ================ 主函数 ================

async def main():
    """主函数"""
    print("Google ADK 多智能体图像分析系统")
    print("=" * 60)
    
    # 图片路径
    image_path = "Transformer.png"  # 修改为你的图片路径
    
    if not os.path.exists(image_path):
        print(f"图片文件不存在: {image_path}")
        print("请确保图片文件存在，或修改 image_path 变量")
        return
    
    # 这里硬编码选择模式，你可以改成交互式选择
    mode = "parallel"  # 改为 "parallel" 使用并行模式
    
    try:
        if mode == "step":
            print("\n使用逐步分析模式...")
            results = await analyze_image_step_by_step(image_path)
        else:
            print("\n使用并行分析模式...")
            results = await analyze_image_parallel(image_path)
        
        print(f"\n分析完成！")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 