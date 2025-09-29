"""
多智能体图像分析系统
"""

import asyncio
import os
import mimetypes
import json
from datetime import datetime
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from dotenv import load_dotenv
load_dotenv(override=True)
from google.adk.agents import SequentialAgent, ParallelAgent # type: ignore

# 从环境变量读取配置
QWEN_MODEL = os.getenv("QWEN_MODEL_NAME")
QWEN_API_KEY = os.getenv("QWEN_API_KEY") 
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL_NAME")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

APP_NAME = "simple_rag_analyzer"
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

# 文档类型定义
SUPPORTED_DOCUMENT_TYPES = {
    "financial_report": "财务报表",
    "invoice": "发票票据", 
    "contract": "合同文档",
    "research_paper": "研究报告",
    "business_chart": "商业图表",
    "receipt": "收据凭证",
    "form": "表单文档",
    "presentation": "演示文档",
    "course_material": "课程资料",
    "technical_doc": "技术文档",
    "other": "其他文档"
}

# 0. 文档分类专家
document_classifier = Agent(
    name="document_classifier",
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY,
    ),
    description="文档类型识别和分类专家",
    instruction=f"""
你是专业的文档分类专家。请分析图像并识别文档类型。

支持的文档类型：
{json.dumps(SUPPORTED_DOCUMENT_TYPES, ensure_ascii=False, indent=2)}

任务：
1. 识别文档的主要类型
2. 评估识别的置信度（0-1）
3. 识别文档的语言和格式
4. 检测文档的质量和清晰度

输出格式（严格JSON）：
{{
    "document_type": "类型代码（如：course_material）",
    "document_name": "类型中文名称（如：课程资料）",
    "confidence": 0.95,
    "language": "中文/英文/混合",
    "format": "表格/文本/图表/混合",
    "quality": "high/medium/low",
    "reasoning": "分类理由和依据"
}}
""",
    output_key="document_classification",
)

# 1. 文本提取专家
text_extractor = Agent(
    name="text_extractor",
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY,
    ),
    description="OCR文本提取专家",
    instruction="""
你是专业的OCR文本提取专家。请仔细提取图片中的所有文字并进行结构化标注。

任务：
1. 提取所有可见文字，包括表格、标题、正文
2. 识别文字的类型和重要性
3. 估算每个文本块的置信度
4. 标注文字的大概位置

输出格式（严格JSON）：
{{
    "extracted_texts": [
        {{
            "text": "具体文字内容",
            "position": "左上角/中央/右下角/顶部/底部等",
            "text_type": "title/subtitle/data/label/content/table/header",
            "confidence": 0.95,
            "importance": "high/medium/low"
        }}
    ],
    "total_confidence": 0.88,
    "extraction_summary": "提取概述",
    "text_count": 15,
    "quality_assessment": "提取质量评估"
}}
""",
    output_key="text_extraction",
)

# 2. 布局分析专家
layout_analyzer = Agent(
    name="layout_analyzer", 
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY,
    ),
    description="文档结构和布局分析专家",
    instruction="""
你是专业的文档结构分析专家。请分析文档的布局和结构。

任务：
1. 识别文档的整体布局结构
2. 检测表格、图表、文本块的位置
3. 分析信息的层次关系
4. 识别关键区域和重点内容

输出格式（严格JSON）：
{{
    "layout_type": "表格型/报告型/图表型/混合型/课程型",
    "structure_elements": [
        {{
            "element_type": "header/table/chart/text_block/footer/title/section",
            "position": "位置描述（如：顶部/左上角/中央等）",
            "size": "large/medium/small",
            "importance": "high/medium/low",
            "description": "元素描述"
        }}
    ],
    "visual_hierarchy": "描述视觉层次和信息组织方式",
    "key_regions": ["重点区域1", "重点区域2"],
    "layout_complexity": "simple/moderate/complex",
    "analysis_confidence": 0.87
}}
""",
    output_key="layout_analysis",
)

# 3. 内容理解专家  
content_analyzer = Agent(
    name="content_analyzer",
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY,
    ),
    description="内容理解和数据提取专家",
    instruction="""
你是专业的内容理解专家。基于前面专家的分析结果，进行深度内容理解和数据提取。

你可以从 session.state['document_classification'] 中获取文档分类信息
你可以从 session.state['text_extraction'] 中获取提取的文字信息
你可以从 session.state['layout_analysis'] 中获取布局分析信息

任务：
1. 判断图像的主要用途和价值
2. 识别关键信息和重要数据
3. 提取结构化的键值对信息
4. 分析数据趋势或关联性（如有）
5. 总结核心要点和发现

输出格式（严格JSON）：
{{
    "content_purpose": "文档的主要用途和目标",
    "key_information": [
        "关键信息点1",
        "关键信息点2",
        "关键信息点3"
    ],
    "key_value_pairs": {{
        "重要字段1": "值1",
        "重要字段2": "值2",
        "日期": "2024-01-15",
        "数量/金额": "具体数值"
    }},
    "data_insights": [
        "数据洞察1：趋势分析",
        "数据洞察2：关联发现"
    ],
    "tables_detected": [
        {{
            "table_description": "表格描述",
            "key_data": "表格中的关键数据",
            "importance": "high/medium/low"
        }}
    ],
    "quality_assessment": {{
        "information_completeness": "high/medium/low",
        "data_reliability": "high/medium/low",
        "analysis_confidence": 0.85
    }},
    "summary": "整体内容总结和核心要点"
}}
""",
    output_key="content_analysis",
)

# 4. RAG数据整理专家（支持溯源）
rag_data_organizer = Agent(
    name="rag_data_organizer",
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY,
    ),
    description="RAG数据整理专家 - 支持精确溯源的多模态数据结构",
    instruction="""
你是高级RAG数据整理专家，专门负责生成支持精确溯源的多模态RAG数据结构。

你可以从 session.state 中获取：
- 'document_classification': 文档分类信息
- 'text_extraction': OCR文本提取结果 (包含extracted_texts数组)
- 'layout_analysis': 布局结构分析结果 (包含structure_elements数组)
- 'content_analysis': 内容理解和数据提取结果

**关键任务**：
1. 从text_extraction.extracted_texts中提取**真实文本内容**生成text_chunks
2. 为每个内容块添加**溯源信息**（位置、类型、置信度）
3. **分类处理**不同类型的内容（文本/表格/图表）
4. 构建**可追溯**的数据结构

输出格式（严格JSON）：
{{
    "document_summary": "一句话概括图像内容和价值",
    "document_classification": {{
        "document_type": "从document_classification中提取的类型代码",
        "document_name": "文档类型中文名称", 
        "confidence": 0.95,
        "reasoning": "分类理由"
    }},
    "extracted_text": "所有提取文本的完整拼接内容",
    "key_data": {{
        "从content_analysis.key_value_pairs中提取真实的键值对": "真实值",
        "重要信息": "实际提取的信息"
    }},
    "text_chunks_with_tracing": [
        {{
            "chunk_id": "chunk_001",
            "content": "从extracted_texts中提取的真实文本内容",
            "source_type": "title/content/table/chart/header/section",
            "position": "从extracted_texts中获取的位置信息",
            "importance": "high/medium/low",
            "confidence": 0.95,
            "char_count": 50,
            "keywords": ["关键词1", "关键词2"]
        }}
    ],
    "table_data": [
        {{
            "table_id": "table_001", 
            "description": "表格描述（从tables_detected提取）",
            "extracted_content": "表格的具体文字内容",
            "position": "表格在文档中的位置",
            "structured_data": {{
                "列名1": "值1",
                "列名2": "值2"
            }},
            "confidence": 0.9,
            "importance": "high/medium/low"
        }}
    ],
    "chart_data": [
        {{
            "chart_id": "chart_001",
            "description": "图表描述",
            "chart_type": "bar/line/pie/flow/other",
            "extracted_text": "图表中的文字内容",
            "position": "图表位置",
            "insights": ["图表洞察1", "图表洞察2"],
            "confidence": 0.85
        }}
    ],
    "layout_structure": {{
        "layout_type": "从layout_analysis提取的布局类型",
        "elements_hierarchy": [
            {{
                "element_type": "header/section/table/chart",
                "position": "位置",
                "content_summary": "内容概要",
                "importance": "high/medium/low"
            }}
        ],
        "reading_order": ["元素1", "元素2", "元素3"]
    }},
    "insights_with_source": [
        {{
            "insight": "从content_analysis.data_insights提取的真实洞察",
            "source_evidence": "支持这个洞察的具体文本证据",
            "confidence": 0.88,
            "insight_type": "trend/comparison/conclusion/recommendation"
        }}
    ],
    "rag_optimized_chunks": [
        "基于extracted_texts重新组织的语义文本块1：标题+相关内容",
        "语义文本块2：特定主题的完整段落",
        "语义文本块3：表格数据+解释文字",
        "语义文本块4：图表信息+分析要点"
    ],
    "metadata": {{
        "analysis_quality": "high/medium/low",
        "total_text_blocks": "从extracted_texts统计的实际数量",
        "total_tables": "实际检测到的表格数量",
        "total_charts": "实际检测到的图表数量", 
        "confidence_score": 0.88,
        "processing_timestamp": "当前时间戳",
        "tracing_enabled": true,
        "recommended_use_cases": ["基于实际内容推荐的应用场景"]
    }}
}}

**重要**：
1. text_chunks_with_tracing必须包含从extracted_texts数组中提取的**真实文本内容**
2. 每个数据块都要有溯源信息（位置、类型、置信度）
3. 分别处理文本、表格、图表三种不同的数据源
4. rag_optimized_chunks要重新组织内容，按语义相关性分组
5. 不要使用模板化内容，要基于实际分析结果生成
"""
)


# 图像内容并行提取（OCR + 布局分析）
image_extractor_agent = ParallelAgent(
    name="image_extractor_workflow", 
    sub_agents=[text_extractor, layout_analyzer],
    description="并行进行OCR文本提取和布局结构分析",
)

# 完整的多智能体分析工作流
# 流程：文档分类 → 并行提取(OCR+布局) → 内容理解 → RAG数据整理
complete_analysis_workflow = SequentialAgent(
    name="complete_analysis_workflow",
    sub_agents=[
        document_classifier,      # 步骤1：文档分类识别
        image_extractor_agent,    # 步骤2：并行提取(OCR + 布局分析)
        content_analyzer,         # 步骤3：内容理解和数据提取
    ],
    description="完整的多智能体文档分析工作流：分类 → 提取 → 理解",
)

# 最终整合智能体
root_agent = SequentialAgent(
    name="document_rag_analyzer",
    sub_agents=[complete_analysis_workflow, rag_data_organizer],
    description="专业文档分析系统：执行完整分析并生成RAG友好的结构化数据",
)

