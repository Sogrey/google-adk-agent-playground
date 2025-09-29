"""
基于Agent架构的溯源多模态RAG系统
使用SequentialAgent组织RAG流程：数据加载 → 向量化 → 检索 → 回答生成
"""

import asyncio
import os
import json
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import faiss
import dashscope
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from dotenv import load_dotenv
load_dotenv(override=True)

# 配置
QWEN_MODEL = os.getenv("QWEN_MODEL_NAME")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL_NAME")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

APP_NAME = "rag_agents_system"
USER_ID = "user"

# 全局FAISS存储
_FAISS_STORAGE = {
    "text_index": None,
    "insight_index": None,
    "text_chunks": None,
    "insights": None,
    "rag_data": None
}

# ================ 工具函数定义 ================

def load_rag_data_tool(json_data: str) -> Dict[str, Any]:
    """RAG数据加载工具 - 处理JSON字符串"""
    print(f"[工具] 加载RAG数据，数据长度: {len(json_data)} 字符")
    
    try:
        # 解析JSON字符串
        data = json.loads(json_data)
        
        # 统计数据量
        text_count = len(data.get("text_chunks_with_tracing", []))
        table_count = len(data.get("table_data", []))
        chart_count = len(data.get("chart_data", []))
        insights_count = len(data.get("insights_with_source", []))
        
        load_result = {
            "load_status": "success",
            "data_summary": {
                "text_chunks_count": text_count,
                "table_count": table_count,
                "chart_count": chart_count,
                "insights_count": insights_count
            },
            "document_info": {
                "document_type": data.get("document_classification", {}).get("document_name", "未知"),
                "summary": data.get("document_summary", "无概要"),
                "confidence": data.get("document_classification", {}).get("confidence", 0.0)
            },
            "rag_data": data,  # 完整数据存储
            "ready_for_vectorization": True,
            "message": f"数据加载成功: {text_count}个文本块, {table_count}个表格, {insights_count}个洞察"
        }
        
        print(f"[工具] 数据加载成功: {text_count}个文本块")
        return load_result
        
    except Exception as e:
        print(f"[工具] 数据加载失败: {str(e)}")
        return {
            "load_status": "failed",
            "error": str(e),
            "ready_for_vectorization": False
        }

def vectorize_content_tool(rag_data: Dict[str, Any]) -> Dict[str, Any]:
    """内容向量化工具"""
    global _FAISS_STORAGE
    print(f"[工具] 开始FAISS向量化处理...")
    
    try:
        vectorization_result = {
            "vectorization_status": "success",
            "vectorized_content": {},
            "indexing_complete": True,
            "ready_for_search": True
        }
        
        # 存储RAG数据到全局变量
        _FAISS_STORAGE["rag_data"] = rag_data
        
        # 1. 向量化text_chunks_with_tracing并创建FAISS索引
        if "text_chunks_with_tracing" in rag_data:
            chunks = rag_data["text_chunks_with_tracing"]
            if chunks:
                print(f"[FAISS] 处理 {len(chunks)} 个文本块...")
                text_contents = [chunk["content"] for chunk in chunks]
                text_embeddings = get_bailian_embedding(text_contents)
                
                # 确保数据类型为float32（FAISS要求）
                text_embeddings = text_embeddings.astype(np.float32)
                
                # 验证向量数据
                if np.any(np.isnan(text_embeddings)) or np.any(np.isinf(text_embeddings)):
                    raise ValueError("向量包含NaN或无穷大值")
                
                # 创建FAISS索引 (使用内积计算，适合归一化向量)
                dimension = text_embeddings.shape[1]
                text_index = faiss.IndexFlatIP(dimension)
                
                # 向量归一化（用于余弦相似度）
                faiss.normalize_L2(text_embeddings)
                
                # 添加向量到索引
                text_index.add(text_embeddings)
                
                # 保存到全局存储（避免序列化问题）
                _FAISS_STORAGE["text_index"] = text_index
                _FAISS_STORAGE["text_chunks"] = chunks
                vectorization_result["vectorized_content"]["text_chunks_vectorized"] = len(chunks)
                
                print(f"[FAISS] 文本索引创建完成: {len(chunks)}个向量, 维度: {dimension}")
        
        # 2. 向量化insights_with_source并创建FAISS索引
        if "insights_with_source" in rag_data:
            insights = rag_data["insights_with_source"]
            if insights:
                print(f"[FAISS] 处理 {len(insights)} 个洞察...")
                insight_texts = [insight["insight"] for insight in insights]
                insight_embeddings = get_bailian_embedding(insight_texts)
                
                # 确保数据类型为float32（FAISS要求）
                insight_embeddings = insight_embeddings.astype(np.float32)
                
                # 验证向量数据
                if np.any(np.isnan(insight_embeddings)) or np.any(np.isinf(insight_embeddings)):
                    raise ValueError("洞察向量包含NaN或无穷大值")
                
                # 创建FAISS索引
                dimension = insight_embeddings.shape[1]
                insight_index = faiss.IndexFlatIP(dimension)
                
                # 向量归一化
                faiss.normalize_L2(insight_embeddings)
                
                # 添加向量到索引
                insight_index.add(insight_embeddings)
                
                # 保存到全局存储（避免序列化问题）
                _FAISS_STORAGE["insight_index"] = insight_index
                _FAISS_STORAGE["insights"] = insights
                vectorization_result["vectorized_content"]["insights_vectorized"] = len(insights)
                
                print(f"[FAISS] 洞察索引创建完成: {len(insights)}个向量, 维度: {dimension}")
        
        vectorization_result["vectorized_content"]["embedding_model"] = "bailian-text-embedding-v1"
        vectorization_result["vectorized_content"]["vector_dimension"] = dimension if 'dimension' in locals() else 1536
        vectorization_result["vectorized_content"]["index_type"] = "FAISS_IndexFlatIP"
        vectorization_result["vectorized_content"]["storage_method"] = "global_storage"
        vectorization_result["message"] = "FAISS向量化完成，索引已存储到全局空间"
        
        indices_count = sum(1 for k in ["text_index", "insight_index"] if _FAISS_STORAGE[k] is not None)
        print(f"[FAISS] 向量化完成，创建了 {indices_count} 个索引")
        return vectorization_result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[工具] FAISS向量化失败: {str(e)}")
        print(f"[调试] 详细错误信息:\n{error_details}")
        
        return {
            "vectorization_status": "failed",
            "error": str(e),
            "error_details": error_details,
            "ready_for_search": False
        }

def search_with_tracing_tool(query: str) -> Dict[str, Any]:
    """带溯源的FAISS检索工具（使用全局存储）"""
    global _FAISS_STORAGE
    print(f"🔍 [FAISS] 执行溯源检索: {query}")
    print(f"提取到的用户问题：{query}")
    try:
        results = {
            "text_matches": [],
            "table_matches": [], 
            "insight_matches": [],
            "query_analysis": {
                "user_intent": f"用户查询关于: {query}",
                "search_strategy": "FAISS向量检索 + 关键词匹配",
                "confidence": 0.9
            },
            "search_engine": "FAISS_GLOBAL"
        }
        
        # 1. FAISS文本块检索
        if (_FAISS_STORAGE["text_index"] is not None and 
            _FAISS_STORAGE["text_chunks"] is not None):
            
            text_index = _FAISS_STORAGE["text_index"]
            chunks = _FAISS_STORAGE["text_chunks"]
            
            print(f"🔤 [FAISS] 检索文本块，索引大小: {text_index.ntotal}")
            
            # 查询向量化并归一化
            query_embedding = get_bailian_embedding([query])
            query_embedding = query_embedding.astype(np.float32)  # FAISS要求float32
            faiss.normalize_L2(query_embedding)
            
            # FAISS检索
            top_k = min(5, text_index.ntotal)  # 检索top-k个结果
            similarities, indices = text_index.search(query_embedding, top_k)
            
            # 处理检索结果
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx != -1 and similarity > 0.3:  # FAISS返回的是余弦相似度
                    chunk = chunks[idx]
                    results["text_matches"].append({
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        "similarity": float(similarity),
                        "source_type": chunk["source_type"],
                        "position": chunk["position"],
                        "confidence": chunk["confidence"],
                        "faiss_rank": i + 1
                    })
            
            print(f"✅ [FAISS] 文本检索完成: {len(results['text_matches'])}个匹配")
        
        # 2. 传统关键词检索表格数据
        if (_FAISS_STORAGE["rag_data"] is not None and 
            "table_data" in _FAISS_STORAGE["rag_data"]):
            
            rag_data = _FAISS_STORAGE["rag_data"]
            for table in rag_data["table_data"]:
                if any(keyword in table["description"].lower() or keyword in table["extracted_content"].lower() 
                       for keyword in query.lower().split()):
                    results["table_matches"].append({
                        "table_id": table["table_id"],
                        "description": table["description"],
                        "relevance": 0.8,
                        "match_type": "keyword"
                    })
            
            print(f"📊 [关键词] 表格检索完成: {len(results['table_matches'])}个匹配")
        
        # 3. FAISS洞察检索
        if (_FAISS_STORAGE["insight_index"] is not None and 
            _FAISS_STORAGE["insights"] is not None):
            
            insight_index = _FAISS_STORAGE["insight_index"]
            insights = _FAISS_STORAGE["insights"]
            
            print(f"💡 [FAISS] 检索洞察，索引大小: {insight_index.ntotal}")
            
            # 查询向量化并归一化
            query_embedding = get_bailian_embedding([query])
            query_embedding = query_embedding.astype(np.float32)  # FAISS要求float32
            faiss.normalize_L2(query_embedding)
            
            # FAISS检索
            top_k = min(3, insight_index.ntotal)
            similarities, indices = insight_index.search(query_embedding, top_k)
            
            # 处理检索结果
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx != -1 and similarity > 0.3:
                    insight = insights[idx]
                    results["insight_matches"].append({
                        "insight": insight["insight"],
                        "source_evidence": insight["source_evidence"],
                        "confidence": insight["confidence"],
                        "similarity": float(similarity),
                        "insight_type": insight.get("insight_type", "general"),
                        "faiss_rank": i + 1
                    })
            
            print(f"✅ [FAISS] 洞察检索完成: {len(results['insight_matches'])}个匹配")
        
        # 统计结果
        results["total_matches"] = len(results["text_matches"]) + len(results["table_matches"]) + len(results["insight_matches"])
        results["search_quality"] = "high" if results["total_matches"] >= 3 else "medium" if results["total_matches"] >= 1 else "low"
        results["ready_for_answer"] = results["total_matches"] > 0
        
        print(f"🎯 [FAISS] 检索完成: {results['total_matches']}个匹配结果")
        print(f"    📝 文本: {len(results['text_matches'])} | 📊 表格: {len(results['table_matches'])} | 💡 洞察: {len(results['insight_matches'])}")
        
        return results
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ [FAISS] 检索失败: {str(e)}")
        print(f"🔍 [调试] 详细错误信息:\n{error_details}")
        
        return {
            "error": str(e),
            "error_details": error_details,
            "total_matches": 0,
            "ready_for_answer": False,
            "search_engine": "FAISS_ERROR"
        }

# ================ RAG智能体定义 ================

# 1. 数据加载专家
data_loader_agent = Agent(
    name="data_loader_agent",
    model=LiteLlm(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
    ),
    description="RAG数据加载和预处理专家",
    instruction="""
你是RAG数据加载专家。从用户消息中提取JSON数据字符串，使用load_rag_data_tool工具解析数据。

任务步骤：
1. 从用户消息中识别和提取JSON数据字符串
2. 调用load_rag_data_tool(json_data_string)解析JSON数据
3. 总结加载结果

输出加载状态报告：
- 数据加载状态（成功/失败）
- 文档类型和概要  
- 各类数据统计（文本块、表格、洞察数量）
- 是否准备好进行向量化处理

注意：传入工具的必须是完整的JSON字符串，不是文件路径。
数据加载完成后将存储在session.state中供后续使用。
""",
    tools=[load_rag_data_tool],
    output_key="data_loading_result"
)

# 2. 向量化专家
vectorization_agent = Agent(
    name="vectorization_agent", 
    model=LiteLlm(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
    ),
    description="FAISS向量化专家",
    instruction="""
你是FAISS向量化专家。从session.state['data_loading_result']获取已加载的RAG数据，使用vectorize_content_tool工具创建FAISS索引。

任务步骤：
1. 从session.state中获取data_loading_result
2. 提取其中的rag_data
3. 调用vectorize_content_tool(rag_data)进行FAISS向量化
4. 总结FAISS索引创建结果

FAISS向量化特点：
- 使用百炼Embedding API生成向量（支持分批处理，每批最多25个）
- 创建IndexFlatIP索引（适合余弦相似度）
- 向量L2归一化处理
- **重要**: 索引存储在全局空间，避免ADK序列化问题

调用工具后，输出向量化状态报告：
- FAISS索引创建状态
- 处理的内容统计（文本块数量、洞察数量）
- 向量维度和索引类型信息
- 全局存储状态
- 是否准备好进行FAISS检索

FAISS索引已安全存储到全局空间，供后续检索使用。
""",
    tools=[vectorize_content_tool],
    output_key="vectorization_result"
)

# 3. 检索专家
retrieval_agent = Agent(
    name="retrieval_agent",
    model=LiteLlm(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
    ),
    description="FAISS多模态检索专家 - 支持精确溯源",
    instruction="""
你是FAISS多模态检索专家，使用search_with_tracing_tool执行高效的向量检索。

**重要**: FAISS索引已存储在全局空间，工具函数会自动访问。

任务步骤：
1. 从用户消息中提取查询问题
2. 调用search_with_tracing_tool(query)执行FAISS向量检索
3. 分析检索结果质量

FAISS检索特点：
- 文本块：使用FAISS IndexFlatIP进行余弦相似度检索
- 表格数据：使用传统关键词匹配
- 洞察信息：使用FAISS IndexFlatIP进行语义检索
- 所有结果都包含faiss_rank排序信息
- 索引存储在全局空间，避免序列化问题

调用工具后，输出检索状态报告：
- 查询意图分析
- FAISS检索结果统计（文本/表格/洞察匹配数量）
- 检索质量评估和排序信息
- 是否准备好生成回答

确保检索结果完整并包含溯源信息和FAISS排序。
""",
    tools=[search_with_tracing_tool],
    output_key="retrieval_result"
)

# 4. 回答生成专家
answer_generation_agent = Agent(
    name="answer_generation_agent",
    model=LiteLlm(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
    ),
    description="FAISS溯源回答生成专家",
    instruction="""
你是FAISS溯源回答生成专家，基于session.state中的FAISS检索结果生成带有精确来源信息的回答。

从session.state获取：
- 'data_loading_result': 文档信息
- 'vectorization_result': FAISS向量化状态（全局存储）
- 'retrieval_result': FAISS检索结果（包含text_matches, table_matches, insight_matches）

FAISS检索结果特点：
- text_matches: 包含faiss_rank排序，similarity为余弦相似度
- table_matches: 关键词匹配，包含match_type
- insight_matches: 包含faiss_rank排序和insight_type
- search_engine: "FAISS_GLOBAL" 表示使用全局存储

任务：
1. 分析FAISS检索到的各类匹配结果
2. 基于匹配内容和FAISS排序生成准确回答
3. 提供完整的溯源信息（chunk_id、位置、FAISS排序、相似度等）
4. 评估回答的可信度

输出格式：
**答案概要**: [一句话回答用户问题]

**详细解答**: 
[基于FAISS检索到的信息进行详细回答，引用具体内容和排序]

**信息来源**:
- 来源1: [chunk_id] - [source_type] - [position] - [内容概要] (FAISS相似度: 0.95, 排序: #1)
- 来源2: [table_id] - [表格描述] (关键词匹配)
- 来源3: [洞察内容] (FAISS相似度: 0.85, 排序: #2, 类型: trend)

**相关数据**:
[如果有结构化数据，列出关键信息]

**补充建议**:
[基于内容提供实用建议]

**回答可信度**: [high/medium/low] (基于FAISS排序、相似度和来源数量综合评估)

注意：确保每个信息都能精确溯源到原始位置，包含FAISS排序信息和相似度分数。
""",
    output_key="final_answer"
)


# ================ 基础工具函数 ================

def get_bailian_embedding(texts: List[str]) -> np.ndarray:
    """调用Embedding API（被工具函数调用）- 支持分批处理"""
    try:
        if not texts:
            raise ValueError("文本列表不能为空")
        
        # 过滤空文本
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            raise ValueError("没有有效的文本内容")
        
        print(f"[向量化] 处理 {len(valid_texts)} 个文本...")
        
        # 每批最多25个文本
        BATCH_SIZE = 25
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(valid_texts), BATCH_SIZE):
            batch_texts = valid_texts[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(valid_texts) + BATCH_SIZE - 1) // BATCH_SIZE
            
            print(f"[批次 {batch_num}/{total_batches}] 处理 {len(batch_texts)} 个文本...")
            
            response = dashscope.TextEmbedding.call(
                model=dashscope.TextEmbedding.Models.text_embedding_v1,
                input=batch_texts
            )
            
            if response.status_code == 200:
                batch_embeddings = []
                for output in response.output['embeddings']:
                    batch_embeddings.append(output['embedding'])
                
                all_embeddings.extend(batch_embeddings)
                print(f"[批次 {batch_num}] 成功获取 {len(batch_embeddings)} 个向量")
                
            else:
                raise Exception(f"向量化错误 (批次 {batch_num}): {response.message}")
            
            # 避免API调用过快
            if i + BATCH_SIZE < len(valid_texts):
                import time
                time.sleep(0.1)  # 100ms延迟
        
        # 合并所有批次的向量
        embeddings_array = np.array(all_embeddings, dtype=np.float64)
        
        # 验证向量
        if embeddings_array.size == 0:
            raise ValueError("获取到空的向量")
        
        print(f"[百炼向量化] 成功获取向量，shape: {embeddings_array.shape}")
        print(f"[统计] 总计 {len(all_embeddings)} 个向量，维度: {embeddings_array.shape[1]}")
        
        return embeddings_array
        
    except Exception as e:
        print(f"[向量化] 失败: {str(e)}")
        raise e


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
upload_and_rag_index = SequentialAgent(
    name="document_rag_analyzer",
    sub_agents=[complete_analysis_workflow, rag_data_organizer, data_loader_agent, vectorization_agent ],
    description="专业文档分析系统：执行完整分析并生成RAG友好的结构化数据",
)

root_agent = SequentialAgent(
    name = 'qa_agent',
    sub_agents=[upload_and_rag_index, retrieval_agent, answer_generation_agent],
    description="QA智能体：上传图像并进行RAG索引，然后进行检索和回答",
)