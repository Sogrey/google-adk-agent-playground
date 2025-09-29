# Google ADK Agent Playground 项目

## 项目概述
这是一个基于Google Agent Development Kit (ADK)的多功能代理开发环境，集成了网页抓取、OCR处理和多代理协作等功能。

## 主要组件

### 1. 网页抓取代理 (02-streamablehttp_agent)
- 基于Firecrawl MCP服务器的专业网页内容抓取
- 支持深度页面分析和结构化数据提取
- 交互式命令行界面

### 2. 多代理OCR系统 (multi_agent_ocr.py)
- 多代理协作的OCR文本识别系统
- 支持图像预处理和文本后处理
- 可扩展的代理工作流

### 3. 事件流处理 (01-stream_event)
- 实时数据流处理框架
- 支持事件驱动架构
- 可扩展的处理器模块

## 技术栈
- **核心框架**: Google ADK
- **AI模型**: DeepSeek Chat, OCR模型
- **服务集成**: Firecrawl MCP
- **开发语言**: Python 3.8+
- **工具链**: Git, Pipenv

## 快速开始

### 环境配置
```bash
git clone git@github.com:Sogrey/google-adk-agent-playground.git
cd google-adk-agent-playground
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 运行网页抓取代理
```bash
python 02-streamablehttp_agent/agent.py
```

### 运行OCR系统
```bash
python multi_agent_ocr.py
```

## 项目结构
```
google-adk-agent-playground/
├── 01-stream_event/        # 事件流处理模块
├── 02-streamablehttp_agent/ # 网页抓取代理
│   └── agent.py            # 主代理程序
├── multi_agent_ocr.py       # OCR多代理系统
├── README.md               # 项目文档
├── .gitignore              # Git忽略规则
└── requirements.txt        # 依赖列表
```

## 贡献指南
1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -m 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建Pull Request

## 许可证
MIT License © 2025 Sogrey