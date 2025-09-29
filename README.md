# Firecrawl 网页抓取代理

## 项目概述
一个基于Google ADK框架的专业网页内容抓取和分析代理，提供高效的网页数据提取服务。

## 功能特点
- 网页内容抓取：支持单URL和批量URL抓取
- 内容提取：从网页中提取结构化数据
- 内容清理：自动清理和格式化抓取内容
- 数据解析：解析HTML、提取文本和链接
- 格式转换：支持JSON、Markdown等多种输出格式
- 交互式命令行界面：简单易用的操作方式

## 技术栈
- Google ADK框架
- DeepSeek Chat模型
- Firecrawl MCP服务器
- Python 3.8+

## 安装指南

### 前置条件
- Python 3.8+
- pip包管理工具

### 安装步骤
1. 克隆仓库：
```bash
git clone https://github.com/Sogrey/google-adk-agent-layground.git
cd google-adk-agent-layground
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
```bash
cp .env.example .env
# 在.env文件中填写您的API密钥和基础URL
```

## 贡献指南
欢迎提交Pull Request或Issue报告问题。

## 许可证
MIT License