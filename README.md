<!-- 添加装饰性样式 -->
<style>
/* 基本样式 */
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  line-height: 1.6;
  color: #333;
  background-color: #f9f9f9;
}

/* 标题样式 */
h1 {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 20px;
  font-size: 2.5em;
  font-weight: bold;
}

h2 {
  color: #2c3e50;
  margin-top: 1.5em;
  margin-bottom: 0.5em;
  border-bottom: 2px solid #3498db;
  padding-bottom: 0.3em;
}

/* 代码块样式 */
pre {
  background-color: #f5f5f5;
  border-radius: 4px;
  padding: 16px;
  overflow: auto;
  font-family: 'Courier New', Courier, monospace;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

code {
  background-color: #f5f5f5;
  padding: 2px 4px;
  border-radius: 3px;
  font-family: 'Courier New', Courier, monospace;
}

/* 表格样式 */
table {
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* 响应式设计 */
@media (max-width: 768px) {
  h1 {
    font-size: 2em;
  }
  
  pre {
    padding: 12px;
  }
}
</style>

<div align="center">
  <br>
  <h1>🔍 社交媒体情感分析系统</h1>
  <p>✨ 多模型支持的智能情感分析平台</p>
  <div style="display: flex; justify-content: center; gap: 10px; margin: 20px 0;">
    <a href="README.md" style="background-color: #4CAF50; border: none; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 4px;">中文</a>
    <a href="README_EN.md" style="background-color: #2196F3; border: none; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 4px;">English</a>
  </div>
  <br>
</div>

---

## 📋 项目简介

这是一个功能全面的社交媒体情感分析系统，支持多种模型进行文本情感和情绪分析。系统提供了友好的Web界面，使用户能够轻松输入文本并获取详细的情感分析结果。

<div align="center">
  <img src="https://via.placeholder.com/600x300?text=情感分析系统演示界面" alt="系统演示界面" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
</div>

## 🌟 主要功能

- **多模型支持**：集成了Hugging Face、OpenAI、豆包和DeepSeek等多种大语言模型
- **实时情感分析**：快速分析文本的情感倾向（积极/消极/中性）及置信度
- **情绪识别**：识别文本中包含的多种情绪（如喜悦、愤怒、悲伤等）
- **批量处理**：支持批量分析多条文本数据
- **可视化展示**：直观展示情感和情绪分析结果
- **异步连接检测**：高效检测模型连接状态，提供更快的响应速度
- **降级机制**：当高级模型不可用时，自动切换到备用分析方案

<div align="center">
  <table style="border-collapse: collapse; width: 100%; max-width: 800px; margin: 20px 0;">
    <tr style="background-color: #f8f9fa;">
      <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">✨ 功能亮点</th>
      <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">💡 技术特点</th>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">多模型集成</td>
      <td style="padding: 12px; border: 1px solid #ddd;">模块化设计，易于扩展</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">实时分析</td>
      <td style="padding: 12px; border: 1px solid #ddd;">异步处理，快速响应</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">批量处理</td>
      <td style="padding: 12px; border: 1px solid #ddd;">高效并行计算</td>
    </tr>
  </table>
</div>

## 🏗️ 系统架构

系统采用模块化设计，主要包含以下组件：

- **Web层**：基于Flask的Web服务，提供用户界面和API接口
- **分析层**：包含传统情感分析器和LLM情感分析器
- **数据层**：处理数据加载、预处理和结果存储
- **可视化层**：展示分析结果的图表和可视化内容

<div align="center">
  <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: left; display: inline-block; white-space: pre;">
  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
  │   Web层     │─────▶│   分析层    │─────▶│   数据层    │─────▶│ 可视化层    │
  │  Flask应用  │◀────│ 情感分析器  │◀────│ 数据处理    │◀────│ 结果展示    │
  └─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
  </pre>
</div>

## 📦 安装指南

### 前置要求

- Python 3.8+
- pip包管理器
- Git

### 安装步骤

1. 克隆项目仓库
   ```bash
   git clone https://github.com/charlesilcn/Emotion-LLM.git
   cd Emotion-LLM
   ```

2. 创建虚拟环境
   ```bash
   python -m venv .venv
   ```

3. 激活虚拟环境
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

5. 配置环境变量
   复制`.env.example`文件为`.env`，并根据需要配置相关参数
   ```bash
   cp .env.example .env
   ```

## 🚀 使用说明

### 启动Web服务

```bash
python app.py
```

服务启动后，访问`http://localhost:5000`即可使用Web界面。

### 命令行使用

使用`main.py`进行批量分析：

```bash
python main.py --input data/raw/sample_social_media.csv --output data/processed/analysis_results.csv
```

## 🤖 模型支持

系统支持多种情感分析模型，包括：

<div align="center">
  <table style="border-collapse: collapse; width: 100%; max-width: 800px; margin: 20px 0;">
    <tr style="background-color: #f8f9fa;">
      <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">模型类型</th>
      <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">支持的模型</th>
      <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">特点</th>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">Hugging Face</td>
      <td style="padding: 12px; border: 1px solid #ddd;">bert-base-chinese, roberta-base-chinese</td>
      <td style="padding: 12px; border: 1px solid #ddd;">开源免费，适合中文文本</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">OpenAI</td>
      <td style="padding: 12px; border: 1px solid #ddd;">GPT-3.5, GPT-4</td>
      <td style="padding: 12px; border: 1px solid #ddd;">分析精度高，需API密钥</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">豆包</td>
      <td style="padding: 12px; border: 1px solid #ddd;">豆包大模型</td>
      <td style="padding: 12px; border: 1px solid #ddd;">对中文语境理解好</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">DeepSeek</td>
      <td style="padding: 12px; border: 1px solid #ddd;">DeepSeek-R1</td>
      <td style="padding: 12px; border: 1px solid #ddd;">开源模型，性能优异</td>
    </tr>
  </table>
</div>

## 📚 API接口文档

系统提供了RESTful API接口，方便其他系统集成：

### 获取支持的模型列表

```bash
GET /api/models
```

响应示例：
```json
{
  "models": ["bert-base-chinese", "roberta-base-chinese", "gpt-3.5", "gpt-4", "doubao", "deepseek-r1"]
}
```

### 情感分析接口

```bash
POST /api/analyze
Content-Type: application/json

{
  "text": "这是一段需要分析情感的文本",
  "model": "bert-base-chinese"
}
```

响应示例：
```json
{
  "text": "这是一段需要分析情感的文本",
  "sentiment": "positive",
  "confidence": 0.92,
  "emotions": [
    {"emotion": "joy", "score": 0.85},
    {"emotion": "trust", "score": 0.32}
  ],
  "model": "bert-base-chinese",
  "timestamp": "2023-11-15T10:30:45Z"
}
```

## 📁 项目结构

项目采用清晰的模块化结构，便于维护和扩展：

```
├── app.py                 # Web服务入口
├── main.py                # 命令行工具入口
├── config.py              # 配置文件
├── requirements.txt       # 依赖列表
├── src/
│   ├── analysis/          # 情感分析模块
│   │   ├── __init__.py
│   │   ├── llm_sentiment_analyzer.py    # LLM情感分析器
│   │   └── traditional_sentiment_analyzer.py  # 传统情感分析器
│   ├── data/              # 数据处理模块
│   │   ├── __init__.py
│   │   ├── data_loader.py        # 数据加载器
│   │   └── preprocessor.py       # 数据预处理器
│   └── visualization/     # 可视化模块
│       ├── __init__.py
│       └── visualizer.py         # 可视化工具
├── data/                  # 数据目录
│   ├── raw/               # 原始数据
│   └── processed/         # 处理后的数据
├── models/                # 模型文件
├── templates/             # Web模板
└── notebooks/             # 分析笔记本
```

## ⚙️ 配置说明

主要配置项在`config.py`和`.env`文件中：

### config.py 主要配置

- `MODEL_CONFIG`: 模型配置字典
- `API_KEYS`: API密钥配置（从环境变量读取）
- `DEFAULT_MODEL`: 默认使用的模型
- `MAX_BATCH_SIZE`: 批处理最大数量
- `ANALYSIS_TIMEOUT`: 分析超时时间

### .env 环境变量配置

复制`.env.example`为`.env`，并设置以下环境变量：

```dotenv
# 数据库连接信息（如需）
DATABASE_URL="sqlite:///emotion_analysis.db"

# API密钥配置
OPENAI_API_KEY="your_openai_api_key"
DOUBAO_API_KEY="your_doubao_api_key"
DEEPSEEK_API_KEY="your_deepseek_api_key"

# 模型配置
DEFAULT_MODEL="bert-base-chinese"
MAX_CONCURRENT_REQUESTS=5
```

## 🛠️ 开发指南

### 添加新模型

1. 在`src/analysis/`目录下创建新的分析器类
2. 实现`analyze`和`batch_analyze`方法
3. 在`config.py`中配置新模型
4. 更新API接口以支持新模型

### 前端开发

前端使用Flask模板引擎，位于`templates/`目录下。修改`index.html`以更新Web界面。

## ⚡ 性能优化

- 使用异步请求处理多个模型
- 实现请求缓存，避免重复分析
- 批处理大量文本以提高效率
- 模型预加载减少首次分析延迟

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 📧 联系方式

- 项目维护者：charlesilcn
- GitHub: https://github.com/charlesilcn/Emotion-LLM

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启Pull Request

## 🙏 鸣谢

Thanks to all developers and users who have contributed to the project!