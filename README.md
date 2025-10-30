<div align="center">
  <br>
  <h1><span class="cn">🔍 社交媒体情感分析系统</span><span class="en">🔍 Social Media Sentiment Analysis System</span></h1>
  <p><span class="cn">✨ 多模型支持的智能情感分析平台</span><span class="en">✨ An intelligent sentiment analysis platform with multi-model support</span></p>
  <div class="language-switcher">
    <button onclick="switchLanguage('cn')" class="cn">中文</button>
    <button onclick="switchLanguage('en')" class="en">English</button>
  </div>
  <br>
</div>

<div class="content cn">

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
  <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: left; display: inline-block;">
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
   git clone https://your-repository-url/social-media-sentiment-analysis.git
   cd social-media-sentiment-analysis
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
      <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">模型名称</th>
      <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">特点</th>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">Hugging Face</td>
      <td style="padding: 12px; border: 1px solid #ddd;">多种免费模型</td>
      <td style="padding: 12px; border: 1px solid #ddd;">本地化运行，无需API密钥</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">OpenAI</td>
      <td style="padding: 12px; border: 1px solid #ddd;">GPT系列模型</td>
      <td style="padding: 12px; border: 1px solid #ddd;">高精度但需要API密钥</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">豆包</td>
      <td style="padding: 12px; border: 1px solid #ddd;">豆包大语言模型</td>
      <td style="padding: 12px; border: 1px solid #ddd;">适合中文场景，需要API密钥</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">DeepSeek</td>
      <td style="padding: 12px; border: 1px solid #ddd;">DeepSeek大语言模型</td>
      <td style="padding: 12px; border: 1px solid #ddd;">专业代码和文本分析</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">本地规则模型</td>
      <td style="padding: 12px; border: 1px solid #ddd;">内置情感词典</td>
      <td style="padding: 12px; border: 1px solid #ddd;">轻量级，离线可用</td>
    </tr>
  </table>
</div>

## 📚 API接口文档

### 获取支持的模型列表

```http
GET /models
```

返回所有支持的模型及其连接状态。

### 情感分析

```http
POST /analyze
Content-Type: application/json

{
  "text": "要分析的文本内容",
  "model_id": "selected_model_id"
}
```

返回文本的情感分析结果。

## 📁 项目结构

```
social-media-sentiment-analysis/
├── app.py                # Flask应用主文件
├── main.py               # 命令行入口
├── config.py             # 配置文件
├── requirements.txt      # 项目依赖
├── src/
│   ├── analysis/         # 分析模块
│   │   ├── __init__.py
│   │   ├── llm_sentiment_analyzer.py    # LLM情感分析器
│   │   └── traditional_sentiment_analyzer.py  # 传统情感分析器
│   ├── data/             # 数据处理模块
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   └── visualization/    # 可视化模块
│       ├── __init__.py
│       └── visualizer.py
├── templates/            # Web模板
│   └── index.html        # 主页面
├── data/
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
└── models/               # 模型存储目录
```

## 🔧 配置说明

主要配置项位于`config.py`文件中：

- **OPENAI_API_KEY**: OpenAI API密钥（可选）
- **HUGGINGFACE_API_KEY**: Hugging Face API密钥（可选）
- **MAX_TOKENS**: 最大生成token数
- **TEMPERATURE**: 生成温度，控制输出随机性
- **BATCH_SIZE**: 批处理大小
- **SENTIMENT_CLASSES**: 情感分类标签
- **EMOTION_CLASSES**: 情绪分类标签
- **CACHE_DIR**: 缓存目录

## 🛠️ 开发指南

### 添加新模型

要添加新的模型支持，请在`src/analysis/llm_sentiment_analyzer.py`中：

1. 在`model_type`参数中添加新模型类型
2. 实现相应的`_model_type_analyze_sentiment`方法
3. 更新`check_connection`方法以支持新模型的连接检查

### 前端开发

前端代码位于`templates/index.html`，使用纯HTML、CSS和JavaScript实现。

## ⚡ 性能优化

- 使用异步连接检测提高响应速度
- 实现模型状态缓存减少重复检查
- 采用延迟初始化策略减少启动时间
- 配置适当的超时设置避免长时间等待

## 📄 许可证

[MIT License](LICENSE)

## 📧 联系方式

如有问题或建议，请联系项目维护者。

## 🎯 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目！

## 💖 鸣谢

感谢所有为项目做出贡献的开发者和用户！

</div>

<div class="content en" style="display: none;">

## 📋 Project Introduction

This is a comprehensive social media sentiment analysis system that supports multiple models for text sentiment and emotion analysis. The system provides a user-friendly web interface, allowing users to easily input text and obtain detailed sentiment analysis results.

<div align="center">
  <img src="https://via.placeholder.com/600x300?text=Sentiment%20Analysis%20System%20Demo" alt="System Demo Interface" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
</div>

## 🌟 Key Features

- **Multi-model Support**: Integrates multiple large language models including Hugging Face, OpenAI, Doubao, and DeepSeek
- **Real-time Sentiment Analysis**: Quickly analyze text sentiment tendency (positive/negative/neutral) and confidence level
- **Emotion Recognition**: Identify multiple emotions contained in text (such as joy, anger, sadness, etc.)
- **Batch Processing**: Support batch analysis of multiple text data
- **Visual Presentation**: Intuitively display sentiment and emotion analysis results
- **Asynchronous Connection Detection**: Efficiently detect model connection status for faster response
- **Degradation Mechanism**: Automatically switch to backup analysis solutions when advanced models are unavailable

<div align="center">
  <table style="border-collapse: collapse; width: 100%; max-width: 800px; margin: 20px 0;">
    <tr style="background-color: #f8f9fa;">
      <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">✨ Feature Highlights</th>
      <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">💡 Technical Features</th>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">Multi-model Integration</td>
      <td style="padding: 12px; border: 1px solid #ddd;">Modular design, easy to extend</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">Real-time Analysis</td>
      <td style="padding: 12px; border: 1px solid #ddd;">Asynchronous processing, fast response</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">Batch Processing</td>
      <td style="padding: 12px; border: 1px solid #ddd;">Efficient parallel computing</td>
    </tr>
  </table>
</div>

## 🏗️ System Architecture

The system adopts a modular design, mainly including the following components:

- **Web Layer**: Flask-based web service providing user interface and API
- **Analysis Layer**: Contains traditional sentiment analyzer and LLM sentiment analyzer
- **Data Layer**: Handles data loading, preprocessing, and result storage
- **Visualization Layer**: Presents analysis results with charts and visualizations

<div align="center">
  <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: left; display: inline-block;">
  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
  │   Web Layer │─────▶│Analysis Layer│─────▶│  Data Layer │─────▶│Visualization│
  │Flask App    │◀────│Sentiment Analyzers│◀────│Data Processing│◀────│Result Display│
  └─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
  </pre>
</div>

## 📦 Installation Guide

### Prerequisites

- Python 3.8+
- pip package manager
- Git

### Installation Steps

1. Clone the project repository
   ```bash
   git clone https://your-repository-url/social-media-sentiment-analysis.git
   cd social-media-sentiment-analysis
   ```

2. Create a virtual environment
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

5. Configure environment variables
   Copy the `.env.example` file to `.env` and configure the relevant parameters as needed
   ```bash
   cp .env.example .env
   ```

## 🚀 Usage Instructions

### Start Web Service

```bash
python app.py
```

After the service starts, visit `http://localhost:5000` to use the web interface.

### Command Line Usage

Use `main.py` for batch analysis:

```bash
python main.py --input data/raw/sample_social_media.csv --output data/processed/analysis_results.csv
```

## 🤖 Model Support

The system supports multiple sentiment analysis models, including:

<div align="center">
  <table style="border-collapse: collapse; width: 100%; max-width: 800px; margin: 20px 0;">
    <tr style="background-color: #f8f9fa;">
      <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">Model Type</th>
      <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">Model Name</th>
      <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">Features</th>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">Hugging Face</td>
      <td style="padding: 12px; border: 1px solid #ddd;">Multiple Free Models</td>
      <td style="padding: 12px; border: 1px solid #ddd;">Local execution, no API key required</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">OpenAI</td>
      <td style="padding: 12px; border: 1px solid #ddd;">GPT Series Models</td>
      <td style="padding: 12px; border: 1px solid #ddd;">High accuracy but requires API key</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">Doubao</td>
      <td style="padding: 12px; border: 1px solid #ddd;">Doubao LLM</td>
      <td style="padding: 12px; border: 1px solid #ddd;">Suitable for Chinese scenes, requires API key</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">DeepSeek</td>
      <td style="padding: 12px; border: 1px solid #ddd;">DeepSeek LLM</td>
      <td style="padding: 12px; border: 1px solid #ddd;">Professional code and text analysis</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;">Local Rule Model</td>
      <td style="padding: 12px; border: 1px solid #ddd;">Built-in Sentiment Dictionary</td>
      <td style="padding: 12px; border: 1px solid #ddd;">Lightweight, offline available</td>
    </tr>
  </table>
</div>

## 📚 API Documentation

### Get Supported Models

```http
GET /models
```

Returns all supported models and their connection status.

### Sentiment Analysis

```http
POST /analyze
Content-Type: application/json

{
  "text": "Text content to analyze",
  "model_id": "selected_model_id"
}
```

Returns the sentiment analysis results of the text.

## 📁 Project Structure

```
social-media-sentiment-analysis/
├── app.py                # Flask application main file
├── main.py               # Command line entry
├── config.py             # Configuration file
├── requirements.txt      # Project dependencies
├── src/
│   ├── analysis/         # Analysis module
│   │   ├── __init__.py
│   │   ├── llm_sentiment_analyzer.py    # LLM sentiment analyzer
│   │   └── traditional_sentiment_analyzer.py  # Traditional sentiment analyzer
│   ├── data/             # Data processing module
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   └── visualization/    # Visualization module
│       ├── __init__.py
│       └── visualizer.py
├── templates/            # Web templates
│   └── index.html        # Main page
├── data/
│   ├── raw/              # Raw data
│   └── processed/        # Processed data
└── models/               # Model storage directory
```

## 🔧 Configuration Instructions

Main configuration items are located in the `config.py` file:

- **OPENAI_API_KEY**: OpenAI API key (optional)
- **HUGGINGFACE_API_KEY**: Hugging Face API key (optional)
- **MAX_TOKENS**: Maximum number of generated tokens
- **TEMPERATURE**: Generation temperature, controls output randomness
- **BATCH_SIZE**: Batch processing size
- **SENTIMENT_CLASSES**: Sentiment classification labels
- **EMOTION_CLASSES**: Emotion classification labels
- **CACHE_DIR**: Cache directory

## 🛠️ Development Guide

### Adding New Models

To add support for a new model, in `src/analysis/llm_sentiment_analyzer.py`:

1. Add the new model type to the `model_type` parameter
2. Implement the corresponding `_model_type_analyze_sentiment` method
3. Update the `check_connection` method to support connection checking for the new model

### Frontend Development

Frontend code is located in `templates/index.html`, implemented using pure HTML, CSS, and JavaScript.

## ⚡ Performance Optimization

- Use asynchronous connection detection to improve response speed
- Implement model state caching to reduce duplicate checks
- Adopt lazy initialization strategy to reduce startup time
- Configure appropriate timeout settings to avoid long waits

## 📄 License

[MIT License](LICENSE)

## 📧 Contact Information

For questions or suggestions, please contact the project maintainers.

## 🎯 Contribution Guide

Feel free to submit Issues and Pull Requests to help improve the project!

## 💖 Acknowledgments

Thanks to all developers and users who have contributed to the project!

</div>

<script>
// 语言切换功能
function switchLanguage(lang) {
  // 隐藏所有内容
  document.querySelectorAll('.content').forEach(content => {
    content.style.display = 'none';
  });
  
  // 显示选中的语言内容
  document.querySelector(`.content.${lang}`).style.display = 'block';
  
  // 保存语言偏好到localStorage
  localStorage.setItem('preferredLanguage', lang);
}

// 页面加载时恢复上次的语言选择
window.onload = function() {
  const savedLang = localStorage.getItem('preferredLanguage') || 'cn';
  switchLanguage(savedLang);
};
</script>

<style>
.language-switcher {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin: 20px 0;
}

.language-switcher button {
  background-color: #4CAF50;
  border: none;
  color: white;
  padding: 10px 20px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  margin: 4px 2px;
  cursor: pointer;
  border-radius: 4px;
  transition: background-color 0.3s;
}

.language-switcher button:hover {
  background-color: #45a049;
}

.content {
  margin: 20px 0;
}

/* 装饰性样式 */
h1 {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 20px;
}

@media (max-width: 768px) {
  .language-switcher {
    flex-direction: column;
    align-items: center;
  }
  
  .language-switcher button {
    width: 100%;
    max-width: 200px;
  }
}

/* 为代码块添加样式 */
pre {
  background-color: #f5f5f5;
  border-radius: 4px;
  padding: 16px;
  overflow: auto;
  font-family: 'Courier New', Courier, monospace;
}

code {
  background-color: #f5f5f5;
  padding: 2px 4px;
  border-radius: 3px;
  font-family: 'Courier New', Courier, monospace;
}
</style>