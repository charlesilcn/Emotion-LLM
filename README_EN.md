<div align="center">
  <br>
  <h1>🔍 Social Media Sentiment Analysis System</h1>
  <p>✨ An intelligent sentiment analysis platform with multi-model support</p>
  <div style="display: flex; justify-content: center; gap: 10px; margin: 20px 0;">
    <a href="README.md" style="background-color: #4CAF50; border: none; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 4px;">中文</a>
    <a href="README_EN.md" style="background-color: #2196F3; border: none; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 4px;">English</a>
  </div>
  <br>
</div>

---

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
```
┌─────────────┐      ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   Web Layer │─────▶│Analysis Layer │─────▶│  Data Layer   │─────▶│Visualization │
│  Flask App  │◀────│Sentiment Analyzers│◀────│Data Processing│◀────│Result Display │
└─────────────┘      └───────────────┘      └───────────────┘      └───────────────┘
```
</div>

## 📦 Installation Guide

### Prerequisites

- Python 3.8+
- pip package manager
- Git

### Installation Steps

1. Clone the project repository
   ```bash
   git clone https://github.com/charlesilcn/Emotion-LLM.git
   cd Emotion-LLM
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
Emotion-LLM/
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