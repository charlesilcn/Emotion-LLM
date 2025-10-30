import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SRC_DIR = os.path.join(BASE_DIR, 'src')

# 确保必要的目录存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# LLM配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# 模型参数
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 2000))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.3))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 100))

# 情感分析配置
SENTIMENT_CLASSES = {
    'positive': '积极',
    'negative': '消极',
    'neutral': '中性'
}

EMOTION_CLASSES = {
    'joy': '喜悦',
    'anger': '愤怒',
    'sadness': '悲伤',
    'fear': '恐惧',
    'surprise': '惊讶',
    'disgust': '厌恶',
    'trust': '信任',
    'anticipation': '期待'
}

# 文本处理配置
MAX_TEXT_LENGTH = 500
MIN_TEXT_LENGTH = 2

# 日志配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.path.join(BASE_DIR, 'app.log')

# 缓存配置
CACHE_DIR = os.path.join(BASE_DIR, '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# 示例数据配置
SAMPLE_DATA_URL = {
    'twitter': 'https://example.com/twitter_sample.csv',
    'reddit': 'https://example.com/reddit_sample.csv'
}