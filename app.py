from flask import Flask, render_template, request, jsonify
import os
import sys
import logging
from typing import Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from src.data.preprocessor import ChineseTextPreprocessor, TextPreprocessor
# 尝试导入LLMSentimentAnalyzer，如果失败则设置为None
try:
    from src.analysis.llm_sentiment_analyzer import LLMSentimentAnalyzer
    logger.info("成功导入LLMSentimentAnalyzer")
except ImportError as e:
    logger.warning(f"导入LLMSentimentAnalyzer失败: {e}，将使用基本功能")
    LLMSentimentAnalyzer = None

# 尝试导入TraditionalSentimentAnalyzer，如果失败则创建一个简单的备用
try:
    from src.analysis.traditional_sentiment_analyzer import TraditionalSentimentAnalyzer
    logger.info("成功导入TraditionalSentimentAnalyzer")
except ImportError as e:
    logger.warning(f"导入TraditionalSentimentAnalyzer失败: {e}，将使用简单备用分析器")
    # 创建一个增强版备用情感分析器类
    class SimpleSentimentAnalyzer:
        def __init__(self):
            # 扩展的情感词典
            self.pos_words = set([
                # 中文积极词
                '好', '棒', '优秀', '喜欢', '开心', '快乐', '满意', '赞', '推荐', '支持',
                '精彩', '棒极了', '完美', '出色', '很好', '不错', '良好', '优秀', '很棒', '厉害',
                '感谢', '感激', '棒', '赞', '爱', '幸福', '愉悦', '舒适', '方便', '高效',
                '物超所值', '惊喜', '满意', '值得', '好评', '完美', '精彩', '出色', '优秀', '成功',
                '喜欢', '热爱', '欣赏', '敬佩', '认同', '肯定', '鼓励', '支持', '赞同', '认可',
                # 英文积极词
                'good', 'great', 'excellent', 'like', 'love', 'happy', 'satisfied', 'awesome',
                'wonderful', 'fantastic', 'amazing', 'perfect', 'outstanding', 'brilliant', 'superb',
                'excellent', 'terrific', 'positive', 'pleased', 'delight', 'joy', 'grateful', 'thanks',
                'appreciate', 'admire', 'approve', 'support', 'encourage', 'success', 'achieve', 'accomplish'
            ])
            
            self.neg_words = set([
                # 中文消极词
                '坏', '差', '糟糕', '讨厌', '生气', '难过', '不满意', '坑', '失望', '反对',
                '垃圾', '差', '不好', '糟糕', '恶心', '讨厌', '生气', '难过', '伤心', '痛苦',
                '失望', '郁闷', '烦躁', '愤怒', '烦', '闷', '累', '困', '无聊', '乏味',
                '贵', '贵了', '不值', '性价比低', '差评', '退货', '退款', '投诉', '举报', '问题',
                '错误', '失败', '漏洞', '缺陷', '缺点', '不足', '缺点', '弱点', '薄弱', '欠缺',
                # 英文消极词
                'bad', 'poor', 'terrible', 'hate', 'angry', 'sad', 'disappointed', 'worst',
                'awful', 'horrible', 'disgusting', 'pathetic', 'terrible', 'negative', 'upset', 'frustrated',
                'angry', 'mad', 'hate', 'dislike', 'reject', 'oppose', 'complain', 'refund', 'return',
                'fail', 'error', 'problem', 'defect', 'fault', 'weakness', 'shortcoming', 'disadvantage'
            ])
            
            # 程度词（增强或减弱情感）
            self.intensifiers = {
                '很': 1.5, '非常': 2.0, '特别': 1.8, '十分': 1.7, '极其': 2.2, '超': 1.6, '太': 1.5,
                '非常': 2.0, '无比': 2.3, '格外': 1.5, '相当': 1.4, '颇为': 1.3,
                'very': 1.5, 'extremely': 2.2, 'really': 1.4, 'so': 1.3, 'quite': 1.2,
                'highly': 1.6, 'particularly': 1.7, 'exceptionally': 2.0
            }
            
            # 否定词（反转情感）
            self.negators = {
                '不', '没', '无', '非', '否', '未', '别', '勿', '没', '不是', '不会', '不行',
                'not', 'no', 'never', 'none', 'neither', 'nor', 'isn\'t', 'wasn\'t', 'aren\'t', 'weren\'t'
            }
            
            # 感叹词（增强情感）
            self.exclamations = {
                '！', '!', '啊', '呀', '哇', '哦', '耶', '哇塞', '天哪',
                'wow', 'oh', 'hey', 'yay', 'omg', 'gosh', 'oh my god'
            }
        
        def analyze_text(self, text):
            """增强版文本情感分析"""
            if not text or not isinstance(text, str):
                return 0.0  # 中性
            
            # 尝试使用jieba进行中文分词，否则使用简单分词
            try:
                import jieba
                words = list(jieba.cut(text))
                # 对于英文部分，转为小写
                words = [word.lower() if all(ord(c) < 128 for c in word) else word for word in words]
            except ImportError:
                # 如果jieba不可用，则使用简单的字符分割（针对中文）和空格分割（针对英文）
                # 对于中文，简单地将每个字符作为一个词
                chinese_chars = []
                english_parts = []
                current_english = ''
                
                for char in text:
                    if ord(char) < 128:  # 英文或符号
                        if current_english and char == ' ':
                            # 英文单词结束
                            english_parts.append(current_english.lower())
                            current_english = ''
                        elif char.isalnum():
                            current_english += char
                        else:
                            # 非字母数字字符作为单独的token
                            if current_english:
                                english_parts.append(current_english.lower())
                                current_english = ''
                            if char.strip():
                                chinese_chars.append(char)
                    else:  # 中文字符
                        if current_english:
                            english_parts.append(current_english.lower())
                            current_english = ''
                        chinese_chars.append(char)
                
                # 处理剩余的英文
                if current_english:
                    english_parts.append(current_english.lower())
                
                # 合并所有单词
                words = chinese_chars + english_parts
            score = 0.0
            pos_count = 0
            neg_count = 0
            
            # 跟踪上下文信息
            negation_active = False
            intensity_factor = 1.0
            exclamation_factor = 1.0
            
            # 计算感叹号数量影响
            exclamation_count = sum(1 for char in text if char in ['!', '！'])
            if exclamation_count > 0:
                exclamation_factor = 1.0 + (exclamation_count * 0.3)  # 每个感叹号增加30%强度
                exclamation_factor = min(exclamation_factor, 2.0)  # 上限为2.0
            
            # 逐词分析
            for i, word in enumerate(words):
                # 检查否定词
                if word in self.negators:
                    negation_active = True
                    continue
                
                # 检查程度词
                if word in self.intensifiers:
                    intensity_factor = self.intensifiers[word]
                    continue
                
                # 检查情感词
                word_score = 0.0
                if word in self.pos_words:
                    word_score = 1.0 * intensity_factor
                    pos_count += 1
                elif word in self.neg_words:
                    word_score = -1.0 * intensity_factor
                    neg_count += 1
                
                # 应用否定
                if negation_active:
                    word_score *= -1
                    negation_active = False  # 否定只影响下一个情感词
                
                # 应用感叹词影响
                word_score *= exclamation_factor
                
                # 添加到总分数
                score += word_score
                
                # 重置强度因子
                intensity_factor = 1.0
            
            # 如果有情感词，归一化分数
            if pos_count + neg_count > 0:
                # 归一化到 -1 到 1 范围
                max_score = pos_count + neg_count  # 理论最大绝对值
                if max_score > 0:
                    score = score / max_score
                    # 限制在 [-1, 1] 范围内
                    score = max(-1.0, min(1.0, score))
            else:
                # 没有找到情感词，尝试检查是否有明显的情感表达
                # 检查是否有多个感叹号
                if exclamation_count >= 3:
                    # 多个感叹号通常表示强烈情感，但难以判断正负
                    # 这里我们假设是积极的（可以根据需要调整）
                    score = 0.5
                # 检查是否有问号（可能表示疑问或困惑）
                question_count = sum(1 for char in text if char in ['?', '？'])
                if question_count > 0:
                    # 问号可能表示中性或轻微负面
                    score -= 0.2
                    score = max(-1.0, min(1.0, score))
            
            return score
        
        def analyze_dataframe(self, df, text_column='content'):
            """分析数据框中的文本"""
            df = df.copy()
            # 使用简单的情感分析
            df['sentiment_score'] = df[text_column].apply(self.analyze_text)
            # 基于得分分类
            df['sentiment'] = df['sentiment_score'].apply(
                lambda score: 'positive' if score > 0 else ('negative' if score < 0 else 'neutral')
            )
            return df
    
    TraditionalSentimentAnalyzer = SimpleSentimentAnalyzer

# 下载NLTK资源
def download_nltk_resources():
    import nltk
    import os
    import logging
    import shutil
    import time
    import threading
    
    logger = logging.getLogger(__name__)
    
    # 定义需要的资源列表
    required_resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    
    def download_with_timeout(resource, timeout=10):
        """带超时的资源下载"""
        result = [False]
        error = [None]
        
        def download_thread():
            try:
                nltk.download(resource, quiet=True)
                result[0] = True
            except Exception as e:
                error[0] = e
        
        thread = threading.Thread(target=download_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        return result[0], error[0]
    
    for resource in required_resources:
        try:
            # 快速检查资源是否已存在
            if resource == 'punkt' or resource == 'punkt_tab':
                try:
                    # 对于tokenizer资源，使用特定路径
                    nltk.data.find(f'tokenizers/{resource}')
                    logger.info(f"NLTK资源 {resource} 已存在")
                except LookupError:
                    # 如果直接查找失败，尝试更具体的检查
                    if resource == 'punkt':
                        # 测试punkt功能
                        from nltk.tokenize import word_tokenize
                        word_tokenize("test")
                        logger.info(f"NLTK资源 {resource} 功能可用")
                    elif resource == 'punkt_tab':
                        # 检查punkt_tab特定路径
                        punkt_tab_found = False
                        for p in nltk.data.path:
                            punkt_tab_dir = os.path.join(p, "tokenizers/punkt_tab/english")
                            if os.path.exists(punkt_tab_dir) and os.listdir(punkt_tab_dir):
                                punkt_tab_found = True
                                break
                        if not punkt_tab_found:
                            raise LookupError("punkt_tab not found")
                        logger.info(f"NLTK资源 {resource} 已存在")
            else:
                # 其他资源使用常规路径
                nltk.data.find(f'corpora/{resource}')
                logger.info(f"NLTK资源 {resource} 已存在")
        except (LookupError, Exception) as e:
            logger.info(f"尝试下载NLTK资源 {resource}...")
            try:
                # 使用超时机制下载
                success, err = download_with_timeout(resource)
                if success:
                    logger.info(f"NLTK资源 {resource} 下载成功")
                else:
                    logger.warning(f"NLTK资源 {resource} 下载超时或失败: {err}")
            except Exception as e:
                logger.warning(f"NLTK资源 {resource} 处理失败: {e}")
        except Exception as e:
            logger.warning(f"检查NLTK资源 {resource} 时出错: {e}")
            # 对于tokenizer资源的特殊处理
            if resource in ['punkt', 'punkt_tab']:
                try:
                    logger.info(f"尝试清理损坏的{resource}资源...")
                    for path in nltk.data.path:
                        resource_path = os.path.join(path, f'tokenizers/{resource}')
                        if os.path.exists(resource_path):
                            if os.path.isdir(resource_path):
                                shutil.rmtree(resource_path)
                            else:
                                os.remove(resource_path)
                except Exception:
                    pass

# 初始化Flask应用
app = Flask(__name__)

# 初始化分析器和预处理器
# 下载必要的NLTK资源
download_nltk_resources()
try:
    # 创建分析器实例
    llm_analyzers = {}
    
    # 支持的模型配置，包含适合中国内地网络环境的选项
    supported_models = {
        'huggingface_english': {
            'model_type': 'huggingface',
            'model_name': 'distilbert-base-uncased-finetuned-sst-2-english',
            'display_name': 'Hugging Face 英文模型'
        },
        'huggingface_chinese': {
            'model_type': 'huggingface',
            'model_name': 'uer/roberta-base-finetuned-jd-binary-chinese',
            'display_name': 'Hugging Face 中文模型'
        },
        'ernie_chinese': {
            'model_type': 'huggingface',
            'model_name': 'nghuyong/ernie-3.0-nano-zh',
            'display_name': '百度 ERNIE 模型'
        },
        'local_chinese': {
            'model_type': 'local',
            'model_name': 'rule-based-chinese',
            'display_name': '本地规则中文模型'
        },
        'local_english': {
            'model_type': 'local',
            'model_name': 'rule-based-english',
            'display_name': '本地规则英文模型'
        },
        'doubao_ernie_bot': {
            'model_type': 'doubao',
            'model_name': 'ERNIE-Bot-4',
            'display_name': '豆包 ERNIE-Bot-4'
        },
        'doubao_turbo': {
            'model_type': 'doubao',
            'model_name': 'ERNIE-Bot-turbo',
            'display_name': '豆包轻量模型'
        },
        'deepseek_chat': {
            'model_type': 'deepseek',
            'model_name': 'deepseek-chat',
            'display_name': 'DeepSeek 对话模型'
        }
    }
    
    # 初始化默认模型（本地规则模型，确保在中国内地环境中可用）
    if LLMSentimentAnalyzer is not None:
        # 使用延迟初始化创建默认模型
        llm_analyzer = LLMSentimentAnalyzer(
            model_name='rule-based-chinese',
            model_type='local',
            init_model=True  # 本地模型立即初始化
        )
        # 延迟创建其他模型实例，只创建对象不初始化模型
        for model_id, config in supported_models.items():
            try:
                # 创建模型对象但不立即初始化模型，提高启动速度
                analyzer = LLMSentimentAnalyzer(
                    model_name=config['model_name'],
                    model_type=config['model_type'],
                    init_model=config['model_type'] == 'local'  # 本地模型立即初始化
                )
                llm_analyzers[model_id] = analyzer
                logger.info(f"成功创建模型对象: {model_id}")
            except Exception as e:
                logger.warning(f"创建模型对象 {model_id} 失败: {e}，将在需要时尝试重新创建")
    else:
        llm_analyzer = None
    
    # 创建传统分析器实例
    traditional_analyzer = TraditionalSentimentAnalyzer()
    
    # 创建中文预处理器
    chinese_preprocessor = ChineseTextPreprocessor()
    
    # 创建英文预处理器
    english_preprocessor = TextPreprocessor(language='english')
    
    logger.info("初始化完成")
except Exception as e:
    logger.error(f"初始化分析器时出错: {e}")

# 判断文本语言（简单实现）
def detect_language(text):
    """简单检测文本语言"""
    # 检查是否包含中文字符
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        return 'chinese'
    return 'english'

# 处理文本并进行情感分析
def analyze_text_sentiment(text, model_id=None):
    """分析单条文本的情感
    
    Args:
        text: 要分析的文本
        model_id: 要使用的模型ID，None表示使用默认模型
    """
    if not text or not text.strip():
        return {
            'success': False,
            'error': '请输入有效的文本',
            'original_text': text,
            'processed_text': None,
            'sentiment': None,
            'score': None,
            'method': None
        }
    
    try:
        # 检测语言
        language = detect_language(text)
        
        # 优化预处理策略：根据语言调整清理参数
        preprocessor = chinese_preprocessor if language == 'chinese' else english_preprocessor
        
        # 改进预处理逻辑，保留更多情感信息
        processed_text = preprocessor.clean_text(
            text,
            remove_urls=True,
            remove_usernames=True,
            remove_hashtags=False,  # 保留话题标签，它们可能包含情感信息
            remove_emojis=False,    # 保留表情符号，它们是重要的情感指示器
            lowercase=(language != 'chinese'),
            remove_punct=False,     # 部分保留标点，如感叹号和问号包含情感信息
            remove_stop=(language != 'chinese'),  # 英文移除停用词，中文保留
            lemmatize_text=(language != 'chinese')
        )
        
        # 如果预处理后文本为空，使用原始文本
        if not processed_text:
            processed_text = text
        
        # 尝试使用指定的LLM分析器
        sentiment_result = None
        method = None
        confidence = 0.0
        
        # 分析结果融合：综合LLM和传统分析器的结果
        llm_score = None
        llm_sentiment = None
        traditional_score = None
        
        # 获取传统分析器的基准结果
        traditional_score = traditional_analyzer.analyze_text(processed_text)
        
        # 选择合适的分析器
        analyzer_to_use = llm_analyzer
        current_model = 'default'
        
        # 根据模型ID选择分析器
        if model_id and model_id in llm_analyzers:
            analyzer_to_use = llm_analyzers[model_id]
            current_model = model_id
        elif language == 'chinese' and 'local_chinese' in llm_analyzers:
            # 默认对中文使用本地模型
            analyzer_to_use = llm_analyzers['local_chinese']
            current_model = 'local_chinese'
        
        # 尝试LLM分析
        if analyzer_to_use is not None:
            try:
                # 确保模型已初始化
                if hasattr(analyzer_to_use, '_initialize_model') and not getattr(analyzer_to_use, '_model_initialized', False):
                    try:
                        logger.info(f"正在初始化模型: {current_model}")
                        analyzer_to_use._initialize_model()
                        analyzer_to_use._model_initialized = True
                        logger.info(f"模型初始化成功: {current_model}")
                    except Exception as init_error:
                        logger.error(f"模型初始化失败 {current_model}: {init_error}")
                        analyzer_to_use._model_initialized = False
                        # 不抛出异常，继续尝试使用传统分析器
                
                # 检查模型连接状态
                if current_model != 'default' and not check_model_connection(current_model):
                    logger.warning(f"模型 {current_model} 连接失败，将使用权重融合方法")
                else:
                    sentiment_result = analyzer_to_use.analyze_sentiment(processed_text)
                    if sentiment_result and 'sentiment' in sentiment_result:
                        # 转换中文情感标签为英文标准格式
                        sentiment_mapping = {
                            '正面': 'positive',
                            '负面': 'negative',
                            '中性': 'neutral'
                        }
                        if sentiment_result['sentiment'] in sentiment_mapping:
                            llm_sentiment = sentiment_mapping[sentiment_result['sentiment']]
                        else:
                            llm_sentiment = sentiment_result['sentiment']
                        
                        # 确保sentiment是有效的值
                        if llm_sentiment not in ['positive', 'negative', 'neutral']:
                            # 如果情感值无效，基于score重新确定
                            llm_score = sentiment_result.get('score', 0)
                            if llm_score > 0:
                                llm_sentiment = 'positive'
                            elif llm_score < 0:
                                llm_sentiment = 'negative'
                            else:
                                llm_sentiment = 'neutral'
                        else:
                            llm_score = sentiment_result.get('score', traditional_score)
                        
                        confidence = sentiment_result.get('confidence', 0.5)
            except Exception as e:
                logger.warning(f"LLM分析器失败: {e}，将使用权重融合方法")
        
        # 融合策略：使用加权投票结合LLM和传统分析器结果
        final_score = 0.0
        final_sentiment = 'neutral'
        
        # 基于语言和可用结果确定最终评分和情感
        if llm_sentiment is not None and llm_score is not None:
            # 融合LLM和传统分析的结果
            # 对于英文文本，LLM权重更高
            # 对于中文文本，平衡两种方法
            llm_weight = 0.7 if language == 'english' else 0.6
            traditional_weight = 1.0 - llm_weight
            
            # 计算加权分数
            final_score = (llm_weight * llm_score) + (traditional_weight * traditional_score)
            method = 'Hybrid'
        else:
            # 仅使用传统分析器结果
            final_score = traditional_score
            method = 'Traditional'
        
        # 优化情感判断阈值，避免过于敏感
        threshold_low = 0.15  # 提高阈值，减少误判
        threshold_high = 0.3  # 更明显的积极情感才判定为positive
        
        if final_score >= threshold_high:
            final_sentiment = 'positive'
        elif final_score <= -threshold_low:
            final_sentiment = 'negative'
        else:
            final_sentiment = 'neutral'
        
        # 准备返回结果
        result = {
            'success': True,
            'original_text': text,
            'processed_text': processed_text,
            'sentiment': final_sentiment,
            'score': round(final_score, 4),  # 保留4位小数
            'confidence': round(confidence, 4),
            'method': method,
            'language': language,
            'model_used': current_model  # 添加使用的模型信息
        }
        
        return result
    
    except Exception as e:
        logger.error(f"分析文本时出错: {e}")
        return {
            'success': False,
            'error': str(e),
            'original_text': text,
            'processed_text': None,
            'sentiment': None,
            'score': None,
            'method': None
        }

# 主页路由
@app.route('/')
def home():
    return render_template('index.html')

# 添加模型连接状态缓存和异步检测支持
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# 模型连接状态缓存
model_status_cache = {}
status_last_checked = {}
STATUS_CACHE_TTL = 30  # 缓存有效期30秒，减少以更快地反映状态变化

# 线程池用于异步模型检测
model_check_executor = ThreadPoolExecutor(max_workers=3)

# 检查模型连接状态
def check_model_connection(model_id):
    """检查模型连接状态，带缓存机制"""
    # 检查缓存是否有效
    current_time = time.time()
    if model_id in model_status_cache and model_id in status_last_checked:
        if current_time - status_last_checked[model_id] < STATUS_CACHE_TTL:
            return model_status_cache[model_id]
    
    try:
        # 检查模型实例是否存在
        if model_id not in llm_analyzers:
            # 尝试创建模型实例
            config = supported_models.get(model_id)
            if config:
                try:
                    analyzer = LLMSentimentAnalyzer(
                        model_name=config['model_name'],
                        model_type=config['model_type'],
                        init_model=False  # 延迟初始化，提高启动速度
                    )
                    llm_analyzers[model_id] = analyzer
                    print(f"成功创建模型实例: {model_id}")
                except Exception as e:
                    print(f"创建模型实例 {model_id} 失败: {e}")
                    model_status_cache[model_id] = False
                    status_last_checked[model_id] = current_time
                    return False
        
        # 使用模型的check_connection方法检查连接
        analyzer = llm_analyzers[model_id]
        if hasattr(analyzer, 'check_connection'):
            # 对于非本地模型，使用异步检查避免阻塞
            if supported_models[model_id]['model_type'] != 'local':
                # 先返回缓存状态或默认值
                if model_id not in model_status_cache:
                    # 首次检查时默认为可用，提高用户体验
                    model_status_cache[model_id] = True  
                    status_last_checked[model_id] = current_time
                
                # 异步检查并更新缓存
                def async_check():
                    try:
                        # 使用快速检查模式提高连接速度
                        # 设置更短的超时时间
                        result = analyzer.check_connection(quick_check=True, timeout=2.0)
                        model_status_cache[model_id] = result
                        status_last_checked[model_id] = time.time()
                    except Exception as e:
                        # 捕获异常但不输出日志，减少日志噪音
                        model_status_cache[model_id] = False
                        status_last_checked[model_id] = time.time()
                
                # 提交异步任务但不等待结果
                model_check_executor.submit(async_check)
                # 立即返回缓存状态，提供快速响应
                return model_status_cache[model_id]
            else:
                # 本地模型直接同步检查
                # 使用快速检查模式
                result = analyzer.check_connection(quick_check=True)
                model_status_cache[model_id] = result
                status_last_checked[model_id] = current_time
                return result
        
        # 如果没有check_connection方法，默认为可用
        model_status_cache[model_id] = True
        status_last_checked[model_id] = current_time
        return True
    except Exception as e:
        print(f"检查模型连接状态失败 {model_id}: {e}")
        model_status_cache[model_id] = False
        status_last_checked[model_id] = current_time
        return False

# 定期刷新模型状态缓存
def refresh_model_status():
    """定期刷新所有模型状态"""
    def refresh_task():
        while True:
            try:
                # 只检查非本地模型的状态
                for model_id, config in supported_models.items():
                    if config['model_type'] != 'local':
                        # 使用更快的刷新策略，每个模型单独异步检查
                        if model_check_executor:
                            model_check_executor.submit(check_model_connection, model_id)
                        else:
                            # 如果没有线程池，就正常调用
                            check_model_connection(model_id)
                # 减少刷新间隔，更快反映状态变化
                time.sleep(STATUS_CACHE_TTL // 2)  # 15秒刷新一次
            except Exception as e:
                print(f"刷新模型状态时出错: {e}")
                # 出错时暂停一段时间再重试
                time.sleep(5)
    
    # 在独立线程中运行刷新任务
    refresh_thread = threading.Thread(target=refresh_task, daemon=True)
    refresh_thread.start()

# 启动定期刷新任务
refresh_model_status()

# 获取支持的模型列表
@app.route('/models', methods=['GET'])
def get_models():
    """获取所有可用模型列表，包含状态信息"""
    model_list = []
    for model_id, model_config in supported_models.items():
        # 动态检查模型连接状态
        connected = check_model_connection(model_id)
        
        model_list.append({
            "id": model_id,
            "name": model_config.get("display_name", model_config["model_name"]),
            "type": model_config["model_type"],
            "connected": connected,
            "status": "connected" if connected else "disconnected"
        })
    
    return jsonify({
        'success': True,
        'models': model_list,
        'default_model': 'local_chinese'
    })

# 情感分析API路由
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' parameter"}), 400
        
        text = data['text'].strip()
        model_id = data.get('model_id', None)
        
        # 验证输入
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        # 检查模型连接状态
        if model_id and model_id in supported_models:
            # 确保模型初始化并检查连接
            if not check_model_connection(model_id):
                logger.warning(f"模型 {supported_models[model_id].get('display_name', model_id)} 连接失败")
                # 不直接返回错误，允许使用备用分析器
        
        # 分析文本
        result = analyze_text_sentiment(text, model_id)
        
        # 如果使用了传统分析器但指定了其他模型，添加提示
        if result['method'] == 'Traditional' and model_id and model_id != 'default':
            result['warning'] = f'模型 {supported_models.get(model_id, {}).get("display_name", model_id)} 不可用，已使用传统分析器'
        
        # 添加使用的模型信息
        if 'model_used' not in result:
            result['model_used'] = model_id or 'default'
        if 'model_display_name' not in result and model_id in supported_models:
            result['model_display_name'] = supported_models[model_id].get("display_name", model_id)
        
        return jsonify(result)
    except ValueError as ve:
        # 处理值错误
        logger.warning(f"值错误: {ve}")
        return jsonify({"error": f"输入数据错误: {str(ve)}"}), 400
    except ConnectionError as ce:
        # 处理连接错误
        logger.error(f"连接错误: {ce}")
        return jsonify({"error": f"模型连接失败: {str(ce)}"}), 503
    except Exception as e:
        # 处理其他未预期的错误
        logger.error(f"分析请求处理错误: {e}")
        return jsonify({"error": "服务器内部错误，请稍后重试"}), 500

# 运行应用
if __name__ == '__main__':
    # 确保templates目录存在
    if not os.path.exists('templates'):
        os.makedirs('templates')
        # 创建基本的HTML模板
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>情感分析系统</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 30px;
        }
        
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #2c3e50;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #2c3e50;
        }
        
        textarea {
            width: 100%;
            height: 150px;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
            resize: vertical;
            font-family: inherit;
        }
        
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
            width: 100%;
        }
        
        button:hover {
            background-color: #2980b9;
        }
        
        button:disabled {
            background-color: #95a5a6;
            cursor: not-allowed;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
        }
        
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 4px;
            background-color: #f9f9f9;
            display: none;
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .result-title {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .sentiment-badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }
        
        .sentiment-positive {
            background-color: #2ecc71;
            color: white;
        }
        
        .sentiment-negative {
            background-color: #e74c3c;
            color: white;
        }
        
        .sentiment-neutral {
            background-color: #95a5a6;
            color: white;
        }
        
        .result-content {
            margin-top: 15px;
        }
        
        .result-item {
            margin-bottom: 10px;
        }
        
        .result-label {
            font-weight: bold;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        
        .result-text {
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 4px;
            word-wrap: break-word;
        }
        
        .score-container {
            margin-top: 15px;
        }
        
        .score-bar {
            width: 100%;
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }
        
        .score-fill {
            height: 100%;
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            transition: all 0.5s ease;
        }
        
        .score-text {
            text-align: center;
            margin-top: 5px;
            font-size: 14px;
            color: #7f8c8d;
        }
        
        .error {
            margin-top: 20px;
            padding: 15px;
            background-color: #ffdddd;
            border-left: 5px solid #e74c3c;
            color: #c0392b;
            display: none;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>情感分析系统</h1>
        
        <div class="form-group">
            <label for="text-input">请输入文本进行情感分析：</label>
            <textarea id="text-input" placeholder="在这里输入中文或英文文本..."></textarea>
        </div>
        
        <button id="analyze-button" onclick="analyzeText()">分析情感</button>
        
        <div id="loading" class="loading">
            <p>正在分析中，请稍候...</p>
        </div>
        
        <div id="error-message" class="error">
            <p id="error-text">发生错误，请重试。</p>
        </div>
        
        <div id="result" class="result">
            <div class="result-header">
                <div class="result-title">分析结果</div>
                <div id="sentiment-badge" class="sentiment-badge">中性</div>
            </div>
            
            <div class="result-content">
                <div class="result-item">
                    <div class="result-label">原始文本：</div>
                    <div id="original-text" class="result-text"></div>
                </div>
                
                <div class="result-item">
                    <div class="result-label">处理后文本：</div>
                    <div id="processed-text" class="result-text"></div>
                </div>
                
                <div class="score-container">
                    <div class="result-label">情感得分：</div>
                    <div class="score-bar">
                        <div id="score-fill" class="score-fill" style="width: 50%; background-color: #95a5a6;"></div>
                    </div>
                    <div id="score-text" class="score-text">0.0 (中性)</div>
                </div>
                
                <div class="result-item">
                    <div class="result-label">分析方法：</div>
                    <div id="method-text" class="result-text"></div>
                </div>
                
                <div class="result-item">
                    <div class="result-label">检测语言：</div>
                    <div id="language-text" class="result-text"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function analyzeText() {
            const text = document.getElementById('text-input').value.trim();
            const button = document.getElementById('analyze-button');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const errorMessage = document.getElementById('error-message');
            
            // 重置状态
            result.style.display = 'none';
            errorMessage.style.display = 'none';
            
            // 验证输入
            if (!text) {
                showError('请输入有效的文本');
                return;
            }
            
            // 显示加载状态
            button.disabled = true;
            loading.style.display = 'block';
            
            // 发送请求到服务器
            fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            })
            .then(response => response.json())
            .then(data => {
                // 隐藏加载状态
                button.disabled = false;
                loading.style.display = 'none';
                
                if (data.success) {
                    // 显示结果
                    displayResult(data);
                } else {
                    // 显示错误
                    showError(data.error || '分析失败，请重试');
                }
            })
            .catch(error => {
                // 隐藏加载状态
                button.disabled = false;
                loading.style.display = 'none';
                
                // 显示错误
                showError('网络错误，请重试');
                console.error('Error:', error);
            });
        }
        
        function showError(message) {
            const errorText = document.getElementById('error-text');
            const errorMessage = document.getElementById('error-message');
            
            errorText.textContent = message;
            errorMessage.style.display = 'block';
        }
        
        function displayResult(data) {
            const result = document.getElementById('result');
            const sentimentBadge = document.getElementById('sentiment-badge');
            const originalText = document.getElementById('original-text');
            const processedText = document.getElementById('processed-text');
            const scoreFill = document.getElementById('score-fill');
            const scoreText = document.getElementById('score-text');
            const methodText = document.getElementById('method-text');
            const languageText = document.getElementById('language-text');
            
            // 设置情感标签
            sentimentBadge.textContent = getSentimentLabel(data.sentiment);
            sentimentBadge.className = `sentiment-badge sentiment-${data.sentiment}`;
            
            // 设置文本内容
            originalText.textContent = data.original_text;
            processedText.textContent = data.processed_text || '无';
            methodText.textContent = data.method;
            languageText.textContent = data.language === 'chinese' ? '中文' : '英文';
            
            // 设置分数条
            const score = data.score || 0;
            const percentage = (score + 1) / 2 * 100; // 将 -1 到 1 的分数映射到 0% 到 100%
            
            // 根据分数设置颜色
            let color;
            if (score > 0) {
                color = `rgb(${Math.floor(46 - score * 46)}, ${Math.floor(204 + score * 51)}, ${Math.floor(113 - score * 113)})`;
            } else {
                color = `rgb(${Math.floor(231 + score * 185)}, ${Math.floor(76 + score * 76)}, ${Math.floor(60 + score * 60)})`;
            }
            
            scoreFill.style.width = `${percentage}%`;
            scoreFill.style.backgroundColor = color;
            scoreText.textContent = `${score.toFixed(2)} (${getSentimentLabel(data.sentiment)})`;
            
            // 显示结果
            result.style.display = 'block';
        }
        
        function getSentimentLabel(sentiment) {
            switch(sentiment) {
                case 'positive':
                    return '积极';
                case 'negative':
                    return '消极';
                case 'neutral':
                    return '中性';
                default:
                    return '未知';
            }
        }
        
        // 添加回车+Ctrl快捷键提交
        document.getElementById('text-input').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                analyzeText();
            }
        });
    </script>
</body>
</html>''')
    
    # 运行应用
    app.run(host='0.0.0.0', port=5000, debug=True)