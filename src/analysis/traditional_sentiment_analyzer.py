import logging
import pandas as pd
import jieba
from typing import List, Dict, Any, Optional

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TraditionalSentimentAnalyzer:
    """传统情感分析器，使用TextBlob和机器学习模型，增强中文支持"""
    
    def __init__(self):
        self.models = {}
        
        # 扩展的中文情感词典
        self.chinese_pos_words = set([
            '好', '棒', '优秀', '喜欢', '开心', '快乐', '满意', '赞', '推荐', '支持',
            '精彩', '棒极了', '完美', '出色', '很好', '不错', '良好', '优秀', '很棒', '厉害',
            '感谢', '感激', '棒', '赞', '爱', '幸福', '愉悦', '舒适', '方便', '高效',
            '物超所值', '惊喜', '满意', '值得', '好评', '完美', '精彩', '出色', '优秀', '成功',
            '喜欢', '热爱', '欣赏', '敬佩', '认同', '肯定', '鼓励', '支持', '赞同', '认可',
            '顺利', '顺利进行', '顺利完成', '顺利通过', '顺利到达', '顺利实现', '顺利开展',
            '强大', '强劲', '强势', '强壮', '强健', '坚韧', '坚韧不拔', '坚定不移', '坚强',
            '舒适', '舒服', '愉快', '畅快', '畅快淋漓', '畅快舒适', '舒适宜人', '舒适安逸',
            '美好', '美妙', '美满', '美丽', '漂亮', '秀丽', '靓丽', '亮丽', '优美', '优雅',
            '高', '高等', '高级', '高档', '高贵', '高档', '高质量', '高品质', '高效率', '高效能',
            '优', '优良', '优秀', '优质', '优雅', '优美', '优胜', '优势', '优先', '优惠',
            '佳', '佳品', '佳作', '佳境', '佳肴', '佳酿', '佳期', '佳偶', '佳宾', '佳话',
            '惊喜', '惊叹', '惊艳', '惊诧', '惊奇', '惊喜交加', '惊喜万分', '惊喜不已',
            '感动', '感激', '感恩', '感谢', '感动不已', '感激不尽', '感恩戴德', '感谢不尽',
            '满意', '满足', '畅快', '畅爽', '满足感', '满意感', '满足度', '满意度',
            '赞', '赞美', '赞赏', '称赞', '赞扬', '赞叹', '赞美', '赞誉', '赞美之词',
            '棒', '很棒', '超棒', '太棒了', '棒极了', '真棒', '太棒', '非常棒', '无比棒',
            '开心', '快乐', '愉快', '欢乐', '欢快', '欢心', '欢欣', '欢喜', '欢快',
            '推荐', '力荐', '推荐给', '强烈推荐', '值得推荐', '大力推荐', '推荐大家',
            '支持', '力挺', '拥护', '赞成', '赞同', '同意', '支持', '支援', '声援',
            '喜欢', '喜爱', '热爱', '爱好', '喜好', '欢喜', '喜悦', '喜欢', '着迷',
            '完美', '完美无瑕', '完美无缺', '完美收官', '完美落幕', '完美呈现', '完美演绎',
            '出色', '出类拔萃', '出类拔群', '出色表现', '出色完成', '出色工作', '出色发挥',
            '高效', '高效率', '高效能', '高效益', '高效工作', '高效完成', '高效处理',
            '成功', '胜利', '获胜', '得胜', '成功', '成就', '成绩', '成果', '成效',
            '方便', '便捷', '便利', '便当', '方便快捷', '便捷高效', '便利实用',
            '舒服', '舒适', '舒坦', '舒适', '舒服', '舒适', '舒适', '舒适', '舒适',
            '值得', '值得拥有', '值得推荐', '值得信赖', '值得期待', '值得关注', '值得购买',
            '给力', '很给力', '非常给力', '超级给力', '相当给力', '太给力了',
            '厉害', '很厉害', '非常厉害', '超级厉害', '相当厉害', '太厉害了',
            '惊喜', '很惊喜', '非常惊喜', '超级惊喜', '相当惊喜', '太惊喜了',
            '开心', '很开心', '非常开心', '超级开心', '相当开心', '太开心了',
            '满意', '很满意', '非常满意', '超级满意', '相当满意', '太满意了',
            '精彩', '很精彩', '非常精彩', '超级精彩', '相当精彩', '太精彩了',
            '棒', '很棒', '非常棒', '超级棒', '相当棒', '太棒了',
            '优秀', '很优秀', '非常优秀', '超级优秀', '相当优秀', '太优秀了',
            '好', '很好', '非常好', '超级好', '相当好', '太好了', '好好',
            '牛', '很牛', '非常牛', '超级牛', '相当牛', '太牛了',
            '赞', '很赞', '非常赞', '超级赞', '相当赞', '太赞了',
            '美', '很美', '非常美', '超级美', '相当美', '太美了',
            '爽', '很爽', '非常爽', '超级爽', '相当爽', '太爽了',
            '棒', '很棒', '非常棒', '超级棒', '相当棒', '太棒了',
            '好', '很好', '非常好', '超级好', '相当好', '太好了', '好好',
        ])
        
        self.chinese_neg_words = set([
            '坏', '差', '糟糕', '讨厌', '生气', '难过', '不满意', '坑', '失望', '反对',
            '垃圾', '差', '不好', '糟糕', '恶心', '讨厌', '生气', '难过', '伤心', '痛苦',
            '失望', '郁闷', '烦躁', '愤怒', '烦', '闷', '累', '困', '无聊', '乏味',
            '贵', '贵了', '不值', '性价比低', '差评', '退货', '退款', '投诉', '举报', '问题',
            '错误', '失败', '漏洞', '缺陷', '缺点', '不足', '缺点', '弱点', '薄弱', '欠缺',
            '差', '很差', '非常差', '超级差', '相当差', '太差了',
            '坏', '很坏', '非常坏', '超级坏', '相当坏', '太坏了',
            '垃圾', '很垃圾', '非常垃圾', '超级垃圾', '相当垃圾', '太垃圾了',
            '糟糕', '很糟糕', '非常糟糕', '超级糟糕', '相当糟糕', '太糟糕了',
            '恶心', '很恶心', '非常恶心', '超级恶心', '相当恶心', '太恶心了',
            '讨厌', '很讨厌', '非常讨厌', '超级讨厌', '相当讨厌', '太讨厌了',
            '生气', '很生气', '非常生气', '超级生气', '相当生气', '太生气了',
            '难过', '很难过', '非常难过', '超级难过', '相当难过', '太难过了',
            '伤心', '很伤心', '非常伤心', '超级伤心', '相当伤心', '太伤心了',
            '痛苦', '很痛苦', '非常痛苦', '超级痛苦', '相当痛苦', '太痛苦了',
            '失望', '很失望', '非常失望', '超级失望', '相当失望', '太失望了',
            '郁闷', '很郁闷', '非常郁闷', '超级郁闷', '相当郁闷', '太郁闷了',
            '烦躁', '很烦躁', '非常烦躁', '超级烦躁', '相当烦躁', '太烦躁了',
            '愤怒', '很愤怒', '非常愤怒', '超级愤怒', '相当愤怒', '太愤怒了',
            '烦', '很烦', '非常烦', '超级烦', '相当烦', '太烦了',
            '闷', '很闷', '非常闷', '超级闷', '相当闷', '太闷了',
            '累', '很累', '非常累', '超级累', '相当累', '太累了',
            '困', '很困', '非常困', '超级困', '相当困', '太困了',
            '无聊', '很无聊', '非常无聊', '超级无聊', '相当无聊', '太无聊了',
            '乏味', '很乏味', '非常乏味', '超级乏味', '相当乏味', '太乏味了',
            '贵', '很贵', '非常贵', '超级贵', '相当贵', '太贵了',
            '不值', '很不值', '非常不值', '超级不值', '相当不值', '太不值了',
            '差评', '很差评', '非常差评', '超级差评', '相当差评', '太差评了',
            '退货', '要退货', '必须退货', '坚决退货', '强烈要求退货',
            '退款', '要退款', '必须退款', '坚决退款', '强烈要求退款',
            '投诉', '要投诉', '必须投诉', '坚决投诉', '强烈投诉',
            '举报', '要举报', '必须举报', '坚决举报', '强烈举报',
            '问题', '有问题', '很多问题', '严重问题', '大问题', '小问题',
            '错误', '有错误', '很多错误', '严重错误', '大错误', '小错误',
            '失败', '失败了', '彻底失败', '完全失败', '很失败', '太失败了',
            '漏洞', '有漏洞', '很多漏洞', '严重漏洞', '大漏洞', '小漏洞',
            '缺陷', '有缺陷', '很多缺陷', '严重缺陷', '大缺陷', '小缺陷',
            '缺点', '有缺点', '很多缺点', '严重缺点', '大缺点', '小缺点',
            '不足', '有不足', '很多不足', '严重不足', '大不足', '小不足',
        ])
        
        # 中文程度词
        self.chinese_intensifiers = {
            '很': 1.5, '非常': 2.0, '特别': 1.8, '十分': 1.7, '极其': 2.2, '超': 1.6, '太': 1.5,
            '非常': 2.0, '无比': 2.3, '格外': 1.5, '相当': 1.4, '颇为': 1.3,
            '超级': 2.1, '极其': 2.2, '极度': 2.1, '异常': 1.9, '异常地': 1.9,
            '很是': 1.4, '真是': 1.5, '实在是': 1.6, '确实是': 1.5, '简直': 1.7,
            '特别地': 1.8, '十分地': 1.7, '非常地': 2.0, '格外的': 1.5, '相当的': 1.4,
        }
        
        # 中文否定词
        self.chinese_negators = {
            '不', '没', '无', '非', '否', '未', '别', '勿', '没', '不是', '不会', '不行',
            '不是', '没有', '并非', '绝非', '并不', '并未', '不曾', '不能', '不要', '不必',
            '不应该', '不可以', '不能够', '不喜欢', '不满意', '不高兴', '不开心',
            '没有', '没', '未', '无', '否', '非', '别', '勿', '莫', '休',
            '不行', '不能', '不会', '不可以', '不可', '不能', '无法', '难以', '不得',
            '不如', '不及', '不足', '不够', '不好', '不对', '不切', '不适', '不宜',
            '不良', '不妥', '不当', '不端', '不良', '不轨', '不法', '不公', '不平',
        }
        
        logger.info("传统情感分析器初始化完成")
    
    def detect_language(self, text: str) -> str:
        """简单检测文本语言"""
        # 检查是否包含中文字符
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'chinese'
        return 'english'
    
    def analyze_text(self, text: str) -> float:
        """分析文本情感，返回情感分数
        
        Args:
            text: 要分析的文本
            
        Returns:
            float: 情感分数，范围[-1, 1]，正数表示积极，负数表示消极
        """
        if not text or not isinstance(text, str):
            return 0.0  # 中性
        
        # 检测语言
        language = self.detect_language(text)
        
        # 根据语言选择不同的分析方法
        if language == 'chinese':
            return self._analyze_chinese_text(text)
        else:
            # 英文使用TextBlob分析
            try:
                blob = TextBlob(text)
                return blob.sentiment.polarity  # TextBlob返回的极性范围[-1, 1]
            except Exception as e:
                logger.warning(f"TextBlob分析失败: {e}")
                return 0.0
    
    def _analyze_chinese_text(self, text: str) -> float:
        """分析中文文本情感，增强版"""
        # 计算标点符号的情感增强作用
        exclamation_count = sum(1 for char in text if char in ['!', '！'])
        question_count = sum(1 for char in text if char in ['?', '？'])
        
        # 感叹号增强情感强度
        punctuation_factor = 1.0 + (exclamation_count * 0.3)  # 每个感叹号增加30%强度
        punctuation_factor = min(punctuation_factor, 2.5)  # 上限为2.5
        
        # 问号可能表示疑问或轻微负面
        question_effect = -0.1 * question_count
        question_effect = max(-0.3, question_effect)  # 最多降低0.3
        
        try:
            # 使用jieba分词
            words = list(jieba.cut(text))
        except ImportError:
            # 如果没有jieba，使用简单的字符级处理
            logger.warning("未安装jieba，使用简单字符级处理")
            words = list(text)
        
        score = 0.0
        pos_count = 0
        neg_count = 0
        
        # 跟踪上下文信息
        negation_active = False
        intensity_factor = 1.0
        sentiment_words_found = False
        
        # 分析每个词
        for i, word in enumerate(words):
            # 检查程度词
            if word in self.chinese_intensifiers:
                intensity_factor = self.chinese_intensifiers[word]
                continue
            
            # 检查否定词
            if word in self.chinese_negators:
                negation_active = True
                continue
            
            # 检查积极词
            if word in self.chinese_pos_words:
                # 提高基础分数，让情感词有更大影响
                word_score = 0.3  # 提高到0.3
                if negation_active:
                    word_score = -word_score
                word_score *= intensity_factor
                score += word_score
                pos_count += 1
                negation_active = False
                intensity_factor = 1.0
                sentiment_words_found = True
            # 检查消极词
            elif word in self.chinese_neg_words:
                # 提高基础分数
                word_score = -0.3  # 提高到-0.3
                if negation_active:
                    word_score = -word_score
                word_score *= intensity_factor
                score += word_score
                neg_count += 1
                negation_active = False
                intensity_factor = 1.0
                sentiment_words_found = True
        
        # 归一化分数到[-1, 1]范围
        if sentiment_words_found:
            # 改进归一化逻辑，更好地反映情感强度
            total_count = pos_count + neg_count
            # 基于情感词数量和分数计算最终结果
            if total_count > 0:
                # 对于少量情感词，保持较高权重
                if total_count <= 3:
                    normalized_score = score
                else:
                    # 对于多个情感词，考虑整体分布
                    normalized_score = score * (1 + min(0.5, (total_count - 3) * 0.1))
                
                # 应用标点符号影响
                normalized_score = normalized_score * punctuation_factor
                normalized_score += question_effect
                
                # 确保在[-1, 1]范围内
                normalized_score = max(-1.0, min(1.0, normalized_score))
                return normalized_score
        
        # 没有找到情感词，但有多个感叹号
        if exclamation_count >= 2:
            return 0.2 * punctuation_factor  # 假设积极情感
        
        # 没有找到情感词，返回中性，但考虑问号影响
        return max(-0.3, min(0.3, question_effect))  # 轻微偏向中性
    
    def analyze_with_textblob(self, text: str) -> Dict[str, Any]:
        """使用TextBlob进行情感分析"""
        if not text or not isinstance(text, str):
            return {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "sentiment": "中性"
            }
        
        # 检测语言
        language = self.detect_language(text)
        
        if language == 'chinese':
            # 中文使用增强的情感分析
            polarity = self._analyze_chinese_text(text)
            subjectivity = 0.5  # 中文默认中等主观性
        else:
            # 英文使用TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        
        # 确定情感标签
        if polarity > 0.1:
            sentiment = "正面"
        elif polarity < -0.1:
            sentiment = "负面"
        else:
            sentiment = "中性"
        
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": sentiment
        }
    
    def analyze_batch_with_textblob(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量使用TextBlob进行情感分析"""
        results = []
        for text in texts:
            results.append(self.analyze_with_textblob(text))
        return results
    
    def train_model(self, X_train: List[str], y_train: List[str], model_name: str = "logistic_regression"):
        """训练情感分析模型"""
        # 将文本转换为TF-IDF特征
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        
        # 训练逻辑回归模型
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_tfidf, y_train)
        
        # 保存模型和向量化器
        self.models[model_name] = {
            "model": model,
            "vectorizer": vectorizer
        }
        
        logger.info(f"模型 {model_name} 训练完成")
        return model_name
    
    def evaluate_model(self, X_test: List[str], y_test: List[str], model_name: str = "logistic_regression") -> Dict[str, Any]:
        """评估模型性能"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model_data = self.models[model_name]
        model = model_data["model"]
        vectorizer = model_data["vectorizer"]
        
        # 转换测试数据
        X_test_tfidf = vectorizer.transform(X_test)
        
        # 预测
        y_pred = model.predict(X_test_tfidf)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            "accuracy": accuracy,
            "classification_report": report
        }
    
    def predict_with_model(self, texts: List[str], model_name: str = "logistic_regression") -> List[str]:
        """使用训练好的模型进行预测"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model_data = self.models[model_name]
        model = model_data["model"]
        vectorizer = model_data["vectorizer"]
        
        # 转换文本
        X_tfidf = vectorizer.transform(texts)
        
        # 预测
        predictions = model.predict(X_tfidf)
        
        return predictions.tolist()
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str, use_ml: bool = False, 
                         model_name: str = "logistic_regression") -> pd.DataFrame:
        """分析DataFrame中的文本列"""
        # 获取文本列表
        texts = df[text_column].tolist()
        
        # 使用TextBlob分析
        textblob_results = self.analyze_batch_with_textblob(texts)
        textblob_df = pd.DataFrame(textblob_results)
        
        # 合并结果
        result_df = df.copy()
        result_df = pd.concat([result_df, textblob_df.add_prefix('textblob_')], axis=1)
        
        # 如果启用机器学习模型预测
        if use_ml and model_name in self.models:
            ml_predictions = self.predict_with_model(texts, model_name)
            result_df[f"ml_sentiment_{model_name}"] = ml_predictions
        
        return result_df
    
    def train_from_dataframe(self, df: pd.DataFrame, text_column: str, label_column: str, 
                           model_name: str = "logistic_regression", test_size: float = 0.2) -> Dict[str, Any]:
        """从DataFrame中训练模型并评估"""
        # 准备数据
        X = df[text_column].tolist()
        y = df[label_column].tolist()
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # 训练模型
        self.train_model(X_train, y_train, model_name)
        
        # 评估模型
        evaluation = self.evaluate_model(X_test, y_test, model_name)
        
        logger.info(f"模型 {model_name} 评估结果: 准确率 = {evaluation['accuracy']:.4f}")
        
        return evaluation

# 示例用法
if __name__ == "__main__":
    # 示例文本
    sample_texts = [
        "I love this product! It's amazing.",
        "This is terrible. I want a refund.",
        "It's okay, not great but not bad either.",
        "The service was excellent and the food was delicious.",
        "I'm very disappointed with the quality."
    ]
    
    # 创建分析器
    analyzer = TraditionalSentimentAnalyzer()
    
    # 使用TextBlob分析
    print("==== TextBlob 情感分析 ====")
    for i, text in enumerate(sample_texts):
        result = analyzer.analyze_with_textblob(text)
        print(f"文本 {i+1}: {text}")
        print(f"极性: {result['polarity']}, 主观性: {result['subjectivity']}, 情感: {result['sentiment']}")
        print("---")
    
    # 批量分析
    print("\n==== 批量分析 ====")
    batch_results = analyzer.analyze_batch_with_textblob(sample_texts)
    for i, result in enumerate(batch_results):
        print(f"文本 {i+1} 结果: {result}")
    
    # 创建示例DataFrame进行演示
    sample_df = pd.DataFrame({
        'text': sample_texts,
        'label': ['positive', 'negative', 'neutral', 'positive', 'negative']
    })
    
    print("\n==== DataFrame 分析 ====")
    result_df = analyzer.analyze_dataframe(sample_df, 'text')
    print(result_df)
    
    # 训练模型演示（需要有标签的数据）
    print("\n==== 模型训练演示 ====")
    try:
        evaluation = analyzer.train_from_dataframe(sample_df, 'text', 'label')
        print(f"模型评估结果: 准确率 = {evaluation['accuracy']:.4f}")
    except Exception as e:
        print(f"模型训练演示失败（因为示例数据量太小）: {e}")