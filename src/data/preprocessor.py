import os
import re
import string
import logging
import pandas as pd
from typing import List, Optional, Dict, Any

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from config import MIN_TEXT_LENGTH, MAX_TEXT_LENGTH

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 下载必要的NLTK资源
def download_nltk_resources():
    """安全下载NLTK资源，针对punkt分词器添加特殊处理"""
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet'
    }
    
    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
            logger.info(f'NLTK资源 {resource_name} 已存在')
        except LookupError:
            try:
                logger.info(f'正在下载NLTK资源 {resource_name}...')
                nltk.download(resource_name)
                logger.info(f'NLTK资源 {resource_name} 下载完成')
            except Exception as e:
                logger.warning(f'下载NLTK资源 {resource_name} 失败: {e}')
                logger.info(f'将使用备用方法处理文本')
        except Exception as e:
            logger.warning(f'检查NLTK资源 {resource_name} 时出错: {e}')
            # 对于punkt资源，尝试更激进的清理和重新下载
            if resource_name == 'punkt':
                logger.info('尝试清理损坏的punkt资源并重新下载...')
                # 尝试删除损坏的资源
                try:
                    import shutil
                    nltk_data_dir = nltk.data.path[0]
                    punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
                    if os.path.exists(punkt_dir):
                        shutil.rmtree(punkt_dir)
                        logger.info('已删除损坏的punkt目录')
                        # 重新下载
                        nltk.download('punkt')
                        logger.info('punkt资源重新下载完成')
                except Exception as e2:
                    logger.warning(f'清理和重新下载punkt资源失败: {e2}')

# 执行资源下载
download_nltk_resources()

class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, language: str = 'english'):
        self.language = language
        
        # 安全初始化NLTK组件，添加错误处理
        try:
            self.stop_words = set(stopwords.words(language))
            logger.info(f'成功加载{language}停用词')
        except Exception as e:
            logger.warning(f'加载停用词失败: {e}')
            self.stop_words = set()  # 使用空集合作为备用
        
        try:
            self.lemmatizer = WordNetLemmatizer()
            logger.info('成功初始化词形还原器')
        except Exception as e:
            logger.warning(f'初始化词形还原器失败: {e}')
            # 定义简单的备用函数
            self.lemmatizer = type('SimpleLemmatizer', (), {'lemmatize': lambda self, x: x})
        
        # 正则表达式模式（不依赖外部资源）
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.username_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.emoji_pattern = re.compile(
            '['
            '\U0001F600-\U0001F64F'  # 表情符号
            '\U0001F300-\U0001F5FF'  # 符号和象形文字
            '\U0001F680-\U0001F6FF'  # 交通和地图符号
            '\U0001F1E0-\U0001F1FF'  # 国旗
            '\U00002702-\U000027B0'  # 装饰符号
            '\U000024C2-\U0001F251'  # 其他符号
            ']+', 
            flags=re.UNICODE
        )
    
    def remove_urls(self, text: str) -> str:
        """移除URL链接"""
        return self.url_pattern.sub('', text)
    
    def remove_usernames(self, text: str) -> str:
        """移除用户名（如@username）"""
        return self.username_pattern.sub('', text)
    
    def remove_hashtags(self, text: str) -> str:
        """移除话题标签"""
        return self.hashtag_pattern.sub('', text)
    
    def remove_emojis(self, text: str) -> str:
        """移除表情符号"""
        return self.emoji_pattern.sub('', text)
    
    def convert_to_lowercase(self, text: str) -> str:
        """转换为小写"""
        return text.lower()
    
    def remove_punctuation(self, text: str) -> str:
        """移除标点符号"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """备用分词方法，不依赖NLTK punkt"""
        # 使用简单的空格分词作为备用
        return text.split()
    
    def remove_stopwords(self, text: str) -> str:
        """移除停用词"""
        try:
            try:
                tokens = word_tokenize(text)
            except Exception as e:
                logger.warning(f'NLTK分词失败: {e}，使用备用分词方法')
                tokens = self._simple_tokenize(text)
            
            if self.stop_words:
                filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
                return ' '.join(filtered_tokens)
            else:
                # 如果没有停用词表，返回原始文本
                return text
        except Exception as e:
            logger.warning(f'移除停用词时出错: {e}')
            # 如果tokenize失败，返回原始文本
            return text
    
    def lemmatize(self, text: str) -> str:
        """词形还原"""
        try:
            try:
                tokens = word_tokenize(text)
            except Exception as e:
                logger.warning(f'NLTK分词失败: {e}，使用备用分词方法')
                tokens = self._simple_tokenize(text)
            
            lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            return ' '.join(lemmatized_tokens)
        except Exception as e:
            logger.warning(f'词形还原时出错: {e}')
            # 如果lemmatize失败，返回原始文本
            return text
    
    def remove_extra_whitespace(self, text: str) -> str:
        """移除多余的空格"""
        return ' '.join(text.split())
    
    def clean_text(self, text: Optional[str], 
                   remove_urls: bool = True,
                   remove_usernames: bool = True,
                   remove_hashtags: bool = False,
                   remove_emojis: bool = True,
                   lowercase: bool = True,
                   remove_punct: bool = True,
                   remove_stop: bool = True,
                   lemmatize_text: bool = True) -> Optional[str]:
        """完整的文本清洗流程"""
        if text is None or not isinstance(text, str) or not text.strip():
            return None
        
        # 基本清洗
        text = self.remove_extra_whitespace(text)
        
        # 根据参数进行清洗
        if remove_urls:
            text = self.remove_urls(text)
        if remove_usernames:
            text = self.remove_usernames(text)
        if remove_hashtags:
            text = self.remove_hashtags(text)
        if remove_emojis:
            text = self.remove_emojis(text)
        if lowercase:
            text = self.convert_to_lowercase(text)
        if remove_punct:
            text = self.remove_punctuation(text)
        if remove_stop:
            text = self.remove_stopwords(text)
        if lemmatize_text:
            text = self.lemmatize(text)
        
        # 最终清理空格
        text = self.remove_extra_whitespace(text)
        
        # 检查文本长度
        if len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:
            return None
        
        return text
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str, 
                         **clean_kwargs) -> pd.DataFrame:
        """处理DataFrame中的文本列"""
        logger.info(f"Processing text column: {text_column}")
        
        # 创建清洗后的文本列
        cleaned_column = f"{text_column}_cleaned"
        df[cleaned_column] = df[text_column].apply(
            lambda x: self.clean_text(x, **clean_kwargs)
        )
        
        # 对于中文文本，我们采用更宽松的策略，避免过度过滤
        if hasattr(self, 'language') and self.language == 'chinese':
            # 中文处理时，我们只移除绝对为空的值
            initial_count = len(df)
            df = df[df[cleaned_column].notna()]
            final_count = len(df)
            logger.info(f"中文文本处理: 保留了 {final_count}/{initial_count} 条记录")
        else:
            # 其他语言保持原有逻辑
            initial_count = len(df)
            df = df.dropna(subset=[cleaned_column])
            final_count = len(df)
            logger.info(f"Removed {initial_count - final_count} rows with invalid text")
        
        return df
    
    def batch_process(self, texts: List[str], **clean_kwargs) -> List[Optional[str]]:
        """批量处理文本列表"""
        return [self.clean_text(text, **clean_kwargs) for text in texts]

# 中文文本预处理器
class ChineseTextPreprocessor(TextPreprocessor):
    """中文文本预处理器"""
    
    def __init__(self):
        # 调用父类初始化，确保所有必要的属性都被正确设置
        super().__init__(language='chinese')
        
        # 安全加载中文停用词，添加错误处理
        try:
            self.stop_words = set(stopwords.words('chinese'))
            logger.info('成功加载中文停用词')
        except Exception as e:
            logger.warning(f'加载中文停用词失败: {e}')
            # 使用空集合作为备用，避免因为停用词问题导致整个处理失败
            self.stop_words = set()
    
    def remove_stopwords(self, text: str) -> str:
        """中文停用词移除"""
        try:
            # 对于中文，我们可以采用更简单的方法，因为停用词表可能不完整
            # 或者分词可能不准确，我们避免过度处理导致文本被清空
            if not self.stop_words:
                return text
            
            # 简单实现，保留大部分字符，只移除明显的停用词
            result = []
            for char in text:
                if char not in self.stop_words or char.strip():
                    result.append(char)
            return ''.join(result)
        except Exception as e:
            logger.warning(f'中文停用词处理出错: {e}')
            # 如果处理失败，返回原始文本，避免丢失数据
            return text
    
    def lemmatize(self, text: str) -> str:
        """中文不需要词形还原，返回原文本"""
        return text
    
    def clean_text(self, text: Optional[str], 
                   remove_urls: bool = True,
                   remove_usernames: bool = True,
                   remove_hashtags: bool = False,
                   remove_emojis: bool = True,
                   lowercase: bool = False,  # 中文不需要小写
                   remove_punct: bool = True,
                   remove_stop: bool = True,
                   lemmatize_text: bool = False) -> Optional[str]:
        """中文文本清洗流程，调整参数以适应中文特性"""
        # 调用父类的clean_text方法，但传入适合中文的参数
        # 注意：我们降低了文本长度要求，因为中文预处理后可能会变短
        cleaned = super().clean_text(
            text,
            remove_urls=remove_urls,
            remove_usernames=remove_usernames,
            remove_hashtags=remove_hashtags,
            remove_emojis=remove_emojis,
            lowercase=lowercase,
            remove_punct=remove_punct,
            remove_stop=remove_stop,
            lemmatize_text=lemmatize_text
        )
        
        # 对于中文，我们放宽文本长度限制，避免过度过滤
        if cleaned and len(cleaned) < 2:  # 中文至少保留2个字符
            return None
        
        return cleaned

# 示例用法
if __name__ == "__main__":
    # 创建示例文本
    sample_texts = [
        "I love this product! @brand #awesome https://example.com",
        "This is terrible. I want a refund.",
        "Just okay, nothing special.",
        "",  # 空文本
        None  # None值
    ]
    
    # 创建预处理器
    preprocessor = TextPreprocessor()
    
    # 处理示例文本
    print("处理后的文本:")
    for i, text in enumerate(sample_texts):
        cleaned = preprocessor.clean_text(text)
        print(f"原始文本 {i+1}: {text}")
        print(f"处理后文本 {i+1}: {cleaned}")
        print("---")