import os
import sys
import logging
import argparse
import pandas as pd
from typing import Optional

# 配置日志 - 优化为课堂展示风格
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from src.data.data_loader import DataLoader
from src.data.preprocessor import TextPreprocessor, ChineseTextPreprocessor
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
    # 创建一个简单的备用情感分析器类
    class SimpleSentimentAnalyzer:
        def __init__(self):
            # 简单的关键词情感词典
            self.pos_words = set(['好', '棒', '优秀', '喜欢', '开心', '快乐', '满意', '赞', '推荐', '支持',
                                'good', 'great', 'excellent', 'like', 'love', 'happy', 'satisfied', 'awesome'])
            self.neg_words = set(['坏', '差', '糟糕', '讨厌', '生气', '难过', '不满意', '坑', '失望', '反对',
                                'bad', 'poor', 'terrible', 'hate', 'angry', 'sad', 'disappointed', 'worst'])
        
        def analyze_text(self, text):
            """简单的文本情感分析"""
            if not text or not isinstance(text, str):
                return 0.0  # 中性
            
            words = text.lower().split()
            pos_count = sum(1 for word in words if word in self.pos_words)
            neg_count = sum(1 for word in words if word in self.neg_words)
            
            # 计算情感得分 (-1 到 1)
            if pos_count + neg_count == 0:
                return 0.0  # 没有情感词
            
            score = (pos_count - neg_count) / (pos_count + neg_count)
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
# 尝试导入SentimentVisualizer，如果失败则创建一个简单的备用
try:
    from src.visualization.visualizer import SentimentVisualizer
    logger.info("成功导入SentimentVisualizer")
except ImportError as e:
    logger.warning(f"导入SentimentVisualizer失败: {e}，将使用简单备用可视化")
    # 创建一个简单的备用可视化类
    class SimpleVisualizer:
        def __init__(self):
            pass
        
        def plot_sentiment_distribution(self, df, sentiment_column='sentiment'):
            """简单显示情感分布统计"""
            logger.info("\n📊 情感分布统计:")
            sentiment_counts = df[sentiment_column].value_counts()
            for sentiment, count in sentiment_counts.items():
                logger.info(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")
            return None
        
        def plot_sentiment_by_category(self, df, category_column, sentiment_column='sentiment'):
            """简单显示按类别的情感分布"""
            if category_column in df.columns:
                logger.info(f"\n📊 按{category_column}的情感分布:")
                grouped = df.groupby([category_column, sentiment_column]).size().unstack(fill_value=0)
                for category in grouped.index:
                    logger.info(f"  {category}:")
                    for sentiment in grouped.columns:
                        count = grouped.loc[category, sentiment]
                        total = grouped.loc[category].sum()
                        if total > 0:
                            logger.info(f"    {sentiment}: {count} ({count/total*100:.1f}%)")
            return None
        
        def plot_sentiment_time_series(self, df, date_column, sentiment_column='sentiment'):
            """简单显示时间序列情感趋势"""
            if date_column in df.columns:
                logger.info(f"\n📊 情感时间趋势:")
                # 这里只是简单的日期计数
                df['date'] = pd.to_datetime(df[date_column]).dt.date
                daily_counts = df.groupby(['date', sentiment_column]).size().unstack(fill_value=0)
                for date in sorted(daily_counts.index):
                    if date in daily_counts.index:
                        logger.info(f"  {date}:")
                        for sentiment in daily_counts.columns:
                            count = daily_counts.loc[date, sentiment]
                            logger.info(f"    {sentiment}: {count}")
            return None
        
        def save_figures(self, figures, output_dir='./output'):
            """简单保存提示"""
            logger.info(f"提示: 由于缺少可视化库，无法保存图表到 {output_dir}")
            return []
    
    SentimentVisualizer = SimpleVisualizer
from config import OPENAI_API_KEY, PROCESSED_DATA_DIR

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='社交媒体数据高级情感分析工具 - 课堂演示版')
    
    # 数据参数
    parser.add_argument('--data', type=str, help='输入数据文件路径')
    parser.add_argument('--text-column', type=str, default='text', help='文本列名')
    parser.add_argument('--output', type=str, default='analysis_results.csv', help='输出结果文件路径')
    
    # 处理参数
    parser.add_argument('--language', type=str, default='chinese', choices=['english', 'chinese'], help='文本语言')
    parser.add_argument('--skip-processing', action='store_true', help='跳过数据预处理')
    
    # 分析参数 - 添加model_type选项，默认使用Hugging Face模型
    parser.add_argument('--model-type', type=str, default='huggingface', choices=['openai', 'huggingface'], 
                      help='模型类型 (默认: huggingface，使用免费模型)')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased-finetuned-sst-2-english', 
                      help='模型名称 (对于Hugging Face，默认使用免费情感分析模型)')
    parser.add_argument('--use-traditional', action='store_true', help='同时使用传统情感分析方法')
    parser.add_argument('--batch-size', type=int, default=100, help='批处理大小')
    
    # 可视化参数
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')
    parser.add_argument('--save-visualizations', action='store_true', help='保存可视化结果')
    
    # 示例数据
    parser.add_argument('--use-sample', action='store_true', help='使用示例数据')
    
    return parser.parse_args()

def load_sample_data() -> pd.DataFrame:
    """加载示例数据（如果没有真实数据）"""
    logger.info("生成示例数据...")
    
    # 创建示例数据
    sample_data = [
        {"text": "我非常喜欢这个产品，质量很好，价格也很合理！", "source": "微博", "date": "2023-06-15"},
        {"text": "服务态度很差，等了很久都没人理我，不会再来了。", "source": "微信", "date": "2023-06-16"},
        {"text": "产品一般般，没有特别惊艳的地方，但也没有明显缺点。", "source": "小红书", "date": "2023-06-17"},
        {"text": "物流速度太快了，昨天才下单今天就收到了，包装也很精美！", "source": "淘宝", "date": "2023-06-18"},
        {"text": "完全不符合描述，实物和图片差距很大，非常失望。", "source": "京东", "date": "2023-06-19"},
        {"text": "性价比很高，超出预期，推荐给大家！", "source": "抖音", "date": "2023-06-20"},
        {"text": "客服很耐心，解决问题很及时，给个赞！", "source": "拼多多", "date": "2023-06-21"},
        {"text": "质量问题很严重，用了一天就坏了，售后服务也不好。", "source": "天猫", "date": "2023-06-22"},
        {"text": "整体还不错，就是物流有点慢，其他都很满意。", "source": "闲鱼", "date": "2023-06-23"},
        {"text": "这是我用过的最差的产品，没有之一，强烈不推荐！", "source": "美团", "date": "2023-06-24"},
    ]
    
    df = pd.DataFrame(sample_data)
    
    # 保存示例数据
    sample_path = os.path.join(PROCESSED_DATA_DIR, 'sample_data.csv')
    df.to_csv(sample_path, index=False, encoding='utf-8')
    logger.info(f"示例数据已保存到: {sample_path}")
    
    return df

def preprocess_data(df: pd.DataFrame, text_column: str, language: str = 'chinese') -> pd.DataFrame:
    """预处理数据"""
    logger.info(f"开始预处理数据，语言: {language}")
    
    # 根据语言选择预处理器
    if language == 'chinese':
        preprocessor = ChineseTextPreprocessor()
    else:
        preprocessor = TextPreprocessor(language=language)
    
    # 处理文本
    processed_df = preprocessor.process_dataframe(
        df, 
        text_column,
        remove_urls=True,
        remove_usernames=True,
        remove_hashtags=False,  # 保留话题标签可能有用
        remove_emojis=True,
        lowercase=(language != 'chinese'),  # 中文不需要小写
        remove_punct=True,
        remove_stop=True,
        lemmatize_text=(language != 'chinese')  # 中文不需要词形还原
    )
    
    logger.info(f"数据预处理完成，保留了 {len(processed_df)} 条有效数据")
    return processed_df

def analyze_sentiment(df: pd.DataFrame, text_column: str, model_name: str, 
                     model_type: str = 'huggingface', use_traditional: bool = False, 
                     batch_size: int = 100) -> pd.DataFrame:
    """执行情感分析"""
    logger.info(f"🔍 开始情感分析，使用模型类型: {model_type}，模型: {model_name}")
    
    # 初始化分析结果DataFrame
    analyzed_df = df.copy()
    
    # 使用LLM进行情感分析 - 优先使用Hugging Face免费模型
    try:
        # 检查LLMSentimentAnalyzer是否可用
        if LLMSentimentAnalyzer is not None:
            try:
                # 创建LLM情感分析器，使用指定的模型类型和名称
                llm_analyzer = LLMSentimentAnalyzer(
                    model_name=model_name, 
                    model_type=model_type,
                    show_progress=True  # 显示进度条，适合课堂展示
                )
                
                # 执行分析
                logger.info(f"📊 正在使用{model_type.upper()}模型分析 {len(df)} 条文本...")
                analyzed_df = llm_analyzer.analyze_dataframe(df, text_column)
                logger.info("✅ LLM情感分析完成")
            except Exception as e:
                logger.error(f"❌ LLM分析器初始化或执行失败: {e}")
                logger.warning("❌ 将回退到传统分析器")
                use_traditional = True
        else:
            logger.warning("❌ LLM分析器不可用，将使用传统分析器")
            use_traditional = True
    except Exception as e:
        logger.error(f"❌ {model_type.upper()}模型分析失败: {e}")
        # 即使LLM分析失败，也尝试使用传统方法
        use_traditional = True
    
    # 使用传统方法进行情感分析
    if use_traditional:
        try:
            logger.info("📈 使用传统情感分析方法作为补充...")
            traditional_analyzer = TraditionalSentimentAnalyzer()
            analyzed_df = traditional_analyzer.analyze_dataframe(analyzed_df, text_column)
            logger.info("✅ 传统情感分析完成")
        except Exception as e:
            logger.error(f"❌ 传统情感分析失败: {e}")
    
    return analyzed_df

def visualize_results(df: pd.DataFrame, save: bool = False) -> None:
    """可视化分析结果"""
    logger.info("开始生成可视化结果")
    
    visualizer = SentimentVisualizer()
    
    # 提取情感分析相关列
    sentiment_column = 'sentiment_sentiment' if 'sentiment_sentiment' in df.columns else 'textblob_sentiment'
    score_column = 'sentiment_score' if 'sentiment_score' in df.columns else 'textblob_polarity'
    
    # 提取情绪列
    emotion_columns = [col for col in df.columns if col.startswith('emotion_score_')]
    
    # 确保有必要的列
    if sentiment_column not in df.columns:
        logger.warning(f"未找到情感列: {sentiment_column}")
        return
    
    # 创建综合仪表盘
    visualizer.create_summary_dashboard(
        df, 
        sentiment_column=sentiment_column,
        score_column=score_column,
        emotion_columns=emotion_columns if emotion_columns else [],
        text_column='text',
        save=save
    )
    
    logger.info("可视化完成")

def main():
    """主函数"""
    try:
        # 欢迎信息
        print("\n" + "="*60)
        print("🎉 社交媒体情感分析系统（课堂演示版）")
        print("📊 支持实时情感和情绪分析")
        print("💻 使用Hugging Face免费模型，无需API密钥")
        print("="*60 + "\n")
        
        # 解析参数
        args = parse_arguments()
        
        # 加载数据
        if args.use_sample:
            logger.info("🔄 正在加载示例数据...")
            df = load_sample_data()
        elif args.data:
            logger.info(f"📁 正在加载数据文件: {args.data}...")
            loader = DataLoader()
            df = loader.load_data(args.data)
        else:
            logger.error("❌ 请指定数据文件或使用 --use-sample 参数")
            return
        
        logger.info(f"✅ 成功加载数据，共 {len(df)} 条记录")
        logger.info(f"📝 文本列: {args.text_column}")
        logger.info(f"🧠 使用模型: {args.model_type} - {args.model}")
        
        # 数据预处理
        if not args.skip_processing:
            logger.info("🧹 正在预处理数据...")
            df = preprocess_data(df, args.text_column, args.language)
            text_column = f"{args.text_column}_cleaned"
            logger.info(f"✅ 数据预处理完成，使用清理后的文本列: {text_column}")
        else:
            text_column = args.text_column
            logger.info(f"⏩ 跳过预处理，直接使用原始文本列: {text_column}")
        
        # 执行情感分析 - 使用指定的模型类型（默认Hugging Face）
        analyzed_df = analyze_sentiment(
            df, 
            text_column,
            args.model,
            model_type=args.model_type,  # 添加模型类型参数
            use_traditional=args.use_traditional,
            batch_size=args.batch_size
        )
        
        # 保存结果 - 添加更友好的输出信息
        output_path = os.path.join(PROCESSED_DATA_DIR, args.output)
        analyzed_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"💾 分析结果已保存到: {output_path}")
        
        # 显示简要统计信息 - 适合课堂展示
        logger.info(f"\n📊 分析结果统计:")
        if 'sentiment_sentiment' in analyzed_df.columns:
            sentiment_counts = analyzed_df['sentiment_sentiment'].value_counts()
            logger.info(f"情感分布:")
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(analyzed_df)) * 100
                logger.info(f"  - {sentiment}: {count}条 ({percentage:.1f}%)")
        
        if 'emotion_primary_emotion' in analyzed_df.columns:
            emotion_counts = analyzed_df['emotion_primary_emotion'].value_counts()
            logger.info(f"主要情绪分布:")
            for emotion, count in emotion_counts.items():
                percentage = (count / len(analyzed_df)) * 100
                logger.info(f"  - {emotion}: {count}条 ({percentage:.1f}%)")
        
        # 可视化
        if args.visualize:
            logger.info("🎨 正在生成可视化结果...")
            visualize_results(analyzed_df, args.save_visualizations)
        
        logger.info("\n🎉 情感分析任务完成！")
        logger.info("✅ 系统使用Hugging Face免费模型，无需API密钥即可运行")
        logger.info("📱 适合课堂实物展示，提供实时分析和友好的可视化效果")
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise

if __name__ == "__main__":
    main()