import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Union, Any
from wordcloud import WordCloud

from config import EMOTION_CLASSES

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class SentimentVisualizer:
    """情感分析结果可视化器"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"可视化结果将保存到: {self.output_dir}")
    
    def sentiment_distribution(self, df: pd.DataFrame, sentiment_column: str, 
                              title: str = "情感分布", 
                              show: bool = True, 
                              save: bool = False, 
                              filename: str = "sentiment_distribution.png") -> plt.Figure:
        """绘制情感分布饼图"""
        plt.figure(figsize=(10, 6))
        
        # 计算情感分布
        sentiment_counts = df[sentiment_column].value_counts()
        
        # 创建饼图
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'])
        plt.axis('equal')  # 确保饼图是圆的
        plt.title(title, fontsize=15)
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"情感分布图已保存到: {filepath}")
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def sentiment_score_distribution(self, df: pd.DataFrame, score_column: str, 
                                   title: str = "情感得分分布",
                                   show: bool = True,
                                   save: bool = False,
                                   filename: str = "sentiment_score_distribution.png") -> plt.Figure:
        """绘制情感得分分布图"""
        plt.figure(figsize=(12, 6))
        
        # 创建直方图
        sns.histplot(df[score_column], kde=True, bins=30, color='#66b3ff')
        plt.axvline(df[score_column].mean(), color='red', linestyle='--', label=f'平均值: {df[score_column].mean():.2f}')
        plt.title(title, fontsize=15)
        plt.xlabel('情感得分', fontsize=12)
        plt.ylabel('频率', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"情感得分分布图已保存到: {filepath}")
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def emotion_radar_chart(self, df: pd.DataFrame, emotion_columns: List[str], 
                           title: str = "情绪雷达图",
                           show: bool = True,
                           save: bool = False,
                           filename: str = "emotion_radar_chart.png") -> go.Figure:
        """绘制情绪雷达图"""
        # 计算每种情绪的平均得分
        emotion_means = df[emotion_columns].mean()
        
        # 准备雷达图数据
        categories = [EMOTION_CLASSES.get(col.split('_')[-1], col) for col in emotion_columns]
        values = emotion_means.tolist()
        
        # 闭合雷达图
        values += values[:1]  # 重复第一个值以闭合雷达图
        categories += categories[:1]
        
        # 创建雷达图
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='平均情绪得分'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=title
        )
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            fig.write_image(filepath, width=800, height=600, scale=2)
            logger.info(f"情绪雷达图已保存到: {filepath}")
        
        if show:
            fig.show()
        
        return fig
    
    def emotion_bar_chart(self, df: pd.DataFrame, emotion_columns: List[str], 
                         title: str = "情绪得分对比",
                         show: bool = True,
                         save: bool = False,
                         filename: str = "emotion_bar_chart.png") -> go.Figure:
        """绘制情绪得分条形图"""
        # 计算每种情绪的平均得分
        emotion_means = df[emotion_columns].mean()
        
        # 准备数据
        categories = [EMOTION_CLASSES.get(col.split('_')[-1], col) for col in emotion_columns]
        values = emotion_means.tolist()
        
        # 创建条形图
        fig = px.bar(
            x=categories,
            y=values,
            color=categories,
            text_auto='.2f',
            title=title,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.update_layout(
            xaxis_title='情绪类型',
            yaxis_title='平均得分',
            yaxis=dict(range=[0, 1])
        )
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            fig.write_image(filepath, width=800, height=600, scale=2)
            logger.info(f"情绪条形图已保存到: {filepath}")
        
        if show:
            fig.show()
        
        return fig
    
    def create_wordcloud(self, texts: List[str], title: str = "关键词云",
                        show: bool = True,
                        save: bool = False,
                        filename: str = "wordcloud.png") -> plt.Figure:
        """创建词云图"""
        # 合并所有文本
        all_text = ' '.join(texts)
        
        # 创建词云
        plt.figure(figsize=(15, 10))
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                             font_path='simhei.ttf' if os.name == 'nt' else None,
                             max_words=200, contour_width=3, contour_color='steelblue').generate(all_text)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title, fontsize=15)
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"词云图已保存到: {filepath}")
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def sentiment_by_category(self, df: pd.DataFrame, category_column: str, sentiment_column: str,
                             title: str = "不同类别的情感分布",
                             show: bool = True,
                             save: bool = False,
                             filename: str = "sentiment_by_category.png") -> plt.Figure:
        """按类别分析情感分布"""
        plt.figure(figsize=(12, 6))
        
        # 创建分组条形图
        sns.countplot(x=category_column, hue=sentiment_column, data=df)
        plt.title(title, fontsize=15)
        plt.xlabel('类别', fontsize=12)
        plt.ylabel('数量', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='情感')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"按类别情感分布图已保存到: {filepath}")
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def sentiment_time_series(self, df: pd.DataFrame, date_column: str, sentiment_column: str,
                            freq: str = 'D',
                            title: str = "情感随时间变化趋势",
                            show: bool = True,
                            save: bool = False,
                            filename: str = "sentiment_time_series.png") -> plt.Figure:
        """绘制情感随时间变化的趋势图"""
        # 确保日期列是datetime类型
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        
        # 按频率分组并计算情感得分的平均值
        time_series = df_copy.set_index(date_column).resample(freq)[sentiment_column].mean()
        
        plt.figure(figsize=(14, 7))
        time_series.plot(color='#66b3ff')
        plt.title(title, fontsize=15)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel(f'平均{sentiment_column}', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"情感时间序列图已保存到: {filepath}")
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def create_summary_dashboard(self, df: pd.DataFrame, sentiment_column: str, score_column: str,
                                emotion_columns: List[str], text_column: Optional[str] = None,
                                save: bool = True,
                                title: str = "情感分析摘要仪表盘") -> Dict[str, Any]:
        """创建综合分析仪表盘"""
        logger.info("创建情感分析摘要仪表盘...")
        
        results = {}
        
        # 情感分布
        results['sentiment_distribution'] = self.sentiment_distribution(
            df, sentiment_column, show=False, save=save,
            filename="dashboard_sentiment_distribution.png"
        )
        
        # 情感得分分布
        results['sentiment_score_distribution'] = self.sentiment_score_distribution(
            df, score_column, show=False, save=save,
            filename="dashboard_sentiment_score_distribution.png"
        )
        
        # 情绪雷达图
        results['emotion_radar'] = self.emotion_radar_chart(
            df, emotion_columns, show=False, save=save,
            filename="dashboard_emotion_radar.png"
        )
        
        # 情绪条形图
        results['emotion_bar'] = self.emotion_bar_chart(
            df, emotion_columns, show=False, save=save,
            filename="dashboard_emotion_bar.png"
        )
        
        # 如果提供了文本列，创建词云
        if text_column and text_column in df.columns:
            # 按情感过滤文本
            positive_texts = df[df[sentiment_column] == '正面'][text_column].tolist()
            negative_texts = df[df[sentiment_column] == '负面'][text_column].tolist()
            
            # 正面情感词云
            results['positive_wordcloud'] = self.create_wordcloud(
                positive_texts, title="正面情感关键词云", show=False, save=save,
                filename="dashboard_positive_wordcloud.png"
            )
            
            # 负面情感词云
            results['negative_wordcloud'] = self.create_wordcloud(
                negative_texts, title="负面情感关键词云", show=False, save=save,
                filename="dashboard_negative_wordcloud.png"
            )
        
        logger.info("仪表盘创建完成")
        return results

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    import numpy as np
    
    # 生成随机数据
    np.random.seed(42)
    n_samples = 100
    
    sample_df = pd.DataFrame({
        'text': [f"这是示例文本 {i}" for i in range(n_samples)],
        'sentiment': np.random.choice(['正面', '负面', '中性'], n_samples),
        'sentiment_score': np.random.uniform(-1, 1, n_samples),
        'emotion_score_喜悦': np.random.uniform(0, 1, n_samples),
        'emotion_score_愤怒': np.random.uniform(0, 1, n_samples),
        'emotion_score_悲伤': np.random.uniform(0, 1, n_samples),
        'emotion_score_恐惧': np.random.uniform(0, 1, n_samples),
        'emotion_score_惊讶': np.random.uniform(0, 1, n_samples),
        'emotion_score_厌恶': np.random.uniform(0, 1, n_samples),
        'category': np.random.choice(['产品', '服务', '价格', '质量'], n_samples),
        'date': pd.date_range(start='2023-01-01', periods=n_samples)
    })
    
    # 创建可视化器
    visualizer = SentimentVisualizer()
    
    # 演示各种可视化
    print("演示情感分布图...")
    visualizer.sentiment_distribution(sample_df, 'sentiment')
    
    print("\n演示情感得分分布图...")
    visualizer.sentiment_score_distribution(sample_df, 'sentiment_score')
    
    emotion_cols = ['emotion_score_喜悦', 'emotion_score_愤怒', 'emotion_score_悲伤', 
                   'emotion_score_恐惧', 'emotion_score_惊讶', 'emotion_score_厌恶']
    
    print("\n演示情绪条形图...")
    visualizer.emotion_bar_chart(sample_df, emotion_cols)
    
    print("\n演示按类别情感分布...")
    visualizer.sentiment_by_category(sample_df, 'category', 'sentiment')
    
    print("\n演示情感时间序列...")
    visualizer.sentiment_time_series(sample_df, 'date', 'sentiment_score')
    
    print("\n创建综合仪表盘...")
    visualizer.create_summary_dashboard(
        sample_df, 
        sentiment_column='sentiment',
        score_column='sentiment_score',
        emotion_columns=emotion_cols,
        text_column='text',
        save=True
    )
    
    print("\n所有可视化演示完成！")