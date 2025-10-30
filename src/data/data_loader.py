import os
import pandas as pd
import json
import logging
from typing import Union, List, Dict, Any

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """社交媒体数据加载器"""
    
    @staticmethod
    def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """加载CSV格式数据"""
        try:
            logger.info(f"Loading CSV data from {file_path}")
            return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    @staticmethod
    def load_json(file_path: str, lines: bool = False, **kwargs) -> pd.DataFrame:
        """加载JSON格式数据"""
        try:
            logger.info(f"Loading JSON data from {file_path}")
            if lines:
                # 按行读取JSON（类似Twitter数据流格式）
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f]
                return pd.DataFrame(data)
            else:
                # 普通JSON文件
                return pd.read_json(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            raise
    
    @staticmethod
    def load_excel(file_path: str, **kwargs) -> pd.DataFrame:
        """加载Excel格式数据"""
        try:
            logger.info(f"Loading Excel data from {file_path}")
            return pd.read_excel(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise
    
    @staticmethod
    def load_data(file_path: str, **kwargs) -> pd.DataFrame:
        """自动根据文件扩展名加载数据"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return DataLoader.load_csv(file_path, **kwargs)
        elif file_ext == '.json':
            return DataLoader.load_json(file_path, **kwargs)
        elif file_ext in ['.xlsx', '.xls']:
            return DataLoader.load_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    @staticmethod
    def load_sample_data(sample_name: str) -> pd.DataFrame:
        """加载示例数据集"""
        sample_path = os.path.join(RAW_DATA_DIR, f"{sample_name}.csv")
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample data {sample_name} not found at {sample_path}")
        return DataLoader.load_csv(sample_path)
    
    @staticmethod
    def save_processed_data(df: pd.DataFrame, filename: str) -> str:
        """保存处理后的数据"""
        save_path = os.path.join(PROCESSED_DATA_DIR, filename)
        try:
            df.to_csv(save_path, index=False, encoding='utf-8')
            logger.info(f"Processed data saved to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    @staticmethod
    def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """验证数据是否包含必要的列"""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    @staticmethod
    def get_available_datasets() -> List[str]:
        """获取可用的数据集列表"""
        available_datasets = []
        if os.path.exists(RAW_DATA_DIR):
            for file in os.listdir(RAW_DATA_DIR):
                if file.endswith(('.csv', '.json', '.xlsx', '.xls')):
                    available_datasets.append(os.path.splitext(file)[0])
        return available_datasets

# 示例用法
if __name__ == "__main__":
    # 列出可用的数据集
    print("可用的数据集:")
    for dataset in DataLoader.get_available_datasets():
        print(f"- {dataset}")