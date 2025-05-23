import os
import logging

def change_log_file(new_path):
    root_logger = logging.getLogger()
    
    # 移除旧的 FileHandler
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()  # 确保旧文件正常关闭
            root_logger.removeHandler(handler)
    
    # 添加新的 FileHandler
    file_handler = logging.FileHandler(new_path)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    root_logger.addHandler(file_handler)
