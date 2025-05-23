import logging

# 设置log文件，可以在main.py中写
logging.basicConfig(
    filename='results/test.log', 
    filemode='a',
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)


# omni_run开始时
query = 'This is the query'
logging.info(f"omni_run start, query: {query}")

# omni_run结束时
result = 'This is the omni_run result' # (只保存最终的结果，类型为str)
answer = 'This is the expected answer' # (只保存最终的结果，类型为str)
logging.info(f"omni_run end, result: {result}")
logging.info(f"omni_run end, answer:{answer}")

# 工具调用完成时
tool_name = 'pdf_tool' # 可选值：['pdf_tool', 'docx_tool', 'web_browser_tool', 'txt_tool', 'csv_tool', 'xlsx_tool', 'python_tool', 'vision_tool', 'video_tool', 'audio_tool']，工具名传入方式可以参考./eval/time_tools.py
tool_time = 10.5 # 单位：秒
logging.info(f"tool_name: {tool_name}, tool_time: {tool_time}") 

# Agent之间通信时
source_agent_name = 'Root Agent'
target_agent_name = 'Agent 1'
message_size = 1024 # Bytes (是所有的通信内容）
packaging_size = 1024 # Bytes (是所有的通信内容减去LLM Response部分)
logging.info(f"source_agent_name: {source_agent_name}, target_agent_name: {target_agent_name}, message_size: {message_size}, packaging_size: {packaging_size}")

# 数据库向量化完成时
embedding_time = 221.22 # 单位：秒
logging.info(f"database_vectorization time: {embedding_time}")

# retrieve完成时
retrieve_time = 1.89 # 单位：秒
logging.info(f"retrieve time: {retrieve_time}")

# rerank完成时 (如果有)
rerank_time = 1.89 # 单位：秒
logging.info(f"rerank time: {rerank_time}")


