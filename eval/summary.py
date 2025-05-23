import weave
import re
import json
import os
from collections import defaultdict
from datetime import datetime

@DeprecationWarning
def process_weave(project_name):
    client = weave.init(project_name)
    all_calls = client.get_calls()
    for call in all_calls:
        if 'openai.chat.completions.create' in call.to_dict()['op_name']:
            print(f"id: {call.id}")
            print(f"trace_id: {call.trace_id}")
            print(f"usage: {call.summary['usage']}")
            print(f"start time: {call.started_at}")
            print(f"send time: {call.ended_at}")


def process_log(file_path, save_path, custom_time_format=None):
    current_entry = None
    
    log_entry_pattern = custom_time_format if custom_time_format is not None else re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}.*')
    
    os.makedirs(save_path, exist_ok=True)
    root_f = open(os.path.join(save_path, file_path.split('/')[-1]), 'a', encoding='utf-8')

    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in f:
            if log_entry_pattern.match(line):
                if current_entry and "- root - INFO -" in current_entry:
                    root_f.write(current_entry)
                current_entry = line
            else:
                if current_entry is not None:
                    current_entry += line
        
        if current_entry and "- root - INFO -" in current_entry:
            root_f.write(current_entry)
    root_f.close()
            

def read_logs_streaming(file_path, custom_time_format=None):
    current_log = []  # 缓存当前log的多行内容

    log_pattern = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - root - INFO -') if custom_time_format is None else custom_time_format

    with open(file_path, 'r',encoding='utf-8') as f:
        for line in f:
            if log_pattern.match(line) and current_log:
                # 如果当前行是新的log开头，且缓存不为空，则返回已缓存的完整log
                yield ''.join(current_log).strip()  # 合并为字符串并去除末尾换行
                current_log = [line]  # 开始新的log缓存
            else:
                current_log.append(line)  # 继续缓存当前log的多行
        
        # 返回最后一条log（文件结束时）
        if current_log:
            yield ''.join(current_log).strip()



def process_root_log(file_path, save_path, reverse=False, weave_op=True ,include_op_name=False, custom_time_format=None, time_lens=23, read_format=re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - root - INFO -')):
    process_num = 0
    already_start = False
    llm_to_file = {}
    llm_to_time = {}
    
    omni_start_time = None

    omni_start_time_id = {}
    omni_start_time_id_file = {}
    log_time_format = custom_time_format if custom_time_format is not None else "%Y-%m-%d %H:%M:%S,%f"
    current_database_vectorization_time =None
    
    
    for current_log in read_logs_streaming(file_path, custom_time_format=read_format):
        
        # omni_run开始时
        if 'omni_run start' in current_log[:time_lens+40]:
            if already_start: 
                # 上一个执行失败了
                current_json_content['failure_run'] = True
                save_json_content = json.dumps(current_json_content)
                f = open(os.path.join(save_path, str(process_num).zfill(5)+'.json'), 'w')
                f.write(save_json_content)
                f.close()
                already_start = False
                process_num += 1


            current_json_content = {
                "query": None,
                "answer": None,
                "result": None,
                "tools": {},
                "retrieve_time": [],
                "rerank_time": [],
                "communication_size": {},
                "communication_pack": {},
                "total_time": [],
                "llm_time": {},
                "llm_tokens": {}
            }
            if current_database_vectorization_time is not None:
                current_json_content['database_vectorization_time'] = current_database_vectorization_time
            current_json_content['query'] = current_log[time_lens+40:]
            if not weave_op:
                omni_start_time = datetime.strptime(current_log[:time_lens], log_time_format)
            already_start = True

        # omni_run结束时
        if 'omni_run end, result' in current_log[:time_lens+39]:
            current_json_content['result'] = current_log[time_lens+39:]
            if not weave_op:
                current_json_content['total_time'] = [(datetime.strptime(current_log[:time_lens], log_time_format) - omni_start_time).total_seconds()]
                omni_start_time = None

        
        # omni_run结束时
        if 'omni_run end, answer' in current_log[:time_lens+39]:
            current_json_content['answer'] = current_log[time_lens+38:]
            save_json_content = json.dumps(current_json_content)
            f = open(os.path.join(save_path, str(process_num).zfill(5)+'.json'), 'w')
            f.write(save_json_content)
            f.close()
            already_start = False
            process_num += 1
        
        # 工具调用完成时
        if 'tool_name' in current_log[:27+time_lens]:
            match = re.search(
                r'tool_name: (?P<tool_name>\w+),\s+tool_time: (?P<tool_time>[\d.]+)', 
                current_log
            )
            tool_name = match.group('tool_name')
            tool_time = float(match.group('tool_time')) 
            if current_json_content['tools'].get(tool_name) is not None:
                current_json_content['tools'][tool_name].append(tool_time)
            else:
                current_json_content['tools'][tool_name] = [tool_time]
        
        # Agent之间通信时
        if 'source_agent_name' in current_log[:time_lens+34]:
            pattern = re.compile(
                r'source_agent_name: (?P<source_agent_name>[^,]+), '
                r'target_agent_name: (?P<target_agent_name>[^,]+), '
                r'message_size: (?P<message_size>\d+), '
                r'packaging_size: (?P<packaging_size>\d+)'
            )
    
            match = pattern.search(current_log)
            key = match.group("source_agent_name") + '_to_' + match.group("target_agent_name")
            
            if current_json_content['communication_size'].get(key) is not None:
                current_json_content['communication_size'][key] += eval(match.group("message_size"))
            else:
                current_json_content['communication_size'][key] = eval(match.group("message_size"))

            if current_json_content['communication_pack'].get(key) is not None:
                current_json_content['communication_pack'][key] += eval(match.group("packaging_size"))
            else:
                current_json_content['communication_pack'][key] = eval(match.group("packaging_size"))

        # 数据库向量化完成时，只有一个，只放在最一开始json里面
        if 'database_vectorization time' in current_log[:time_lens+44]:
            current_database_vectorization_time = eval(current_log[time_lens+46:].strip())
            
        
        # retrieve完成时
        if 'retrieve time' in current_log[:time_lens+30]:
            current_json_content['retrieve_time'].append(current_log[time_lens+31:].strip())

        # rerank完成时
        if 'rerank time' in current_log[:time_lens+28]:
            current_json_content['rerank_time'].append(current_log[29+time_lens:].strip())

        # LLM start
        if 'LLM completion start' in current_log[:37+time_lens]:
            if include_op_name:
                pattern = re.compile(
                    r'LLM completion start, '
                    r'id:(?P<id>[a-f0-9-]+), '
                    r'timestamp: (?P<timestamp>\d+\.\d+), '  # 浮点数
                    r'is_omni_run_trace: (?P<is_omni_run_trace>True|False), '  # 布尔值
                    r'op_name:\s*(?P<op_name>\S+)'
                )
                match = pattern.search(current_log)
                id = match.group("id")
                start_time_stamp = eval(match.group("timestamp"))
                is_omni_run_trace = eval(match.group("is_omni_run_trace"))
                op_name = match.group("op_name")
                if is_omni_run_trace:
                    omni_start_time_id[id] = eval(match.group("timestamp"))
                    omni_start_time_id_file[id] = str(process_num).zfill(5)+'.json'
                    continue
                if 'openai.chat.completions' not in op_name:
                    if 'omni_run' in op_name:
                        omni_start_time_id[id] = eval(match.group("timestamp"))
                        omni_start_time_id_file[id] = str(process_num).zfill(5)+'.json'
                    continue
            else:
                pattern = re.compile(
                    r'LLM completion start, '
                    r'id:(?P<id>[a-f0-9-]+), '
                    r'timestamp: (?P<timestamp>\d+\.\d+), '  # 浮点数
                    r'is_omni_run_trace: (?P<is_omni_run_trace>True|False)'  # 布尔值
                )

                match = pattern.search(current_log)
                id = match.group("id")
                is_omni_run_trace = eval(match.group("is_omni_run_trace"))
                start_time_stamp = eval(match.group("timestamp"))
                if is_omni_run_trace:
                    omni_start_time_id[id] = eval(match.group("timestamp"))
                    omni_start_time_id_file[id] = str(process_num).zfill(5)+'.json'
                    continue
                
            llm_to_file[id] = str(process_num).zfill(5)+'.json'
            llm_to_time[id] = start_time_stamp
            
            
        
        # LLM end
        if 'LLM name' in current_log[:time_lens+27]:
            pattern = re.compile(
                    r'LLM name: (?P<llm_name>[^,]+), '
                    r'prompt_tokens: (?P<prompt_tokens>[^,]+), '
                    r'completion_tokens: (?P<completion_tokens>\d+), '
                    r'total_tokens: (?P<total_tokens>\d+), '
                    r'id: (?P<id>[a-f0-9-]+), '
                    r'timestamp: (?P<end_time_stamp>\d+\.\d+)'
                )
            match = pattern.search(current_log)
            prompt_tokens = eval(match.group("prompt_tokens"))
            completion_tokens = eval(match.group("completion_tokens"))
            total_tokens = eval(match.group("total_tokens"))
            llm_name = match.group("llm_name")
            id = match.group("id")
            end_time_stamp = eval(match.group("end_time_stamp"))
            if llm_to_file.get(id) is not None:
                if reverse:
                    prompt_tokens, completion_tokens = completion_tokens, prompt_tokens
                start_file = llm_to_file[id]
                if start_file == str(process_num).zfill(5)+'.json':
                    if current_json_content['llm_time'].get(llm_name) is not None:
                        current_json_content['llm_time'][llm_name].append(end_time_stamp - llm_to_time[id])
                        current_json_content['llm_tokens'][llm_name]['prompt'].append(prompt_tokens)
                        current_json_content['llm_tokens'][llm_name]['completion'].append(completion_tokens)
                        current_json_content['llm_tokens'][llm_name]['total'].append(total_tokens)
                    else:
                        current_json_content['llm_time'][llm_name] = [end_time_stamp - llm_to_time[id]]
                        current_json_content['llm_tokens'][llm_name] = {
                            "prompt": [prompt_tokens],
                            "completion": [completion_tokens],
                            "total": [total_tokens]
                        }

                else:
                    f = open(os.path.join(save_path, start_file), 'r')
                    pre_json_content = json.load(f)
                    f.close()
                    if pre_json_content['llm_time'].get(llm_name) is not None:
                        pre_json_content['llm_time'][llm_name].append(end_time_stamp - llm_to_time[id])
                        pre_json_content['llm_tokens'][llm_name]['prompt'].append(prompt_tokens)
                        pre_json_content['llm_tokens'][llm_name]['completion'].append(completion_tokens)
                        pre_json_content['llm_tokens'][llm_name]['total'].append(total_tokens)
                    else:
                        pre_json_content['llm_time'][llm_name] = [end_time_stamp - llm_to_time[id]]
                        pre_json_content['llm_tokens'][llm_name] = {
                            "prompt": [prompt_tokens],
                            "completion": [completion_tokens],
                            "total": [total_tokens]
                        }
                    save_json_content = json.dumps(pre_json_content)
                    f = open(os.path.join(save_path, start_file), 'w')
                    f.write(save_json_content)
                    f.close()
            elif omni_start_time_id_file.get(id) is not None:
                start_file = omni_start_time_id_file[id]
                if start_file == str(process_num).zfill(5)+'.json':
                    current_json_content['total_time'] = [eval(match.group('end_time_stamp')) - omni_start_time_id[id]]
                else:
                    f = open(os.path.join(save_path, start_file), 'r')
                    pre_json_content = json.load(f)
                    f.close()
                    pre_json_content['total_time'] = [eval(match.group('end_time_stamp')) - omni_start_time_id[id]]
                    save_json_content = json.dumps(pre_json_content)
                    f = open(os.path.join(save_path, start_file), 'w')
                    f.write(save_json_content)
                    f.close()
                    eval(match.group('end_time_stamp')) - omni_start_time_id[id]

            
def summary_log(
        file_path, 
        save_path,
        reverse,
        weave_op,
        millisecond,
        include_op_name,
        time_lens
        ):
    """
    Args:
        file_path: .log文件的位置
        reverse: 是否需要反转prompt_tokens和completion_token
        
        ----是否在omni_run上面加weave.op装饰器----
        weave_op: 有没有加装饰器，可以检查一下日志里面is_omni_run字段有没有True


        ----如果没有在omni_run上面加weave.op将使用日志时间戳测总时间，需要指明时间戳格式----
        millisecond: 日志中的时间格式是否包含毫秒，如果是“2025-05-09 15:34:10,459 - root - INFO -”这种格式就设为True

        include_op_name: LLM completion start的日志中是否存在include_op_name, 如有置为True
        time_lens: 日志中的时间前缀的占用的位数
    """
    
    time_format_log = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}') if millisecond else re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
    time_format_root_log = "%Y-%m-%d %H:%M:%S,%f" if millisecond else "%Y-%m-%d %H:%M:%S"
    time_lens = 23 if millisecond else 19
    
    process_log(file_path=file_path, save_path=save_path, custom_time_format=time_format_log)
    process_root_log(file_path=os.path.join(save_path, file_path.split('/')[-1]), save_path=save_path, weave_op=weave_op, reverse=reverse, time_lens=time_lens, include_op_name=include_op_name, custom_time_format=time_format_root_log, read_format=time_format_log)





# summary_log('results/llamaindex-moa-alpacaeval.log', 'results/llamaindex-moa-alpacaeval', weave_op=True, reverse=True, millisecond=True, include_op_name=False, time_lens=23)
# summary_log('results/llamaindex-react-gaia.log', 'results/llamaindex-react-gaia/', weave_op=True, reverse=True, millisecond=True, include_op_name=True, time_lens=23)
# summary_log('results/llamaindex-react-human-r2.log', 'results/llamaindex-react-human-r2/', weave_op=True, reverse=False, millisecond=True, include_op_name=True, time_lens=23)


# # extract root log
# process_log('results/example.log', 'results/example/')

# # log to json
# process_root_log('./results/example/example.log', './results/example/', reverse=True, include_op_name=False, custom_time_format=None, time_lens=23)


# extract root log
# process_log('results/agentbench_langchain_moa.log', 'results/agentbench_langchain_moa/', custom_time_format=re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*'))

# log to json