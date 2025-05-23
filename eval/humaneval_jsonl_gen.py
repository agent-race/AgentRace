import os
import json

import jsonlines
json_dir = 'results/llamaindex-react-human-r3/'
num = 0

# jsonl_file = open('results/phidata-react-human_eval.jsonl', 'a')
import re

def extract_code_blocks(text):
    # 匹配 ``` 之间的内容（非贪婪模式，支持多行）
    try:
        pattern = r'```(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)  # re.DOTALL 使 . 匹配换行符
        return matches[0]
    except:
        return text


for idx, file in enumerate(os.listdir(json_dir)):
    if file[-1] == 'g':
        continue
    f = open(os.path.join(json_dir, file), 'r')
    data = json.load(f)
    if data.get('failure_run'):
        continue
    
    task_id = 'HumanEval/' + str(num)
    data['result']
    # {"task_id": "test/0", "completion": "\treturn 1"}
    # jsonl_file.write(str({"task_id": task_id, "completion": extract_code_blocks(data['result'])})+'\n')
    with jsonlines.open("results/llamaindex-react-human-r3.jsonl", mode="a") as writer:
        writer.write({"task_id": task_id, "completion": extract_code_blocks(data['result'])})
    num += 1

# jsonl_file.close()