"""
生成评估输入
"""

import json
import os

res = []
json_dir = 'results/llamaindex-moa-alpacaeval'


all_ins = []

for idx, file in enumerate(os.listdir(json_dir)):
    if file[-1] == 'g':
        continue
    f = open(os.path.join(json_dir, file), 'r')
    data = json.load(f)
    if data.get('failure_run'):
        print('failure:',idx)
        continue
    all_ins.append(data['query'])
    res.append(
        {
            "instruction": None, # 后面会添加
            "output":data['result'],
            # "generator":"example",
            # "dataset":"helpful_base",
            # "datasplit":"eval"
        }

    )

"""
将instruction和dataset对齐
"""


from datasets import load_dataset

alpacaeval = load_dataset("tatsu-lab/alpaca_eval", 'alpaca_eval', trust_remote_code=True)['eval']
for idx, r in enumerate(res):

    r["instruction"] = alpacaeval[idx]['instruction']


save_json_content = json.dumps(res)
f = open(json_dir + '-eval.json', 'w')
f.write(save_json_content)
f.close()



"""
生成评估结果
"""

import httpx
import openai

client = httpx.Client(verify=False)  # 禁用 SSL 验证
openai.default_http_client = client  # 传递给 OpenAI 客户端



from alpaca_eval import evaluate

res = evaluate(model_outputs='results/llamaindex-moa-alpacaeval-eval.json', 
                      is_return_instead_of_print=True, 
                    #   annotators_config="alpaca_eval_gpt4",
                      annotators_config="alpaca_eval_gpt4_turbo_fn",
                      output_path='./llamaindex-moa-alpacaevel',
                      
                    #   is_recompute_metrics_only=True
                )

