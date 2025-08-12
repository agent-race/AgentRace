import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from agents.autogen_agent import autogen_agent
from eval.token_count import init_token_count
from eval.utils import change_log_file
init_token_count(offline_mode=True)
import weave
weave.init('autogen')
import asyncio
import nest_asyncio
import logging
import time

import logging
logging.basicConfig(
    filename='', 
    filemode='a',
    level=logging.INFO, 
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

async def react_gaia():
    from datasets import load_dataset
    test=load_dataset("data/GAIA/gaia.py",name="2023_all")
    test_data=test['validation']
    for gaia_data in test_data:
        agent = autogen_agent(model_name="gpt-4o",agent_type="ReAct")
        task_id = gaia_data['task_id']
        question = gaia_data['Question']
        level = gaia_data['Level']
        answer = gaia_data['Final answer']
        file_name = gaia_data['file_name']
        file_path = gaia_data['file_path']
        annotator_metadata = gaia_data['Annotator Metadata']
        if file_name!="":
            query = f'question: {question}, file_name: {file_name}, file_path: {file_path}'
        else:
            query = f'question: {question}'
        logging.info(f"omni_run start, query: {query}")
        try:
            res = await agent.omni_run(input_text=query)
            text_messages = res.messages[len(res.messages)-1].content
        except Exception as e:
            text_messages = e
            print("An error occurred:", e)
        logging.info(f"omni_run end, result: {text_messages}")
        logging.info(f"omni_run end, answer:{answer}")
        time.sleep(1)

async def react_human_eval():
    from datasets import load_dataset
    import builtins
    from autogen_agentchat.messages import TextMessage
    flag=0
    human_eval = load_dataset("openai_humaneval")
    for human_eval_data in human_eval['test']:
        task_id = human_eval_data['task_id']
        prompt = human_eval_data['prompt']
        canonical_solution = human_eval_data['canonical_solution']
        test = human_eval_data['test']
        entry_point = human_eval_data['entry_point']
        query = prompt+"\n# Complete the function. Only return code. No explanation, no comments, no markdown."
        agent = autogen_agent(model_name="gpt-4o",agent_type="ReAct")
        logging.info(f"omni_run start, query: {query}")
        try:
            res = await agent.omni_run(input_text=query)
            text_messages = res.messages[len(res.messages)-1].content
            logging.info(f"omni_run end, result: {text_messages}")
            logging.info(f"omni_run end, answer:{canonical_solution}")
            time.sleep(1)
        except Exception as e:
            print("An error occurred:", e)

async def moa_test():
    from data.alpaca_eval import load_alpaca_eval
    test=load_alpaca_eval()
    agent = autogen_agent(model_name=None,agent_type="MoA")
    for alpaca_eval_data in test:
        query=alpaca_eval_data["instruction"]
        answer=alpaca_eval_data["output"]
        logging.info(f"omni_run start, query: {query}")
        try:
            res = await agent.omni_run(input_text=query)
        except Exception as e:
            print("An error occurred:", e)
            res=e
        logging.info(f"omni_run end, result: {res}")
        logging.info(f"omni_run end, answer:{answer}")
        time.sleep(1)
        
def mmlu_RAG():
    change_log_file(os.path.join(parent_dir,"results","autogen","log","autogen_rag.log"))
    from data.mmlu import load_mmlu
    data = load_mmlu()
    from agents.autogen_agent import autogen_agent
    agent=autogen_agent("OpenAI","RAG")
    for i in range(0,len(data),4):
        query = data[i][0]
        try:
            logging.info(f"omni_run start, query: {query}")
            res = asyncio.run(agent.omni_run(query))
            logging.info(f"omni_run end, result: {res}")
            logging.info(f"omni_run end, answer:{data[i][1]}")
        except Exception as e:
            print("An error occurred:\n", e)
            print("The task number whith the error is:", i)
            agent=autogen_agent("OpenAI","RAG")
        time.sleep(1)

async def react_vqa():
    change_log_file(os.path.join(parent_dir,"results","autogen","log","autogen_vqa.log"))
    from data.vqa import load_vqa
    datas = load_vqa()
    i=0
    for data in datas:
        if i%5!=0:
            i+=1
            continue
        if i>5000:
            break
        i+=1
        agent = autogen_agent(model_name="gpt-4o",agent_type="ReAct")
        id = data['question_id']
        question = data['question']
        answer = data['answers']
        query = f"You need to naswer the question from the image.The path is:/root/AgentBench/data/VQA/image/{id}.png\nQuestion is: {question}\nAnswer the quetions just use easy words.Answer normalization (all chars lowercase, no period except as decimal point, number words —> digits, strip articles (a, an the)) "
        logging.info(f"omni_run start, query: {query}")
        try:
            res = await agent.omni_run(input_text=query)
            text_messages = res.messages[len(res.messages)-1].content
        except Exception as e:
            text_messages = e
            print("An error occurred:", e)
        logging.info(f"omni_run end, result: {text_messages}")
        logging.info(f"omni_run end, answer:{answer}")
        time.sleep(1)