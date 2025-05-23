import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from agents.langchain_agent import LangchainAgent
from eval.token_count import init_token_count
init_token_count(offline_mode=True)
import weave
weave.init('langchain')
import asyncio
import nest_asyncio
import logging
import time

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
    agent = LangchainAgent(model_name="OpenAI",agent_type="ReAct")
    for gaia_data in test_data:
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
            res = await agent.omni_run(question=query)
        except Exception as e:
            res=e
            print("An error occurred:", e)
        logging.info(f"omni_run end, result: {res}")
        logging.info(f"omni_run end, answer:{answer}")
        time.sleep(1)

async def react_human_eval():
    from datasets import load_dataset
    human_eval = load_dataset("openai_humaneval")
    for human_eval_data in human_eval['test']:
        agent = LangchainAgent(model_name="OpenAI",agent_type="ReAct")
        task_id = human_eval_data['task_id']
        prompt = human_eval_data['prompt']
        canonical_solution = human_eval_data['canonical_solution']
        test = human_eval_data['test']
        entry_point = human_eval_data['entry_point']
        query = prompt+"\n# Complete the function. Only return code. No explanation, no comments, no markdown."
        logging.info(f"omni_run start, query: {query}")
        try:
            res = await agent.omni_run(question=query)
            logging.info(f"omni_run end, result: {res}")
            logging.info(f"omni_run end, answer:{canonical_solution}")
            time.sleep(1)
        except Exception as e:
            print("An error occurred:", e)

async def moa_test():
    from data.alpaca_eval import load_alpaca_eval
    test=load_alpaca_eval()
    agent = LangchainAgent(model_name=None,agent_type="MoA")
    for alpaca_eval_data in test:
        query=alpaca_eval_data["instruction"]
        answer=alpaca_eval_data["output"]
        logging.info(f"omni_run start, query: {query}")
        try:
            res = await agent.omni_run(question=query)
            logging.info(f"omni_run end, result: {res}")
        except Exception as e:
            print("An error occurred:", e)
        logging.info(f"omni_run end, answer:{answer}")
        time.sleep(1)

def mmlu_test():
    from eval.utils import change_log_file
    change_log_file(os.path.join(parent_dir,"results","langchain","log","langchain_rag.log"))
    from data.mmlu import load_mmlu
    data = load_mmlu()
    agent=LangchainAgent("OpenAI","RAG")
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
        time.sleep(1)