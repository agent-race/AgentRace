import sys
import os
import json
import tarfile
import weave
import shutil
import logging
import time
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from data.mmlu import load_mmlu,download_mmlu
from agents.agentscope_agent import AgentScopeAgent,AgentScopeAgentRun
from eval.summary import summary_log
from eval.token_count import init_token_count
init_token_count(offline_mode=True)
import os
import dotenv
dotenv.load_dotenv(os.path.join(parent_dir,".env"))


openai_api_key=os.getenv("OPENAI_API_KEY")

def rag_test():
    dir=os.path.join(parent_dir,"data","MMLU","test")
    if not os.path.exists(dir):
        download_mmlu()
        with tarfile.open(os.path.join(parent_dir,"data","MMLU","data.tar"), 'r') as tar:
            tar.extractall(path=os.path.join(parent_dir,"data","MMLU"))
        for root, dirs, files in os.walk(os.path.join(parent_dir,"data","MMLU","data")):
            rel_path = os.path.relpath(root, os.path.join(parent_dir,"data","MMLU","data"))
            target_root = os.path.join(os.path.join(parent_dir,"data","MMLU"), rel_path)
            os.makedirs(target_root, exist_ok=True)
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_root, file)
                shutil.move(src_file, dst_file)

        os.remove(os.path.join(parent_dir,"data","MMLU","data.tar"))
        shutil.rmtree(os.path.join(parent_dir,"data","MMLU","data"))
    test=load_mmlu()
    [agent,embedding_time]=AgentScopeAgent(
        agent_type="RAG",
        api_key=openai_api_key,
    )
    #print(embedding_time)#92.47222409397364s
    file_path=os.path.join(parent_dir,"results","agentscope","log","rag.json")
    numa=0
    numb=0
    numc=0
    numd=0
    nume=0
    numf=0
    for i in range(len(test)): 
        if i%4==0 :
            max_retries = 5
            retry_count = 0
            while retry_count < max_retries:
                try:
                    res=AgentScopeAgentRun(
                        agent_type="RAG",
                        agent=agent,
                        query=test[i][0],
                    )
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"Retry #{retry_count} failed:{e}")
                    time.sleep(2)
            else:
                print("Multiple retries failed. Please check the issue. The task number with the error is ",i)

            
            for item in res[4]["text_and_embedding"]:
                if item["model_name"]=="gpt-4o-2024-08-06":
                    gpt={
                        "prompt": [item["prompt_tokens"]-numa],
                        "completion": [item["completion_tokens"]-numb],
                        "total": [item["total_tokens"]-numc],
                    }
                    numa=item["prompt_tokens"]
                    numb=item["completion_tokens"]
                    numc=item["total_tokens"]
                if item["model_name"]=="text-embedding-3-large":
                    large={
                        "prompt": [item["prompt_tokens"]-numd],
                        "completion": [item["completion_tokens"]-nume],
                        "total": [item["total_tokens"]-numf],
                    }
                    numd=item["prompt_tokens"]
                    nume=item["completion_tokens"]
                    numf=item["total_tokens"]
            if os.path.getsize(file_path) > 0:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
            else:
                data_list=[]
            output={
                "query": test[i][0],
                "answer": test[i][1],
                "result": res[0],
                "tools" : {
                },
                "embedding_time":[embedding_time],
                "retrieve_time": [res[2]],
                "rerank_time": [],
                "communication_size": {
                    "root_to_agent1": 0,
                    "root_to_agent2": 0,
                    "root_to_agent3":0,
                    "agent1_to_aggregation" :0,
                    "agent2_to_aggregation":0,
                    "agent3_to_aggregation":0,
                },
                "communication_pack": {
                    "root_to_agent1": 0,
                    "root_to_agent2": 0,
                    "root_to_agent3":0,
                    "agent1_to_aggregation" :0,
                    "agent2_to_aggregation":0,
                    "agent3_to_aggregation":0,
                },
                
                "total_time": [res[1]],

                
                "llm_time": {
                    "gpt-4o":res[3],
                },

                "llm_tokens": {
                    "gpt-4o":gpt,
                    "text-embedding-3-large":large,
                }
            }
            data_list.append(output)
            print(i)
            with open(file_path, 'w', encoding='utf-8') as f:    
                json.dump(data_list, f, ensure_ascii=False, indent=2)

    #breakdown
    os.makedirs(os.path.join(parent_dir,"results","agentscope","rag"), exist_ok=True)            
    with open(os.path.join(parent_dir,"results","agentscope","log","rag.json"), 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    for i in range(len(data_list)):
        out_path = os.path.join(parent_dir,"results","agentscope","rag", f"{i:05d}.json")
        with open(out_path, 'w', encoding='utf-8') as f:    
            json.dump(data_list[i], f, ensure_ascii=False, indent=2)


from data.alpaca_eval import load_alpaca_eval

def moa_test(scale):
    logging.basicConfig(
        filename=os.path.join(parent_dir,"results","agentscope","log","moa"+str(scale)+".log"), 
        filemode='a',
        level=logging.INFO, 
        format='%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    test=load_alpaca_eval()
    agents=AgentScopeAgent(
        agent_type="MoA",
        moa_scale=scale,
        api_key=openai_api_key
    )

    
    weave.init(project_name="agentscope_moa_test")

    if scale==3:
        for i in range(len(test)):
            
            max_retries = 5
            retry_count = 0

            while retry_count < max_retries:
                try:
                    logging.info(f"omni_run start, query: {test[i]['instruction']}")
                    res=AgentScopeAgentRun(
                        agent=agents,
                        agent_type="MoA",
                        query=test[i]["instruction"],
                        moa_scale=scale,
                    )
                    logging.info(f"omni_run end, result: {res}")
                    logging.info(f"omni_run end, answer:{test[i]['output']}")
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"Retry #{retry_count} failed:{e}")
                    time.sleep(2)
            else:
                print("Multiple retries failed. Please check the issue. The task number with the error is ",i)
                exit(0)
    else:
        for i in range(100):
            
            max_retries = 5
            retry_count = 0

            while retry_count < max_retries:
                try:
                    logging.info(f"omni_run start, query: {test[i]['instruction']}")
                    res=AgentScopeAgentRun(
                        agent=agents,
                        agent_type="MoA",
                        query=test[i]["instruction"],
                        moa_scale=scale
                    )
                    logging.info(f"omni_run end, result: {res}")
                    logging.info(f"omni_run end, answer:{test[i]['output']}")
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"Retry #{retry_count} failed:{e}")
                    time.sleep(2)
            else:
                print("Multiple retries failed. Please check the issue. The task number with the error is ",i)
                exit(0)
            
            




from data.gaia import load_gaia,download_gaia

def react_GAIA_test():
    logging.basicConfig(
        filename=os.path.join(parent_dir,"results","agentscope","log","react-gaia.log"), 
        filemode='a',
        level=logging.INFO, 
        format='%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if not os.path.exists(os.path.join(parent_dir,"data","GAIA","2023")):
        download_gaia()
    test=load_gaia()['validation']
    
    
    weave.init(project_name="agentscope_react_gaia_test")
    for i in range(len(test)):
        if test[i].get('file_name')=="":
            q = f'question: {test[i]["Question"]}'
        else:
            q=f'question: {test[i]["Question"]}, file_name: {test[i]["file_name"]}, file_path: {os.path.join(parent_dir,"data","GAIA","2023","validation",test[i]["file_name"])}'
        
        
        
        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            try:
                agent=AgentScopeAgent(
                    agent_type="ReAct",
                    api_key=openai_api_key,
                )
                logging.info(f"omni_run start, query: {q}")
                res=AgentScopeAgentRun(
                    agent=agent,
                    agent_type="ReAct",
                    query=q
                )
                logging.info(f"omni_run end, result: {res}")
                logging.info(f"omni_run end, answer:{test[i]['Final answer']}")
                break
            except Exception as e:
                retry_count += 1
                print(f"Retry #{retry_count} failed:{e}")
                time.sleep(2)
        else:
            print("Multiple retries failed. Please check the issue. The task number with the error is ",i)
            exit(0)

        
        
        print(i)
        
    

from data.humaneval import load_humaneval

def react_humaneval_test(repeat):
    logging.basicConfig(
        filename=os.path.join(parent_dir,"results","agentscope","log","react-humaneval"+str(repeat)+".log"), 
        filemode='a',
        level=logging.INFO, 
        format='%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    test=load_humaneval()
    
    
    weave.init(project_name="agentscope_react_humaneval_test")
    for taskid,task in test.items():

        q=task["prompt"]+"\n# Implement the function correctly.At the end, output the complete code. "
        
        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            try:
                agent=AgentScopeAgent(
                    agent_type="ReAct",
                    api_key=openai_api_key,
                )
                logging.info(f"omni_run start, query: {q}")
                res=AgentScopeAgentRun(
                    agent=agent,
                    agent_type="ReAct",
                    query=q,
                )
                logging.info(f"omni_run end, result: {res}")
                logging.info(f"omni_run end, answer:{task['canonical_solution']}")
                break
            except Exception as e:
                retry_count += 1
                print(f"Retry #{retry_count} failed:{e}")
                time.sleep(2)
        else:
            print("Multiple retries failed. Please check the issue. The task number with the error is ",taskid)
            exit(0)
        print(taskid)



    
    
