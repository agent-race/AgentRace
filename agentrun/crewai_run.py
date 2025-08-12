import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import logging
import time
from agents.crewai_agent import CrewAIAgent
import weave
from eval.token_count import init_token_count
init_token_count(offline_mode=True)
from typing import List
from eval.utils import change_log_file
from dotenv import load_dotenv
load_dotenv()
from eval.summary import summary_log

def react_gaia_eval():
    j=0
    client=weave.init('crewai-react-gaia')
    filename=os.path.join(parent_dir,"results","crewai","log","react_gaig.log")
    logging.basicConfig(
    filename=filename, 
    filemode='a',
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
    )
    react_agent = CrewAIAgent(
        agent_type='ReAct'
    )
    from datasets import load_dataset
    gaia_dataset = load_dataset("gaia-benchmark/GAIA", "2023_all")['validation']
    logging.info(f"we need run {len(gaia_dataset)}")
    for gaia_data in gaia_dataset:
        try:
            task_id = gaia_data['task_id']
            question = gaia_data['Question']
            level = gaia_data['Level']
            file_name = gaia_data['file_name']
            file_path = gaia_data['file_path']
            annotator_metadata = gaia_data['Annotator Metadata']
            if file_name!="":
                query = f'question: {question}, file_name: {file_name}, file_path: {file_path}'
            else:
                query = f'question: {question}'
            logging.info(f"omni_run start, query: {query}")
            result = react_agent.omni_run(task=query)
            logging.info(f"omni_run end, result: {result}")
            logging.info(f"omni_run end, answer: {annotator_metadata}")
            client.flush()
            j+=1
            if j%10==0:
                time.sleep(30)
            # break
        except Exception as e:
            return f"Error: {e}.\nNow,we run the num {j} end." 
    summary_log(
    file_path=filename,
    save_path=os.path.join(parent_dir,"results","crewai","react_gaia"),
    weave_op=False,          
    reverse=False,           
    millisecond=True,       
    include_op_name=True,   
    time_lens=23            
    )


def react_human_eval():
    from datasets import load_dataset
    alpacaeval = load_dataset("tatsu-lab/alpaca_eval", 'alpaca_eval', trust_remote_code=True)
    client=weave.init('crewai-react-human_eval')
    filename=os.path.join(parent_dir,"results","crewai","log","react_humaneval.log")
    logging.basicConfig(
    filename=filename, 
    filemode='a',
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
    )
    react_agent = CrewAIAgent(
        agent_type='ReAct'
    )
    from datasets import load_dataset
    human_eval = load_dataset("openai_humaneval")
    j=0
    logging.info(f"the num is {len(human_eval['test'])}")
    for human_eval_data in human_eval['test']:
        try:
            task_id = human_eval_data['task_id']
            prompt = human_eval_data['prompt']
            canonical_solution = human_eval_data['canonical_solution']
            test = human_eval_data['test']
            entry_point = human_eval_data['entry_point']
            # logging.info(f"data:{human_eval_data}")
            query = f'question: {prompt}.,you need to give me the true python code.And you need to use the python tool to do it.'
            logging.info(f"omni_run start, query: {query}")
            result = react_agent.omni_run(task=prompt)
            logging.info(f"omni_run end, r esult: {result}")
            logging.info(f"omni_run end, answer: {canonical_solution}")
            client.flush()
            logging.info(f"now,we run the {j} end")
            j+=1
            if j==100:
                time.sleep(30)
        except Exception as e:
            return f"Error: {e}.\nNow,we run the num {j} end." 
    summary_log(
    file_path=filename,
    save_path=os.path.join(parent_dir,"results","crewai","react_humaneval"),
    weave_op=False,          
    reverse=False,           
    millisecond=True,       
    include_op_name=True,   
    time_lens=23            
    )


        

def rag_eval():
    logging.basicConfig(
    filename=os.path.join(parent_dir,"results","crewai","log","rag_mmlu.log"),  
    filemode='a',
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
    )
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from agents.support import create_vector_db, load_vector_db
    time1=time.time()
    create_vector_db()
    db_instance=load_vector_db()
    time2=time.time()
    embedding_time = time2 - time1
    logging.info(f"database_vectorization time: {embedding_time}")
    def vector_search(query: str) -> List[str]:
            try:
                query=query
                k=5
                return db_instance.search(query, k)
            except Exception as e:
                return f"Error: {e}" 
    # client=weave.init(project_name="crewai_RAG_test")
    client=weave.init(project_name="crewai_RAG_1.0")
    agent=CrewAIAgent(
        agent_type="RAG"
    )
    from data.mmlu import load_mmlu
    test=load_mmlu()
    j=0
    begin=0
    begin=2890*4
    end=len(test)
    for i in range(begin,end):
        try:
            if i%4==0:
                logging.info(f"Now,we run the {i}")
                question = test[i][0]
                # subject = mmlu_data['subject']
                # choices = mmlu_data['choices']
                answer = test[i][1]
                query=question
                logging.info(f"omni_run start, query: {query}")
                time1=time.time()
                vector_search_result = vector_search(query)
                time2=time.time()
                retrieve_time = time2 - time1
                logging.info(f"retrieve time: {retrieve_time}")
                res=agent.omni_run_RAG(
                    task=query,
                    vector_search_result=vector_search_result,
                )
                # print(answer)
                logging.info(f"omni_run end, result: {res}")
                logging.info(f"omni_run end, answer: {answer}")
                client.flush()
                j+=1
                if j==100:
                    time.sleep(30)
        except Exception as e:
            return f"Error: {e}.\nNow,we run the num {j} end." 
    summary_log(
    file_path=os.path.join(parent_dir,"results","crewai","log","rag_mmlu.log"),
    save_path=os.path.join(parent_dir,"results","crewai","rag_mmlu.log"),
    weave_op=False,          
    reverse=False,           
    millisecond=True,       
    include_op_name=True,   
    time_lens=23            
    )
        
def moa_eval(moa_num=3):
    client=weave.init('crewai-moa')
    logging.basicConfig(
    filename=os.path.join(parent_dir,"results","crewai","log","moa.log"),  
    filemode='a',
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
    )
    moa_agent = CrewAIAgent(
        agent_type="MoA",
        moa_num=moa_num   #change MOA num here
    )
    from datasets import load_dataset
    alpacaeval = load_dataset("tatsu-lab/alpaca_eval", 'alpaca_eval', trust_remote_code=True)
    i= 0
    for alpacaeval_data in alpacaeval['eval']:
        try:
            instruction = alpacaeval_data['instruction']
            output = alpacaeval_data['output']
            generator = alpacaeval_data['generator']
            dataset = alpacaeval_data['dataset']
            query = f'instruction: {instruction}'
            logging.info(f"omni_run start, query: {query}")
            result = moa_agent.omni_run_MOA(task=instruction)
            logging.info(f"omni_run end, result: {result}")
            logging.info(f"omni_run end, answer: {output}")
            client.flush()
            logging.info(f"Now,we run the {i} over")
            i+=1
            time.sleep(30)
        # break
        except Exception as e:
            return f"Error: {e}.\nNow,we run the num {i} end." 
    summary_log(
    file_path=os.path.join(parent_dir,"results","crewai","log","moa.log"),
    save_path=os.path.join(parent_dir,"results","crewai","moa3.log"),
    weave_op=False,          
    reverse=False,           
    millisecond=True,       
    include_op_name=True,   
    time_lens=23            
    )

def vqa_test():
    client=weave.init('crewai-vqa')
    import os
    filename=os.path.join(parent_dir,"results","crewai","log","vqa-v3.log")
    # 创建目录（如果不存在）
    dir_path = os.path.dirname(filename)
    os.makedirs(dir_path, exist_ok=True)
    # 创建文件（如果不存在）
    if not os.path.exists(filename):
        with open(filename, 'w'):
            pass
    if not os.access(os.path.dirname(filename), os.W_OK):
        raise PermissionError(f"Cannot write to directory: {os.path.dirname(filename)}")
    logging.basicConfig(
        filename=filename,  
        filemode='a',
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    from datasets import load_dataset
    dataset = load_dataset(
        "lmms-lab/OK-VQA"
    )
    datas=dataset['val2014']
    react_agent = CrewAIAgent(
               agent_type='ReAct'
     )
    i=0
    print(datas)
    for data in datas:
            if i%5!=0:
                i+=1
                continue
            question=data['question']
            question_id=data['question_id']
            answers=data['answers']
            query = f"You need to naswer the question from the image.The path is:../data/VQA/image/{question_id}.png\nQuestion is: {question}\nAnswer the quetions just use easy words.Answer normalization (all chars lowercase, no period except as decimal point, number words —> digits, strip articles (a, an the)).You just need to use the vision tool for one time.What's more , you need to try your best yo understand the output of your tool.You need to Carefully analyze what the outcome of the problem is, as the tool output may not be very accurate. So if you can, you can make the output of your tool more detailed. If the output is too simple, you can tell me I don't know. If the tool is wrong, you can tell me the tool error. "
            logging.info(f"omni_run start, query: {query}")
            result = react_agent.omni_run(task=query)
            logging.info(f"omni_run end, result: {result}")
            logging.info(f"omni_run end, answer: {answers}")
            client.flush()
            logging.info(f"Now,we run the {i} over")
            i+=1
            if i%100==0:
                time.sleep(10)
    
# vqa_test()
# rag_test()
# moa_test()
# react_human_eval()
# react_gaia_eval()

# import argparse
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description='llamaindex-runner',
#         epilog='react_gaia, react_humaneval, rag_mmlu, moa_alpacaeval'
#     )
#     parser.add_argument('--react_gaia', action='store_true', help='run react on gaia')
#     parser.add_argument('--react_humaneval', action='store_true', help='run react on humaneval')
#     parser.add_argument('--rag_mmlu', action='store_true', help='run rag on mmlu')
#     parser.add_argument('--moa_alpacaeval', action='store_true', help='run moa on alpacaeval')
#     args = parser.parse_args()
#     if args.react_gaia:
#         react_gaia_eval()
#     if args.react_humaneval:
#         react_human_eval()
#     if args.rag_mmlu:
#         rag_eval()
#     if args.moa_alpacaeval:
#         moa_eval()

