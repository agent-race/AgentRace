import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
sys.path.append(os.getcwd())
from eval.token_count import init_token_count
init_token_count(offline_mode=True)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import os
import dotenv
dotenv.load_dotenv(os.path.join(parent_dir,".env"))

import logging
logging.basicConfig(
        filename='results/phidata-initial-logs.log', 
        filemode='a',
        level=logging.INFO, 
        format='%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
from eval.utils import change_log_file

from agents.phidata_agent import PhidataAgent
import weave
import eval.token_count
import time
import tarfile
import shutil
import ast

"""
ReAct
"""

def react_gaia(checkpoint=None):
    weave.init('phidata-react-gaia')
    change_log_file('results/phidata-react-gaia.log')
    react_agent = PhidataAgent('gpt-4o', 'ReAct')

    from datasets import load_dataset
    gaia_dataset = load_dataset("gaia-benchmark/GAIA", "2023_all")['validation']
    is_finished = [False for _ in gaia_dataset]
    if checkpoint is not None:
        with open('react-gaia-phidata.txt', 'r') as f:
            is_finished = ast.literal_eval(f.read())


    for idx, gaia_data in enumerate(gaia_dataset):
        if is_finished[idx]:
            continue
        while not is_finished[idx]:
            try:
                task_id = gaia_data['task_id']
                question = gaia_data['Question']
                level = gaia_data['Level']
                final_answer = gaia_dataset['Final answer']
                file_name = gaia_data['file_name']
                file_path = gaia_data['file_path']
                annotator_metadata = gaia_data['Annotator Metadata']

                query = f'question: {question}' if file_name == '' else f'question: {question}, file_name: {file_name}, file_path: {file_path}'

                logging.info(f"omni_run start, query: {query}")
                result = react_agent.omni_run(query)
                logging.info(f"omni_run end, result: {result}")
                logging.info(f"omni_run end, answer:{final_answer}")
                is_finished[idx] = True
                with open('react-gaia-phidata.txt', 'w') as f:
                    f.write(str(is_finished))
            except:
                is_finished[idx] = False
                time.sleep(10)
                react_agent = PhidataAgent('gpt-4o', 'ReAct')


    


def react_human_eval(checkpoint=None):
    weave.init('phidata-react-human_eval')
    change_log_file('results/phidata-react-human.log')
    react_agent = PhidataAgent('gpt-4o', 'ReAct')

    from datasets import load_dataset
    human_eval = load_dataset("openai_humaneval")['test']

    is_finished = [False for _ in human_eval]
    if checkpoint is not None:
        with open('react-humaneval-phidata.txt', 'r') as f:
            is_finished = ast.literal_eval(f.read())

    for idx, human_eval_data in enumerate(human_eval):
        if is_finished[idx]:
            continue
        while not is_finished[idx]:
            try:
                task_id = human_eval_data['task_id']
                prompt = human_eval_data['prompt']
                canonical_solution = human_eval_data['canonical_solution']
                test = human_eval_data['test']
                entry_point = human_eval_data['entry_point']

                query = f'{prompt}'
                logging.info(f"omni_run start, query: {query}")
                result = react_agent.omni_run(prompt)
                logging.info(f"omni_run end, result: {result}")
                logging.info(f"omni_run end, answer:{canonical_solution}")
                is_finished[idx] = True
                with open('react-humaneval-phidata.txt', 'w') as f:
                    f.write(str(is_finished))
            except:
                is_finished[idx] = False
                time.sleep(10)
                react_agent = PhidataAgent('gpt-4o', 'ReAct')


        


"""
RAG
"""


def rag_mmlu(checkpoint=None):
    weave.init('phidata-rag')
    change_log_file("results/phidata-rag-mmlu-chunk3.log")
    rag_agent = PhidataAgent('gpt-4o', 'RAG')

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from data.mmlu import load_mmlu, download_mmlu
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

    # ['abstract_algebra', 'all', 'anatomy', 'astronomy', 'auxiliary_train', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    # mmlu_dataset = load_dataset("cais/mmlu", "all")
    is_finished = [False for _ in test]
    if checkpoint is not None:
        with open(checkpoint, 'r') as f:
            is_finished = ast.literal_eval(f.read())

    for idx, mmlu_data in enumerate(test):
        if is_finished[idx] or idx % 4 != 0:
            is_finished[idx] = True
            continue
        while not is_finished[idx]:
            try:
                # ['question', 'subject', 'choices', 'answer']
                question = mmlu_data[0]
                answer = mmlu_data[1]

                query = f'{question}'
                logging.info(f"omni_run start, query: {query}")
                result = rag_agent.omni_run(question)
                logging.info(f"omni_run end, result: {result}")
                logging.info(f"omni_run end, answer:{answer}")
                # break # just for test

                is_finished[idx] = True
                with open(checkpoint if  checkpoint is not None else 'rag-phidata.txt', 'w') as f:
                    f.write(str(is_finished))
            except:
                is_finished[idx] = False
                time.sleep(5)
                # rag_agent = None
                # while rag_agent is None:
                #     rag_agent = PhidataAgent('gpt-4o', 'RAG')
                #     break




"""
MoA
"""
def moa_alpacaeval(checkpoint=None):
    weave.init('phidata-moa')
    change_log_file('results/phidata-moa-alpacaeval.log')
    moa_agent = PhidataAgent('gpt-4o', 'MoA')

    from datasets import load_dataset
    # ['alpaca_eval', 'alpaca_eval_gpt4_baseline', 'alpaca_eval_all_outputs', 'alpaca_farm_human_annotations', 'alpaca_farm_human_crossannotations', 'alpaca_eval_annotations_alpaca_eval_gpt4', 'alpaca_eval_annotations_claude']
    alpacaeval = load_dataset("tatsu-lab/alpaca_eval", 'alpaca_eval', trust_remote_code=True)['eval']
    is_finished = [False for _ in alpacaeval]
    if checkpoint is not None:
        with open('moa-checkpoint-phidata.txt', 'r') as f:
            is_finished = ast.literal_eval(f.read())

    for idx, alpacaeval_data in enumerate(alpacaeval):
        if is_finished[idx]:
            continue
        # ['instruction', 'output', 'generator', 'dataset']
        while not is_finished[idx]:
            try:
                instruction = alpacaeval_data['instruction']
                output = alpacaeval_data['output']
                generator = alpacaeval_data['generator']
                dataset = alpacaeval_data['dataset']
                query = f'instruction: {instruction}'


                logging.info(f"omni_run start, query: {query}")
                result = moa_agent.omni_run(instruction)
                logging.info(f"omni_run end, result: {result}")
                logging.info(f"omni_run end, answer:{output}")
                is_finished[idx] = True
                with open('moa-checkpoint-phidata.txt', 'w') as f:
                    f.write(str(is_finished))
            except:
                is_finished[idx] = False
                time.sleep(5)
                moa_agent = PhidataAgent('gpt-4o', 'MoA')
                



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='phidata-runner',
        epilog='react_gaia, react_humaneval, rag_mmlu, moa_alpacaeval'
    )
    parser.add_argument('--react_gaia', action='store_true', help='run react on gaia')
    parser.add_argument('--react_humaneval', action='store_true', help='run react on humaneval')
    parser.add_argument('--rag_mmlu', action='store_true', help='run rag on mmlu')
    parser.add_argument('--moa_alpacaeval', action='store_true', help='run moa on alpacaeval')
    parser.add_argument("-checkpoint", help="checkpoint dir", type=str)
    args = parser.parse_args()
    if args.react_gaia:
        react_gaia()
    if args.react_humaneval:
        react_human_eval()
    if args.rag_mmlu:
        rag_mmlu(args.checkpoint)
    if args.moa_alpacaeval:
        moa_alpacaeval(args.checkpoint)
