import os
import json
import sys

import re
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from data.HUMANEVAL.human_eval.evaluation import evaluate_functional_correctness


def moa_eval(path,scale):
    json_files = sorted([f for f in os.listdir(path) if f.endswith('.json')])
    data_list = []
    for file_name in json_files:
        file_path = os.path.join(path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if "failure_run" not in data: 
                    data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"文件 {file_name} 解码失败：{e}")
    print("moa evaluation,scale:",scale)            
    print("sample size:",len(data_list))

    total_time=0

    llm_time_llama=0
    llm_time_qwen=0
    llm_time_ds=0
    llm_time_gpt=0

    prompt_tokens_llama=0
    completion_tokens_llama=0
    total_tokens_llama=0

    prompt_tokens_qwen=0
    completion_tokens_qwen=0
    total_tokens_qwen=0

    prompt_tokens_ds=0
    completion_tokens_ds=0
    total_tokens_ds=0

    prompt_tokens_gpt=0
    completion_tokens_gpt=0
    total_tokens_gpt=0

    size1=0
    size2=0
    size3=0
    size4=0
    size5=0
    size6=0

    pack1=0
    pack2=0
    pack3=0
    pack4=0
    pack5=0
    pack6=0

    for item in data_list:
        total_time+=item["total_time"][0]

        llm_time_llama+=sum(item["llm_time"]["meta-llama/Llama-3.3-70B-Instruct-Turbo"])
        llm_time_qwen+=sum(item["llm_time"]["Qwen/Qwen2.5-7B-Instruct-Turbo"])
        llm_time_ds+=sum(item["llm_time"]["deepseek-ai/DeepSeek-V3"])
        llm_time_gpt+=sum(item["llm_time"]["gpt-4o-2024-08-06"])

        prompt_tokens_llama+=sum(item["llm_tokens"]["meta-llama/Llama-3.3-70B-Instruct-Turbo"]["prompt"])
        completion_tokens_llama+=sum(item["llm_tokens"]["meta-llama/Llama-3.3-70B-Instruct-Turbo"]["completion"])
        total_tokens_llama+=sum(item["llm_tokens"]["meta-llama/Llama-3.3-70B-Instruct-Turbo"]["total"])

        prompt_tokens_qwen+=sum(item["llm_tokens"]["Qwen/Qwen2.5-7B-Instruct-Turbo"]["prompt"])
        completion_tokens_qwen+=sum(item["llm_tokens"]["Qwen/Qwen2.5-7B-Instruct-Turbo"]["completion"])
        total_tokens_qwen+=sum(item["llm_tokens"]["Qwen/Qwen2.5-7B-Instruct-Turbo"]["total"])

        prompt_tokens_ds+=sum(item["llm_tokens"]["deepseek-ai/DeepSeek-V3"]["prompt"])
        completion_tokens_ds+=sum(item["llm_tokens"]["deepseek-ai/DeepSeek-V3"]["completion"])
        total_tokens_ds+=sum(item["llm_tokens"]["deepseek-ai/DeepSeek-V3"]["total"])

        prompt_tokens_gpt+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["prompt"])
        completion_tokens_gpt+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["completion"])
        total_tokens_gpt+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["total"])

        for i in range(scale//3):
            size1+=item["communication_size"]["root_to_agent"+str(i*3+1)]
            size2+=item["communication_size"]["root_to_agent"+str(i*3+2)]
            size3+=item["communication_size"]["root_to_agent"+str(i*3+3)]
            size4+=item["communication_size"]["agent"+str(i*3+1)+"_to_aggregation"]
            size5+=item["communication_size"]["agent"+str(i*3+2)+"_to_aggregation"]
            size6+=item["communication_size"]["agent"+str(i*3+3)+"_to_aggregation"]

            pack1+=item["communication_pack"]["root_to_agent"+str(i*3+1)]
            pack2+=item["communication_pack"]["root_to_agent"+str(i*3+2)]
            pack3+=item["communication_pack"]["root_to_agent"+str(i*3+3)]
            pack4+=item["communication_pack"]["agent"+str(i*3+1)+"_to_aggregation"]
            pack5+=item["communication_pack"]["agent"+str(i*3+2)+"_to_aggregation"]
            pack6+=item["communication_pack"]["agent"+str(i*3+3)+"_to_aggregation"]


    total_time=total_time/len(data_list)

    llm_time_llama=llm_time_llama/len(data_list)
    llm_time_qwen=llm_time_qwen/len(data_list)
    llm_time_ds=llm_time_ds/len(data_list)
    llm_time_gpt=llm_time_gpt/len(data_list)

    prompt_tokens_llama=prompt_tokens_llama/len(data_list)
    completion_tokens_llama=completion_tokens_llama/len(data_list)
    total_tokens_llama=total_tokens_llama/len(data_list)

    prompt_tokens_qwen=prompt_tokens_qwen/len(data_list)
    completion_tokens_qwen=completion_tokens_qwen/len(data_list)
    total_tokens_qwen=total_tokens_qwen/len(data_list)

    prompt_tokens_ds=prompt_tokens_ds/len(data_list)
    completion_tokens_ds=completion_tokens_ds/len(data_list)
    total_tokens_ds=total_tokens_ds/len(data_list)

    prompt_tokens_gpt=prompt_tokens_gpt/len(data_list)
    completion_tokens_gpt=completion_tokens_gpt/len(data_list)
    total_tokens_gpt=total_tokens_gpt/len(data_list)

    
    size1=size1/len(data_list)
    size2=size2/len(data_list)
    size3=size3/len(data_list)
    size4=size4/len(data_list)
    size5=size5/len(data_list)
    size6=size6/len(data_list)

    pack1=pack1/len(data_list)
    pack2=pack2/len(data_list)
    pack3=pack3/len(data_list)
    pack4=pack4/len(data_list)
    pack5=pack5/len(data_list)
    pack6=pack6/len(data_list)

    print("------------------------------")
    print("evaluation per query")
    print("total_time:",total_time)
    print("llm_time_llama:",llm_time_llama)
    print("llm_time_qwen:",llm_time_qwen)
    print("llm_time_ds:",llm_time_ds)
    print("llm_time_gpt:",llm_time_gpt)
    print("prompt_tokens_llama:",prompt_tokens_llama)
    print("completion_tokens_llama:",completion_tokens_llama)
    print("total_tokens_llama:",total_tokens_llama)
    print("prompt_tokens_qwen",prompt_tokens_qwen)
    print("completion_tokens_qwen:",completion_tokens_qwen)
    print("total_tokens_qwen:",total_tokens_qwen)
    print("prompt_tokens_ds:",prompt_tokens_ds)
    print("completion_tokens_ds:",completion_tokens_ds)
    print("total_tokens_ds:",total_tokens_ds)
    print("prompt_tokens_gpt:",prompt_tokens_gpt)
    print("completion_tokens_gpt:",completion_tokens_gpt)
    print("total_tokens_gpt:",total_tokens_gpt)
    print("size1:",size1)
    print("size2:",size2)
    print("size3:",size3)
    print("size4:",size4)
    print("size5:",size5)
    print("size6:",size6)
    print("pack1:",pack1)
    print("pack2:",pack2)
    print("pack3:",pack3)
    print("pack4:",pack4)
    print("pack5:",pack5)
    print("pack6:",pack6)
    print("------------------------------")
    


def rag_eval(path):
    json_files = sorted([f for f in os.listdir(path) if f.endswith('.json')])
    data_list = []
    for file_name in json_files:
        file_path = os.path.join(path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if "failure_run" not in data: 
                    data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"文件 {file_name} 解码失败：{e}")
    print("rag evaluation")
    print("sample size:",len(data_list))

    
    
    total_time=0
    retrieve_time=0
    llm_time=0
    prompt_tokens=0
    completion_tokens=0
    total_tokens=0

    total_time1=0
    retrieve_time1=0
    llm_time1=0
    prompt_tokens1=0
    completion_tokens1=0
    total_tokens1=0

    correct=0
    for item in data_list:
        if item["answer"]==item["result"][0]:
            correct+=1
            total_time1+=item["total_time"][0]
            retrieve_time1+=item["retrieve_time"][0][0]
            llm_time1+=sum(item["llm_time"]["gpt-4o"])
            prompt_tokens1+=sum(item["llm_tokens"]["gpt-4o"]["prompt"])
            completion_tokens1+=sum(item["llm_tokens"]["gpt-4o"]["completion"])
            total_tokens1+=sum(item["llm_tokens"]["gpt-4o"]["total"])
        total_time+=item["total_time"][0]
        retrieve_time+=item["retrieve_time"][0][0]
        llm_time+=sum(item["llm_time"]["gpt-4o"])
        prompt_tokens+=sum(item["llm_tokens"]["gpt-4o"]["prompt"])
        completion_tokens+=sum(item["llm_tokens"]["gpt-4o"]["completion"])
        total_tokens+=sum(item["llm_tokens"]["gpt-4o"]["total"])

    embedding_time=data_list[0]["embedding_time"][0]

    total_time=total_time/len(data_list)
    retrieve_time=retrieve_time/len(data_list)
    llm_time=llm_time/len(data_list)
    prompt_tokens=prompt_tokens/len(data_list)
    completion_tokens=completion_tokens/len(data_list)
    total_tokens=total_tokens/len(data_list)

    total_time1=total_time1/correct
    retrieve_time1=retrieve_time1/correct
    llm_time1=llm_time1/correct
    prompt_tokens1=prompt_tokens1/correct
    completion_tokens1=completion_tokens1/correct
    total_tokens1=total_tokens1/correct
    pri=correct/len(data_list)

    print("------------------------------")
    print("embedding time:",embedding_time)
    print("------------------------------")
    print("evaluation per query")
    print("total_time:",total_time)
    print("retrieve_time:",retrieve_time)
    print("llm_time:",llm_time)
    print("prompt_tokens:",prompt_tokens)
    print("completion_tokens:",completion_tokens)
    print("total_tokens:",total_tokens)
    print("------------------------------")
    print("evaluation per correct query")
    print("total_time:",total_time1)
    print("retrieve_time:",retrieve_time1)
    print("llm_time:",llm_time1)
    print("prompt_tokens:",prompt_tokens1)
    print("completion_tokens:",completion_tokens1)
    print("total_tokens:",total_tokens1)
    print("accuracy:",pri)
    print("------------------------------")
    


def gaia_eval(path,agentsocpe):
    '''
    path:the result folder
    agentscope: a bool value, if the result is from agentscope,please set it true,else set it false
    '''
    json_files = sorted([f for f in os.listdir(path) if f.endswith('.json')])
    data_list = []
    for file_name in json_files:
        file_path = os.path.join(path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if "failure_run" not in data: 
                    data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"文件 {file_name} 解码失败：{e}")

    print("react-gaia evaluation")
    print("sample size:",len(data_list))

    total_time=0
    llm_time=0
    prompt_tokens=0
    completion_tokens=0
    total_tokens=0


    python_code_tool=0
    csv_load=0
    xlsx_load=0
    docs_load=0
    vedio_load=0
    txt_load=0
    pdf_load=0
    audio_load=0
    image_load=0
    search_tool=0

    total_time1=0
    llm_time1=0
    prompt_tokens1=0
    completion_tokens1=0
    total_tokens1=0


    python_code_tool1=0
    csv_load1=0
    xlsx_load1=0
    docs_load1=0
    vedio_load1=0
    txt_load1=0
    pdf_load1=0
    audio_load1=0
    image_load1=0
    search_tool1=0

    correct=0


    for item in data_list:
        total_time+=item["total_time"][0]

        if "python_code_tool" in item["tools"]:
            python_code_tool+=sum(item["tools"]["python_code_tool"])
        if "csv_load" in item["tools"]:
            csv_load+=sum(item["tools"]["csv_load"])
        if "xlsx_load" in item["tools"]:
            xlsx_load+=sum(item["tools"]["xlsx_load"])
        if "docs_load" in item["tools"]:
            docs_load+=sum(item["tools"]["docs_load"])
        if "vedio_load" in item["tools"]:
            vedio_load+=sum(item["tools"]["vedio_load"])
        if "txt_load" in item["tools"]:
            txt_load+=sum(item["tools"]["txt_load"])
        if "pdf_load" in item["tools"]:
            pdf_load+=sum(item["tools"]["pdf_load"])
        if "audio_load" in item["tools"]:
            audio_load+=sum(item["tools"]["audio_load"])
        if "image_load" in item["tools"]:
            image_load+=sum(item["tools"]["image_load"])
        if "search_tool" in item["tools"]:
            search_tool+=sum(item["tools"]["search_tool"])
        
        if "python_tool" in item["tools"]:
            python_code_tool+=sum(item["tools"]["python_tool"])
        if "csv_tool" in item['tools']:
            csv_load+=sum(item["tools"]["csv_tool"])
        if "xlsx_tool" in item['tools']:
            xlsx_load+=sum(item["tools"]["xlsx_tool"])
        if "docx_tool" in item['tools']:
            docs_load+=sum(item["tools"]["docx_tool"])
        if "vedio_tool" in item['tools']:
            vedio_load+=sum(item["tools"]["vedio_tool"])
        if "txt_tool" in item['tools']:
            txt_load+=sum(item["tools"]["txt_tool"])
        if "pdf_tool" in item['tools']:
            pdf_load+=sum(item["tools"]["pdf_tool"])
        if "audio_tool" in item['tools']:
            audio_load+=sum(item["tools"]["audio_tool"])
        if "vision_tool" in item['tools']:
            image_load+=sum(item["tools"]["vision_tool"])
        if "web_browser_tool" in item['tools']:
            search_tool+=sum(item["tools"]["web_browser_tool"])

        llm_time+=sum(item["llm_time"]["gpt-4o-2024-08-06"])

        prompt_tokens+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["prompt"])
        completion_tokens+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["completion"])
        total_tokens+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["total"])
        if item["passed"]==True:
            total_time1+=item["total_time"][0]
            if "python_code_tool" in item["tools"] or "python_tool" in item["tools"]:
                python_code_tool1+=sum(item["tools"]["python_code_tool"])
            if "csv_load" in item["tools"]:
                csv_load1+=sum(item["tools"]["csv_load"])
            if "xlsx_load" in item["tools"]:
                xlsx_load1+=sum(item["tools"]["xlsx_load"])
            if "docs_load" in item["tools"]:
                docs_load1+=sum(item["tools"]["docs_load"])
            if "vedio_load" in item["tools"]:
                vedio_load1+=sum(item["tools"]["vedio_load"])
            if "txt_load" in item["tools"]:
                txt_load1+=sum(item["tools"]["txt_load"])
            if "pdf_load" in item["tools"]:
                pdf_load1+=sum(item["tools"]["pdf_load"])
            if "audio_load" in item["tools"]:
                audio_load1+=sum(item["tools"]["audio_load"])
            if "image_load" in item["tools"]:
                image_load1+=sum(item["tools"]["image_load"])
            if "search_tool" in item["tools"]:
                search_tool1+=sum(item["tools"]["search_tool"])

            if "python_tool" in item["tools"]:
                python_code_tool+=sum(item["tools"]["python_tool"])
            if "csv_tool" in item['tools']:
                csv_load1+=sum(item["tools"]["csv_tool"])
            if "xlsx_tool" in item['tools']:
                xlsx_load1+=sum(item["tools"]["xlsx_tool"])
            if "docx_tool" in item['tools']:
                docs_load1+=sum(item["tools"]["docx_tool"])
            if "vedio_tool" in item['tools']:
                vedio_load1+=sum(item["tools"]["vedio_tool"])
            if "txt_tool" in item['tools']:
                txt_load1+=sum(item["tools"]["txt_tool"])
            if "pdf_tool" in item['tools']:
                pdf_load1+=sum(item["tools"]["pdf_tool"])
            if "audio_tool" in item['tools']:
                audio_load1+=sum(item["tools"]["audio_tool"])
            if "vision_tool" in item['tools']:
                image_load1+=sum(item["tools"]["vision_tool"])
            if "web_browser_tool" in item['tools']:
                search_tool1+=sum(item["tools"]["web_browser_tool"])

            llm_time1+=sum(item["llm_time"]["gpt-4o-2024-08-06"])

            prompt_tokens1+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["prompt"])
            completion_tokens1+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["completion"])
            total_tokens1+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["total"])
            correct+=1

    

    total_time=total_time/len(data_list)

    llm_time=llm_time/len(data_list)

    prompt_tokens=prompt_tokens/len(data_list)
    completion_tokens=completion_tokens/len(data_list)
    total_tokens=total_tokens/len(data_list)

    python_code_tool=python_code_tool/len(data_list)
    csv_load=csv_load/len(data_list)
    xlsx_load=xlsx_load/len(data_list)
    docs_load=docs_load/len(data_list)
    vedio_load=vedio_load/len(data_list)
    txt_load=txt_load/len(data_list)
    pdf_load=pdf_load/len(data_list)
    audio_load=audio_load/len(data_list)
    image_load=image_load/len(data_list)
    search_tool=search_tool/len(data_list)
    tool_time=python_code_tool+csv_load+xlsx_load+docs_load+vedio_load+txt_load+pdf_load+audio_load+image_load+search_tool

    total_time1=total_time1/correct

    llm_time1=llm_time1/correct

    prompt_tokens1=prompt_tokens1/correct
    completion_tokens1=completion_tokens1/correct
    total_tokens1=total_tokens1/correct

    python_code_tool1=python_code_tool1/correct
    csv_load1=csv_load1/correct
    xlsx_load1=xlsx_load1/correct
    docs_load1=docs_load1/correct
    vedio_load1=vedio_load1/correct
    txt_load1=txt_load1/correct
    pdf_load1=pdf_load1/correct
    audio_load1=audio_load1/correct
    image_load1=image_load1/correct
    search_tool1=search_tool1/correct

    tool_time1=python_code_tool1+csv_load1+xlsx_load1+docs_load1+vedio_load1+txt_load1+pdf_load1+audio_load1+image_load1+search_tool1

    print("------------------------------")
    print("evaluation per query")
    print("total_time:",total_time)
    if agentsocpe:
        print("llm_time:",llm_time-audio_load-image_load)
    else:
        print("llm_time:",llm_time)
    print("prompt_tokens:",prompt_tokens)
    print("completion_tokens:",completion_tokens)
    print("total_tokens:",total_tokens)
    print("web_browser_tool:",search_tool)
    print("pdf_tool:",pdf_load)
    print("csv_tool:",csv_load)
    print("xlsx_tool:",xlsx_load)
    print("txt_tool:",txt_load)
    print("docx_tool:",docs_load)
    print("audio_tool:",audio_load)
    print("vision_tool:",image_load)
    print("vedio_tool:",vedio_load)
    print("python_tool:",python_code_tool)
    print("tool_time:",tool_time)
    print("------------------------------")
    print("evaluation per correct query")
    print("total_time:",total_time1)
    if agentsocpe:
        print("llm_time:",llm_time1-audio_load1-image_load1)
    else:
        print("llm_time:",llm_time1)
    print("prompt_tokens:",prompt_tokens1)
    print("completion_tokens:",completion_tokens1)
    print("total_tokens:",total_tokens1)
    print("web_browser_tool:",search_tool1)
    print("pdf_tool:",pdf_load1)
    print("csv_tool:",csv_load1)
    print("xlsx_tool:",xlsx_load1)
    print("txt_tool:",txt_load1)
    print("docx_tool:",docs_load1)
    print("audio_tool:",audio_load1)
    print("vision_tool:",image_load1)
    print("vedio_tool:",vedio_load1)
    print("python_tool:",python_code_tool1)
    print("tool_time:",tool_time1)
    print("accuracy:",correct/len(data_list))
    print("------------------------------")


    python_code_tool=0
    csv_load=0
    xlsx_load=0
    docs_load=0
    vedio_load=0
    txt_load=0
    pdf_load=0
    audio_load=0
    image_load=0
    search_tool=0



    python_code_tool_num=0
    csv_load_num=0
    xlsx_load_num=0
    docs_load_num=0
    vedio_load_num=0
    txt_load_num=0
    pdf_load_num=0
    audio_load_num=0
    image_load_num=0
    search_tool_num=0


    for item in data_list:
        if "python_code_tool" in item["tools"]:
            python_code_tool+=sum(item["tools"]["python_code_tool"])
            python_code_tool_num+=len(item["tools"]["python_code_tool"])
        if "csv_load" in item["tools"]:
            csv_load+=sum(item["tools"]["csv_load"])
            csv_load_num+=len(item["tools"]["csv_load"])
        if "xlsx_load" in item["tools"]:
            xlsx_load+=sum(item["tools"]["xlsx_load"])
            xlsx_load_num+=len(item["tools"]["xlsx_load"])
        if "docs_load" in item["tools"]:
            docs_load+=sum(item["tools"]["docs_load"])
            docs_load_num+=len(item["tools"]["docs_load"])
        if "vedio_load" in item["tools"]:
            vedio_load+=sum(item["tools"]["vedio_load"])
            vedio_load_num+=len(item["tools"]["vedio_load"])
        if "txt_load" in item["tools"]:
            txt_load+=sum(item["tools"]["txt_load"])
            txt_load_num+=len(item["tools"]["txt_load"])
        if "pdf_load" in item["tools"]:
            pdf_load+=sum(item["tools"]["pdf_load"])
            pdf_load_num+=len(item["tools"]["pdf_load"])
        if "audio_load" in item["tools"]:
            audio_load+=sum(item["tools"]["audio_load"])
            audio_load_num+=len(item["tools"]["audio_load"])
        if "image_load" in item["tools"]:
            image_load+=sum(item["tools"]["image_load"])
            image_load_num+=len(item["tools"]["image_load"])
        if "search_tool" in item["tools"]:
            search_tool+=sum(item["tools"]["search_tool"])
            search_tool_num+=len(item["tools"]["search_tool"])

        if "python_tool" in item["tools"]:
            python_code_tool+=sum(item["tools"]["python_tool"])
            python_code_tool_num+=len(item["tools"]["python_tool"])
        if "csv_tool" in item["tools"]:
            csv_load+=sum(item["tools"]["csv_tool"])
            csv_load_num+=len(item["tools"]["csv_tool"])
        if "xlsx_tool" in item["tools"]:
            xlsx_load+=sum(item["tools"]["xlsx_tool"])
            xlsx_load_num+=len(item["tools"]["xlsx_tool"])
        if "docx_tool" in item["tools"]:
            docs_load+=sum(item["tools"]["docx_tool"])
            docs_load_num+=len(item["tools"]["docx_tool"])
        if "vedio_tool" in item["tools"]:
            vedio_load+=sum(item["tools"]["vedio_tool"])
            vedio_load_num+=len(item["tools"]["vedio_tool"])
        if "txt_tool" in item["tools"]:
            txt_load+=sum(item["tools"]["txt_tool"])
            txt_load_num+=len(item["tools"]["txt_tool"])
        if "pdf_tool" in item["tools"]:
            pdf_load+=sum(item["tools"]["pdf_tool"])
            pdf_load_num+=len(item["tools"]["pdf_tool"])
        if "audio_tool" in item["tools"]:
            audio_load+=sum(item["tools"]["audio_tool"])
            audio_load_num+=len(item["tools"]["audio_tool"])
        if "vision_tool" in item["tools"]:
            image_load+=sum(item["tools"]["vision_tool"])
            image_load_num+=len(item["tools"]["vision_tool"])
        if "web_browser_tool" in item["tools"]:
            search_tool+=sum(item["tools"]["web_browser_tool"])
            search_tool_num+=len(item["tools"]["web_browser_tool"])
        


        



    print("tool time per call")
    python_code_tool=python_code_tool/python_code_tool_num
    csv_load=csv_load/csv_load_num
    xlsx_load=xlsx_load/xlsx_load_num
    docs_load=docs_load/docs_load_num
    vedio_load=vedio_load/vedio_load_num
    txt_load=txt_load/txt_load_num
    pdf_load=pdf_load/pdf_load_num
    audio_load=audio_load/audio_load_num
    image_load=image_load/image_load_num
    search_tool=search_tool/search_tool_num



    print("web_browser_tool:",search_tool)
    print("pdf_tool:",pdf_load)
    print("csv_tool:",csv_load)
    print("xlsx_tool:",xlsx_load)
    print("txt_tool:",txt_load)
    print("docx_tool:",docs_load)
    print("audio_tool:",audio_load)
    print("vision_tool:",image_load)
    print("vedio_tool:",vedio_load)
    print("python_tool:",python_code_tool)
    print("------------------------------")



def extract_code_from_markdown(markdown_str):
    match = re.search(r"```python\n(.*?)```", markdown_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return markdown_str



def humaneval_eval(path):
    json_files = sorted([f for f in os.listdir(path) if f.endswith('.json')])
    data_list = []
    i=0
    for file_name in json_files:
        file_path = os.path.join(path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if "failure_run" not in data: 
                    code={
                        "task_id":"HumanEval/"+str(i),
                        "completion":extract_code_from_markdown(data["result"])
                    }
                    with open("samples.jsonl", "a") as f:
                        f.write(json.dumps(code) + "\n")
                    data_list.append(data)
                    i=i+1
            except json.JSONDecodeError as e:
                print(f"文件 {file_name} 解码失败：{e}")
        
    print("react-humaneval evaluation")
    print("sample size:",len(data_list))
    evaluate_functional_correctness("samples.jsonl")
    res = []
    with open("samples.jsonl_results.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 避免空行报错
                res.append(json.loads(line))
    total_time=0
    llm_time=0
    python_code_tool=0
    prompt_tokens=0
    completion_tokens=0
    total_tokens=0

    total_time1=0
    llm_time1=0
    python_code_tool1=0
    prompt_tokens1=0
    completion_tokens1=0
    total_tokens1=0
    correct=0

    for item,pa in zip(data_list,res):
        if pa["passed"]==True:
            correct+=1
            total_time1+=item["total_time"][0]
            llm_time1+=sum(item["llm_time"]["gpt-4o-2024-08-06"])
            if "python_code_tool" in item["tools"]:
                python_code_tool1+=sum(item["tools"]["python_code_tool"])
            if "python_tool" in item["tools"]:
                python_code_tool1+=sum(item["tools"]["python_tool"])
            prompt_tokens1+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["prompt"])
            completion_tokens1+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["completion"])
            total_tokens1+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["total"])

        total_time+=item["total_time"][0]
        llm_time+=sum(item["llm_time"]["gpt-4o-2024-08-06"])
        if "python_code_tool" in item["tools"]:
            python_code_tool+=sum(item["tools"]["python_code_tool"])
        if "python_tool" in item["tools"]:
            python_code_tool+=sum(item["tools"]["python_tool"])
        prompt_tokens+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["prompt"])
        completion_tokens+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["completion"])
        total_tokens+=sum(item["llm_tokens"]["gpt-4o-2024-08-06"]["total"])


    total_time=total_time/len(data_list)
    python_code_tool=python_code_tool/len(data_list)
    llm_time=llm_time/len(data_list)
    prompt_tokens=prompt_tokens/len(data_list)
    completion_tokens=completion_tokens/len(data_list)
    total_tokens=total_tokens/len(data_list)

    total_time1=total_time1/correct
    python_code_tool1=python_code_tool1/correct
    llm_time1=llm_time1/correct
    prompt_tokens1=prompt_tokens1/correct
    completion_tokens1=completion_tokens1/correct
    total_tokens1=total_tokens1/correct
    pri=correct/len(data_list)


    print("------------------------------")
    print("evaluation per query")
    print("total_time:",total_time)
    print("llm_time:",llm_time)
    print("prompt_tokens:",prompt_tokens)
    print("completion_tokens:",completion_tokens)
    print("total_tokens:",total_tokens)
    print("python_tool:",python_code_tool)
    print("------------------------------")
    print("evaluation per correct query")
    print("total_time:",total_time1)
    print("llm_time:",llm_time1)
    print("prompt_tokens:",prompt_tokens1)
    print("completion_tokens:",completion_tokens1)
    print("total_tokens:",total_tokens1)
    print("python_tool:",python_code_tool1)
    print("accuracy:",pri)
    print("------------------------------")
    os.remove("samples.jsonl")
    os.remove("samples.jsonl_results.jsonl")


