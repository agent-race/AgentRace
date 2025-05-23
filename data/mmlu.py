import os
import csv
from huggingface_hub import snapshot_download


base_dir = os.path.dirname(os.path.abspath(__file__))
dir = os.path.join(base_dir, 'MMLU')


def download_mmlu():
    snapshot_download(
        repo_id="cais/mmlu",
        repo_type="dataset",
        allow_patterns="*.tar",
        local_dir=dir,
        local_dir_use_symlinks=False
    )

def parse_csv_to_prompt_answer_array(file_path):
    prompt_answer_array = []
    
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if len(row) >= 6:  # 确保每行至少有6列
                prompt = f"Question:{row[0]}\nA.{row[1]}\nB.{row[2]}\nC.{row[3]}\nD.{row[4]}\nAnswer with A, B, C, or D only."
                answer = row[5]
                prompt_answer_array.append([prompt, answer])  # 将prompt和answer作为一行添加到数组中
    
    return prompt_answer_array

def merge_csv_files_in_folder(folder_path):
    merged_array = []
    
    # 遍历文件夹中的所有文件
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".csv"):  # 只处理CSV文件
            file_path = os.path.join(folder_path, filename)
            
            # 解析当前CSV文件
            prompt_answer_array = parse_csv_to_prompt_answer_array(file_path)
            
            # 将当前文件的数组合并到总数组中
            merged_array.extend(prompt_answer_array)
    
    return merged_array

def load_mmlu():
    data = merge_csv_files_in_folder(os.path.join(base_dir,"MMLU","test"))
    return data