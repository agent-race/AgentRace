from datasets import load_dataset
from huggingface_hub import snapshot_download
import os


base_dir = os.path.dirname(os.path.abspath(__file__))
dir = os.path.join(base_dir, 'GAIA')

#需要设置临时环境变量HF_TOKEN
def download_gaia():
    snapshot_download(
        repo_id="gaia-benchmark/GAIA",
        repo_type="dataset",
        local_dir=dir,
        local_dir_use_symlinks=False
    )

def load_gaia():
    data=load_dataset(path=os.path.join(dir,"GAIA.py"),name="2023_all",trust_remote_code=True)
    return data

def load_gaia_by_level(level):
    LEVELS={"2023_level1",
            "2023_level2",
            "2023_level3"}
    data=load_dataset(path=os.path.join(dir,"GAIA.py"),name=LEVELS[level-1],trust_remote_code=True)
    return data
