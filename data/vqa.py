from datasets import load_dataset
import os
from pathlib import Path

# 设置环境变量加速下载
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.aliyun.com"  # 使用HF镜像

# 加载数据集
dataset = load_dataset("lmms-lab/OK-VQA")
print(dataset)

# 定义图片保存路径
save_dir = "./VQA/image"
Path(save_dir).mkdir(parents=True, exist_ok=True)  # 自动创建目录（如果不存在）

# 处理验证集数据
datas = dataset['val2014']
for data in datas:
    image = data['image']
    image_id = data['question_id']
    # 保存图片为PNG格式
    image_path = os.path.join(save_dir, f"{image_id}.png")
    image.save(image_path, "PNG")
    print(f"we png {image_id}ok")