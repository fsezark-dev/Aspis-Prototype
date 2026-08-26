import pandas as pd
import glob

from huggingface_hub import snapshot_download
local_dir = snapshot_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    repo_type="dataset",
    allow_patterns="raw_meta_Electronics/*.parquet",
)

files = glob.glob(f"{local_dir}/raw_meta_Electronics/*.parquet")
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df.to_json("dataset/electronics/meta.jsonl", orient="records", lines=True)
