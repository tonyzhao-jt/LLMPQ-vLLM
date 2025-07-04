'''
    Download model with hint.
'''
# load the hf token
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()
# load from dot env
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("REPO_ID")
CACHE_DIR = os.getenv("CACHE_DIR", "/data/llms/hub/")

# load from user inputs
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=REPO_ID)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--hf_token", type=str, default=HF_TOKEN)
    return parser.parse_args()


args = parse_args()
REPO_ID = args.repo_id
CACHE_DIR = args.cache_dir
HF_TOKEN = args.hf_token

snapshot_download(
    REPO_ID, revision="main", cache_dir=CACHE_DIR, use_auth_token=HF_TOKEN
)
