# cache_utils.py

import os
import pickle

CACHE_DIR = "information_retrieval/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(name):
    return os.path.join(CACHE_DIR, name)

def save_pickle(name, obj):
    with open(cache_path(name), "wb") as f:
        pickle.dump(obj, f)

def load_pickle(name):
    path = cache_path(name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def exists(name):
    return os.path.exists(cache_path(name))
