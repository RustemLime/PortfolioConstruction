# context_store.py
# Простейшее key-value хранилище
DATA_CACHE = {}

def save_data(key, data):
    DATA_CACHE[key] = data
    return key

def get_data(key):
    return DATA_CACHE.get(key)