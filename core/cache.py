import time

CACHE = {}
CACHE_TTL = 300
CACHE_MAX_SIZE = 1000
# reuse model sessions dict here for simplicity
MODEL_SESSIONS = {}

def get_from_cache(key: str):
    now = time.time()
    if key in CACHE:
        val, ts = CACHE[key]
        if now - ts < CACHE_TTL:
            return val
        del CACHE[key]
    return None

def add_to_cache(key: str, value):
    now = time.time()
    CACHE[key] = (value, now)
    if len(CACHE) > CACHE_MAX_SIZE:
        oldest = min(CACHE.items(), key=lambda x: x[1][1])[0]
        del CACHE[oldest]
