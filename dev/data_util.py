import zlib
import pickle
import redis


##################################################
def get_keys(redis):
    return redis.getkeys()


##################################################
def compress(data) -> bytes:
    return zlib.compress(pickle.dumps(data), 9)


##################################################
def decompress(comp_data: bytes):
    return pickle.loads(zlib.decompress(comp_data))


##################################################
def RedisDb(host="localhost", port="6379"):
    return redis.Redis(host=host, port=port)


##################################################
def r_write(key, val, redis):
    return redis.set(key, val)
    

##################################################
def r_read(key, redis):
    return redis.get(key)

