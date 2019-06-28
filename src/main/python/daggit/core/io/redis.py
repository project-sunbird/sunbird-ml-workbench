import redis

def setRediskey(key, val, host, port, password):

    """
    This function writes a key value pair into Redis cache. It is a wrapper on set operation of redis-py.
    :param key(str): The key.
    :param val(str): The value assigned to key.
    :param host(str): redis server host. default:'localhost'.
    :param port(str): redis server port. default: 6379'.
    :param password(str): redis server password. default: None.
    :returns: The detected language for the given text.
    """
    try:
        r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
        msg = r.set(key, val)   
    except Exception as e:
        print(e)
        
def getRediskey(key, host, port, password):
    """
    This function reads the value from Redis cache based on the key. It is a wrapper on  get operation of redis-py .
    :param key(str): The key.
    :param host(str): redis server host. default:'localhost'.
    :param port(str): redis server port. default: 6379'.
    :param password(str): redis server password. default: None.
    :returns: The detected language for the given text.
    """
    try:
        r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
        msg = r.get(key)     
        return msg
    except Exception as e:
        print(e)

