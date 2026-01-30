import os
from redis import Redis
from rq import Queue

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
conn = Redis.from_url(redis_url)
q = Queue("conformation", connection=conn)