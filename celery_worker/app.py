import configparser
from celery import Celery
import redis

# Get the broker URL and result backend from the configuration file
config = configparser.ConfigParser()
config.read('configuration.cfg')
broker_url = config.get('celery', 'broker_url')
result_backend = config.get('celery', 'result_backend')

app = Celery('batch_tasks', broker=broker_url)
app.conf.result_backend = result_backend

# Import tasks
import celery_worker.tasks