from django.conf import settings

import rpy2.rinterface as rinterface
from celery import Celery, Task

celery = Celery('tasks',broker='amqp://guest@localhost//', backend='redis://localhost:6379/5')

@celery.tasks
class Rpy2Task(Task):   
    def __init__(self):
        self.name = "rpy2"

    def run(self, args):    
        rinterface.initr()
        r_func = rinterface.baseenv['source']('logistic.R')
        r_func[0](args) 
        pass

Rpy2Task = celery.register_task(Rpy2Task())
async_result = Rpy2Task.delay()