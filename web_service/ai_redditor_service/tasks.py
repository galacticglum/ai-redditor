'''
Celery tasks.

'''

from ai_redditor_service.extensions import celery

@celery.task
def add(x, y):
    return x + y