'''
Celery worker module.

'''

from ai_redditor_service import create_app
from ai_redditor_service.extensions import _init_celery

# Initialize celery instance with flask app
celery = _init_celery(create_app())