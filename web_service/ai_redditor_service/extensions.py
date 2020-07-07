import time
from celery import Celery
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
import ai_redditor_service.template_filters as template_filters

db = SQLAlchemy()
migrate = Migrate(db=db)
celery = Celery(
    'ai_redditor_service',
    include=['ai_redditor_service.tasks']
)

def init_app(app):
    '''
    Initializes extensions with a Flask app context.

    '''

    db.init_app(app)
    template_filters.init_app(app)
    
    _init_migrate(app)
    _init_celery(app)

def _init_celery(app):
    '''
    Initializes a :class:`celery.Celery` object instance with a Flask app.

    '''

    celery.conf.update(
        app.config,
        result_backend=app.config['CELERY_RESULT_BACKEND'],
        broker_url=app.config['CELERY_BROKER_URL']
    )

    class ContextTask(celery.Task):
        '''
        A Celery task that wraps the task execution
        in a Flask application context.

        '''

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

def _init_migrate(app):
    is_sqlite = app.config['SQLALCHEMY_DATABASE_URI'].startswith('sqlite:')
    migrate.init_app(app, render_as_batch=is_sqlite)