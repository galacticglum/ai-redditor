from celery import Celery
from flask_cors import CORS
from flask_migrate import Migrate
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
import ai_redditor_service.template_filters as template_filters

db = SQLAlchemy()
cors = CORS()
migrate = Migrate(db=db)
celery = Celery(
    'ai_redditor_service',
    include=['ai_redditor_service.tasks']
)

socketio = SocketIO(cookie=None)

def init_app(app):
    '''
    Initializes extensions with a Flask app context.

    '''

    db.init_app(app)
    cors.init_app(app)
    template_filters.init_app(app)
    
    _init_migrate(app)
    _init_celery(app)
    socketio.init_app(
        app, message_queue=app.config['SOCKETIO_MESSAGE_QUEUE'],
        logger=app.config['SOCKETIO_ENABLE_LOGGING'],
        engineio_logger=app.config['ENGINEIO_ENABLE_LOGGING']
    )

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
