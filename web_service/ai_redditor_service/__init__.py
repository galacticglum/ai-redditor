import mimetypes
from pathlib import Path
from flask import Flask

def create_app(instance_config_filename='local_config.py', test_config=None):
    '''
    Creates the Flask app.

    '''
    
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object('ai_redditor_service.config')
    if test_config is None:
        app.config.from_pyfile(instance_config_filename, silent=True)
    else:
        app.config.from_mapping(test_config)

    
    try:
        Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    # Extensions has to be imported first so that the
    # celery base task is properly initialized
    from ai_redditor_service import extensions
    extensions.init_app(app)

    from ai_redditor_service import routes, models, cli
    routes.init_app(app)
    cli.init_app(app)

    # Register mimetypes
    mimetypes.add_type('text/css', '.css')
    mimetypes.add_type('text/javascript', '.js')

    return app