from pathlib import Path
from flask import Flask

def create_app(instance_config_filename='local_config.py', test_config=None):
    '''
    Creates the Flask app.

    '''
    
    app = Flask(__name__, instance_relative_config=True)
    if test_config is None:
        app.config.from_pyfile(instance_config_filename, silent=True)
    else:
        app.config.from_mapping(test_config)

    app.config.from_object('ai_redditor_service.config')

    try:
        Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    # from ai_redditor_service import routes, models, extensions, views, cli
    from ai_redditor_service import extensions, cli

    extensions.init_app(app)
    # routes.init_app(app)
    # views.init_app(app)
    cli.init_app(app)

    return app