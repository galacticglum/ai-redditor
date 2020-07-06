from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from ai_redditor_service.gpt2 import ModelType, load_model
import ai_redditor_service.template_filters as template_filters

db = SQLAlchemy()
migrate = Migrate(db=db)

GPT2_MODELS = None

def init_app(app):
    '''
    Initializes extensions with a Flask app context.

    '''

    db.init_app(app)
    template_filters.init_app(app)
    
    _init_migrate(app)

    # Load the GPT2 models
    GPT2_MODELS = {
        ModelType.TIFU: load_model(
            app.config['TIFU_MODEL_PATH'],
            app.config.get('TIFU_TOKENIZER_PATH', None)
        ),
        ModelType.WP: load_model(
            app.config['WP_MODEL_PATH'],
            app.config.get('WP_TOKENIZER_PATH', None)
        ),
        ModelType.PHC: load_model(
            app.config['PHC_MODEL_PATH'],
            app.config.get('PHC_TOKENIZER_PATH', None)
        )
    }

    app.logger.info('Loaded GPT2 models ({})'.format(', '.join(str(x) for x in GPT2_MODELS.keys())))

def _init_migrate(app):
    is_sqlite = app.config['SQLALCHEMY_DATABASE_URI'].startswith('sqlite:')
    migrate.init_app(app, render_as_batch=is_sqlite)