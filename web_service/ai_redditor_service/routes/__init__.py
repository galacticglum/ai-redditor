from ai_redditor_service.routes import api

def init_app(app):
    '''
    Initialize routes with Flask app context.

    '''

    app.register_blueprint(api.bp)