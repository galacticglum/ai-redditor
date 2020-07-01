from ai_redditor_service.routes import api, main

def init_app(app):
    '''
    Initialize routes with Flask app context.

    '''

    app.register_blueprint(main.bp)
    app.register_blueprint(api.bp)
