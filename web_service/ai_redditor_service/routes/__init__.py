from ai_redditor_service.routes import main, api

def init_app(app):
    '''
    Initialize routes with Flask app context.

    '''

    # Register blueprints
    app.register_blueprint(main.bp)
    app.register_blueprint(api.bp)