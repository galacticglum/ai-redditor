from ai_redditor_service.routes import main

def init_app(app):
    '''
    Initialize routes with Flask app context.

    '''

    # Register blueprints
    app.register_blueprint(main.bp)