'''
The entrypoint for uWSGI.

'''

from ai_redditor_service import create_app
from ai_redditor_service.extensions import socketio

# Initialize the app
app = create_app()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0')