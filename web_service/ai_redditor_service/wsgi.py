'''
The entrypoint for uWSGI.

'''

from ai_redditor_service import create_app

# Initialize the app
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0')