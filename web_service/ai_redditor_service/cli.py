import click
from flask.cli import with_appcontext
from ai_redditor_service.extensions import db

def init_app(app):
    '''
    Initializes Click commands with a Flask app context.

    :note:
        This will integrate the commands with the "flask run" command.

    '''

    app.cli.add_command(_init_db_command)

@click.command('init-db')
@with_appcontext
def _init_db_command():
    confirmation = click.confirm('Are you sure you would like to continue? This will drop and recreate all tables in the database.')
    if confirmation:
        db.create_all()
        click.echo('Initialized the database: dropped and recreated all tables.')  

