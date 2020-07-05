import json
import click
from pathlib import Path
from flask.cli import with_appcontext
from ai_redditor_service.extensions import db
from ai_redditor_service.models import TIFURecord, WPRecord, PHCRecord

def init_app(app):
    '''
    Initializes Click commands with a Flask app context.

    :note:
        This will integrate the commands with the "flask run" command.

    '''

    app.cli.add_command(_init_db_command)
    app.cli.add_command(_load_fixture_command)

@click.command('init-db')
@with_appcontext
def _init_db_command():
    confirmation = click.confirm('Are you sure you would like to continue? This will drop and recreate all tables in the database.')
    if confirmation:
        db.create_all()
        click.echo('Initialized the database: dropped and recreated all tables.')  

def _load_tifu_fixture():
    pass

def _load_wp_fixture():
    pass

def _load_phc_fixture():
    pass

_TYPE_HANDLER_MAP = {
    'tifu': _load_tifu_fixture,
    'wp': _load_wp_fixture,
    'phc': _load_phc_fixture
}

@click.command('load-fixture')
@click.argument('record_type', type=click.Choice(_TYPE_HANDLER_MAP.keys(), case_sensitive=False))
@click.argument('fixture_filename', type=Path)
@with_appcontext
def _load_fixture_command(record_type, fixture_filename):
    _TYPE_HANDLER_MAP[record_type]()