import tqdm
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
    app.cli.add_command(_build_record_refs)

@click.command('init-db')
@with_appcontext
def _init_db_command():
    confirmation = click.confirm('Are you sure you would like to continue? This will drop and recreate all tables in the database.')
    if confirmation:
        db.drop_all()
        db.create_all()
        click.echo('Initialized the database: dropped and recreated all tables.')  

def _load_tifu_fixture(data, progress_bar):
    for entry in data:
        progress_bar.update(1)

        record = TIFURecord(**entry)
        db.session.add(record)
    
    db.session.commit()

def _load_wp_fixture(data, progress_bar):
    for entry in data:
        progress_bar.update(1)

        record =  WPRecord(**entry)
        db.session.add(record)

    db.session.commit()

def _load_phc_fixture(data, progress_bar):
    for entry in data:
        progress_bar.update(1)

        record =  PHCRecord(**entry)
        db.session.add(record)

    db.session.commit()

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
    if not fixture_filename.is_file():
        raise ValueError('\'{}\' is not a file!'.format(
            fixture_filename.resolve()
        ))

    with open(fixture_filename) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError('\'{}\' does not contain a JSON-array!'.format(
            fixture_filename.resolve()
        ))
    
    with tqdm.tqdm(total=len(data)) as progress_bar:
        _TYPE_HANDLER_MAP[record_type](data, progress_bar)

_RECORD_TYPES = {
    'tifu': TIFURecord,
    'wp': WPRecord,
    'phc': PHCRecord
}

@click.command('build-record-refs')
@click.argument('record_type', type=click.Choice(_RECORD_TYPES.keys(), case_sensitive=False))
@with_appcontext
def _build_record_refs(record_type):
    confirmation = click.confirm(
        'Are you sure you would like to continue? '
        'This will recreate all {} record reference index tables.'.format(record_type.upper())
    )

    if not confirmation: return
    
    record_class = _RECORD_TYPES[record_type]
    # Remove all entries from reference index tables
    record_class._generated_record_ref_class.query.delete()
    record_class._dataset_record_ref_class.query.delete()

    for record in tqdm.tqdm(record_class.query.all()):
        if record.is_custom: continue
        if record.is_generated:
            record_ref = record._generated_record_ref_class(record_id=record.id)
        else:
            record_ref = record._dataset_record_ref_class(record_id=record.id)

        db.session.add(record_ref)
    db.session.commit()