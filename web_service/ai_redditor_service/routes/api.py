import json

from ai_redditor_service.models import Record
from ai_redditor_service.extensions import db
from flask_expects_json import expects_json
from flask import Blueprint, jsonify, current_app as app, g

bp = Blueprint('api', __name__, url_prefix='/api')

def error_response(message, status_code, **kwargs):
    '''
    Creates a :class:`flask.Response` with the specified status
    code containing a JSON object containing an error message 
    and other metadata. 

    :param message:
        A message explaining the error.
    :param status_code:
        The error's status code.

    '''
    
    response = jsonify(error=message, success=False, **kwargs)
    response.status_code = status_code

    return response

generate_tifu_record_schema = {
    'type': 'object',
    'properties': {
        'title': {
            'type': [ 'string', 'null' ],
            'default': None,
            'description': 'The title of the TIFU post.'
        },
        'body': {
            'type': [ 'string', 'null' ],
            'default': None,
            'description': 'The body of the TIFU post.'
        }
    },
    'dependencies': {
        'body': { 'required': ['title'] }
    }
}

@bp.route('/r/tifu/generate', methods=['POST'])
@expects_json(generate_tifu_record_schema)
def generate_tifu_record():
    '''
    Generates a TIFU record.

    '''

    title, body = g.data['title'], g.data['body']
    has_prompt = title is not None or body is not None

    return '<OOOGABOOGA>'