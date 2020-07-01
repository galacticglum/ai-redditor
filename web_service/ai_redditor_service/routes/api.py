import json

from ai_redditor_service.models import Record
from ai_redditor_service.extensions import db
from flask_expects_json import expects_json
from flask import Blueprint, jsonify, current_app as app

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

# generate_record_schema = {
#     'type': 'object',
#     'additional'
# }

@bp.route('/r/generate', methods=['POST'])
# @expects_json(generate_record_schema)
def generate_record():
    '''
    Generates a record.

    :note:
        An optional JSON payload can be provided to the endpoint
        that indicates a prompt input for which the model should
        start generating from.

    '''

    pass