from celery.result import AsyncResult
from flask_expects_json import expects_json
from flask import Blueprint, current_app, g, jsonify

import ai_redditor_service.tasks as tasks
from ai_redditor_service.models import RecordType
from ai_redditor_service.extensions import celery as celery_app

bp = Blueprint('api', __name__, url_prefix='/api')

def error_response(message, status_code, **kwargs):
    response = jsonify(error=message, success=False, **kwargs)
    response.status_code = status_code

    return response

_RECORD_PROMPT_PREFIXES = {
    RecordType.TIFU: 'TIFU',
    RecordType.WP: '[WP]',
    RecordType.PHC: ''
}

generate_schema = {
    'type': 'object',
    'properties': {
        'prompt': {
            'type': ['string', 'null'],
            'default': None
        }
    }
}

@bp.route('/r/<any(tifu, wp, phc):record_type>/generate', methods=['POST'])
@expects_json(generate_schema, fill_defaults=True)
def generate_record(record_type):
    # Convert record type argument to enum
    record_type = RecordType[record_type.upper()]

    prompt = g.data['prompt']
    if prompt is None:
        prompt_prefix = _RECORD_PROMPT_PREFIXES[record_type]
        prompt = current_app.config['GPT2_BOS_TOKEN'] + prompt_prefix
    
    result = tasks.generate_record.delay(record_type, prompt=prompt, samples=1) 
    response_message = 'Queued up {} record generation.'.format(record_type)
    return jsonify(task_id=result.id, message=response_message, success=True), 202

@bp.route('/r/generate/<string:task_id>')
def generate_record_task_status(task_id):
    result_handle = AsyncResult(task_id, app=celery_app)
    is_ready = result_handle.ready()

    kwargs = {
        'is_ready': is_ready,
        'state': result_handle.state,
    }

    if is_ready:
        kwargs['uuid'] = result_handle.result[0]

    status_code = 201 if is_ready else 202
    return jsonify(success=True, **kwargs), status_code