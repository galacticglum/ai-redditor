from celery.result import AsyncResult
from flask_expects_json import expects_json
from flask import Blueprint, current_app, g, jsonify, url_for

import ai_redditor_service.tasks as tasks
from ai_redditor_service.models import RecordType, RECORD_MODEL_CLASSES
from ai_redditor_service.extensions import celery as celery_app

bp = Blueprint('api', __name__, url_prefix='/api')

def error_response(message, status_code, **kwargs):
    response = jsonify(error=message, success=False, **kwargs)
    response.status_code = status_code

    return response

get_random_record_schema = {
    'type': 'object',
    'properties': {
        'is_custom': {
            'type': ['boolean', 'null'],
            'default': None
        }, 
        'is_generated': {
            'type': ['boolean', 'null'],
            'default': True
        }
    }
}

@bp.route('/r/<any(tifu, wp, phc):record_type>/random', methods=['POST'])
@expects_json(get_random_record_schema, fill_defaults=True)
def get_random_record(record_type):
    '''
    Gets a random record of the specified type. An optional JSON object
    can be specified to filter the records that are sampled.

    :param ``is_custom``:
        A boolean value indicating whether the record was user generated with
        a custom prompt. If not specified or null (None), all values of this
        field are accepted. Defaults to None.       
    :param ``is_generated``:
        A boolean value indicating whether the record was generated by the 
        GPT2 model, or if it is an original record from a dataset. If not
        specified or null (None), all values of this field are accepeted.
        Defaults to True.

    '''

    filter_kwargs = {}
    is_custom = g.data['is_custom']
    if is_custom is not None:
        filter_kwargs['is_custom'] = is_custom
    
    is_generated = g.data['is_generated']
    if is_generated is not None:
        filter_kwargs['is_generated'] = is_generated

    # Convert record type argument to enum
    record_type = RecordType[record_type.upper()]
    record = RECORD_MODEL_CLASSES[record_type].select_random(**filter_kwargs)
    if record is None:
        return error_response('No {} record could be found with the provided constraints'.format(
            record_type.name
        ), 404)

    return jsonify(record.to_dict()), 201

_RECORD_PROMPT_PREFIXES = {
    RecordType.TIFU: 'TIFU ',
    RecordType.WP: '[WP] ',
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
    '''
    Generates a record of the specified type. An optional JSON object
    can be specified to this endpoint providing a prompt to the model.

    '''

    # Convert record type argument to enum
    record_type = RecordType[record_type.upper()]
    bos_token = current_app.config['GPT2_BOS_TOKEN']
    prompt_prefix = _RECORD_PROMPT_PREFIXES[record_type]

    prompt = g.data['prompt']
    if prompt is None:
        prompt = bos_token + prompt_prefix
    else:
        prompt = prompt.strip()
        
        # Make sure that the prompt starts with the bos token and prompt prefix
        if prompt.startswith(bos_token):
            prompt = prompt.replace(bos_token,  '')
        
        if prompt.startswith(prompt_prefix):
            prompt = prompt.replace(prompt_prefix, '')

        prompt = bos_token + prompt_prefix + prompt
    
    result = tasks.generate_record.delay(record_type, prompt=prompt, samples=1) 
    response_message = 'Queued up {} record generation.'.format(record_type.name)

    return jsonify(
        task_id=result.id,
        task_status_endpoint=url_for('api.generate_record_task_status', task_id=result.id),
        message=response_message,
        success=True
    ), 202

_RECORD_ROUTE_MAP = {
    RecordType.TIFU: 'tifu',
    RecordType.WP: 'writingprompts',
    RecordType.PHC: 'phc',
}

@bp.route('/r/generate/<string:task_id>')
def generate_record_task_status(task_id):
    '''
    Gets the status of a record generation task given its id.

    '''

    result_handle = AsyncResult(task_id, app=celery_app)
    is_ready = result_handle.ready()

    kwargs = {
        'is_ready': is_ready,
        'state': result_handle.state,
    }

    if is_ready:
        record_type, uuids = result_handle.result
        kwargs['uuid'] = uuids[0]

        route = 'main.{}_page'.format(_RECORD_ROUTE_MAP[record_type])
        kwargs['permalink'] = url_for(route, uuid=uuids[0])

    status_code = 201 if is_ready else 202
    return jsonify(success=True, **kwargs), status_code