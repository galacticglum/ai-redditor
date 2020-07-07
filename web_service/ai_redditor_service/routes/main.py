import re
import time
from sqlalchemy import func
from flask import Blueprint, redirect, url_for, render_template, abort, current_app

from ai_redditor_service.extensions import db
from ai_redditor_service.forms import GeneratePostForm
from ai_redditor_service.string_utils import unescape_unicode
from ai_redditor_service.models import TIFURecord, WPRecord, PHCRecord
from ai_redditor_service.gpt2 import (
    ModelType,
    DecodeFormat,
    generate as gpt2_model_generate
)

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return redirect(url_for('main.tifu_page'))

def _select_random(model_class, **filter_kwargs):
    return model_class.query.filter_by(**filter_kwargs).order_by(func.random()).first()

def get_special_tokens_match_pattern(special_tokens):
    return re.compile('|'.join(re.escape(token) for token in special_tokens))

@bp.route('/tifu', defaults={'uuid': None}, methods=('GET', 'POST'))
@bp.route('/tifu/<string:uuid>', methods=('GET', 'POST'))
def tifu_page(uuid):
    generate_form = GeneratePostForm()
    if generate_form.validate_on_submit():
        prompt_title = generate_form.title.data
        prompt_post_body = generate_form.post_body.data

        model, tokenizer =  current_app.config['GPT2_MODELS'][ModelType.TIFU]
        prompt = '{}{}<|eq_tok|>'.format(tokenizer.bos_token, prompt_title)
        if prompt_post_body:
            prompt += prompt_post_body
        
        start_time = time.time()
        output = gpt2_model_generate(
            model, tokenizer,
            DecodeFormat.QUERY_ANSWER,
            prompt=prompt, samples=1
        )

        print('Generated in {:.2f} seconds'.format(time.time() - start_time))

        # Clean the model output:
        #  * remove specials tokens
        #  * decode unicode
        special_tokens_match_pattern = get_special_tokens_match_pattern(tokenizer.all_special_tokens)
        groups = {
            key: unescape_unicode(special_tokens_match_pattern.sub('', value)) \
                for key, value in output[0].groups.items()
        }

        record = TIFURecord(groups['prompt'], groups['response'], is_custom=True)

        # Add to database
        db.session.add(record)
        db.session.commit()
    elif uuid is None:
        record = _select_random(TIFURecord, is_custom=False)
    else:
        record = TIFURecord.query.filter_by(uuid=uuid).first()
        if record is None:
            abort(404)
    
    return render_template('tifu.html', record=record, generate_form=generate_form)

@bp.route('/wp', defaults={'uuid': None}, methods=('GET', 'POST'))
@bp.route('/wp/<string:uuid>', methods=('GET', 'POST'))
def writingprompts_page(uuid):
    generate_form = GeneratePostForm()
    if generate_form.validate_on_submit():
        # TODO: Use form data to generate post
        pass

    if uuid is None:
        record = _select_random(WPRecord, is_custom=False)
    else:
        record = WPRecord.query.filter_by(uuid=uuid).first()
        if record is None:
            abort(404)
    
    return render_template('writingprompts.html', record=record, generate_form=generate_form)

@bp.route('/phc')
def phc_page():
    return render_template('phc.html')

# CELERY STUFF
# TODO: FIX ME
from flask import request
import ai_redditor_service.tasks as tasks
from celery.result import AsyncResult
from ai_redditor_service.extensions import celery as celery_app

@bp.route('/add_task')
def test_celery():
    result = tasks.add.delay(
        request.args.get('x', type=int),
        request.args.get('y', type=int)
    )

    return str(result.id)

@bp.route('/add_task/<string:task_id>')
def task_status(task_id):
    result = AsyncResult(task_id, app=celery_app)
    is_ready = result.ready()
    if is_ready:
        return '{} (code: {})'.format(result.get(), result.state)
    else:
        return 'task not ready :('