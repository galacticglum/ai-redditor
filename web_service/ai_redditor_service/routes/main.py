import re
import time
from sqlalchemy import func
from flask import Blueprint, redirect, url_for, render_template, abort, current_app

from ai_redditor_service.forms import GeneratePostForm
from ai_redditor_service.models import TIFURecord, WPRecord, PHCRecord

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return redirect(url_for('main.tifu_page'))

def _select_random(model_class, **filter_kwargs):
    return model_class.query.filter_by(**filter_kwargs).order_by(func.random()).first()

@bp.route('/tifu', defaults={'uuid': None})
@bp.route('/tifu/<string:uuid>')
def tifu_page(uuid):
    if uuid is None:
        record = _select_random(TIFURecord, is_custom=False)
    else:
        record = TIFURecord.query.filter_by(uuid=uuid).first()
        if record is None:
            abort(404)
    
    return render_template('tifu.html', record=record, generate_form=GeneratePostForm())

@bp.route('/wp', defaults={'uuid': None}, methods=('GET', 'POST'))
@bp.route('/wp/<string:uuid>', methods=('GET', 'POST'))
def writingprompts_page(uuid):
    if uuid is None:
        record = _select_random(WPRecord, is_custom=False)
    else:
        record = WPRecord.query.filter_by(uuid=uuid).first()
        if record is None:
            abort(404)
    
    return render_template('writingprompts.html', record=record, generate_form=GeneratePostForm())

@bp.route('/phc')
def phc_page():
    return render_template('phc.html')