import re
import time
from sqlalchemy import func
from flask import Blueprint, redirect, url_for, render_template, abort, current_app

from ai_redditor_service.forms import GeneratePostForm, GeneratePHCForm
from ai_redditor_service.models import TIFURecord, WPRecord, PHCRecord

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return redirect(url_for('main.tifu_page'))

def _record_route(record_class, template_name, generate_form, uuid=None):
    if uuid is None:
        record = record_class.select_random(is_custom=False, is_generated=True)
    else:
        record = record_class.query.filter_by(uuid=uuid).first()
        if record is None:
            abort(404)
        
    return render_template(template_name, record=record, generate_form=generate_form)

@bp.route('/tifu', defaults={'uuid': None})
@bp.route('/tifu/<string:uuid>')
def tifu_page(uuid):
    return _record_route(TIFURecord, 'tifu.html', GeneratePostForm(), uuid=uuid)

@bp.route('/wp', defaults={'uuid': None})
@bp.route('/wp/<string:uuid>')
def writingprompts_page(uuid):
    return _record_route(WPRecord, 'writingprompts.html', GeneratePostForm(), uuid=uuid)

@bp.route('/phc', defaults={'uuid': None})
@bp.route('/phc/<string:uuid>')
def phc_page(uuid):
    # TODO: change form to one specific for pornhub comments
    return _record_route(PHCRecord, 'phc.html', GeneratePHCForm(), uuid=uuid)