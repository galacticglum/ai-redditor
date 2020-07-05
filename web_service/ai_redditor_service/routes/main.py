import markdown
from sqlalchemy import func
from flask import Blueprint, redirect, url_for, render_template

from ai_redditor_service.forms import GeneratePostForm
from ai_redditor_service.models import TIFURecord, WPRecord, PHCRecord

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return redirect(url_for('main.tifu_page'))

def _select_random(model_class, **filter_kwargs):
    return model_class.query.filter_by(**filter_kwargs).order_by(func.random()).first()

@bp.route('/tifu', methods=('GET', 'POST'))
def tifu_page():
    generate_form = GeneratePostForm()
    if generate_form.validate_on_submit():
        # TODO: Use form data to generate post
        pass

    record = _select_random(TIFURecord, is_custom=False)
    return render_template('tifu.html', record=record, generate_form=generate_form)

@bp.route('/wp')
def writingprompts_page():
    generate_form = GeneratePostForm()
    if generate_form.validate_on_submit():
        # TODO: Use form data to generate post
        pass

    record = _select_random(WPRecord, is_custom=False)
    return render_template('writingprompts.html', record=record, generate_form=generate_form)

@bp.route('/phc')
def phc_page():
    return render_template('phc.html')