from ai_redditor_service.forms import GeneratePostForm
from flask import Blueprint, redirect, url_for, render_template

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return redirect(url_for('main.tifu_page'))

@bp.route('/tifu', methods=('GET', 'POST'))
def tifu_page():
    generate_form = GeneratePostForm()
    if generate_form.validate_on_submit():
        # TODO: Use form data to generate post
        pass

    return render_template('tifu.html', generate_form=generate_form)

@bp.route('/wp')
def writingprompts_page():
    return render_template('writingprompts.html')

@bp.route('/phc')
def phc_page():
    return render_template('phc.html')