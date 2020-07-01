from flask import Blueprint, redirect, url_for, render_template

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return redirect(url_for('main.tifu_page'))

@bp.route('/tifu')
def tifu_page():
    return render_template('tifu.html')

@bp.route('/wp')
def writingprompts_page():
    return render_template('writingprompts.html')

@bp.route('/phc')
def phc_page():
    return render_template('phc.html')