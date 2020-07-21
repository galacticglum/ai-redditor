from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Length

class GeneratePostForm(FlaskForm):
    '''
    A form for generating a standard Reddit post.

    :note:
        A standard Reddit post, in reference to this model, is in
        a query-answer format.

    '''

    post_title = StringField('post_title', validators=[DataRequired()])
    post_body = TextAreaField('post_body')
    submit = SubmitField('Submit')

class GeneratePHCForm(FlaskForm):
    '''
    A form for generating a PornHub comment.

    '''

    author = StringField('author')
    likes = IntegerField('likes')
    comment = TextAreaField('comment')
    submit = SubmitField('Submit')