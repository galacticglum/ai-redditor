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

    title = StringField('title', validators=[DataRequired()])
    post_body = TextAreaField('post_body')
    submit = SubmitField('Submit')

class GeneratePHCForm(FlaskForm):
    '''
    A form for generating a PornHub comment.

    '''

    author_username = StringField('title', validators=[DataRequired()])
    likes = IntegerField('likes', validators=[DataRequired()])
    comment = TextAreaField('comment')
    submit = SubmitField('Submit')