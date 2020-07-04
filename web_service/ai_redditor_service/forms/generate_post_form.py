from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length

class GeneratePostForm(FlaskForm):
    '''
    A form for generating a standard Reddit post.

    :note:
        A standard Reddit post, in reference to this model, is in
        a query-answer format. This format can be thought of a pair
        of strings (referred to as the query and answer respectively)
        where the answer is a mapping of the query into some other
        translation space. For example, the query might be a sentence in
        English and the answer might be the same sentence translated to German.

        In the case of Reddit post text generation, the query is the title of the post
        (such as a writingprompt) and the answer is the corresponding post selftext or
        comment (such as a story written from the prompt).

    '''

    title = StringField('title', validators=[DataRequired()])
    post_body = TextAreaField('post_body')
    submit = SubmitField('Submit')