from ai_redditor_service.extensions import db

class Record(db.Model):
    '''
    The base model class for all records.

    :ivar id:
        The primary-key integer id of the record.
    :ivar uuid:
        The hexadecimal UUID of the record.
    '''

    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(32))
    is_custom = db.Column(db.Boolean)

class TIFURecord(Record):
    '''
    A TIFU post record.

    :ivar post_title:
        The title of the post.
    :ivar post_body:
        The body text (self text) of the post.

    '''

    __tablename__ = 'tifu_record'
    post_title = db.Column(db.String)
    post_body = db.Column(db.String)

class WPRecord(Record):
    '''
    A writingprompt and response pair record.

    :ivar prompt:
        The writingprompt.
    :ivar prompt_response:
        The response to the writingprompt. 

    '''

    __tablename__ = 'wp_record'
    prompt = db.Column(db.String)
    prompt_response = db.Column(db.String)

class PHCRecord(Record):
    '''
    A pornhub comment record.

    :ivar author_username:
        The username of the author.
    :ivar likes:
        The number of likes the comment has.
    :ivar comment:
        The comment text.

    '''

    __tablename__ = 'phc_record'
    author_username = db.Column(db.String)
    likes = db.Column(db.Integer)
    comment = db.Column(db.String)