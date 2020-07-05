import uuid
from ai_redditor_service.extensions import db

class RecordMixin(object):
    '''
    The base model class for all records.

    :ivar id:
        The primary-key integer id of the record.
    :ivar uuid:
        The hexadecimal UUID of the record.
    :ivar is_custom:
        Whether the record was generated with a custom prompt.

    '''

    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(32), unique=True, index=True)
    is_custom = db.Column(db.Boolean)

    def __init__(self, uuid=None, is_custom=False):
        '''
        Base constructor for Record models.

        '''

        if uuid is None:
            uuid = uuid.uuid4().hex

        self.uuid = uuid
        self.is_custom = is_custom
        
class TIFURecord(db.Model, RecordMixin):
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

    def __init__(self, post_title, post_body, uuid=None, is_custom=False):
        '''
        Initializes an instance of :class:`TIFURecord`.

        :param post_title:
            The title of the post.
        :param post_body:
            The body text (self text) of the post.
        :param uuid:
            A 128-bit hexademical string representing the unique integer id of this record.
            Defaults to None, meaning that the UUID will be randomly generated.
        :param is_custom:
            Whether the record was generated with a custom prompt. Defaults to False.
        
        '''

        super(RecordMixin, self).__init__(uuid, is_custom)
        self.post_title = post_title
        self.post_body = post_body

class WPRecord(db.Model, RecordMixin):
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

    def __init__(self, prompt, prompt_response, uuid=None, is_custom=False):
        '''
        Initializes an instance of :class:`WPRecord`.

        :param prompt:
            The writingprompt.
        :param prompt_response:
            The response to the writingprompt. 
        :param uuid:
            A 128-bit hexademical string representing the unique integer id of this record.
            Defaults to None, meaning that the UUID will be randomly generated.
        :param is_custom:
            Whether the record was generated with a custom prompt. Defaults to False.
        
        '''

        super(RecordMixin, self).__init__(uuid, is_custom)
        self.prompt = prompt
        self.prompt_response = prompt_response

class PHCRecord(db.Model, RecordMixin):
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

    def __init__(self, author_username, likes, comment, uuid=None, is_custom=False):
        '''
        Initializes an instance of :class:`WPRecord`.

        :param author_username:
            The username of the author.
        :param likes:
            The number of likes the comment has.
        :param comment:
            The comment text.
        :param uuid:
            A 128-bit hexademical string representing the unique integer id of this record.
            Defaults to None, meaning that the UUID will be randomly generated.
        :param is_custom:
            Whether the record was generated with a custom prompt. Defaults to False.
        
        '''

        super(RecordMixin, self).__init__(uuid, is_custom)
        self.author_username = author_username
        self.likes = likes
        self.comment = comment