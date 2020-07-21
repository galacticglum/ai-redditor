from flask import abort
from jsonschema import validate, ValidationError
from flask_expects_json.default_validator import DefaultValidatingDraft4Validator

def unescape_unicode(string):
    '''
    Unescape a string encoded with unicode_escape.

    :param string:
        A string value containing escaped unicode characters.
    
    '''

    return string.encode().decode('unicode_escape', errors='ignore')

def merge_dicts(x, y):
    '''
    Merges two dictionary objects.

    '''
    
    z = x.copy()
    z.update(y)
    return z

def validate_json(data, schema, force=False, fill_defaults=False):
    '''
    Validates a JSON schema.

    '''

    try:
        if fill_defaults:
            DefaultValidatingDraft4Validator(schema).validate(data)
        else:
            validate(data, schema)
    except ValidationError as exception:
        return abort(400, exception)

def all_empty(iterable):
    '''
    Returns whether the specified iterable is all empty.

    '''
    
    return all(not bool(x) for x in iterable)
