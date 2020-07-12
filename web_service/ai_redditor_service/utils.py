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