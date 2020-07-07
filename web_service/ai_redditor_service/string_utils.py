def unescape_unicode(string):
    '''
    Unescape a string encoded with unicode_escape.

    :param string:
        A string value containing escaped unicode characters.
    
    '''

    return string.encode().decode('unicode_escape', errors='ignore')