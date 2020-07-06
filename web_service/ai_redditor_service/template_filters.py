import re
import urllib
import markdown
from flask import Markup

def init_app(app):
    '''
    Initializes template filters with a Flask app context.

    '''

    @app.template_filter('markdown')
    def markdown_filter(string):
        return markdown.markdown(string, extensions=['nl2br'])

    @app.template_filter('sanitize_for_md')
    def sanitize_for_md_filter(string):
        string = re.sub(r'</p>(?=.*</p>)', '<br/><br/>', string, flags=re.DOTALL)
        string = re.sub(r'<p>|</p>', '', string)
        return string

    @app.template_filter('urlencode')
    def urlencode_filter(value):
        print(value)
        if isinstance(value, Markup):
            value = value.unescape()

        value = urllib.parse.quote_plus(value.encode('utf8'))
        return Markup(value)