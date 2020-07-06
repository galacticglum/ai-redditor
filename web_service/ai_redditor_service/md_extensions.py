import re
import markdown
from flask import Markup

def init_app(app):
    '''
    Initializes extensions with a Flask app context.

    '''

    @app.template_filter('markdown')
    def markdown_filter(string):
        return markdown.markdown(string, extensions=['nl2br'])

    @app.template_filter('sanitize_for_md')
    def sanitize_for_md(string):
        string = re.sub(r'</p>(?=.*</p>)', '<br/><br/>', string, flags=re.DOTALL)
        string = re.sub(r'<p>|</p>', '', string)
        return string