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

    @app.template_filter('remove_p_tags')
    def remove_p_tags_filter(string):
        string = re.sub(r'</p>(?=.*</p>)', '<br/><br/>', string, flags=re.DOTALL)
        string = re.sub(r'<p>|</p>', '', string)
        return string