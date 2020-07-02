'''
Scrapes PornHub video comments from r/PornHubComments.
'''

import re
import json
import praw
import psaw
import prawcore
import argparse
import datetime
import requests
import html5lib
from tqdm import tqdm
from pathlib import Path
from io_utils import prompt
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description='Scrapes PornHub video comments from r/PornHubComments.')
parser.add_argument('output', type=Path, help='The output filename.')
parser.add_argument('--client-id', type=str, help='The Reddit OAuth client id.', default=None)
parser.add_argument('--client-secret', type=str, help='The Reddit OAuth client secret.', default=None)
parser.add_argument('--user-agent', type=str, help='The Reddit user agent.', default='airedditor.pornhub_comment_scraper (by u/GalacticGlum)')
parser.add_argument('-c', '--credentials', type=Path, help='The filepath of a JSON file containing the Reddit credentials (client ' +
                    'id and secret). If both client id/secret and this is specified, this takes precedence.', default=None)
parser.add_argument('-l', '--limit', type=int, help='The number of posts to get. Defaults to infinite.', default=None)
parser.add_argument('--dump-batch', type=int, help='The number of submissions to batch process before writing to file.', default=8)
parser.add_argument('--no-indent-json', dest='indent_json', action='store_false',
                    help='Don\'t indent the output JSON file. Default behaviour is to indent.')
args = parser.parse_args()

client_id = args.client_id
client_secret = args.client_secret
if args.credentials.exists() and args.credentials.is_file():
    with open(args.credentials) as credentials_file:
        credentials = json.load(credentials_file)
        client_id = credentials.get('client_id', client_id)
        client_secret = credentials.get('client_secret', client_secret)

# Make sure that we have a client id AND a client secret.
assert client_id is not None and client_secret is not None

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=args.user_agent)
api = psaw.PushshiftAPI(reddit)

args.output.parent.mkdir(parents=True, exist_ok=True)

if args.output.exists():
    if prompt('The file \'{}\' already exists! '.format(args.output) + 
                'Continuing will overwrite the file. Are you sure?'):
        args.output.unlink()

def _add_schema_to_url(url, schema='https'):
    '''
    Prepends the specified URL schema to the specified URL if it doesn't
    already have a schema (ftp, http, or https).

    '''

    if not re.match('(?:http|ftp|https)://', url):
        url = '{}://'.format(schema) + url
    return url

# Get top comments (i.e. sorted in descending order of score) from the phc-bot
# The phc-bot provides the source for the comments based on r/pornhubcomments
comments = api.search_comments(author='phc-bot', sort_type='score', sort='desc')
results = []

# Maintain a collection of all scraped URLs
visited = set()
with tqdm(total=args.limit) as progress_bar:
    for index, comment in enumerate(comments):
        progress_bar.update(1)

        progress_bar.set_description('Post {} (score={})'.format(comment.submission.id, comment.submission.score))
        progress_bar.refresh()

        urls = re.findall(r'(?P<url>(https?://)?pornhub\.com/view_video.php*[^\s]+)', comment.body)
        if len(urls) == 0: continue
        # findall returns a tuple containing each capture group result; however,
        # we only care about the URL as a whole, not its components, so we can drop
        # the rest of the tuple and only keep the first element (the whole URL).
        urls = [_add_schema_to_url(x[0]) for x in urls]
        for i, url in enumerate(urls):
            if url.endswith(')'):    
                # The url shouldn't end with a closing bracket.
                # This is an artifact of markdown formatting.
                urls[i] = url[:-1]

        for url in urls:
            if url in visited: continue
            visited.add(url)
            
            progress_bar.write('- Scraping {}'.format(url))
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html5lib')

            for phc in soup.findAll('div', {'class': 'commentBlock'}):
                phc_body = phc.find('div', {'class': 'commentMessage'}).find('span').text.strip()
                # Filter for invalid comments. This seem to exist in the HTML
                # but are hidden on the website...
                if phc_body == '[[commentMessage]]': continue  
                
                # Get likes on comments
                phc_likes = int(phc.find('span', {'class': 'voteTotal'}).text)
                phc_author = phc.find('a', {'class': 'usernameLink'}) or \
                    phc.find('span', {'class': 'usernameLink'})

                if phc_author is None:
                    phc_author = None
                else:
                    phc_author = phc_author.text

                phc_data = {
                    'author': phc_author,
                    'likes': phc_likes,
                    'body': phc_body
                }

                results.append(phc_data)
            
        if index % args.dump_batch == 0:
            with open(args.output, 'w+') as output_file:
                json.dump(results, output_file, indent=args.indent_json)
        
            progress_bar.write('Writing to file (batch {})'.format(index // args.dump_batch))