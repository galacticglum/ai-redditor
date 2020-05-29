'''
A tool for scraping Reddit posts and comments.

'''

import json
import praw
import psaw
import argparse
import datetime
from tqdm import tqdm
from pathlib import Path

def prompt(question, default='yes'):
    '''
    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    '''

    valid = {
        'yes': True, 'y': True, 'ye': True,
        'no': False, 'n': False
    }

    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError('invalid default answer: \'%s\'' % default)

    while True:
        print(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print('Please respond with \'yes\' or \'no\' (or \'y\' or \'n\').\n')

parser = argparse.ArgumentParser(description='A tool for scraping Reddit posts and comments.')
parser.add_argument('subreddit', type=str, help='The subreddit to search in.')
parser.add_argument('output', type=str, help='The output filename.')
parser.add_argument('--client-id', type=str, help='The Reddit OAuth client id.', default=None)
parser.add_argument('--client-secret', type=str, help='The Reddit OAuth client secret.', default=None)
parser.add_argument('--user-agent', type=str, help='The Reddit user agent.', default='airedditor.reddit_scraper (by u/GalacticGlum)')
parser.add_argument('-c', '--credentials', type=str, help='The filepath of a JSON file containing the Reddit credentials (client ' +
                    'id and secret). If both client id/secret and this is specified, this takes precedence.', default=None)
parser.add_argument('-l', '--limit', type=int, help='The number of posts to get. Defaults to infinite.', default=None)
parser.add_argument('-q', '--query', type=str, help='A search query to perform on the post.', default=None)
parser.add_argument('-k', '--top-k', type=int, help='The number of top comments to get. If negative, all comments are scraped.', default=3)
parser.add_argument('--dump-batch', type=int, help='The number of submissions to batch process before writing to file.', default=8)
parser.add_argument('--no-indent-json', dest='indent_json', action='store_false',
                    help='Don\'t indent the output JSON file. Default behaviour is to indent.')
args = parser.parse_args()

client_id = args.client_id
client_secret = args.client_secret
credentials_filepath = Path(args.credentials)
if credentials_filepath.exists() and credentials_filepath.is_file():
    with open(credentials_filepath) as credentials_file:
        credentials = json.load(credentials_file)
        client_id = credentials.get('client_id', client_id)
        client_secret = credentials.get('client_secret', client_secret)

# Make sure that we have a client id AND a client secret.
assert client_id is not None and client_secret is not None

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=args.user_agent)
api = psaw.PushshiftAPI(reddit)

SUBMISSION_SERIALIZE_ATTRIBUTES = [
    'created_utc', 'id', 'name', 'permalink',
    'score', 'title', 'upvote_ratio', 'url'
]

COMMENT_SERIALIZE_ATTRIBUTES = [
    'body', 'created_utc', 'id', 'is_submitter',
    'link_id', 'parent_id', 'permalink', 'score', 'subreddit_id'
]

def _serialize_reddit_object(obj, attributes):
    data = {attribute: getattr(obj, attribute) for attribute in attributes}
    if obj.author is not None:
        data['author'] = obj.author.name
        data['author_id'] = obj.author.id
    data['subreddit'] = obj.subreddit.name
    data['subreddit_id'] = obj.subreddit.id

    return data

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
if output_path.exists() and prompt('The file \'{}\' already exists! '.format(output_path) +
    'Continuing will overwrite the file. Are you sure?'):
    output_path.unlink()
    
submissions = api.search_submissions(q=args.query, subreddit=args.subreddit, sort_type='score', sort='desc', limit=args.limit)
with tqdm(total=args.limit) as progress_bar:
    results = []
    for index, submission in enumerate(submissions):
        # Update the progress bar.
        progress_bar.set_description('Post {} (score={})'.format(submission.id, submission.score))
        progress_bar.refresh()

        submission.comment_sort = 'top'
        submission.comment_limit = None if args.top_k < 0 else args.top_k
        submission.comments.replace_more(0 or args.top_k)

        submission_data = _serialize_reddit_object(submission, SUBMISSION_SERIALIZE_ATTRIBUTES)
        submission_data['comments'] = []

        comments = list(submission.comments)[:args.top_k]
        for comment in comments:
            comment_data = _serialize_reddit_object(comment, COMMENT_SERIALIZE_ATTRIBUTES)
            submission_data['comments'].append(comment_data)

        results.append(submission_data)

        if index % args.dump_batch == 0:
            with open(output_path, 'w+') as output_file:
                json.dump(results, output_file, indent=args.indent_json)
        
            progress_bar.write('Writing to file (batch {})'.format(index // args.dump_batch))

        # Increment progress bar counter
        progress_bar.update(1)