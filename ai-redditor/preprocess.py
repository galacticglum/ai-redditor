'''
Preprocess scraped Reddit posts into translations of the form "source=target".

'''

import json
import argparse
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser(description='Preprocess scraped Reddit posts into translations of the form "source=target".')
parser.add_argument('input', help='A path to the JSON file containing the Reddit posts.', type=str)
parser.add_argument('output', help='The output filename.', type=str)
parser.add_argument('source_attribute', help='The name of the source attribute. Should be a field or array. ' +
                    'In the case that the mapped attribute is an array, a separate translation will be ' +
                    'created for each value in the array.', type=str)
parser.add_argument('target_attribute', help='The name of the target attribute. Should be a field or array. ' +
                    'In the case that the mapped attribute is an array, a separate translation will be ' +
                    'created for each value in the array.', type=str)
parser.add_argument('--start-token', help='The start BOS (beginning of sentence) token. Defaults to \'<BOS>\'.',
                    type=str, default='<|bos|>')
parser.add_argument('--translate-token', help='The translation token. Defaults to \'<TRANSLATE_TOKEN>\'.',
                    type=str, default='<|eq_tok|>')
parser.add_argument('--end-token', help='The end EOS (end of sentence) token. Defaults to \'<EOS>\'.',
                    type=str, default='<|eos|>')
parser.add_argument('--no-filter', dest='filter', action='store_false', help='Don\'t filter posts for ' +
                    '"[deleted]", "[removed]", and other useless values. Default behaviour is to filter.')

# A list of common Reddit bots to ignore.
_DEFAULT_USER_BLACKLIST = [
    '6l4z3', # AutoModerator,
    'nb6f1'  # WritingPromptsRobot
]

parser.add_argument('--user-blacklist', nargs='+', type=str, default=_DEFAULT_USER_BLACKLIST,
                    help='A list of Reddit user IDs whose submissions and comments to ignore.')
args = parser.parse_args()

def filter_text(text):
    '''
    Filter for [deleted] and [removed].

    '''

    text = text.strip()
    return text == '' or text == '[deleted]' or text == '[removed]'

def get_text(attribute, value):
    '''
    Get text of a Reddit object.

    '''

    return value['body'] if attribute == 'comments' else value

def is_blacklisted(user_id):
    '''
    Checks if the specified ``user_id`` is blacklisted.

    '''

    return user_id in args.user_blacklist

with open(Path(args.input), 'r') as input_file, \
    open(Path(args.output), 'wb+') as output_file:

    inputs = json.load(input_file)
    for data in tqdm(inputs):
        # Treat both sources and target as list, even if it is not a JSON array.
        # This simplifies the logic for iterating over them.

        # Map sources to list
        sources = data[args.source_attribute]
        if not isinstance(sources, list):
            sources = [sources]

        # Map targets to list
        targets = data[args.target_attribute]
        if not isinstance(targets, list):
            targets = [targets]

        for source in sources:
            if 'author_id' in source and is_blacklisted(source['author_id']): continue

            # If the source is invalid, we can skip filtering the target.
            source_text = get_text(args.source_attribute, source)
            if args.filter and filter_text(source_text): continue

            for target in targets:
                if 'author_id' in target and is_blacklisted(target['author_id']): continue

                target_text = get_text(args.target_attribute, target)
                if args.filter and filter_text(target_text): continue
      
                output_file.write('{}{}{}{}{}'.format(
                    args.start_token,
                    source_text,
                    args.translate_token,
                    target_text,
                    args.end_token
                ).encode('unicode_escape'))
                output_file.write(b'\n')