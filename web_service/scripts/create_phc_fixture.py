'''
Creates a PHCRecord fixture from generated samples.

'''

import re
import json
import argparse
from pathlib import Path
from create_fixture_utils import get_special_tokens_match_pattern

parser = argparse.ArgumentParser(description='Creates a PHCRecord fixture from generated samples.')
parser.add_argument('input_filename', type=Path, help='The input JSON file.')
parser.add_argument('output_filename', type=Path, help='The output JSON file.')
parser.add_argument('--special-tokens', dest='special_tokens_map_filename', type=Path,
                    default=None, help='A JSON file containing a dictionary of special tokens.')
parser.add_argument('--no-indent-json', dest='indent_json', action='store_false',
                    help='Don\'t indent the output JSON file. Default behaviour is to indent.')
parser.add_argument('--default-likes', type=int, default=0, help='The default likes on a post if none is provided.')
args = parser.parse_args()

special_tokens_match_pattern = get_special_tokens_match_pattern(args.special_tokens_map_filename)
with open(args.input_filename) as input_file, \
     open(args.output_filename, 'w+') as output_file:
    data = json.load(input_file)
    output_data = []
    for sample in data:
        author_username = special_tokens_match_pattern.sub('', sample['groups']['author'])
        comment_body = special_tokens_match_pattern.sub('', sample['groups']['comment_body'])

        # Remove all non-numeric characters from the
        # likes string, since its an integer value.
        likes = re.sub('[^0-9]', '', sample['groups']['likes'])
        if not likes:
            likes = args.default_likes
        else:
            likes = int(likes)

        output_data.append({
            'likes': likes,
            'author_username': author_username,
            'comment': comment_body
        })

    json.dump(output_data, output_file, indent=args.indent_json)
