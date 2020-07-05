'''
Creates a PHCRecord fixture from generated samples.

'''

import re
import time
import json
import random
import argparse
from pathlib import Path
from create_fixture_utils import (
    get_special_tokens_match_pattern,
    unescape_unicode
)

parser = argparse.ArgumentParser(description='Creates a PHCRecord fixture from generated samples.')
parser.add_argument('input_filename', type=Path, help='The input JSON file.')
parser.add_argument('output_filename', type=Path, help='The output JSON file.')
parser.add_argument('--special-tokens', dest='special_tokens_map_filename', type=Path,
                    default=None, help='A JSON file containing a dictionary of special tokens.')
parser.add_argument('--no-indent-json', dest='indent_json', action='store_false',
                    help='Don\'t indent the output JSON file. Default behaviour is to indent.')
parser.add_argument('--default-likes-min', type=int, default=0, help='The minimum range on default likes.')
parser.add_argument('--default-likes-max', type=int, default=25, help='The maximum range on default likes.')
parser.add_argument('--seed', type=int, default=None, help='The seed of the random engine.')
args = parser.parse_args()

# Set seed to reproduce results.
if args.seed is None:
    # We use the current time as the seed rather than letting numpy seed
    # since we want to achieve consistent results across sessions.
    # Source: https://stackoverflow.com/a/45573061/7614083
    t = int(time.time() * 1000.0)
    args.seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) <<  8) + ((t & 0x000000ff) << 24)

random.seed(args.seed)
print('- Set seed to {}'.format(args.seed))

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
            likes = random.randint(args.default_likes_min, args.default_likes_max)
        else:
            likes = int(likes)
            
        output_data.append({
            'likes': likes,
            'author_username': unescape_unicode(author_username),
            'comment': unescape_unicode(comment_body)
        })

    json.dump(output_data, output_file, indent=args.indent_json)
