'''
Creates a TIFURecord fixture from generated samples.

'''

import re
import json
import argparse
import functools
from pathlib import Path
from collections.abc import Iterable

parser = argparse.ArgumentParser(description='Creates a TIFURecord fixture from generated samples.')
parser.add_argument('input_filename', type=Path, help='The input JSON file.')
parser.add_argument('output_filename', type=Path, help='The output JSON file.')
parser.add_argument('--special-tokens', dest='special_tokens_map_filename', type=Path,
                    default=None, help='A JSON file containing a dictionary of special tokens.')
parser.add_argument('--no-indent-json', dest='indent_json', action='store_false',
                    help='Don\'t indent the output JSON file. Default behaviour is to indent.')
args = parser.parse_args()

special_tokens = []
if args.special_tokens_map_filename is not None:
    with open(args.special_tokens_map_filename) as special_tokens_file:
        values = list(json.load(special_tokens_file).values())

        # Flatten nested list:
        # We do this because the special tokens map contains an "additional_special_tokens"
        # field which consists of a list of additional special tokens. When we convert the
        # values from the special tokens map dictionary, the additional special tokens ends
        # up creating an element of type list, which is what we want to flatten out.
        for item in values:
            if isinstance(item, list):
                special_tokens.extend(item)
            else:
                special_tokens.append(item)

special_tokens_match_pattern = re.compile('|'.join(re.escape(token) for token in special_tokens))

with open(args.input_filename) as input_file, \
     open(args.output_filename, 'w+') as output_file:
    data = json.load(input_file)
    output_data = []
    for sample in data:
        prompt = special_tokens_match_pattern.sub('', sample['prompt'])
        response = special_tokens_match_pattern.sub('', sample['response'])

        output_data.append({
            'post_title': prompt,
            'post_body': response
        })

    json.dump(output_data, output_file, indent=args.indent_json)