'''
Creates a TIFURecord fixture from generated samples.

'''

import json
import argparse
from pathlib import Path
from create_fixture_utils import (
    get_special_tokens_match_pattern,
    unescape_unicode
)

parser = argparse.ArgumentParser(description='Creates a TIFURecord fixture from generated samples.')
parser.add_argument('input_filename', type=Path, help='The input JSON file.')
parser.add_argument('output_filename', type=Path, help='The output JSON file.')
parser.add_argument('--special-tokens', dest='special_tokens_map_filename', type=Path,
                    default=None, help='A JSON file containing a dictionary of special tokens.')
parser.add_argument('--no-indent-json', dest='indent_json', action='store_false',
                    help='Don\'t indent the output JSON file. Default behaviour is to indent.')
args = parser.parse_args()

special_tokens_match_pattern = get_special_tokens_match_pattern(args.special_tokens_map_filename)
with open(args.input_filename) as input_file, \
     open(args.output_filename, 'w+') as output_file:
    data = json.load(input_file)
    output_data = []
    for sample in data:
        prompt = special_tokens_match_pattern.sub('', sample['prompt'])
        response = special_tokens_match_pattern.sub('', sample['response'])

        output_data.append({
            'post_title': unescape_unicode(prompt),
            'post_body': unescape_unicode(response)
        })

    json.dump(output_data, output_file, indent=args.indent_json)