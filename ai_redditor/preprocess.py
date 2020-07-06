'''
Preprocess scraped Reddit posts into translations of the form "source=target".

'''

import time
import json
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from string import Formatter

parser = argparse.ArgumentParser(description='Preprocess scraped Reddit posts into translations of the form "source=target".')
parser.add_argument('input', help='A path to the JSON file containing the Reddit posts.', type=Path)
parser.add_argument('output', help='The output filename.', type=Path)
parser.add_argument('--source-attribute', help='The name of the source attribute. Should be a field or array. ' +
                    'In the case that the mapped attribute is an array, a separate translation will be ' +
                    'created for each value in the array.', type=str, default=None)
parser.add_argument('--target-attribute', help='The name of the target attribute. Should be a field or array. ' +
                    'In the case that the mapped attribute is an array, a separate translation will be ' +
                    'created for each value in the array.', type=str, default=None)
parser.add_argument('--start-token', help='The start BOS (beginning of sentence) token. Defaults to \'<BOS>\'.',
                    type=str, default='<|bos|>')
parser.add_argument('--translate-token', help='The translation token. Defaults to \'<TRANSLATE_TOKEN>\'.',
                    type=str, default='<|eq_tok|>')
parser.add_argument('--end-token', help='The end EOS (end of sentence) token. Defaults to \'<EOS>\'.',
                    type=str, default='<|eos|>')
parser.add_argument('--format', help='The format of the processed examples. Uses Python formatting notation to inject JSON attributes. ' +
                    'For example, the format string \"{author}<|eq_tok|>{post}\" will replace \"{author}\" and \"{post}\" with their ' +
                    'respective values in the JSON object. Special tokens should be included directly in this format string. ' +
                    'This will override the --source-attribute and --target-attribute mappings.', type=str, default=None)
parser.add_argument('--no-filter', dest='filter', action='store_false', help='Don\'t filter posts for ' +
                    '"[deleted]", "[removed]", and other useless values. Default behaviour is to filter.')
parser.add_argument('--no-split', dest='split', action='store_false', help='Indicates whether the dataset ' +
                    'should be split into train and test sets. Defaults to True.')
parser.add_argument('--test-split-ratio', default=0.30, help='The ratio of the dataset that is allocated ' +
                    'to testing. Defaults to 30%%.', type=float)
parser.add_argument('--dataset-ratio', default=1.0, help='The ratio of samples to include in the dataset. ' +
                    'This will generate a dataset file from the first X percent of samples. Defaults to 100%%.',
                    type=float)
parser.add_argument('--shuffle', action='store_true', help='Shuffle the samples before outputting datasets.')
parser.add_argument('--seed', type=int, default=None, help='The seed of the random engine.')

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

# Set seed
if args.seed is None:
    # We use the current time as the seed rather than letting numpy seed
    # since we want to achieve consistent results across sessions.
    # Source: https://stackoverflow.com/a/45573061/7614083
    t = int(time.time() * 1000.0)
    args.seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) <<  8) + ((t & 0x000000ff) << 24)

random.seed(args.seed)
print('- Set seed to {}'.format(args.seed))

if (args.source_attribute is None or args.target_attribute is None) and args.format is None:
    raise ValueError('Both --source-attribute/--target-attribute AND --format cannot be None.')

train_set_output = args.output.parent / (args.output.stem + '_train' + args.output.suffix)
test_set_output = args.output.parent / (args.output.stem + '_test' + args.output.suffix)
with open(args.input, 'r') as input_file:
    inputs = json.load(input_file)
    samples = []
    for data in tqdm(inputs):
        if args.format is None:
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
        
                    samples.append('{}{}{}{}{}'.format(
                        args.start_token,
                        source_text,
                        args.translate_token,
                        target_text,
                        args.end_token
                    ).encode('unicode_escape') + b'\n')
        else:
            format_variables = [name for _, name , _, _ in Formatter().parse(args.format) if name is not None]
            format_kwargs = {format_variable: data[format_variable] for format_variable in format_variables}
            samples.append(args.format.format(**format_kwargs).encode('unicode_escape') + b'\n')

    dataset_size = int(len(samples) * args.dataset_ratio)
    if args.shuffle:
        samples = random.sample(samples, dataset_size)
        print(
            '- Using random sample consisting of {}%% ({} samples) '.format(
                args.dataset_ratio * 100,
                dataset_size
            ) + 'of the dataset for preprocessing.'
        )
    else:
        samples = samples[:dataset_size]
        print(
            '- Using the first {} samples ({}%% '.format(
                dataset_size,
                args.dataset_ratio * 100
            ) + 'of the dataset) for preprocessing.'
        )

    train_samples_amount = int(len(samples) * (1 - args.test_split_ratio))

    # Write train set
    with open(train_set_output, 'wb+') as train_output_file:
        for train_sample in samples[:train_samples_amount]:
            train_output_file.write(train_sample)
    
    # Write test set
    with open(test_set_output, 'wb+') as test_output_file:
        for test_sample in samples[train_samples_amount:]:
            test_output_file.write(test_sample)

    print('- Created training dataset consisting of {} examples.'.format(train_samples_amount))
    print('- Created testing dataset consisting of {} examples.'.format(len(samples) - train_samples_amount))