import re
import time
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import (
    set_seed,
    AutoTokenizer, 
    AutoModelWithLMHead
)

parser = argparse.ArgumentParser(description='Generate text from a model with an language modelling head.')
parser.add_argument('model_name_or_path', type=str, help='The model checkpoint for weights initialization (i.e. a pretrained model).')
parser.add_argument('--prompt', type=str, default=None, help='A prompt for the model. Leave to None for no prompt.')
parser.add_argument('--samples', type=int, default=1, help='The number of samples to generate. Defaults to 1.')
parser.add_argument('--top-k', type=int, default=300, help='The number of highest probability vocabulary tokens ' +
                    'to keep for top-k-filtering. Between 1 and infinity. Default to 300.')
parser.add_argument('--top-p', type=float, default=1, help='The cumulative probability of parameter highest probability ' +
                    'vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.')
parser.add_argument('--num-return-sequences', type=int, default=10, help='The number of sequences to return per iteration. ' + 
                    'Defaults to 10.')
parser.add_argument('--max-iterations', type=int, default=5, help='Maximum number of iterations. Defaults to 5.')
parser.add_argument('--max-length', type=int, default=250, help='Maximum number of tokens to generate in a single iteration.')
parser.add_argument('--seed', type=int, default=None, help='The seed of the random engine.')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path)

print('- Loaded model and tokenizer from \'{}\''.format(args.model_name_or_path))

# Set seed to reproduce results.
if args.seed is None:
    # We use the current time as the seed rather than letting numpy seed
    # since we want to achieve consistent results across sessions.
    # Source: https://stackoverflow.com/a/45573061/7614083
    t = int(time.time() * 1000.0)
    args.seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) <<  8) + ((t & 0x000000ff) << 24)

set_seed(args.seed)

print('- Set seed to {}'.format(args.seed))

TRANSLATE_TOKEN = '<|eq_tok|>'
split_pattern = (
    f'^{re.escape(tokenizer.bos_token)}(?P<prompt>.+?)'
    f'(?:{re.escape(TRANSLATE_TOKEN)}(?P<response>.+?))*'
    f'{re.escape(tokenizer.eos_token)}'
)

split_regex = re.compile(split_pattern, flags=re.MULTILINE | re.DOTALL)

if args.prompt is None:
    args.prompt = '{} [WP]'.format(tokenizer.bos_token)
 
prompt_ids = tokenizer.encode(args.prompt, return_tensors='pt').to(model.device)

print('- Encoded prompt into token ids')

results = []
current_iteration = 0
with tqdm(total=args.samples) as progress_bar:
    while len(results) < args.samples and current_iteration < args.max_iterations:
        current_iteration += 1
        generated = model.generate(
            prompt_ids,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            top_k=args.top_k, top_p=args.top_p,
            num_return_sequences=args.num_return_sequences,
            max_length=args.max_length + len(prompt_ids), do_sample=True
        )

        for i in range(generated.size()[0]):
            if len(results) >= args.samples: break

            sentence_tokens = generated[i, :].tolist()
            decoded = tokenizer.decode(sentence_tokens)

            match = split_regex.match(decoded)
            if not match:
                progress_bar.write('- Could not split generated sequence into parts. Skipping...')
                continue

            prompt = match.group('prompt')
            response = match.group('response')
            if prompt is None or response is None:
                progress_bar.write('- Generated sequence has no prompt or response. Skipping...')
                continue

            progress_bar.update(1)
            results.append((prompt, response))

print(results)
