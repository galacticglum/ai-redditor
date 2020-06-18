import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelWithLMHead

parser = argparse.ArgumentParser(description='Generate text from a model with an language modelling head.')
parser.add_argument('model', type=str, dest='model_name_or_path',
                    help='The model checkpoint for weights initialization (i.e. a pretrained model).')
parser.add_argument('--prompt', type=str, default=None, help='A prompt for the model. Leave to None for no prompt.')
parser.add_argument('--samples', type=int, default=1, help='The number of samples to generate. Defaults to 1.')
parser.add_argument('--top-k', type=int, default=50, help='The number of highest probability vocabulary tokens ' +
                    'to keep for top-k-filtering. Between 1 and infinity. Default to 50.')
parser.add_argument('--top-p', type=float, default=1, help='The cumulative probability of parameter highest probability ' +
                    'vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path)

prompt_ids = None
if args.prompt is not None:
    prompt_ids = tokenizer.tokenize(args.prompt)

output = model.generate(
    input_ids=prompt_ids, do_sample=True,
    top_k=args.top_k, top_p=args.top_p,
    num_return_sequences=args.samples
)

for index, sample in enumerate(output):
    decoded_sample = tokenizer.decode(sample)    
    print('Sample {}: {}'.format(index + 1, decoded_sample))