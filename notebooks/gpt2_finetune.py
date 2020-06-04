import argparse
import gpt_2_simple as gpt2
from pathlib import Path

parser = argparse.ArgumentParser(description='Finetune a GPT2 model on text data.')
parser.add_argument('mode', help='Either \'train\' or \'generate\'.', type=str)
parser.add_argument('dataset', help='The path to the dataset file.', type=str)
parser.add_argument('--start-token', help='The beginning of sentence token.', type=str, default='<BOS>')
parser.add_argument('--translate-token', help='The translate token.', type=str, default='<TRANSLATE_TOKEN>')
parser.add_argument('--end-token', help='The end of sentence token.', type=str, default='<EOS>')
parser.add_argument('--model-name', help='The name of the model. Either \'124M\' or \'355M\'.', default='124M')
parser.add_argument('--steps', help='The number of steps to train for.', default=1000)
parser.add_argument('--prompt', help='Text to prompt the generator with.', type=str, default='')
args = parser.parse_args()

if not (Path('models') / args.model_name).is_dir():
    gpt2.download_gpt2(model_name=args.model_name)

session = gpt2.start_tf_sess()
if args.mode == 'train':
    gpt2.finetune(session, args.dataset, model_name=args.model_name, steps=args.steps)
elif args.mode == 'generate':
    text = gpt2.generate(session, prefix=START_TOKEN + args.prompt, truncate=END_TOKEN, return_as_list=True)[0]
    print(text)