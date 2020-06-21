'''
Train or fine-tune a GPT-2 model,

'''

import re
import time
import json
import math
import torch
import shutil
import logging
import argparse
import functools
from pathlib import Path
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    WEIGHTS_NAME,
    CONFIG_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
    get_linear_schedule_with_warmup
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

def get_special_tokens(filepath=None):
    '''
    Gets a dictionary mapping of the special tokens.

    :param filepath:
        A path to a JSON file containing a dictionary of special tokens.
        If not specified, default values are used.

    '''

    _DEFAULT_SPECIAL_TOKENS = {
        'bos_token': '<|bos|>', # Beginning of sentence token
        'eos_token': '<|eos|>', # End of sentence token
        'pad_token': '<|pad|>', # Padding token
        'additional_special_tokens': [
            '<|eq_tok|>' # Query/answer separator token (translation separator token).
        ]
    }
    
    # If not JSON file has been provided, use the default special tokens.
    if filepath is None: return _DEFAULT_SPECIAL_TOKENS

    # Load special tokens from JSON and fill in any missing values
    with open(filepath, 'r') as file:
        data = json.load(file)
    
        for key in _DEFAULT_SPECIAL_TOKENS:
            if key in data: continue
            data[key] = _DEFAULT_SPECIAL_TOKENS[key]

    return data

def get_checkpoints(directory, prefix='checkpoint', use_mtime=False, sort_ascending=True):
    '''
    Gets all the checkpoints contained in the specified directory, sorted by
    number or, if specified, by modified time.

    :param directory:
        The path to the directory containing the checkpoints.
    :param prefix:
        The prefix of all checkpoint files. Default to "checkpoint".
    :param use_mtime:
        Whether to sort by modification time instead of checkpoint number.
        Defaults to False.
    :param sort_ascending:
        Whether to sort in ascending or descending order. Defaults to True,
        meaning that the checkpoints will be sorted in ascending order.
    :returns:
        A list of Path-like objects containing the location of each checkpoint
        in the specified `directory`.

    '''

    checkpoints = []
    checkpoint_files = Path(directory).glob('{}-*'.format(prefix))
    for filepath in checkpoint_files:
        if use_mtime:
            # Sort by modified time
            checkpoints.append((filepath.lstat().st_mtime, filepath))
        else:
            # Sort by checkpoint number
            matches = re.match(r'.*{}-([0-9]+)'.format(prefix), str(filepath))
            if matches and matches.groups():
                checkpoints.append((int(matches.groups()[0]), filepath))

    return [checkpoint for _, checkpoint in sorted(checkpoints, key=lambda x: x[0])]

def rotate_checkpoints(directory, max_checkpoints=None, prefix='checkpoint', use_mtime=False):
    '''
    Rotate checkpoints saved in the specified directory.

    :param directory:
        The path to the directory containing the checkpoints.
    :param max_checkpoints:
        The maximum number of checkpoints to keep before deleting older ones.
        A negative or None value means that there is no limit.
    :param prefix:
        The prefix of all checkpoint files. Default to "checkpoint".
    :param use_mtime:
        Whether to sort by modification time instead of checkpoint number.
        Defaults to False.

    '''

    if not max_checkpoints: return
    if max_checkpoints <= 0: return

    checkpoints = get_checkpoints(directory, prefix, use_mtime)
    if len(checkpoints) <= max_checkpoints: return

    n = max(0, len(checkpoints) - max_checkpoints)
    for checkpoint in checkpoints[:n]:
        shutil.rmtree(checkpoint)

def get_dataset(filepath, tokenizer, block_size, line_by_line=False, overwrite_cache=False):
    '''
    Load a dataset from the specified filepath.

    :param filepath:
        The filepath of the dataset.
    :param tokenizer:
        The tokenizer to parse the dataset with.
    :param block_size:
        The length of a single input sequence (block).
    :param line_by_line:
        Indicates whether distinct lines of text in the dataset are to be handled as
        separate sequence (i.e. whether to add the BOS adn EOS tokens to each line).
        Defaults to False.
    :param overwrite_cache:
        Overwrite the cached training and evaluation sets. Defaults to False.
    :returns:
        A :class:`torch.utils.data.Dataset` object.

    '''

    if line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=filepath, block_size=block_size)
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=filepath,
            block_size=block_size, overwrite_cache=overwrite_cache
        )

def mask_tokens(inputs, tokenizer, args):
    '''
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

    '''

    if tokenizer.mask_token is None:
        raise ValueError(
            'This tokenizer does not have a mask token which is necessary for masked language modeling. '
            'Remove the --mlm flag if you want to use this tokenizer.'
        )

    labels = inputs.clone()

    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]

    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def _collate(tokenizer, examples):
    '''
    Collate examples for model input.

    '''

    if tokenizer._pad_token is None:
        return pad_sequence(examples, batch_first=True)
        
    return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

def train(args, dataset, model, tokenizer):
    '''
    Train the model.

    '''

    if args.local_rank in [-1, 0]:
        summary_writer = SummaryWriter()
    
    batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    collate_func = functools.partial(_collate, tokenizer)
    sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_func)

    if args.global_steps > 0:
        total_steps = args.global_steps
        max_epochs = args.global_steps // (len(data_loader) // args.gradient_accumulation_steps) + 1
    else:
        total_steps = len(data_loader) // args.gradient_accumulation_steps * args.epochs
    
    # Prepare optimizer and learning rate scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [param for n, param in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [param for n, param in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.epsilon) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=total_steps
    )

    # Check if saved optimizer or scheduler states exist
    optimizer_checkpoint = Path(args.model_name_or_path) / 'optimizer.pt' 
    scheduler_checkpoint = Path(args.model_name_or_path) / 'scheduler.pt' 
    if args.model_name_or_path and optimizer_checkpoint.is_file() and scheduler_checkpoint.is_file():
        optimizer.load_state_dict(torch.load(str(optimizer_checkpoint)))
        scheduler.load_state_dict(torch.load(str(scheduler_checkpoint)))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError('Failed to enable NVIDIA Apex. Please install Apex from https://www.github.com/nvidia/apex to use fp16 training.')

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    logging.info('***** Running training *****')
    logging.info('  Num Examples: {}'.format(len(dataset)))
    logging.info('  Num Epochs: {}'.format(args.epochs))
    logging.info('  Batch size per GPU ({}): {}'.format(args.n_gpu, args.per_gpu_train_batch_size))
    logging.info('  Total batch size: {}'.format(
        batch_size * args.gradient_accumulation_steps * (
            torch.distributed.get_world_size() if args.local_rank != -1 else 1
        )
    ))

    logging.info('  Gradient accumulation steps: {}'.format(args.gradient_accumulation_steps))
    logging.info('  Total training steps: {}'.format(total_steps))

    global_step = 0
    epochs_trained = 0
    steps_in_current_epoch = 0

    if args.model_name_or_path and Path(args.model_name_or_path).exists():
        try:
            # Continue training from checkpoint
            checkpoint_suffix = str(args.model_name_or_path).split('-')[-1].split('/')[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(data_loader) // args.gradient_accumulation_steps)
            steps_in_current_epoch = global_step % (len(data_loader) // args.gradient_accumulation_steps)

            logging.info('  Resuming training from checkpoint {} (epoch={}, global step={})'.format(
                checkpoint_suffix, epochs_trained, global_step
            ))

            logging.info('  Will skip the first {} steps in the first epoch'.format(steps_in_current_epoch))

        except ValueError:
            logging.info('  Starting training...')

    training_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, 'module') else model
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    set_seed(args.seed)

    disable = args.local_rank not in [-1, 0]
    train_iterator = trange(epochs_trained, int(args.epochs), desc='Epoch', disable=disable)
    for _ in train_iterator:
        epoch_iterator = tqdm(data_loader, desc='Step', disable=disable, unit_scale=batch_size, unit='examples')
        for step, batch in enumerate(epoch_iterator):
            if steps_in_current_epoch > 0:
                steps_in_current_epoch -= 1
                continue

            inputs, labels = mask_tokens(batch, tokenizer, args) if args.use_masked_loss else (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, masked_lm_labels=labels) if args.use_masked_loss else model(inputs, labels=labels)

            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            training_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_gradient_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
                
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.log_frequency > 0 and global_step % args.log_frequency == 0:
                    if args.local_rank == -1 and args.eval_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            summary_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    
                    summary_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    summary_writer.add_scalar('loss', (training_loss - logging_loss) / args.log_frequency, global_step)

                    logging_loss = training_loss

                if args.local_rank in [-1, 0] and args.save_frequency > 0 and global_step % args.save_frequency == 0:
                    checkpoint_prefix = 'checkpoint'
                    output_dir = args.outdir / '{}-{}'.format(checkpoint_prefix, global_step)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, output_dir / 'training_args.bin')
                    logging.info('Saving model checkpoint to {}'.format(output_dir))

                    rotate_checkpoints(args.outdir, max_checkpoints=args.max_checkpoints, prefix=checkpoint_prefix)

                    torch.save(optimizer.state_dict(), output_dir / 'optimizer.pt')
                    torch.save(scheduler.state_dict(), output_dir / 'scheduler.pt')
                    logging.info('Saving optimizer and scheduler states to {}'.format(output_dir))

            if args.global_steps > 0 and global_step > args.global_steps:
                epoch_iterator.close()
                break

        if args.global_steps > 0 and global_step > args.global_steps:
            train_iterator.close()
            break
    
    if args.local_rank in [-1, 0]:
        summary_writer.close()

    return global_step, training_loss / global_step

def evaluate(args, dataset, model, tokenizer, prefix=''):
    '''
    Evaluate the model.

    '''

    if args.local_rank in [-1, 0]:
        args.outdir.mkdir(parents=True, exist_ok=True)
    
    batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    collate_func = functools.partial(_collate, tokenizer)
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_func)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logging.info('***** Running evaluation {} *****'.format(prefix))
    logging.info('  Num Examples: {}'.format(len(dataset)))
    logging.info('  Batch size: {}'.format(batch_size))

    eval_loss = 0.0
    eval_steps = 0
    model.eval()
    
    for batch in tqdm(data_loader, desc='Evaluating', unit_scale=batch_size, unit='examples'):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.use_masked_loss else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.use_masked_loss else model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()

        eval_steps += 1

    eval_loss /= eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    loss = torch.tensor(eval_loss)

    result = {
        'perplexity': perplexity,
        'loss': loss
    }

    with open(args.outdir / (prefix + 'eval_results.txt'), 'w+') as file:
        logging.info('***** Evaluation results {} *****'.format(prefix))
        for key in sorted(result.keys()):
            logging.info('  {}: {}'.format(key, str(result[key])))
            file.write('{}={}'.format(key, str(result[key])))
    
    return result

parser = argparse.ArgumentParser('Train or fine-tune a GPT-2 model using casual language (CLM) loss.')
parser.add_argument('train_dataset', type=Path, help='The preprocessed training dataset file.')
parser.add_argument('--eval-dataset', type=Path, help='The preprocessed evalutation dataset file.')
parser.add_argument('--line-by-line', dest='line_by_line', action='store_true',
                    help='Indicates whether distinct lines of text in the input dataset are to be handled as ' +
                    'separate input sequences (i.e. whether to add the BOS and EOS tokens to each line).')
parser.add_argument('--mlm', dest='use_masked_loss', action='store_true', help='Train with masked-language modelling ' +
                    '(CLM) loss rather than casual language model loss.')
parser.add_argument('--mlm-probability', type=float, default=0.15, help='Ratio of tokens to mask for MLM loss.')
parser.add_argument('--block-size', type=int, default=-1, help='Optional input sequence length after tokenization. ' +
                    'The training dataset will be truncated in blocks of this size. Defaults to -1, meaning that ' +
                    'the maximum input sequence length of the model for single sentence inputs will be used.')
parser.add_argument('--overwrite-cache', action='store_true', help='Overwrite the cached training and evaluation sets.')
parser.add_argument('--model', type=str, default='gpt2', dest='model_name_or_path',
                    help='The model checkpoint for weights initialization (i.e. a pretrained model). ' +
                    'Leave as None to train a model from scratch.')
parser.add_argument('--model-type', type=str, default='gpt2', help='The model architecture. Defaults to GPT2.')
parser.add_argument('--outdir', type=Path, default=Path('./output'), help='The directory to save model checkpoints.')
parser.add_argument('--overwrite-outdir', action='store_true', help='Overwrrite the output directory.')
parser.add_argument('--restore', dest='restore_checkpoint', action='store_true', help='Whether to resume training from the lastest checkpoint.')
parser.add_argument('--cache-dir', type=Path, default=None, help='The location to store pretrained models.')
parser.add_argument('--do-train', action='store_true', help='Whether to run training.')
parser.add_argument('--do-eval', action='store_true', help='Whether to evaluate the model on the evaluation set.')
parser.add_argument('--eval-during-training', action='store_true', help='Whether to run evaluation during training at each logging step.')
parser.add_argument('--eval-all-checkpoints', action='store_true', help='Evaluate all checkpoints starting with the same prefix as the input model checkpoint.')
parser.add_argument('--model-config', default=None, type=str, help='Optional model config name or path if not the same as the model ' +
                    'checkpoint path. If both are None, a new config will be initialized.')
parser.add_argument('--tokenizer', default=None, type=str, help='Optional pretrained tokenizer name or path if not the same as the model ' + 
                    'checkpoint path. If both are None, a new tokenizer will be initialized.')
parser.add_argument('--per-gpu-train-batch-size', default=4, type=int, help='Batch size per device (i.e. GPU, CPU, or TPU) while training.')
parser.add_argument('--per-gpu-eval-batch-size', default=4, type=int, help='Batch size per device (i.e. GPU, CPU, or TPU) while evaluationg.')
parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Number of steps to accumulate before performing a backpropagation pass.')
parser.add_argument('--learning-rate', type=float, default=5e-5, help='The initial learning rate for the Adam optimizer.')
parser.add_argument('--weight-decay', type=float, default=0, help='The rate at which weights decay.')
parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon for Adam optimizer.')
parser.add_argument('--max-gradient-norm', type=float, default=1.0, help='Maximum gradient norm.')
parser.add_argument('--epochs', type=float, default=10.0, help='Total number of training epochs to perform.')
parser.add_argument('--global-steps', type=int, default=-1, help='The number of training steps to perform. ' +
                    'Overrides epochs parameter if non-negative value.')
parser.add_argument('--lr-warmup-steps', type=int, default=0, help='The duration, in training steps, of the linear warmup on the learning rate.')
parser.add_argument('--log-frequency', type=int, default=500, help='The frequency, in training steps, at which the model metrics are logged.')
parser.add_argument('--save-frequency', type=int, default=500, help='The frequency, in training steps, at which the model checkpoint is saved.')
parser.add_argument('--max-checkpoints', type=int, default=None, help='The maximum number of checkpoints to keep before deleting older ones. ' +
                    'A negative or None value means that there is no limit.')
parser.add_argument('--no-cuda', dest='no_cuda', action='store_true', help='Disable CUDA devices even when they are available.')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='Use 16-bit (mixed) precision floats.')
parser.add_argument('--fp16-opt-level', type=str, default='O1', help='Apex AMP optimization level. See https://nvidia.github.io/apex/amp.html.')
parser.add_argument('--local-rank', type=int, default=-1, help='The rank of the process; for distributed training. ' +
                    'A value of -1 means no distributed training.')
parser.add_argument('--debug-server-ip', type=str, default='', help='The IP of the PTVSD server.')
parser.add_argument('--debug-server-port', type=str, default='', help='The port of the PTVSD server.')
parser.add_argument('--special-tokens', type=Path, default=None, help='A JSON file containing a dictionary of special tokens. ' +
                    'If not specified, the default special tokens are used (<|bos|>, <|eos|>, <|pad|>, and <|eq_tok|>).')
parser.add_argument('--seed', type=int, default=None, help='The seed of the random engine.')
args = parser.parse_args()

if args.model_type in ['bert', 'roberta', 'distilbert', 'camembert'] and not args.use_masked_loss:
    raise ValueError(
        'BERT and RoBERTa-like models require masked language model heads. '
        'They must be run with masked language modelliing, using the --mlm flag.'
    )

if args.eval_dataset is None and args.do_eval:
    raise ValueError(
        'Cannot evaluate model without an evaluation dataset. '
        'Either supply an evaluation dataset or remove the --do-eval flag.'
    )

if args.restore_checkpoint:
    checkpoints = get_checkpoints(args.outdir)
    if len(checkpoints) == 0:
        raise ValueError(
            'Cannot restore the latest checkpoint: no checkpoint was found in the output directory (\'\').' \
                .format(args.outdir)
        )
    else:
        args.model_name_or_path = str(checkpoints[-1])
elif args.outdir.exists() and any(args.outdir.iterdir()) and args.do_train and not args.overwrite_outdir:
    raise ValueError(
        'Output directory (\'{}\') already exists and is non-empty. Use --overwrite-outdir to disable this.' \
                .format(args.outdir)
    )

# Setup PTVSD (Python Tools for Visual Studio Debugging) server
if args.debug_server_ip and args.debug_server_port:
    import ptvsd

    print('Waiting for PTVSD server to attach...')
    ptvsd.enable_attach(address=(args.debug_server_ip, args.debug_server_port), redirect_output=True)
    ptvsd.wait_for_attach()

# Setup CUDA, GPU, and distributed training
if args.local_rank == -1 or args.no_cuda:
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    args.n_gpu = 0 if args.no_cuda else 1
else:
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = 1

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)

logging.warning(
    'Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bit floating-point precision: %s',
    args.local_rank,
    args.device,
    args.n_gpu,
    bool(args.local_rank != -1),
    args.fp16,
)

# Set seed
if args.seed is None:
    # We use the current time as the seed rather than letting numpy seed
    # since we want to achieve consistent results across sessions.
    # Source: https://stackoverflow.com/a/45573061/7614083
    t = int(time.time() * 1000.0)
    args.seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) <<  8) + ((t & 0x000000ff) << 24)

set_seed(args.seed)

if args.local_rank not in [-1, 0]:
    # Start barrier to ensure that the model is only downloaded once.
    torch.distributed.barrier()

if args.model_config:
    config = AutoConfig.from_pretrained(args.model_config, cache_dir=args.cache_dir)
elif args.model_name_or_path:
    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
else:
    config = CONFIG_MAPPING[args.model_type]()
    logging.warning('No config found: creating a new config instance from scatch.')

if args.tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir)
elif args.model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
else:
    raise ValueError(
        'Instantiating a new tokenizer from scratch is not support; however, it can be done from another script.'
        'Use the --tokenizer command line argument, providing it with the location of the script, to load the tokenizer.'
    )

if args.block_size <= 0:
    args.block_size = tokenizer.max_len
else:
    args.block_size = min(args.block_size, tokenizer.max_len)

if args.model_name_or_path:
    model = AutoModelWithLMHead.from_pretrained(
        args.model_name_or_path,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir
    )
else:
    logging.info('Training {} model from scratch'.format(args.model_type))
    model = AutoModelWithLMHead.from_config(config)

# Add special tokens and resize the model
logging.info('Initializing tokenizer with special tokens and resizing the model\'s token embeddings.')
tokenizer.add_special_tokens(get_special_tokens(args.special_tokens))
model.resize_token_embeddings(len(tokenizer))

model.to(args.device)
if args.local_rank == 0:
    # End barrier
    torch.distributed.barrier()

if args.do_train:
    if args.local_rank not in [-1, 0]:
        # Start barrier to make sure that the dataset is only processed once among the distributed training pool.
        torch.distributed.barrier()

    train_dataset = get_dataset(
        args.train_dataset, tokenizer, args.block_size,
        line_by_line=args.line_by_line, overwrite_cache=args.overwrite_cache
    )

    if args.local_rank == 0:
        # End barrier
        torch.distributed.barrier()

    global_step, loss = train(args, train_dataset, model, tokenizer)
    logging.info('  global_step: {}, average loss: {}'.format(global_step, loss))

if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    if args.local_rank in [-1, 0]:
        args.outdir.mkdir(parents=True, exist_ok=True)
    
    logging.info('Saving model checkpoint to {}'.format(args.outdir))
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    torch.save(args, args.outdir / 'training_args.bin')

    absolute_outdir = str(args.outdir.absolute())
    model = AutoModelWithLMHead.from_pretrained(absolute_outdir)
    tokenizer = AutoTokenizer.from_pretrained(absolute_outdir)
    model.to(args.device)

if args.do_eval and args.local_rank in [-1, 0]:
    eval_dataset = get_dataset(
        args.eval_dataset, tokenizer, args.block_size,
        line_by_line=args.line_by_line, overwrite_cache=args.overwrite_cache
    )

    results = {}
    checkpoints = [args.outdir]
    if args.eval_all_checkpoints:
        checkpoints = list(args.outdir.glob('**/{}'.format(WEIGHTS_NAME)))
        logging.getLogger('transformers.modelling_utils').setLevel(logging.WARN)

    logging.info('Evaluating the following checkpoints: {}'.format(checkpoints))
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[1] if len(checkpoints) > 1 else ''
        prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ''

        model = AutoModelWithLMHead.from_pretrained(str(checkpoint.absolute()))
        model.to(args.device)

        result = evaluate(args, eval_dataset, model, tokenizer, prefix=prefix)
        result = dict((key + '_{}'.format(global_step), value) for key, value in result.items())
        results.update(result)
        
    logging.info('Evaluation results: {}'.format(results))