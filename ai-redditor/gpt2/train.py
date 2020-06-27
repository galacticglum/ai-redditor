'''
Train or fine-tune a GPT-2 model,

'''

import re
import time
import json
import math
import torch
import logging
import argparse
from pathlib import Path

from transformers import (
    WEIGHTS_NAME,
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)

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

def main():
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
    parser.add_argument('--logdir', type=Path, default=None, help='The Tensorboard logging directory.')
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
    parser.add_argument('--train-batch-size', '--per-device-train-batch-size', dest='per_device_train_batch_size', default=8, type=int, help='Batch size per device (i.e. GPU, CPU, or TPU) while training.')
    parser.add_argument('--eval-batch-size', '--per-device-eval-batch-size', dest='per_device_eval_batch_size', default=8, type=int, help='Batch size per device (i.e. GPU, CPU, or TPU) while evaluationg.')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Number of steps to accumulate before performing a backpropagation pass.')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='If True, use gradient checkpointing to save memory at the expense of slower backward pass.')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='The initial learning rate for the Adam optimizer.')
    parser.add_argument('--weight-decay', type=float, default=0, help='The rate at which weights decay.')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon for Adam optimizer.')
    parser.add_argument('--max-gradient-norm', type=float, default=1.0, help='Maximum gradient norm.')
    parser.add_argument('--epochs', type=float, default=10.0, help='Total number of training epochs to perform.')
    parser.add_argument('--global-steps', type=int, default=-1, help='The number of training steps to perform. ' +
                        'Overrides epochs parameter if non-negative value.')
    parser.add_argument('--lr-warmup-steps', type=int, default=0, help='The duration, in training steps, of the linear warmup on the learning rate.')
    parser.add_argument('--log-frequency', type=int, default=500, help='The frequency, in training steps, at which the model metrics are logged.')
    parser.add_argument('--log-first-step', action='store_true', help='Log and evaluate the first global step. Defaults to False.')
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
    # We support both --tpu--num-cores and --tpu_num_cores for compatibility
    # with the XLA launcher script, which injects the --tpu_num_cores argument.
    parser.add_argument('--tpu-num-cores', '--tpu_num_cores', type=int, default=None, help='Number of TPU cores.')
    parser.add_argument('--tpu-metrics-debug', action='store_true', help='Whether to print TPU debug metrics. Defaults to False.')
    parser.add_argument('--dataloader-drop-last', dest='dataloader_drop_last', action='store_true', help='Drop the last incomplete ' +
                        'batch if it is not divisible by the batch size. Defaults to False.')
    args = parser.parse_args()

    if args.logdir is None:
        args.logdir = args.outdir / 'logs'

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
        format='%(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    )

    logging.info(
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

    if args.model_config:
        config = AutoConfig.from_pretrained(args.model_config, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logging.warning('No config found: creating a new config instance from scatch.')

    # Inject gradient checkpoint value into config
    if hasattr(config, 'gradient_checkpointing'):
        config.gradient_checkpointing = args.gradient_checkpointing
    else:
        logging.warning(
            'Config does not have the \'gradient_checkpointing\' attribute. '
            'Gradient checkpointing will NOT be used.'
        )

    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            'Instantiating a new tokenizer from scratch is not support; however, it can be done from another script.'
            'Use the --tokenizer command line argument, providing it with the location of the script, to load the tokenizer.'
        )

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

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)
    
    # Add special tokens and resize the model
    logging.info('Initializing tokenizer with special tokens and resizing the model\'s token embeddings.')
    tokenizer.add_special_tokens(get_special_tokens(args.special_tokens))
    model.resize_token_embeddings(len(tokenizer))

    # Load datasets
    train_dataset = get_dataset(
        args.train_dataset, tokenizer, args.block_size,
        line_by_line=args.line_by_line, overwrite_cache=args.overwrite_cache
    ) if args.do_train else None

    eval_dataset = get_dataset(
        args.eval_dataset, tokenizer, args.block_size,
        line_by_line=args.line_by_line, overwrite_cache=args.overwrite_cache
    ) if args.do_eval else None

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=args.use_masked_loss, mlm_probability=args.mlm_probability
    )

    # Project arguments to a TrainingArguments class.
    training_args = TrainingArguments(
        output_dir=str(args.outdir.absolute()),
        overwrite_output_dir=args.overwrite_outdir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        evaluate_during_training=args.eval_during_training,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=args.epsilon,
        max_grad_norm=args.max_gradient_norm,
        num_train_epochs=args.epochs,
        max_steps=args.global_steps,
        warmup_steps=args.lr_warmup_steps,
        logging_dir=str(args.logdir.absolute()),
        logging_first_step=args.log_first_step,
        logging_steps=args.log_frequency,
        save_steps=args.save_frequency,
        save_total_limit=args.max_checkpoints,
        no_cuda=args.no_cuda,
        seed=args.seed,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        local_rank=args.local_rank,
        tpu_num_cores=args.tpu_num_cores,
        tpu_metrics_debug=args.tpu_metrics_debug
    )

    # For some reason, the TrainingArguments __init__ method does not
    # recognize dataloader_drop_last as a valid kwarg (it's possibly not implemented?).
    # So, as a temporary workaround, we just set the field directly.
    training_args.dataloader_drop_last = args.dataloader_drop_last

    def _on_save_model(trainer, output_dir):
        tokenizer.save_pretrained(output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
        # Callback to save tokenizer when checkpointing...
        on_save_model=_on_save_model
    )

    if args.do_train:
        model_path = (
            args.model_name_or_path
            if args.model_name_or_path is not None and Path(args.model_name_or_path).is_dir()
            else None
        )

        trainer.train(model_path=model_path)
        trainer.save_model()      
        if trainer.is_world_master():
            tokenizer.save_pretrained(args.outdir)

    if args.do_eval:
        checkpoints = [args.outdir]
        if args.eval_all_checkpoints:
            checkpoints = list(args.outdir.glob('**/{}'.format(WEIGHTS_NAME)))
            logging.getLogger('transformers.modelling_utils').setLevel(logging.WARN)

        logging.info('Evaluating the following checkpoints: {}'.format(checkpoints))
        for checkpoint in checkpoints:
            model = AutoModelWithLMHead.from_pretrained(str(checkpoint.absolute()))
            # Recreate trainer with model from checkpoint
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                prediction_loss_only=True
            )

            eval_output = trainer.evaluate()
            perplexity = math.exp(eval_output['eval_loss'])
            result = {'perplexity': perplexity, 'loss': eval_output['eval_loss']}
            
            prefix = str(checkpoint).split('/')[-1] if str(checkpoint).find('checkpoint') != -1 else ''
            logging.info('***** Evaluation results {} *****'.format(prefix))

            if trainer.is_world_master():
                with open(args.outdir / (prefix + 'eval_results.txt'), 'w+') as file:
                    for key in sorted(result.keys()):
                        logging.info('  {}: {}'.format(key, str(result[key])))
                        file.write('{}={}\n'.format(key, str(result[key])))

def _mp_fn(index):
    # For xla spawn (TPUs)
    main()

if __name__ == '__main__':
    main()