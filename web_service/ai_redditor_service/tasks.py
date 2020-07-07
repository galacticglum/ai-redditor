'''
Celery tasks.

'''

import re
import time

from flask import current_app
from celery.utils import cached_property, log
from ai_redditor_service.extensions import celery
from ai_redditor_service.string_utils import unescape_unicode
from ai_redditor_service.gpt2 import (
    ModelType,
    ModelDecodeFormat,
    load_model,
    generate as gpt2_model_generate
)

logger = log.get_task_logger(__name__)

class GPT2GenerateTask(celery.Task):
    '''
    Celery task for generating text using the GPT2 models.

    '''

    @cached_property
    def _models(self):
        '''
        A dictionary mapping each :class:`ai_redditor_service.gpt2.ModelType`
        to its respective :class:`transformers.PreTrainedModel` instance
        and :class:`transformers.PreTrainedTokenizer` instance. 

        '''

        # Load the GPT2 models
        start_time = time.time()
        models = {
            ModelType.TIFU: load_model(
                current_app.config['TIFU_MODEL_PATH'],
                current_app.config.get('TIFU_TOKENIZER_PATH', None)
            ),
            ModelType.WP: load_model(
                current_app.config['WP_MODEL_PATH'],
                current_app.config.get('WP_TOKENIZER_PATH', None)
            ),
            ModelType.PHC: load_model(
                current_app.config['PHC_MODEL_PATH'],
                current_app.config.get('PHC_TOKENIZER_PATH', None)
            )
        }

        logger.info('Loaded GPT2 models ({}) ({} seconds)'.format(
            ', '.join(str(x) for x in models.keys()),
            round(time.time() - start_time, 2)
        ))

        return models

    @cached_property
    def _special_tokens_match_pattern(self):
        '''
        A dictionary mapping each :class:`ai_redditor_service.gpt2.ModelType`
        to a compiled regular expression pattern for matching special tokens.

        '''

        match_patterns = {}
        for model_type, value in self._models.items():
            _, tokenizer = value
            match_patterns[model_type] = re.compile(
                '|'.join(re.escape(token) for token in tokenizer.all_special_tokens)
            )
        
        return match_patterns

@celery.task(base=GPT2GenerateTask)
def gpt2_generate(model_type, decode_format, **kwargs):
    model, tokenizer = gpt2_generate._models[model_type]

    start_time = time.time()
    outputs = gpt2_model_generate(model, tokenizer, decode_format, **kwargs)
    special_token_pattern = gpt2_generate._special_tokens_match_pattern[model_type]

    # Clean the model output and map to a list of dictionaries
    return [
        { key: unescape_unicode(special_token_pattern.sub('', value)) \
            for key, value in x.groups.items()
        } for x in outputs
    ]