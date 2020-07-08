'''
Celery tasks.

'''

import re
import time

from flask import current_app
from celery.utils import cached_property, log
from ai_redditor_service.extensions import celery, db
from ai_redditor_service.string_utils import unescape_unicode
from ai_redditor_service.gpt2 import (
    ModelDecodeFormat,
    load_model,
    generate as gpt2_model_generate
)

from ai_redditor_service.models import (
    RecordType, 
    TIFURecord, 
    WPRecord, 
    PHCRecord
)

logger = log.get_task_logger(__name__)

class SqlAlchemyTask(celery.Task):
    '''
    Celery task that ensures that the connection of
    the database is closed on task completion.

    '''

    abstract = True

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        '''
        After each Celery task, teardown the database session.

        '''

        db.session.remove()

class GPT2GenerateTask(SqlAlchemyTask):
    '''
    Celery task for generating text using the GPT2 models.

    '''

    abstract = True
    
    @cached_property
    def _models(self):
        '''
        A dictionary mapping each :class:`ai_redditor_service.models.RecordType`
        to its respective :class:`transformers.PreTrainedModel` instance
        and :class:`transformers.PreTrainedTokenizer` instance. 

        '''

        no_cuda = current_app.config.get('GPT2_NO_CUDA', False),
        quantize = current_app.config.get('GPT2_QUANTIZE', False)
        start_time = time.time()

        models = {
            RecordType.TIFU: load_model(
                current_app.config['TIFU_MODEL_PATH'],
                current_app.config.get('TIFU_TOKENIZER_PATH', None),
                no_cuda=no_cuda, quantize=quantize
            ),
            RecordType.WP: load_model(
                current_app.config['WP_MODEL_PATH'],
                current_app.config.get('WP_TOKENIZER_PATH', None),
                no_cuda=no_cuda, quantize=quantize
            ),
            RecordType.PHC: load_model(
                current_app.config['PHC_MODEL_PATH'],
                current_app.config.get('PHC_TOKENIZER_PATH', None),
                no_cuda=no_cuda, quantize=quantize
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
        A dictionary mapping each :class:`ai_redditor_service.models.RecordType`
        to a compiled regular expression pattern for matching special tokens.

        '''

        match_patterns = {}
        for model_type, value in self._models.items():
            _, tokenizer = value
            match_patterns[model_type] = re.compile(
                '|'.join(re.escape(token) for token in tokenizer.all_special_tokens)
            )
        
        return match_patterns

class RecordGenerateConfig:
    '''
    Configuration values for generating a record.

    '''

    def __init__(self, decode_format, group_to_record_func, min_length=250, max_length=1024):
        self.decode_format = decode_format
        self.group_to_record_func = group_to_record_func
        self.min_length = min_length
        self.max_length = max_length

# Maps record type to a value configuration
_RECORD_GENERATE_CONFIGS = {
    RecordType.TIFU: RecordGenerateConfig(
        ModelDecodeFormat.QUERY_ANSWER,
        lambda groups, *args, **kwargs: TIFURecord(
            groups['prompt'], groups['response'],
            *args, **kwargs
        )
    ),
    RecordType.WP: RecordGenerateConfig(
        ModelDecodeFormat.QUERY_ANSWER,
        lambda groups, *args, **kwargs: WPRecord(
            groups['prompt'], groups['response'],
            *args, **kwargs
        )
    ),
    RecordType.PHC: RecordGenerateConfig(
        ModelDecodeFormat.PHC,
        lambda groups, *args, **kwargs: PHCRecord(
            groups['author'], int(re.sub('[^0-9]', '', groups['likes']) or 0),
            groups['comment_body'], *args, **kwargs
        ), min_length=10, max_length=200
    )
}

@celery.task(base=GPT2GenerateTask)
def generate_record(record_type, **kwargs):
    record_config = _RECORD_GENERATE_CONFIGS[record_type]
    model, tokenizer = generate_record._models[record_type]

    outputs = gpt2_model_generate(
        model, tokenizer, record_config.decode_format,
        translate_token=current_app.config['GPT2_TRANSLATE_TOKEN'],
        end_of_likes_token=current_app.config['GPT2_END_OF_LIKES_TOKEN'],
        min_length=record_config.min_length, max_length=record_config.max_length,
        **kwargs
    )

    # The records are custom if the prompt keyword argument
    # is present AND non-empty.
    is_custom = 'prompt' in kwargs and bool(kwargs['prompt'])
    special_token_pattern = generate_record._special_tokens_match_pattern[record_type]

    record_uuids = []
    for output in outputs:
        # Clean the model output
        groups = { key: unescape_unicode(special_token_pattern.sub('', value)) \
            for key, value in output.groups.items()
        }

        record = record_config.group_to_record_func(groups, is_custom=is_custom)
        record_uuids.append(record.uuid)
        db.session.add(record)

    db.session.commit()
    return record_type, record_uuids