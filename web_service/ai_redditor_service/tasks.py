'''
Celery tasks.

'''

import re
import time

from flask import current_app
from celery.utils import cached_property, log
from ai_redditor_service.extensions import celery, db
from ai_redditor_service.utils import unescape_unicode
from ai_redditor_service.gpt2 import (
    ModelDecodeFormat,
    load_model,
    generate as gpt2_model_generate,
    PHC_LINK_PATTERN,
    _get_decode_regex_mapping,
    _get_decode_special_tokens_mapping
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
    def models(self):
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
    def special_tokens_match_pattern(self):
        '''
        A dictionary mapping each :class:`ai_redditor_service.models.RecordType`
        to a compiled regular expression pattern for matching special tokens.

        '''

        match_patterns = {}
        for model_type, value in self.models.items():
            _, tokenizer = value
            match_patterns[model_type] = re.compile(
                '|'.join(re.escape(token) for token in tokenizer.all_special_tokens)
            )
        
        return match_patterns

    @cached_property
    def translate_token(self):
        '''
        The query/answer separator token (translation separator token).

        '''

        return current_app.config['GPT2_TRANSLATE_TOKEN']

    @cached_property
    def end_of_likes_token(self):
        '''
        The special token specifying the end of likes.
        
        '''

        return current_app.config['GPT2_END_OF_LIKES_TOKEN']

    def _load_decode_regex_mapping(self, strict):
        regex_mappings = {}
        for model_type, value in self.models.items():
            _, tokenizer = value
            regex_mappings[model_type] =  _get_decode_regex_mapping(
                strict, tokenizer.bos_token, tokenizer.eos_token,
                self.translate_token, self.end_of_likes_token
            )

        return regex_mappings

    @cached_property
    def decode_strict_regex_mapping(self):
        '''
        A dictionary mapping each :class:`ai_redditor_service.models.RecordType`
        to another dictionary containing the strict regex patterns (i.e. matching
        groups cannot be empty) for splitting the model output into groups of data
        based on the decode format, for each
        :class:`ai_redditor_service.gpt2.ModelDecodeFormat`.

        '''
        
        return self._load_decode_regex_mapping(True)

    @cached_property
    def decode_non_strict_regex_mapping(self):
        '''
        A dictionary mapping each :class:`ai_redditor_service.models.RecordType`
        to another dictionary containing the non-strict regex patterns (i.e. 
        matching groups can be empty) for splitting the model output into groups 
        of data based on the decode format, for each
        :class:`ai_redditor_service.gpt2.ModelDecodeFormat`.

        '''
        
        return self._load_decode_regex_mapping(False)

    @cached_property
    def decode_special_tokens_mapping(self):
        '''
        A dictionary mapping each :class:`ai_redditor_service.models.RecordType`
        to another dictionary containing the special tokens, in order of appearance
        in the decode format, for each :class:`ai_redditor_service.gpt2.ModelDecodeFormat`.

        '''

        special_tokens_mapping = {}
        for model_type, value in self.models.items():
            _, tokenizer = value
            special_tokens_mapping[model_type] = _get_decode_special_tokens_mapping(
                tokenizer.bos_token, tokenizer.eos_token,
                self.translate_token, self.end_of_likes_token
            )

        return special_tokens_mapping

class RecordGenerateConfig:
    '''
    Configuration values for generating a record.

    '''

    def __init__(self, decode_format, group_to_record_func, min_length=250, max_length=1024):
        self.decode_format = decode_format
        self._group_to_record_func = group_to_record_func
        self.min_length = min_length
        self.max_length = max_length

    def group_to_record(self, prompt, generated_groups,decode_non_strict_regex,
                        decode_special_tokens, *args, **kwargs):  
        if prompt is not None:
            # Prepare prompt for splitting into groups.
            # The regex pattern expects that ALL special tokens are present,
            # meaning that we need to pad the end of the prompt with the missing tokens.
            special_tokens = decode_special_tokens[self.decode_format]
            prompt += ''.join(token for token in special_tokens if token not in prompt)
            # Split the prompt into match groups
            decode_regex = decode_non_strict_regex[self.decode_format]
            prompt_groups = decode_regex.match(prompt).groupdict()
        else:
            prompt_groups = {}

        return self._group_to_record_func(prompt_groups, generated_groups, *args, **kwargs)

# Maps record type to a value configuration
_RECORD_GENERATE_CONFIGS = {
    RecordType.TIFU: RecordGenerateConfig(
        ModelDecodeFormat.QUERY_ANSWER,
        lambda prompt_groups, generated_groups, *args, **kwargs: TIFURecord(
            generated_groups['prompt'], generated_groups['response'],
            post_title_prompt_end=len(prompt_groups.get('prompt', '')),
            post_body_prompt_end=len(prompt_groups.get('response', '')),
            *args, **kwargs
        )
    ),
    RecordType.WP: RecordGenerateConfig(
        ModelDecodeFormat.QUERY_ANSWER,
        lambda prompt_groups, generated_groups, *args, **kwargs: WPRecord(
            generated_groups['prompt'], generated_groups['response'],
            prompted_prompt_end=len(prompt_groups.get('prompt', '')),
            prompted_response_end=len(prompt_groups.get('response', '')),
            *args, **kwargs
        )
    ),
    RecordType.PHC: RecordGenerateConfig(
        ModelDecodeFormat.PHC,
        lambda prompt_groups, generated_groups, *args, **kwargs: PHCRecord(
            generated_groups['author'],
            int(re.sub('[^0-9]', '', generated_groups['likes']) or 0),
            generated_groups['comment_body'],
            prompted_author_username_end=len(prompt_groups.get('author', '')),
            is_likes_prompted=bool(prompt_groups.get('likes', None)),
            prompted_comment_end=len(prompt_groups.get('comment_body', '')),
            *args, **kwargs
        ), min_length=10, max_length=200
    )
}

@celery.task(base=GPT2GenerateTask)
def generate_record(record_type, **kwargs):
    record_config = _RECORD_GENERATE_CONFIGS[record_type]
    model, tokenizer = generate_record.models[record_type]

    use_link_filter = True
    if record_type == RecordType.PHC and kwargs.get('prompt', None):
        # We only use the link filter if the prompt DOES NOT have links; otherwise,
        # if it does, we want to bypass the link filter to allow the prompt.
        use_link_filter = len(PHC_LINK_PATTERN.findall(kwargs['prompt'])) == 0
        
    decode_strict_regex_mapping = generate_record.decode_strict_regex_mapping[record_type]
    outputs = gpt2_model_generate(
        model, tokenizer, record_config.decode_format,
        translate_token=generate_record.translate_token,
        end_of_likes_token=generate_record.end_of_likes_token,
        min_length=record_config.min_length,
        max_length=record_config.max_length,
        use_link_filter=use_link_filter,
        decode_strict_regex_mapping=decode_strict_regex_mapping,
        **kwargs
    )

    # The records are custom if the prompt keyword argument
    # is present AND non-empty.
    is_custom = 'prompt' in kwargs and bool(kwargs['prompt'])
    special_token_pattern = generate_record.special_tokens_match_pattern[record_type]

    record_uuids = []
    for output in outputs:
        # Clean the model output
        groups = { key: unescape_unicode(special_token_pattern.sub('', value)) \
            for key, value in output.groups.items()
        }

        prompt = None or kwargs['prompt']
        decode_non_strict_regex = generate_record.decode_non_strict_regex_mapping[record_type]
        decode_special_tokens = generate_record.decode_special_tokens_mapping[record_type]

        record = record_config.group_to_record(
            prompt, groups, decode_non_strict_regex,
            decode_special_tokens, is_custom=is_custom
        )
        
        record_uuids.append(record.uuid)
        db.session.add(record)

    db.session.commit()
    return record_type, record_uuids