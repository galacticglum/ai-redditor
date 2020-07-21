'''
Celery tasks.

'''

import re
import copy
import time
import traceback

from celery import states
from flask import current_app
from celery.utils import cached_property, log
from ai_redditor_service.extensions import celery, db
from ai_redditor_service.utils import unescape_unicode, all_empty
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

    def on_failure(self, exception, task_id, args, kwargs, einfo):
        '''
        Invoked when the task raises an exception.

        '''

        self.update_state(state=states.FAILURE, meta={
            'exc_type': type(exception).__name__,
            'exc_message': traceback.format_exc().split('\n'),
            'error': str(exception)
        })

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

    @cached_property
    def decode_strict_regex_mapping(self):
        '''
        A dictionary mapping each :class:`ai_redditor_service.models.RecordType`
        to another dictionary containing the strict regex patterns (i.e. matching
        groups cannot be empty) for splitting the model output into groups of data
        based on the decode format, for each
        :class:`ai_redditor_service.gpt2.ModelDecodeFormat`.

        '''
        
        regex_mappings = {}
        for model_type, value in self.models.items():
            _, tokenizer = value
            regex_mappings[model_type] =  _get_decode_regex_mapping(
                True, tokenizer.bos_token, tokenizer.eos_token,
                self.translate_token, self.end_of_likes_token
            )

        return regex_mappings

    @cached_property
    def log_debug_info(self):
        '''
        Indicates whether to log debug information when generating records.

        '''

        return current_app.config['RECORD_GENERATION_LOG_DEBUG_INFO']

class RecordGenerateConfig:
    '''
    Configuration values for generating a record.

    '''

    def __init__(self, decode_format, group_to_record_func, min_length=250, max_length=1024):
        self.decode_format = decode_format
        self._group_to_record_func = group_to_record_func
        self.min_length = min_length
        self.max_length = max_length

    def group_to_record(self, prompt_object, generated_groups, *args, **kwargs):  
        return self._group_to_record_func(prompt_object, generated_groups, *args, **kwargs)

def _sanitize_likes(likes_str):
    '''
    Sanitize the likes group of the model output.

    '''

    return int(re.sub('[^0-9-]', '', likes_str) or 0)

# Maps record type to a value configuration
_RECORD_GENERATE_CONFIGS = {
    RecordType.TIFU: RecordGenerateConfig(
        ModelDecodeFormat.QUERY_ANSWER,
        lambda prompt_object, generated_groups, *args, **kwargs: TIFURecord(
            generated_groups['prompt'], generated_groups['response'],
            post_title_prompt_end=len(prompt_object.get('post_title', '')),
            post_body_prompt_end=len(prompt_object.get('post_body', '')),
            *args, **kwargs
        )
    ),
    RecordType.WP: RecordGenerateConfig(
        ModelDecodeFormat.QUERY_ANSWER,
        lambda prompt_object, generated_groups, *args, **kwargs: WPRecord(
            generated_groups['prompt'], generated_groups['response'],
            prompted_prompt_end=len(prompt_object.get('post_title', '')),
            prompted_response_end=len(prompt_object.get('post_body', '')),
            *args, **kwargs
        )
    ),
    RecordType.PHC: RecordGenerateConfig(
        ModelDecodeFormat.PHC,
        lambda prompt_object, generated_groups, *args, **kwargs: PHCRecord(
            generated_groups['author'],
            _sanitize_likes(generated_groups['likes']),
            generated_groups['comment_body'],
            prompted_author_username_end=len(prompt_object.get('author', '')),
            is_likes_prompted=bool(prompt_object.get('likes', None)),
            prompted_comment_end=len(prompt_object.get('comment_body', '')),
            *args, **kwargs
        ), min_length=10, max_length=200
    )
}

# Prefix for each record type
_RECORD_PROMPT_PREFIXES = {
    RecordType.TIFU: 'TIFU ',
    RecordType.WP: '[WP] '
}


def _qa_prompt_to_string(record_type, prompt_object):
    prompt_prefix = _RECORD_PROMPT_PREFIXES[record_type]
    _, tokenizer = generate_record.models[record_type]

    # Check if the prompt object is empty or None; if so,
    # we simply provide the <|bos|> and prompt prefix as the
    # input to the model.
    if not bool(prompt_object) or all_empty(prompt_object.values()):
        return tokenizer.bos_token + prompt_prefix

    if not bool(prompt_object.get('post_title', None)):
        record_type_name = RecordType(record_type).name
        raise ValueError(
            f'Invalid prompt provided when trying to generate {record_type_name} record. '
            'The \'post_title\' field is required but was not provided.'
        )

    # Make sure that the post title starts with the prefix
    if not prompt_object['post_title'].startswith(prompt_prefix):
        prompt_object['post_title'] = prompt_prefix + prompt_object['post_title']
    
    prompt = tokenizer.bos_token + prompt_object['post_title']
    if bool(prompt_object.get('post_body', None)):
        prompt += generate_record.translate_token + prompt_object['post_body']

    return prompt    

def _phc_prompt_to_string(record_type, prompt_object):
    model, tokenizer = generate_record.models[record_type]

    # Check if the prompt object is empty or None, or if it contains
    # keys with all empty values. If so, we simply provide the <|bos|>
    # token as the input to the model.
    if not bool(prompt_object) or all_empty(prompt_object.values()):
        return tokenizer.bos_token

    # A prompt field depends on all the ones preceeding it
    # (due to the nature of the input format to the model).
    if bool(prompt_object.get('comment_body', None)):
        required_fields = ['likes', 'author']
    elif bool(prompt_object.get('author', None)):
        required_fields = ['likes']
    else:
        required_fields = []

    # Find all the missing fields
    def _is_field_missing(field, data):
        return field not in data or not isinstance(data[field], int) and not bool(data.get(field, None))

    missing_fields = [
        field for field in required_fields if _is_field_missing(field, prompt_object)
    ]

    if generate_record.log_debug_info:
        logger.warning(f'{required_fields} are required; {missing_fields} are missing.')

    if len(missing_fields) > 0:
        # Generate another record to populate the missing fields
        record_config = _RECORD_GENERATE_CONFIGS[RecordType.PHC]
        decode_strict_regex_mapping = generate_record.decode_strict_regex_mapping[RecordType.PHC]
        
        prompt = tokenizer.bos_token
        if 'likes' not in missing_fields:
            # Include the prompted likes, if given, to get more accurate field values.
            prompt += str(prompt_object['likes']) + generate_record.end_of_likes_token

        start_time = time.time()
        outputs = gpt2_model_generate(
            model, tokenizer, record_config.decode_format,
            translate_token=generate_record.translate_token,
            end_of_likes_token=generate_record.end_of_likes_token,
            min_length=record_config.min_length,
            max_length=record_config.max_length,
            decode_strict_regex_mapping=decode_strict_regex_mapping,
            prompt=prompt, samples=1
        )

        if generate_record.log_debug_info:
            end_time = time.time()
            logger.warning('Generating secondary record with prompt \'{}\'; took {:.2f} seconds'.format(
                prompt, end_time - start_time
            ))

        # Copy prompt object so that we only modify it within this function
        prompt_object = copy.copy(prompt_object)

        # Populate the missing fields
        for field in missing_fields:
            value = outputs[0].groups[field]
            if field == 'likes':
                value = _sanitize_likes(value)
            
            prompt_object[field] = value

    prompt = tokenizer.bos_token + str(prompt_object['likes'])
    if bool(prompt_object.get('author', None)):
        prompt += generate_record.end_of_likes_token + prompt_object['author']

    if bool(prompt_object.get('comment_body', None)):
        prompt += generate_record.translate_token + prompt_object['comment_body']
    
    return prompt

_PROMPT_OBJECT_TO_STRING = {
    RecordType.TIFU: _qa_prompt_to_string,
    RecordType.WP: _qa_prompt_to_string,
    RecordType.PHC: _phc_prompt_to_string
}

@celery.task(base=GPT2GenerateTask)
def generate_record(record_type, prompt_object=None, **kwargs):
    record_config = _RECORD_GENERATE_CONFIGS[record_type]
    model, tokenizer = generate_record.models[record_type]

    if prompt_object is None:
        prompt_object = dict()
    
    # Strip all prompt string values of trailing whitespace
    prompt_object = {
        key: (value.strip() if isinstance(value, str) else value) \
            for key, value in prompt_object.items()
    }

    # Convert the prompt object to a string
    prompt = _PROMPT_OBJECT_TO_STRING[record_type](record_type, prompt_object)

    use_link_filter = True
    is_custom = prompt is not None
    if record_type == RecordType.PHC and is_custom:
        # We only use the link filter if the prompt DOES NOT have links; otherwise,
        # if it does, we want to bypass the link filter to allow the prompt.
        use_link_filter = len(PHC_LINK_PATTERN.findall(prompt)) == 0

    start_time = time.time()
    decode_strict_regex_mapping = generate_record.decode_strict_regex_mapping[record_type]
    outputs = gpt2_model_generate(
        model, tokenizer, record_config.decode_format,
        translate_token=generate_record.translate_token,
        end_of_likes_token=generate_record.end_of_likes_token,
        min_length=record_config.min_length,
        max_length=record_config.max_length,
        use_link_filter=use_link_filter,
        decode_strict_regex_mapping=decode_strict_regex_mapping,
        prompt=prompt, **kwargs
    )

    if generate_record.log_debug_info:
        end_time = time.time()
        logger.warning('Generating primary record with prompt \'{}\'; took {:.2f} seconds'.format(
            prompt, end_time - start_time
        ))

    special_token_pattern = generate_record.special_tokens_match_pattern[record_type]

    record_uuids = []
    for output in outputs:
        # Clean the model output
        groups = { key: unescape_unicode(special_token_pattern.sub('', value)) \
            for key, value in output.groups.items()
        }

        record = record_config.group_to_record(prompt_object, groups, is_custom=is_custom)
        
        record_uuids.append(record.uuid)
        db.session.add(record)

    db.session.commit()
    return record_type, record_uuids
