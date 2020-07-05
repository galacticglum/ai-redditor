import re
import json

def get_special_tokens_match_pattern(special_tokens_map_filename):
    special_tokens = []
    if special_tokens_map_filename is not None:
        with open(special_tokens_map_filename) as special_tokens_file:
            values = list(json.load(special_tokens_file).values())

            # Flatten nested list:
            # We do this because the special tokens map contains an "additional_special_tokens"
            # field which consists of a list of additional special tokens. When we convert the
            # values from the special tokens map dictionary, the additional special tokens ends
            # up creating an element of type list, which is what we want to flatten out.
            for item in values:
                if isinstance(item, list):
                    special_tokens.extend(item)
                else:
                    special_tokens.append(item)

    return re.compile('|'.join(re.escape(token) for token in special_tokens))