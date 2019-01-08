import json
def load_component_map(component_map_file):
    json_data=open(component_map_file).read()
    component_map = json.loads(json_data)
    return component_map

def tokens_to_words(tokens, component_map):
    '''
    Takes a list of tokens (a5b9, for example) and converts to relevant word
    string
    '''
    import pdb
    column_names = ['a', 'b', 'c', 'd', 'e']
    sentence_lists = {}
    token_words = []
    for ind, token in enumerate(tokens):
        word_ind = int(token[1])
        column_key = column_names[ind]
        column_words = component_map[column_key]
        token_words.append(column_words[word_ind])
    return token_words
