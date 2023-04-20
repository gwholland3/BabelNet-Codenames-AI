import os
import queue
import requests

import json as jsonlib
import networkx as nx


CACHED_BN_SUBGRAPHS_DIR = 'babelnet_bots/data/cached_babelnet_subgraphs'
CACHED_LEMMA_SYNSETS_DIR = 'babelnet_bots/data/cached_lemma_synsets'
CACHED_SYNSET_INFO_DIR = 'babelnet_bots/data/cached_synset_info'
CACHED_OUTGOING_EDGES_DIR = 'babelnet_bots/data/cached_outgoing_edges'
API_KEY_FILEPATH = 'babelnet_bots/bn_api_key.txt'

BN_DOMAIN = 'babelnet.io'
BN_VERSION = 'v8'
LEMMA_SYNSETS_URL = f'https://{BN_DOMAIN}/{BN_VERSION}/getSynsetIds'
SYNSET_INFO_URL = f'https://{BN_DOMAIN}/{BN_VERSION}/getSynset'
OUTGOING_EDGES_URL = f'https://{BN_DOMAIN}/{BN_VERSION}/getOutgoingEdges'

REQUIRED_BN_HEADERS = {'Accept-Encoding': 'gzip'}
LANG = 'EN'


with open(API_KEY_FILEPATH) as f:
    api_key = f.read().strip()


def retrieve_bn_subgraph(word):
    cached_bn_subgraph_filename = f'{CACHED_BN_SUBGRAPHS_DIR}/{word}'

    if not os.path.exists(cached_bn_subgraph_filename):
        G = construct_bn_subgraph(word)
        with open(cached_bn_subgraph_filename, 'w') as f:
            f.write(G)

    else:
        with open(cached_bn_subgraph_filename) as f:
            G = f.read()

    return G


def construct_bn_subgraph(word, max_path_len=10):
    G = nx.DiGraph()
    synset_queue = queue.SimpleQueue()

    word_synset_ids = get_synsets_containing_lemma(word)
    for word_synset_id in word_synset_ids:
        # get_synset_info() won't return None here because we know the synset 
        # contains at least one English sense, that of the lemma
        synset_info = get_synset_info(word_synset_id)
        G.add_node(word_synset_id, **synset_info)
        synset_queue.put(word_synset_id)
    visited_synsets = set(word_synset_ids)
    skipped_synsets = set()

    for path_len in range(max_path_len):
        print("Level: " + str(path_len+1))
        next_level_synset_queue = queue.SimpleQueue()
        while not synset_queue.empty():
            synset_id = synset_queue.get()
            outgoing_edges = get_outgoing_edges(synset_id)
            for edge_info, target_synset_id in outgoing_edges:
                if target_synset_id in skipped_synsets:
                    continue
                if target_synset_id not in visited_synsets:
                    synset_info = get_synset_info(target_synset_id)
                    if synset_info:
                        G.add_node(target_synset_id, **synset_info)
                        next_level_synset_queue.put(target_synset_id)
                        visited_synsets.add(target_synset_id)
                    else:
                        print(f"Synset {target_synset_id} does not contain an English word sense... skipping")
                        skipped_synsets.add(target_synset_id)
                        continue
                G.add_edge(synset_id, target_synset_id, **edge_info)
        synset_queue = next_level_synset_queue

    return G


def get_synsets_containing_lemma(lemma):
    cached_lemma_synsets_filename = f'{CACHED_LEMMA_SYNSETS_DIR}/{lemma}.txt'

    if not os.path.exists(cached_lemma_synsets_filename):
        synset_ids = request_lemma_synsets(lemma)
        with open(cached_lemma_synsets_filename, 'w') as f:
            f.write('\n'.join(synset_ids))

    else:
        with open(cached_lemma_synsets_filename) as f:
            synset_ids = f.read().splitlines()
       
    return synset_ids


def request_lemma_synsets(lemma):
    params = {
        'lemma': lemma,
        'searchLang': LANG,
        'key': api_key
    }
    json = query_babelnet(LEMMA_SYNSETS_URL, params)

    return [synset['id'] for synset in json]


def query_babelnet(url, params):
    res = requests.get(url, params=params, headers=REQUIRED_BN_HEADERS)
    json = res.json()

    if 'message' in json and 'limit' in json['message']:
        raise ValueError(json['message'])

    return json


def get_synset_info(synset_id):
    cached_synset_info_filename = f'{CACHED_SYNSET_INFO_DIR}/{synset_id}.json'

    if not os.path.exists(cached_synset_info_filename):
        synset_info = request_synset_info(synset_id)
        with open(cached_synset_info_filename, 'w') as f:
            jsonlib.dump(synset_info, f, separators=(',', ':'))

    else:
        with open(cached_synset_info_filename) as f:
            synset_info = jsonlib.load(f)

    return synset_info


def request_synset_info(synset_id):
    params = {
        'id': synset_id,
        'key': api_key,
        'targetLang': LANG
    }
    json = query_babelnet(SYNSET_INFO_URL, params)

    # Some synsets have no English senses, so we skip them
    return prune_synset_info(json) if len(json['senses']) > 0 else None


def prune_synset_info(json):
    synset_info = {
        'pos': json['senses'][0]['properties']['pos'],
        'type': json['synsetType'],
        'domains': json['domains'],
        'isKeyConcept': json['bkeyConcepts'],
        'senses': [prune_sense_info(sense) for sense in json['senses']],
        'glosses': [prune_gloss_info(gloss) for gloss in json['glosses']],
        'examples': [prune_example_info(example) for example in json['examples']],
        'labelTags': extract_label_tags(json['tags'])
    }

    return synset_info


def prune_sense_info(json):
    sense_props = json['properties']
    sense_info = extract_fields(sense_props, ('fullLemma', 'simpleLemma', 'source'))
    sense_info['isKeySense'] = sense_props['bKeySense']
    sense_info['lemma'] = sense_props['lemma']['lemma']
    sense_info['type'] = sense_props['lemma']['type']

    return sense_info


def extract_fields(d, field_list):
    return {field: d[field] for field in field_list}


def prune_gloss_info(json):
    return extract_fields(json, ('gloss', 'source'))


def prune_example_info(json):
    return extract_fields(json, ('example', 'source'))


def extract_label_tags(tags):
    return [
        tag['DATA']['label'] 
        for tag in tags
        if type(tag) is dict and tag['CLASSNAME'].endswith('LabelTag') and tag['DATA']['language'] == LANG
    ]


def get_outgoing_edges(synset_id):
    cached_outgoing_edges_filename = f'{CACHED_OUTGOING_EDGES_DIR}/{synset_id}.json'

    if not os.path.exists(cached_outgoing_edges_filename):
        outgoing_edges = request_outgoing_edges(synset_id)
        with open(cached_outgoing_edges_filename, 'w') as f:
            jsonlib.dump(outgoing_edges, f, separators=(',', ':'))

    else:
        with open(cached_outgoing_edges_filename) as f:
            outgoing_edges = jsonlib.load(f)

    return outgoing_edges


def request_outgoing_edges(synset_id):
    params = {
        'id': synset_id,
        'key': api_key,
    }
    json = query_babelnet(OUTGOING_EDGES_URL, params)

    return [prune_edge_info(edge) for edge in json if edge['language'] in (LANG, 'MUL')]


def prune_edge_info(json):
    edge_info = extract_fields(json['pointer'], ('name', 'shortName', 'relationGroup', 'isAutomatic'))

    return edge_info, json['target']
