import os
import queue
import requests

import gzip
import json as jsonlib
import networkx as nx

from . import package_fp


CACHED_BN_SUBGRAPHS_DIR = f'{package_fp}/data/cached_babelnet_subgraphs'
CACHED_LEMMA_SYNSETS_DIR = f'{package_fp}/data/cached_lemma_synsets'
CACHED_SYNSET_INFO_DIR = f'{package_fp}/data/cached_synset_info'
CACHED_OUTGOING_EDGES_DIR = f'{package_fp}/data/cached_outgoing_edges'
API_KEY_FILEPATH = f'{package_fp}/bn_api_key.txt'

BN_DOMAIN = 'babelnet.io'
BN_VERSION = 'v8'
LEMMA_SYNSETS_URL = f'https://{BN_DOMAIN}/{BN_VERSION}/getSynsetIds'
SYNSET_INFO_URL = f'https://{BN_DOMAIN}/{BN_VERSION}/getSynset'
OUTGOING_EDGES_URL = f'https://{BN_DOMAIN}/{BN_VERSION}/getOutgoingEdges'

REQUIRED_BN_HEADERS = {'Accept-Encoding': 'gzip'}
LANG = 'EN'

# Labeled constants
SRC_SYNSET = True
NOT_SRC_SYNSET = False


offline = False
if offline:
    """
    This code won't work if you haven't installed and set up the BabelNet package to read from a local
    copy of the BabelNet indices, see: https://babelnet.org/guide
    """

    import babelnet as bn
    from babelnet import Language, BabelSynsetID
    from babelnet.synset import SynsetType
    from babelnet.data.relation import BabelPointer

    HYPERNYM_RELATIONSHIP_TYPES = [
        BabelPointer.ANY_HYPERNYM,
        BabelPointer.HYPERNYM,
        BabelPointer.HYPERNYM_INSTANCE,
        BabelPointer.WIBI_HYPERNYM,
        BabelPointer.WIKIDATA_HYPERNYM
    ]
    HYPO_AND_HYPERNYM_RELATIONSHIP_TYPES = HYPERNYM_RELATIONSHIP_TYPES + [
        BabelPointer.ANY_HYPONYM,
        BabelPointer.HYPONYM,
        BabelPointer.HYPONYM_INSTANCE,
        BabelPointer.WIBI_HYPONYM,
        BabelPointer.WIKIDATA_HYPONYM,
        BabelPointer.WIKIDATA_HYPONYM_INSTANCE
    ]


online = False
if online:
    # Only needed/used in the online scraping at the bottom of this file and the old BabelNet code in `babelnet_bots.py`
    with open(API_KEY_FILEPATH) as f:
        api_key = f.read().strip()
else:
    api_key = None


def retrieve_bn_subgraph(word):
    cached_bn_subgraph_filename = f'{CACHED_BN_SUBGRAPHS_DIR}/{word}.gz'

    if not os.path.exists(cached_bn_subgraph_filename):
        # Construct and cache a subgraph of BabelNet for the given word if there is not already a cached subgraph available
        G = construct_bn_subgraph_offline(word)
        with gzip.open(cached_bn_subgraph_filename, 'wt', encoding='UTF-8') as zipfile:
            jsonlib.dump(nx.node_link_data(G), zipfile)

    else:
        with gzip.open(cached_bn_subgraph_filename, 'rt', encoding='UTF-8') as zipfile:
            G = nx.node_link_graph(jsonlib.load(zipfile))

    return G


def construct_bn_subgraph_offline(word, max_path_len=10):
    G = nx.DiGraph(source_synset_ids=[])  # Directed graph because edges in BabelNet are not symmetric
    synset_queue = queue.SimpleQueue()
    visited_synsets = set()
    skipped_synsets = set()
    outgoing_edges = dict()

    # Initialize queue with just the source synsets containing the word
    word_synsets = bn.get_synsets(word, from_langs=[Language.EN])
    for word_synset in word_synsets:
        word_synset_id = str(word_synset.id)

        """
        get_synset_info() won't return `None` here because we know the synset 
        contains at least one English sense, that of the lemma
        """
        synset_info, synset_outgoing_edges = get_synset_info_offline(word_synset, SRC_SYNSET)

        # Add source synset info to graph
        G.graph['source_synset_ids'].append(word_synset_id)
        G.add_node(word_synset_id, **synset_info)

        outgoing_edges[word_synset_id] = synset_outgoing_edges
        synset_queue.put(word_synset_id)
        visited_synsets.add(word_synset_id)

    for path_len in range(max_path_len):
        print("Level: " + str(path_len+1))
        next_level_synset_queue = queue.SimpleQueue()
        while not synset_queue.empty():
            synset_id = synset_queue.get()

            # Retrieve all relevant neighbor synsets
            synset_outgoing_edges = outgoing_edges[synset_id]

            print(f"Expanding from synset with {len(synset_outgoing_edges)} edges: " + synset_id)
            for edge_info, target_synset_id in synset_outgoing_edges:
                if target_synset_id in skipped_synsets:
                    continue
                if target_synset_id not in visited_synsets:
                    target_synset = bn.get_synset(BabelSynsetID(target_synset_id))
                    target_synset_info, target_synset_outgoing_edges = get_synset_info_offline(target_synset, NOT_SRC_SYNSET)

                    # `target_synset_info` will be `None` if it has been determined to be an undesirable synset to visit
                    if target_synset_info:
                        G.add_node(target_synset_id, **target_synset_info)
                        outgoing_edges[target_synset_id] = target_synset_outgoing_edges
                        next_level_synset_queue.put(target_synset_id)
                        visited_synsets.add(target_synset_id)
                    else:
                        skipped_synsets.add(target_synset_id)
                        continue

                # Add edge to the graph even if target synset has already been visited, since the edge is unique
                G.add_edge(synset_id, target_synset_id, **edge_info)
        synset_queue = next_level_synset_queue

    return G


def get_synset_info_offline(synset, is_src_synset):
    senses = [get_sense_info(sense) for sense in synset.senses(language=Language.EN)]

    # Check if synset has any English senses
    if len(senses) == 0:
        # print(f"Synset {synset.id} does not contain an English word sense... skipping")
        return None, None

    # Check to make sure synset is a concept (as opposed to a named entity or unknown)
    # Source synsets are allowed to be non-concepts
    if not is_src_synset and synset.type != SynsetType.CONCEPT:
        # print(f"Synset {synset.id} is not a concept... skipping")
        return None, None

    synset_info = {
        'pos': str(synset.pos),
        'type': str(synset.type),
        'domains': {str(domain): val for domain, val in synset.domains.items()},
        'isKeyConcept': synset.is_key_concept,
        'senses': senses,
        'mainSense': get_sense_info(synset.main_sense(language=Language.EN)),
        'glosses': [get_gloss_info(gloss) for gloss in synset.glosses(language=Language.EN)],
        'mainGloss': get_gloss_info(synset.main_gloss(language=Language.EN)),
        'examples': [get_example_info(example) for example in synset.examples(language=Language.EN)],
        'mainExample': get_example_info(synset.main_example(language=Language.EN))
    }

    # Filter outgoing edges to just hypernym relationships if not a source synset
    all_synset_outgoing_edges = synset.outgoing_edges() if is_src_synset else synset.outgoing_edges(*HYPERNYM_RELATIONSHIP_TYPES)
    # Prune out edges we don't want to follow
    synset_outgoing_edges = [get_edge_info(edge) for edge in all_synset_outgoing_edges if should_follow_edge(edge)]

    return synset_info, synset_outgoing_edges


def get_sense_info(sense):
    sense_info = {
        'fullLemma': sense.full_lemma,
        'normalizedLemma': sense.normalized_lemma,
        'source': str(sense.source),
        'isKeySense': sense.is_key_sense,
        'lemma': sense.lemma.lemma,
        'lemmaType': str(sense.lemma.lemma_type),
        'isAutomatic': sense.is_automatic_translation
    }

    return sense_info


def get_gloss_info(gloss):
    if gloss is None:
        return None

    gloss_info = {
        'gloss': gloss.gloss,
        'source': str(gloss.source)
    }

    return gloss_info


def get_example_info(example):
    if example is None:
        return None

    example_info = {
        'example': example.example,
        'source': str(example.source)
    }

    return example_info


def get_edge_info(edge):
    edge_info = {
        'name': edge.pointer.relation_name,
        'shortName': edge.pointer.short_name,
        'relationGroup': str(edge.pointer.relation_group)
    }

    return edge_info, edge.target


def should_follow_edge(edge):
    if edge.language not in (Language.EN, Language.MUL):
        return False

    pointer = edge.pointer

    # Automatic edges tend to be bad quality
    if pointer.is_automatic:
        return False

    return True


"""
WARNING: 

THE BELOW CODE FOR ONLINE API SCRAPING IS NOT AS ROBUST
AS THE ABOVE CODE FOR OFFLINE LOCAL INDEX SCRAPING
"""

def construct_bn_subgraph_online(word, max_path_len=10):
    G = nx.DiGraph()
    synset_queue = queue.SimpleQueue()

    word_synset_ids = get_synsets_containing_lemma(word)
    for word_synset_id in word_synset_ids:
        """
        get_synset_info() won't return `None` here because we know the synset 
        contains at least one English sense, that of the lemma
        """
        synset_info = get_synset_info(word_synset_id)

        # Add source synset info to graph
        G.graph['source_synset_ids'].append(word_synset_id)
        G.add_node(word_synset_id, **synset_info)

        synset_queue.put(word_synset_id)
    visited_synsets = set(word_synset_ids)
    skipped_synsets = set()

    for path_len in range(max_path_len):
        print("Level: " + str(path_len+1))
        next_level_synset_queue = queue.SimpleQueue()
        while not synset_queue.empty():
            synset_id = synset_queue.get()

            # Retrieve all relevant neighbor synsets
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

                # Add edge to the graph even if target synset has already been visited, since the edge is unique
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
