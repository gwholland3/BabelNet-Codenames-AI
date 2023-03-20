import gzip
from itertools import combinations
import os
import re
import requests
import string
import pickle

# Gensim
from gensim.corpora import Dictionary
import gensim.downloader as api

from codenames_bots import Spymaster, FieldOperative

# nltk
from nltk.stem import PorterStemmer

# Graphing
import networkx as nx

from babelnet_bots.utils import get_dict2vec_score


babelnet_relationships_limits = {
    "HYPERNYM": float("inf"),
    "OTHER": 0,
    "MERONYM": 20,
    "HYPONYM": 20,
}

punctuation = re.compile("[" + re.escape(string.punctuation) + "]")

stopwords = [
    'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 
    'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 
    'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 
    'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 
    'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 
    'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 
    'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 
    'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 
    'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 
    'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 
    'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 
    'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 
    'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 
    'further', 'was', 'here', 'than', 'get', 'put'
]

idf_lower_bound = 0.0006
FREQ_WEIGHT = 2
DICT2VEC_WEIGHT = 2

API_KEY_FILEPATH = 'babelnet_bots/bn_api_key.txt'


"""
Configuration for the bot
"""
class Configuration():
    def __init__(
        self,
        verbose=False,
        split_multi_word=True,
        disable_verb_split=True,
        length_exp_scaling=None,
    ):
        self.verbose = verbose
        self.split_multi_word = split_multi_word
        self.disable_verb_split = disable_verb_split
        self.length_exp_scaling = length_exp_scaling


class BabelNetSpymaster(Spymaster):

    # Constants
    VERB_SUFFIX = "v"
    NOUN_SUFFIX = "n"
    ADJ_SUFFIX = "a"

    # File paths to cached babelnet query results
    file_dir = 'data/babelnet_v6/'
    synset_main_sense_file = file_dir + 'synset_to_main_sense.txt'
    synset_senses_file = file_dir + 'synset_to_senses.txt'
    synset_glosses_file = file_dir + 'synset_to_glosses.txt'
    synset_metadata_file = file_dir + 'synset_to_metadata.txt'

    default_single_word_label_scores = (1, 1.1, 1.1, 1.2)

    def __init__(self, game_words):
        with open(API_KEY_FILEPATH) as f:
            self.api_key = f.read()

        # Initialize variables
        self.configuration = Configuration()
        self.configuration.single_word_label_scores = self.default_single_word_label_scores
        print("Spymaster Configuration: ", self.configuration.__dict__)

        (
            self.synset_to_main_sense,
            self.synset_to_senses,
            self.synset_to_definitions,
            self.synset_to_metadata,
        ) = self._load_synset_data_v5()

        self.num_docs, self.word_to_df = self._load_document_frequencies()  # dictionary of word to document frequency

        # Used to get word stems
        self.stemmer = PorterStemmer()

        self.game_words = game_words
        self.weighted_nns = dict()
        for word in self.game_words:
            self.get_weighted_nns(word)

        if self.configuration.verbose:
            print("NEAREST NEIGHBORS:")
            for word, clues in self.weighted_nns.items():
                print(word)
                print(sorted(clues, key=lambda k: clues[k], reverse=True)[:5])

    """
    Pre-process steps
    """

    def _load_synset_data_v5(self):
        """Load synset_to_main_sense"""
        synset_to_main_sense = {}
        synset_to_senses = {}
        synset_to_definitions = {}
        synset_to_metadata = {}
        if os.path.exists(self.synset_main_sense_file):
            with open(self.synset_main_sense_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    synset, main_sense = parts[0], parts[1]
                    synset_to_main_sense[synset] = main_sense
                    synset_to_senses[synset] = set()
        if os.path.exists(self.synset_senses_file):
            with open(self.synset_senses_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    assert len(parts) == 5
                    synset, full_lemma, simple_lemma, source, pos = parts
                    if source == "WIKIRED":
                        continue
                    synset_to_senses[synset].add(simple_lemma)
        if os.path.exists(self.synset_glosses_file):
            with open(self.synset_glosses_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    assert len(parts) == 3
                    synset, source, definition = parts
                    if synset not in synset_to_definitions:
                        synset_to_definitions[synset] = set()
                    if source == "WIKIRED":
                        continue
                    synset_to_definitions[synset].add(definition)
        if os.path.exists(self.synset_metadata_file):
            with open(self.synset_metadata_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    assert len(parts) == 3
                    synset, keyConcept, synsetType = parts
                    synset_to_metadata[synset] = {
                        "keyConcept": keyConcept,
                        "synsetType": synsetType,
                    }

        return (
            synset_to_main_sense,
            synset_to_senses,
            synset_to_definitions,
            synset_to_metadata,
        )

    def _load_document_frequencies(self):
        """
        Sets up a dictionary from words to their document frequency
        """
        if (os.path.exists("data/word_to_df.pkl")) and (os.path.exists("data/text8_num_documents.txt")):
            with open('data/word_to_df.pkl', 'rb') as f:
                word_to_df = pickle.load(f)
            with open('data/text8_num_documents.txt', 'rb') as f:
                for line in f:
                    num_docs = int(line.strip())
                    break
        else:
            dataset = api.load("text8")
            dct = Dictionary(dataset)
            id_to_doc_freqs = dct.dfs
            num_docs = dct.num_docs
            word_to_df = {dct[id]: id_to_doc_freqs[id]
                          for id in id_to_doc_freqs}
            with open('data/text8_num_documents.txt', 'w') as f:
                f.write(str(num_docs))
            with open('data/word_to_df.pkl', 'wb') as f:
                pickle.dump(word_to_df, f)

        return num_docs, word_to_df

    def give_clue(self, team_words, opp_words, bystanders, assassin):
        """
        Required Codenames method
        """
        penalty = 1

        # potential clue candidates are the intersection of weighted_nns[word] for each word in team_words
        # we need to repeat this for the (|team_words| C n) possible words we can give a clue for

        best_score = float('-inf')

        for n_target_words in range(2, 3):
            for potential_target_words in combinations(team_words, n_target_words):
                clue, score = self.get_clue_for_target_words(
                    potential_target_words, 
                    opp_words, 
                    bystanders,
                    assassin,
                    penalty
                )
                if score > best_score:
                    best_clue = clue
                    best_score = score
                    target_words = potential_target_words

        n_target_words = len(target_words)

        if self.configuration.verbose:
            print(f"Clue: {best_clue}, {n_target_words} ({target_words})")

        return best_clue, n_target_words

    def get_clue_for_target_words(self, target_words, opp_words, bystanders, assassin, penalty=1.0):
        potential_clues = set()
        for target_word in target_words:
            nns = self.weighted_nns[target_word].keys()
            potential_clues.update(nns)

        best_score = float('-inf')

        for clue in potential_clues:
            # Don't consider clues which are a substring of any board words
            if not self.is_valid_clue(clue):
                continue

            babelnet_score = 0
            for target_word in target_words:
                if clue in self.weighted_nns[target_word]:
                    babelnet_score += self.weighted_nns[target_word][clue]
                else:
                    babelnet_score += -1

            detect_score = self.get_detect_score(clue, target_words, opp_words)

            # Give embedding methods the opportunity to rescale the score using their own heuristics
            embedding_score = self.rescale_score(target_words, clue, opp_words.union(bystanders).union(set(assassin)))

            # TODO: add an agressiveness factor
            total_score = babelnet_score + detect_score + embedding_score

            if total_score > best_score:
                best_clue = clue
                best_score = total_score

        return best_clue, best_score

    def is_valid_clue(self, clue):
        """
        No need to remove board words from potential_clues elsewhere
        since we check for validity here
        """
        for board_word in self.game_words:
            # Check if clue or board_word are substring of each other, or if they share the same word stem
            if clue in board_word or board_word in clue or self.stemmer.stem(clue) == self.stemmer.stem(board_word) or not clue.isalpha():
                return False

        return True

    def get_detect_score(self, clue, target_words, opp_words):
        # The larger the idf is, the more uncommon the word
        idf = (1.0 / self.word_to_df[clue]) if clue in self.word_to_df else 1.0

        # Prune out super common words (e.g. "get", "go")
        if clue in stopwords or idf < idf_lower_bound:
            idf = 1.0

        freq = -idf
        dict2vec_score = get_dict2vec_score(target_words, clue, opp_words)

        return FREQ_WEIGHT * freq + DICT2VEC_WEIGHT * dict2vec_score

    def rescale_score(self, target_words, clue, non_team_words):
        """
        :param target_words: potential board words we could apply this clue to
        :param clue: potential clue
        :param opp_words: opponent's words
        returns: using IDF and dictionary definition heuristics, how much to add to the score for this potential clue give these board words
        """

        max_non_team_word_similarity = float("-inf")
        found_clue = False
        for non_team_word in non_team_words:
            if non_team_word in self.weighted_nns and clue in self.weighted_nns[non_team_word]:
                similarity = self.weighted_nns[non_team_word][clue]
                found_clue = True
                if similarity > max_non_team_word_similarity:
                    max_non_team_word_similarity = similarity
        # If we haven't encountered our potential clue in any of the non-team word's nearest neighbors, set max_non_team_word_similarity to 0
        if found_clue == False:
            max_non_team_word_similarity = 0.0

        return 0.5 * max_non_team_word_similarity

    def get_weighted_nns(self, word, filter_entities=True):
        """
        :param word: the codeword to get weighted nearest neighbors for
        returns: a dictionary mapping nearest neighbors (str) to distances from codeword (int)
        """
        def should_add_relationship(relationship, level):
            if relationship != 'HYPERNYM' and level > 1:
                return False
            return relationship in babelnet_relationships_limits.keys() and \
                count_by_relation_group[relationship] < babelnet_relationships_limits[relationship]

        def _single_source_paths_filter(G, firstlevel, paths, cutoff, join):
            level = 0                  # the current level
            nextlevel = firstlevel
            while nextlevel and cutoff > level:
                thislevel = nextlevel
                nextlevel = {}
                for v in thislevel:
                    for w in G.adj[v]:
                        if len(paths[v]) >= 3 and G.edges[paths[v][1], paths[v][2]]['relationship'] != G.edges[v, w]['relationship']:
                            continue
                        if w not in paths:
                            paths[w] = join(paths[v], [w])
                            nextlevel[w] = 1
                level += 1
            return paths

        def single_source_paths_filter(G, source, cutoff=None):
            if source not in G:
                raise nx.NodeNotFound("Source {} not in G".format(source))

            def join(p1, p2):
                return p1 + p2
            if cutoff is None:
                cutoff = float('inf')
            nextlevel = {source: 1}     # list of nodes to check at next level
            # paths dictionary  (paths to key from source)
            paths = {source: [source]}
            return dict(_single_source_paths_filter(G, nextlevel, paths, cutoff, join))

        count_by_relation_group = {
            key: 0 for key in babelnet_relationships_limits.keys()}

        G = nx.DiGraph()
        with gzip.open(self.file_dir + word + '.gz', 'r') as f:
            for line in f:
                source, target, language, short_name, relation_group, is_automatic, level = line.decode(
                    "utf-8").strip().split('\t')

                if should_add_relationship(relation_group, int(level)) and is_automatic == 'False':
                    G.add_edge(source, target, relationship=short_name)
                    count_by_relation_group[relation_group] += 1

        nn_w_dists = {}
        nn_w_synsets = {}
        dictionary_definitions_for_word = []
        with open(self.file_dir + word + '_synsets', 'r') as f:
            for line in f:
                synset = line.strip()
                try:
                    # get all paths starting from source, filtered
                    paths = single_source_paths_filter(
                        G, source=synset, cutoff=10
                    )
                    # NOTE: if we want to filter intermediate nodes, we need to call
                    # get_cached_labels_from_synset_v5 for all nodes in path.
                    if self.configuration.length_exp_scaling is not None:
                        scaling_func = lambda x : self.configuration.length_exp_scaling ** x
                    else:
                        scaling_func = lambda x : x
                    lengths = {neighbor: scaling_func(len(path))
                               for neighbor, path in paths.items()}
                except nx.NodeNotFound as e:
                    if self.configuration.verbose:
                        print(e)
                    continue
                for neighbor, length in lengths.items():
                    neighbor_main_sense, neighbor_senses, neighbor_metadata = self.get_cached_labels_from_synset_v5(
                        neighbor, get_metadata=filter_entities)
                    # Note: this filters entity clues, not intermediate entity nodes
                    if filter_entities and neighbor_metadata["synsetType"] != "CONCEPT":
                        if self.configuration.verbose:
                            print("skipping non-concept:", neighbor, neighbor_metadata["synsetType"])
                        continue

                    split_multi_word = self.configuration.split_multi_word
                    if self.configuration.disable_verb_split and synset.endswith(self.VERB_SUFFIX):
                        split_multi_word = False

                    single_word_labels = self.get_single_word_labels_v5(
                        neighbor_main_sense,
                        neighbor_senses,
                        split_multi_word=split_multi_word,
                    )
                    for single_word_label, label_score in single_word_labels:
                        if single_word_label not in nn_w_dists:
                            nn_w_dists[single_word_label] = length * label_score
                            nn_w_synsets[single_word_label] = neighbor
                        else:
                            if nn_w_dists[single_word_label] > (length * label_score):
                                nn_w_dists[single_word_label] = length * label_score
                                nn_w_synsets[single_word_label] = neighbor

                main_sense, sense, _ = self.get_cached_labels_from_synset_v5(
                    synset)

                # get definitions
                if synset in self.synset_to_definitions:
                    dictionary_definitions_for_word.extend(
                        self.stemmer.stem(word.lower().translate(str.maketrans('', '', string.punctuation)))
                        for definition in self.synset_to_definitions[synset]
                        for word in definition.split()
                    )

        nn_w_dists = {k: 1.0 / (v + 1) for k, v in nn_w_dists.items() if k != word}

        self.weighted_nns[word] = nn_w_dists

    """
    Babelnet methods
    """

    def get_cached_labels_from_synset_v5(self, synset, get_metadata=False):
        """This actually gets the main_sense but also writes all senses/glosses"""
        if (
                synset not in self.synset_to_main_sense
                or (get_metadata and synset not in self.synset_to_metadata)
        ):
            print("getting query", synset)
            labels_json = self.get_labels_from_synset_v5_json(synset)
            self.write_synset_labels_v5(synset, labels_json)

        main_sense = self.synset_to_main_sense[synset]
        senses = self.synset_to_senses[synset]
        metadata = self.synset_to_metadata[synset] if get_metadata else {}
        return main_sense, senses, metadata

    def write_synset_labels_v5(self, synset, json):
        """Write to synset_main_sense_file, synset_senses_file, and synset_glosses_file"""
        if synset not in self.synset_to_main_sense:
            with open(self.synset_main_sense_file, "a") as f:
                if "mainSense" not in json:
                    if self.configuration.verbose:
                        print("no main sense for", synset)
                    main_sense = synset
                else:
                    main_sense = json["mainSense"]
                f.write("\t".join([synset, main_sense]) + "\n")
                self.synset_to_main_sense[synset] = main_sense

        if synset not in self.synset_to_senses:
            self.synset_to_senses[synset] = set()
            with open(self.synset_senses_file, "a") as f:
                self.synset_to_senses[synset] = set()
                if "senses" in json:
                    for sense in json["senses"]:
                        properties = sense["properties"]
                        line = [
                            synset,
                            properties["fullLemma"],
                            properties["simpleLemma"],
                            properties["source"],
                            properties["pos"],
                        ]
                        f.write("\t".join(line) + "\n")
                        if properties["source"] != "WIKIRED":
                            self.synset_to_senses[synset].add(properties["simpleLemma"])

        if synset not in self.synset_to_definitions:
            self.synset_to_definitions[synset] = set()
            with open(self.synset_glosses_file, "a") as f:
                if "glosses" in json:
                    if len(json["glosses"]) == 0:
                        f.write("\t".join([synset, "NONE", "NONE"]) + "\n")
                    else:
                        for gloss in json["glosses"]:
                            line = [synset, gloss["source"], gloss["gloss"]]
                            f.write("\t".join(line) + "\n")
                            if gloss["source"] != "WIKIRED":
                                self.synset_to_definitions[synset].add(gloss["gloss"])

        if synset not in self.synset_to_metadata:
            with open(self.synset_metadata_file, "a") as f:
                keyConcept = "NONE"
                synsetType = "NONE"
                if "bkeyConcepts" in json:
                    keyConcept = str(json["bkeyConcepts"])
                if "synsetType" in json:
                    synsetType = json["synsetType"]
                f.write("\t".join([synset, keyConcept, synsetType]) + "\n")
                self.synset_to_metadata[synset] = {
                    "keyConcept": keyConcept,
                    "synsetType": synsetType,
                }

    def get_labels_from_synset_v5_json(self, synset):
        url = 'https://babelnet.io/v5/getSynset'
        params = {
            'id': synset,
            'key': self.api_key
        }
        headers = {'Accept-Encoding': 'gzip'}
        res = requests.get(url=url, params=params, headers=headers)
        if "message" in res.json() and "limit" in res.json()["message"]:
            raise ValueError(res.json()["message"])
        return res.json()

    def parse_lemma_v5(self, lemma):
        lemma_parsed = lemma.split('#')[0]
        parts = lemma_parsed.split('_')
        single_word = len(parts) == 1 or parts[1].startswith('(')
        return parts[0], single_word

    def get_single_word_labels_v5(self, lemma, senses, split_multi_word=False):
        main_single, main_multi, other_single, other_multi = self.configuration.single_word_label_scores
        single_word_labels = []
        parsed_lemma, single_word = self.parse_lemma_v5(lemma)
        if single_word:
            single_word_labels.append((parsed_lemma, main_single))
        elif split_multi_word:
            single_word_labels.extend(
                zip(parsed_lemma.split("_"), [main_multi for _ in parsed_lemma.split("_")])
            )

        for sense in senses:
            parsed_lemma, single_word = self.parse_lemma_v5(sense)
            if single_word:
                single_word_labels.append((parsed_lemma, other_single))
            elif split_multi_word:
                single_word_labels.extend(
                    zip(parsed_lemma.split("_"), [other_multi for _ in parsed_lemma.split("_")])
                )
        if len(single_word_labels) == 0:
            # can only happen if split_multi_word = False
            assert not split_multi_word
            return [(lemma.split("#")[0], 1)]
        return single_word_labels


class BabelNetFieldOperative(FieldOperative):

    # TODO: Implement
    def make_guess(self, words, clue):
        return 'word'

