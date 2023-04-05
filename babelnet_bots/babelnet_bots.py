import gzip
from itertools import combinations
import os
import re
import requests
import string
import pickle
from scipy.spatial import distance

# Gensim
from gensim.corpora import Dictionary
import gensim.downloader as api

from codenames_bots import Spymaster, FieldOperative

from nltk.stem import WordNetLemmatizer

# Graphing
import networkx as nx


babelnet_relationships_limits = {
    "HYPERNYM": float("inf"),
    "HYPONYM": 20,
    "MERONYM": 20,
}

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


class BabelNetSpymaster(Spymaster):

    # Constants
    VERB_SUFFIX = "v"
    NOUN_SUFFIX = "n"
    ADJ_SUFFIX = "a"

    # File paths to cached babelnet query results
    bn_data_dir = 'data/babelnet_v6/'
    synset_main_sense_file = bn_data_dir + 'synset_to_main_sense.txt'
    synset_senses_file = bn_data_dir + 'synset_to_senses.txt'
    synset_metadata_file = bn_data_dir + 'synset_to_metadata.txt'

    verbose = False
    split_multi_word = True
    disable_verb_split = True
    length_exp_scaling = None
    single_word_label_scores = (1, 1.1, 1.1, 1.2)

    given_clues = []

    def __init__(self, game_words):
        with open(API_KEY_FILEPATH) as f:
            self.api_key = f.read()

        (
            self.synset_to_main_sense,
            self.synset_to_senses,
            self.synset_to_metadata,
        ) = self._load_synset_data_v5()

        with open('data/word_to_dict2vec_embeddings', 'rb') as f:
            self.dict2vec_embeddings = pickle.load(f)

        # Dictionary of word to document frequency
        self.num_docs, self.word_to_df = self._load_document_frequencies()

        # Used to get word lemmas
        self.lemmatizer = WordNetLemmatizer()

        self.game_words = game_words
        self.weighted_nns = dict()
        for word in self.game_words:
            self.get_weighted_nns(word)

        if self.verbose:
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
        if os.path.exists(self.synset_metadata_file):
            with open(self.synset_metadata_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    assert len(parts) == 3
                    synset, key_concept, synset_type = parts
                    synset_to_metadata[synset] = {
                        "key_concept": key_concept,
                        "synset_type": synset_type,
                    }

        return (
            synset_to_main_sense,
            synset_to_senses,
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

        # Keep track of the best-looking clue across all possibilities
        best_score = float('-inf')

        # Check all combinations of target words
        for n_target_words in range(2, 3):
            for potential_target_words in combinations(team_words, n_target_words):
                clue, score = self.get_clue_for_target_words(
                    potential_target_words,
                    opp_words,
                    bystanders,
                    assassin,
                )
                if score > best_score:
                    best_clue = clue
                    best_score = score
                    target_words = potential_target_words

        n_target_words = len(target_words)

        if self.verbose:
            print(f"Clue: {best_clue}, {n_target_words} ({target_words})")

        self.given_clues.append(best_clue)

        return best_clue, n_target_words

    def get_clue_for_target_words(self, target_words, opp_words, bystanders, assassin):
        potential_clues = set()
        for target_word in target_words:
            nns = self.weighted_nns[target_word].keys()
            potential_clues.update(nns)

        best_score = float('-inf')

        for clue in potential_clues:
            # Don't give the same clue twice
            if clue in self.given_clues:
                continue

            # Make sure clue would be valid according to Codenames rules
            if not self.is_valid_clue(clue):
                continue

            """
            babelnet_score is a score based on the clue's distance to the target words
            in the BabelNet graph
            """
            babelnet_score = 0
            for target_word in target_words:
                if clue in self.weighted_nns[target_word]:
                    babelnet_score += self.weighted_nns[target_word][clue]
                else:
                    babelnet_score += -1

            """
            opp_word_penalty is a penalty for clues that are similar to opponent words
            """
            non_team_words = opp_words.union(bystanders).union({assassin})
            opp_word_penalty = self.penalize(clue, non_team_words)

            """
            detect_score is a score based on the clue's rarity and its dictionary
            similarity to target words
            """
            detect_score = self.get_detect_score(clue, target_words, opp_words)

            # TODO: add an agressiveness factor
            total_score = babelnet_score - 0.5 * opp_word_penalty + detect_score

            if total_score > best_score:
                best_clue = clue
                best_score = total_score

        return best_clue, best_score

    def is_valid_clue(self, clue):
        """
        A valid clue must be one word consisting of letters only. 
        It also can't be any form of an unguessed board word 
        (such as "broken" when "break" is on the board), nor can 
        either the clue or an unguessed board word be a substring
        of each other.
        """
        for board_word in self.game_words:
            if clue in board_word or board_word in clue:
                return False
            if self.lemmatizer.lemmatize(clue) == self.lemmatizer.lemmatize(board_word) or not clue.isalpha():
                return False

        return True

    def penalize(self, clue, non_team_words):
        """
        :param clue: potential clue
        :param non_team_words: all words not belonging to own team
        """

        max_non_team_word_similarity = 0.0
        for non_team_word in non_team_words:
            if clue in self.weighted_nns[non_team_word]:
                similarity = self.weighted_nns[non_team_word][clue]
                if similarity > max_non_team_word_similarity:
                    max_non_team_word_similarity = similarity

        return max_non_team_word_similarity

    def get_detect_score(self, clue, target_words, opp_words):
        """
        returns: using IDF and dictionary definition heuristics, how much to add to the score
        for this potential clue give these board words
        """

        # The larger the idf is, the more uncommon the word
        idf = (1.0 / self.word_to_df[clue]) if clue in self.word_to_df else 1.0

        # Prune out super common words (e.g. "get", "go") and super rare words
        if clue in stopwords or idf < idf_lower_bound:
            idf = 1.0

        freq = -idf
        dict2vec_score = self.get_dict2vec_score(clue, target_words, opp_words)

        return FREQ_WEIGHT * freq + DICT2VEC_WEIGHT * dict2vec_score

    def get_dict2vec_score(self, clue, target_words, opp_words):
        """
        :param target_words: the board words intended for the potential clue
        :param clue: potential candidate clue
        :param opp_words: the opponent's words on the board
        returns: the similarity of the two input embedding vectors using their cosine distance
        """

        if clue not in self.dict2vec_embeddings:
            return 0.0

        clue_embedding = self.dict2vec_embeddings[clue]
        target_word_similarity_sum = 0.0
        for target_word in target_words:
            if target_word in self.dict2vec_embeddings:
                target_word_embedding = self.dict2vec_embeddings[target_word]
                dict2vec_similarity = self.get_similarity(clue_embedding, target_word_embedding)
                target_word_similarity_sum += dict2vec_similarity

        max_opp_similarity = float('-inf')
        for opp_word in opp_words:
            if opp_word in self.dict2vec_embeddings:
                opp_word_embedding = self.dict2vec_embeddings[opp_word]
                opp_word_similarity = self.get_similarity(clue_embedding, opp_word_embedding)
                if opp_word_similarity > max_opp_similarity:
                    max_opp_similarity = opp_word_similarity

        return target_word_similarity_sum - max_opp_similarity

    def get_similarity(self, embedding1, embedding2):
        """
        :param embedding1: a dict2vec word embedding
        :param embedding2: another dict2vec word embedding
        returns: the cosine similarity of the two input embedding vectors
        """

        cosine_distance = distance.cosine(embedding1, embedding2)

        # Convert from cosine distance to cosine similarity
        cosine_similarity = 1 - cosine_distance

        return cosine_similarity

    def get_weighted_nns(self, word, filter_entities=True):
        """
        :param filter_entities: idk yet
        :param word: the codeword to get weighted nearest neighbors for
        returns: a dictionary mapping nearest neighbors (str) to distances from codeword (int)
        """

        def should_add_relationship(relationship, level):
            if level > 1 and relationship != 'HYPERNYM':
                return False
            if relationship not in babelnet_relationships_limits:
                return False
            return count_by_relation_group[relationship] < babelnet_relationships_limits[relationship]

        def single_source_paths_filter(G, source, cutoff=float('inf')):
            # Not all synsets have outgoing relations
            if source not in G:
                raise nx.NodeNotFound("Source {} not in G".format(source))

            def join(p1, p2):
                return p1 + p2

            """
            if cutoff is None:
                cutoff = float('inf')
                """
            nextlevel = {source: 1}     # list of nodes to check at next level
            # paths dictionary  (paths to key from source)
            paths = {source: [source]}
            return dict(_single_source_paths_filter(G, nextlevel, paths, cutoff, join))

        def _single_source_paths_filter(G, firstlevel, paths, cutoff, join):
            level = 0                  # the current level
            nextlevel = firstlevel
            while nextlevel and level < cutoff:
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

        count_by_relation_group = {relationship: 0 for relationship in babelnet_relationships_limits}

        G = nx.DiGraph()

        # Add edges based on relations stored in cached file that are outgoing from all synsets that the word belongs to
        with gzip.open(self.bn_data_dir + word + '.gz', 'r') as f:
            for line in f:
                (
                    source,
                    target,
                    language,
                    short_name,
                    relation_group,
                    is_automatic,
                    level
                ) = line.decode('utf-8').strip().split('\t')

                if is_automatic == 'False' and should_add_relationship(relation_group, int(level)):
                    G.add_edge(source, target, relationship=short_name)
                    count_by_relation_group[relation_group] += 1

        nn_w_dists = {}
        nn_w_synsets = {}
        with open(self.bn_data_dir + word + '_synsets', 'r') as f:
            for line in f:
                synset = line.strip()
                try:
                    # get all paths starting from source, filtered
                    paths = single_source_paths_filter(
                        G, source=synset, cutoff=10
                    )
                    # NOTE: if we want to filter intermediate nodes, we need to call
                    # get_cached_labels_from_synset_v5 for all nodes in path.
                    if self.length_exp_scaling is not None:
                        scaling_func = lambda x: self.length_exp_scaling ** x
                    else:
                        scaling_func = lambda x: x
                    lengths = {neighbor: scaling_func(len(path))
                               for neighbor, path in paths.items()}
                except nx.NodeNotFound as e:
                    if self.verbose:
                        print(e)
                    continue
                for neighbor, length in lengths.items():
                    neighbor_main_sense, neighbor_senses, neighbor_metadata = self.get_cached_labels_from_synset_v5(
                        neighbor, get_metadata=filter_entities
                    )
                    # Note: this filters entity clues, not intermediate entity nodes
                    if filter_entities and neighbor_metadata["synset_type"] != "CONCEPT":
                        if self.verbose:
                            print("skipping non-concept:", neighbor, neighbor_metadata["synset_type"])
                        continue

                    split_multi_word = self.split_multi_word
                    if self.disable_verb_split and synset.endswith(self.VERB_SUFFIX):
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

        nn_w_dists = {k: 1.0 / (v + 1) for k, v in nn_w_dists.items() if k != word}

        self.weighted_nns[word] = nn_w_dists

    """
    Babelnet methods
    """

    def get_cached_labels_from_synset_v5(self, synset, get_metadata=False):
        """This actually gets the main_sense but also writes all senses"""
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

    def write_synset_labels_v5(self, synset, json):
        """Write to synset_main_sense_file and synset_senses_file"""
        if synset not in self.synset_to_main_sense:
            with open(self.synset_main_sense_file, "a") as f:
                if "mainSense" not in json:
                    if self.verbose:
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

        if synset not in self.synset_to_metadata:
            with open(self.synset_metadata_file, "a") as f:
                key_concept = "NONE"
                synset_type = "NONE"
                if "bkey_concepts" in json:
                    key_concept = str(json["bkey_concepts"])
                if "synset_type" in json:
                    synset_type = json["synset_type"]
                f.write("\t".join([synset, key_concept, synset_type]) + "\n")
                self.synset_to_metadata[synset] = {
                    "key_concept": key_concept,
                    "synset_type": synset_type,
                }

    def get_single_word_labels_v5(self, lemma, senses, split_multi_word=False):
        main_single, main_multi, other_single, other_multi = self.single_word_label_scores
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

    def parse_lemma_v5(self, lemma):
        lemma_parsed = lemma.split('#')[0]
        parts = lemma_parsed.split('_')
        single_word = len(parts) == 1 or parts[1].startswith('(')
        return parts[0], single_word


class BabelNetFieldOperative(FieldOperative):

    # TODO: Implement
    def make_guess(self, words, clue):
        return 'word'
