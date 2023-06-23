"""
NOTE: This module was edited for the BabelNet-Codenames-AI repo to fix a small bug
"""

"""This module manages the BabelNet Lucene index."""

import logging
import os
import platform
import traceback
from collections import OrderedDict, defaultdict
from typing import Tuple, Dict, List, Optional, Sequence, Set

import lucene
from java.nio.file import Paths
from ordered_set import OrderedSet
from org.apache.lucene.document import Document
from org.apache.lucene.document import Field

# from org.apache.lucene.document import Field.Store
from org.apache.lucene.document import StoredField
from org.apache.lucene.document import StringField
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.index import MultiReader
from org.apache.lucene.index import Term
from org.apache.lucene.search import BooleanClause

# from org.apache.lucene.search import BooleanClause.Occur
from org.apache.lucene.search import BooleanQuery

# from org.apache.lucene.search import BooleanQuery.Builder
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search import TermQuery
from org.apache.lucene.search import WildcardQuery
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.store import SimpleFSDirectory

from babelnet import _utils
from babelnet._impl import BabelNetIndexField, BabelNetIndexImageField
from babelnet.conf import _config
from babelnet.data import phonetics
from babelnet.data.category import BabelCategory
from babelnet.data.domain import BabelDomain
from babelnet.data.example import BabelExample
from babelnet.data.frame import VerbAtlasFrameID
from babelnet.data.gloss import BabelGloss
from babelnet.data.image import BabelImage, BabelImageComparator
from babelnet.data.license import BabelLicense
from babelnet.data.phonetics import BabelSensePhonetics, BabelAudio
from babelnet.data.qcode import QcodeID
from babelnet.data.source import BabelSenseSource
from babelnet.data.tag import StringTag, Tag, LabelTag
from babelnet.data.tokens import *
from babelnet.indices.index import BNIndex
from babelnet.iterators.iter_impl import *
from babelnet.iterators.abstract_iter import *
from babelnet.language import Language
from babelnet.resources import (
    _InternalBabelSynsetID,
    BabelSynsetID,
    InvalidSynsetIDError,
    ResourceWithLemmaID,
    ResourceID,
    WordNetSynsetID,
)
from babelnet.sense import WordNetSense, BabelSense
from babelnet.synset import SynsetType, _OfflineBabelSynset, BabelSynset
from babelnet.tag_utils import get_tag
from babelnet.versions import BabelVersion, WordNetVersion

lucene.initVM()
_log = logging.getLogger(__name__)
"""logger"""

_MAXIMUM_NUMBER_OF_SYNSETS = 5000
_MAXIMUM_NUMBER_OF_IMAGES = 5000
_USE_IMAGE_FILTER = False

_SEPARATOR = ":"
"""Synset id separator (e.g., bn:00000001, UNR:00000001)"""

_OEWN_PREFIX: str = "oewn:"
_WN2020_PREFIX: str = "wn2020:"
_WN30_PREFIX: str = "wn:"


class OfflineIndex(BNIndex):
    """
    The offline version of the BNIndex.
    """
    def __init__(self):
        """init method"""
        self._babelnet = None
        self._lexicon = None
        self._dictionary = None
        self._graph = None
        self._image = None
        self._info = None
        self._wordnet_info = None
        self._languages = _config.LANGUAGES

        self._license_to_dictionaries = {}
        """Dictionaries (other licenses)."""

        self._license_to_glosses_and_examples = {}
        """Glosses (other licenses)."""

        self._load()

    def version(
        self,
    ) -> BabelVersion:
        try:
            if self._dictionary.getIndexReader().numDocs() > 0:
                # Il campo versione è sempre il primo documento dell'indice dict
                # FIMXE: sistemare il compo di versione
                version_string = self._dictionary.doc(0).get(
                    BabelNetIndexField.VERSION.name
                )
                if version_string is not None:
                    return BabelVersion[version_string]
                else:
                    return BabelVersion.V5_1
            else:
                return BabelVersion.PRE_2_0
        except lucene.JavaError:
            return BabelVersion.UNKNOWN

    def mapping_from_babelsynsetids(
        self,
        *ids: BabelSynsetID,
    ) -> Dict[BabelSynsetID, List[_InternalBabelSynsetID]]:
        id2license_ids = {}
        check_ids = set()
        for id_ in ids:
            if id_ in check_ids:
                continue
            mapping_doc = None
            if isinstance(id_, _InternalBabelSynsetID):
                mapping_doc = self._mapping_document(
                    id_.id, BabelNetIndexField.LICENSE_ID
                )
            elif isinstance(id_, BabelSynsetID):
                mapping_doc = self._mapping_document(id_.id, BabelNetIndexField.ID)
            if mapping_doc is None:
                continue
            id_babelnet = mapping_doc.get(BabelNetIndexField.ID.name)
            other_license_ids = mapping_doc.getValues(
                BabelNetIndexField.LICENSE_ID.name
            )
            try:
                id_bn = BabelSynsetID(id_babelnet)
                check_ids.add(id_bn)
                license_ids = []
                for license_id in other_license_ids:
                    app_license_id = _InternalBabelSynsetID(license_id)
                    license_ids.append(app_license_id)
                    check_ids.add(app_license_id)
                id2license_ids[id_bn] = license_ids
            except InvalidSynsetIDError:
                traceback.print_exc()
        return id2license_ids

    def _mapping_document(
        self, id_: str, field: BabelNetIndexField
    ) -> Optional[Document]:
        """Get a mapping Document from a License or BabelNet ID.

        @param id_: The input id.
        @type id_: str
        @param field: The kind of id.
        @type field: BabelNetIndexField

        @return: the document.
        @rtype: Optional[Document]
        """
        q = TermQuery(Term(field.name, id_))
        try:
            # interroga l'indice e restituisce il synset se esiste
            docs = self._babelnet.search(q, 1)

            # nessun synset trovato
            if docs.totalHits == 0:
                return None

            # restituisce il synset
            doc = self._babelnet.doc(docs.scoreDocs[0].doc)
            return doc
        except lucene.JavaError as e:
            print(e)
            return None

    def _info_document(self, id_: str) -> Document:
        """
        Get an info Document from a BabelNet ID.

        This function interrogates the index and returns the Document where the id_ is found.

        @param id_: the id
        @rtype: str

        @return: the Document
        @rtype: Document

        @raises JavaError: Error raised from Java
        """
        if self._info is None:
            return Document()
        q = TermQuery(Term(BabelNetIndexField.ID.name, id_))

        # interroga l'indice e restituisce il synset se esiste
        docs = self._info.search(q, 1)

        # nessun synset trovato
        if docs.totalHits == 0:
            return Document()

        # restituisce il synset
        doc = self._info.doc(docs.scoreDocs[0].doc)
        return doc

    def license_ids_from_dictionary_documents(self, q) -> List[_InternalBabelSynsetID]:
        """Get the set of license ids of Documents in all indices from a given query.

        @param q: a Lucene Query used to obtain the Lucene documents.

        @return: the list of the license ids's (as _InternalBabelSynsetID) from the dictionary document
        @rtype: List[_InternalBabelSynsetID]
        """
        document_ids = OrderedSet()
        for bl in BabelLicense:
            if bl not in self._license_to_dictionaries:
                continue

            # search q
            dict_license = self._license_to_dictionaries[bl]

            try:
                docs_license = dict_license.search(q, _MAXIMUM_NUMBER_OF_SYNSETS)

                # results
                for score_doc in docs_license.scoreDocs:
                    doc = dict_license.doc(score_doc.doc)

                    license_id = None
                    try:
                        license_id = doc.get(BabelNetIndexField.ID.name)
                    except lucene.JavaError as e:
                        print(e)

                    document_ids.add(_InternalBabelSynsetID(license_id))
            except lucene.JavaError as e:
                print(e)

        # all id licenses for the query q
        return list(document_ids)

    def _dictionary_docs(
        self, id_: BabelSynsetID, *ids: _InternalBabelSynsetID
    ) -> List[Document]:
        """Get the list of Documents by mapping ID.

        @param id_: A BabelNetID.
        @param ids: License IDs of the various synset pieces.

        @return: a list of documents
        @rtype: List[Document]
        """
        docs = []
        # all id babelnet in other license
        for license_id in ids:
            bl = license_id.babel_license
            if bl not in self._license_to_dictionaries:
                continue
            dict_license = self._license_to_dictionaries[bl]
            doc = self._doc_from_index_by_id(license_id, dict_license)
            if doc is None:
                continue

            # remove license id
            doc.removeField(BabelNetIndexField.ID.name)
            # add babelnet id
            doc.add(StringField(BabelNetIndexField.ID.name, id_.id, Field.Store.YES))
            docs.append(doc)
        return docs

    def _doc_from_index_by_id(
        self, id_: BabelSynsetID, dict_index: IndexSearcher
    ) -> Optional[Document]:
        """Get a index-document lucene identifier (Babel synset ID).

        @param id_: The Babel synset ID for a specific concept.
        @param dict_index: IndexSearcher.

        @return: The Document
        @rtype: Optional[Document]
        """
        q = TermQuery(Term(BabelNetIndexField.ID.name, id_.id))

        # interroga l'indice e restituisce il synset se esiste
        try:
            docs = dict_index.search(q, 1)

            # nessun synset trovato
            if docs.totalHits == 0:
                return None

            # restituisce il synset
            return dict_index.doc(docs.scoreDocs[0].doc)

        except lucene.JavaError as e:
            print(e)
            return None

    def license_ids_for_word(
        self, word: str, poses: Set["POS"], langs: Set[Language], normalizer: bool
    ) -> List[_InternalBabelSynsetID]:
        word = word.replace(" ", "_")
        if langs is not None:
            langs.add(Language.MUL)
        q_lemma_builder = BooleanQuery.Builder()

        if langs:
            for l in langs:
                lang = l.name
                lang_word = lang + _SEPARATOR + word.lower()
                q_lemma_builder.add(
                    BooleanClause(
                        TermQuery(
                            Term(BabelNetIndexField.LANGUAGE_LEMMA.name, lang_word)
                        ),
                        BooleanClause.Occur.SHOULD,
                    )
                )
                if normalizer:
                    q_lemma_builder.add(
                        BooleanClause(
                            TermQuery(
                                Term(
                                    BabelNetIndexField.LANGUAGE_LEMMA_NORMALIZER.name,
                                    lang_word,
                                )
                            ),
                            BooleanClause.Occur.SHOULD,
                        )
                    )
        else:
            lang = "*"
            lang_word = lang + _SEPARATOR + word
            occor = (
                BooleanClause.Occur.SHOULD if normalizer else BooleanClause.Occur.MUST
            )

            q_lemma_builder.add(
                BooleanClause(
                    WildcardQuery(
                        Term(BabelNetIndexField.LANGUAGE_LEMMA.name, lang_word)
                    ),
                    occor,
                )
            )
            if normalizer:
                q_lemma_builder.add(
                    BooleanClause(
                        WildcardQuery(
                            Term(
                                BabelNetIndexField.LANGUAGE_LEMMA_NORMALIZER.name,
                                lang_word,
                            )
                        ),
                        occor,
                    )
                )

        q_final_builder = BooleanQuery.Builder()
        q_final_builder.add(q_lemma_builder.build(), BooleanClause.Occur.MUST)

        q_pos_builder = BooleanQuery.Builder()
        if poses is not None:

            for pos in poses:
                q_pos_builder.add(
                    BooleanClause(
                        TermQuery(Term(BabelNetIndexField.POS.name, pos.tag)),
                        BooleanClause.Occur.SHOULD,
                    )
                )
            q_final_builder.add(q_pos_builder.build(), BooleanClause.Occur.MUST)

        return self.license_ids_from_dictionary_documents(q_final_builder.build())

    def _multi_dictionary_docs(
        self, id2id_licenses: Dict[BabelSynsetID, List[_InternalBabelSynsetID]]
    ) -> Dict[BabelSynsetID, Set[Document]]:
        """ """
        id2docs = defaultdict(OrderedSet)
        for key, val in id2id_licenses.items():
            id2docs[key].update(self._dictionary_docs(key, *val))
        return id2docs

    def synsets(
        self,
        id2license_ids: Dict[BabelSynsetID, List[_InternalBabelSynsetID]],
        target_languages: Set[Language],
    ) -> List[BabelSynset]:
        id2docs = self._multi_dictionary_docs(id2license_ids)
        synsets = []
        for ids in id2docs:
            doc = self._merge_dictionary_documents(id2docs[ids])

            babel_synset = self._synset_from_all_documents(
                doc, id2license_ids[ids], target_languages
            )
            synsets.append(babel_synset)
        return synsets

    def synsets_from_wordnet_offset(
        self,
        offset: WordNetSynsetID,
    ) -> List[_InternalBabelSynsetID]:
        """Get the set of ids license of Documents in all indices from a given
        WordNetSynsetID.

        @param offset: A WordNet offset.
        @type offset: WordNetSynsetID

        @return: A list of InternalBabelSynsetIDs associated to the given constraints.
        @rtype: List[_InternalBabelSynsetID]
        """

        # remove wn
        q = TermQuery(
            Term(BabelNetIndexField.WORDNET_OFFSET.name, offset.simple_offset)
        )
        return self.license_ids_from_dictionary_documents(q)

    def wordnet_offsets(
        self, offset: str, version: WordNetVersion
    ) -> List[WordNetSynsetID]:
        list_offset = []

        # BN5 doesn't have that directory
        if self._wordnet_info is None:
            return list_offset

        q_off = TermQuery(Term(BabelNetIndexField.OLD_WORDNET_OFFSET.name, offset))
        q_vers = TermQuery(Term(BabelNetIndexField.VERSION.name, version.value))

        search_offset = (
            BooleanQuery.Builder()
            .add(q_off, BooleanClause.Occur.MUST)
            .add(q_vers, BooleanClause.Occur.MUST)
            .build()
        )

        # interroga l'indice e restituisce il synset se esiste
        docs_found = self._wordnet_info.search(
            search_offset, _MAXIMUM_NUMBER_OF_SYNSETS
        )

        if docs_found.totalHits == 0:
            return list_offset

        # results
        for scoreDoc in docs_found.scoreDocs:
            doc = self._wordnet_info.doc(scoreDoc.doc)
            old_offset = doc.get(BabelNetIndexField.WORDNET_OFFSET.name)
            list_offset.append(WordNetSynsetID("wn:" + old_offset))
        return list_offset

    def licence_ids_from_resource_id(
        self,
        resource_id: ResourceID,
    ) -> List[_InternalBabelSynsetID]:
        if isinstance(resource_id, ResourceWithLemmaID):
            q_builder = BooleanQuery.Builder()
            title = resource_id.title.replace(" ", "_")
            q_builder.add(
                BooleanClause(
                    TermQuery(Term(BabelNetIndexField.LEMMA.name, title)),
                    BooleanClause.Occur.MUST,
                )
            )

            if resource_id.pos is not None:
                q_builder.add(
                    BooleanClause(
                        TermQuery(
                            Term(BabelNetIndexField.POS.name, resource_id.pos.tag)
                        ),
                        BooleanClause.Occur.MUST,
                    )
                )
            q = q_builder.build()
        elif isinstance(resource_id, WordNetSynsetID):
            q = TermQuery(
                Term(BabelNetIndexField.WORDNET_OFFSET.name, resource_id.simple_offset)
            )
        else:
            q = TermQuery(Term(BabelNetIndexField.SENSE_ID.name, resource_id.id))
        return self.license_ids_from_dictionary_documents(q)

    def synsets_from_wikipedia_title(
        self,
        language: Language,
        title: str,
        id2license_ids: Dict[BabelSynsetID, List[_InternalBabelSynsetID]],
    ) -> List[BabelSynset]:
        """Given a Wikipedia title, get its BabelSynsets.

        @param language: The language of the input Wikipedia title.
        @type language: Language
        @param title: The title of the Wikipedia page.
        @type title: str
        @param id2license_ids: a dict that maps BabelSynsetID to a list of _InternalBabelSynsetID
        @type id2license_ids: Dict[BabelSynsetID, List[_InternalBabelSynsetID]]

        @return: The list of BabelSynsets associated to the given Wikipedia title.
        @rtype: List[BabelSynset]
        """
        id2docs = self._multi_dictionary_docs(id2license_ids)
        synsets = []
        for ids in id2docs:
            doc = self._merge_dictionary_documents(id2docs[ids])

            lemmas = doc.getFields(str(BabelNetIndexField.LEMMA))
            lemmas_languages = doc.getFields(str(BabelNetIndexField.LEMMA_LANGUAGE))
            lemmas_source = doc.getFields(str(BabelNetIndexField.LEMMA_SOURCE))

            for i in range(len(lemmas)):
                if (
                    (
                        language is None
                        or Language[lemmas_languages[i].stringValue()] is language
                    )
                    and lemmas[i].stringValue().replace(" ", "_")
                    == title.replace(" ", "_")
                    and BabelSenseSource[
                        lemmas_source[i].stringValue()
                    ].is_from_wikipedia
                ):
                    synsets.append(
                        self._synset_from_all_documents(doc, id2license_ids[ids], None)
                    )
        return synsets

    def synsets_from_resource_id(
        self,
        id_: ResourceID,
        id2license_ids: Dict[BabelSynsetID, List[_InternalBabelSynsetID]],
        target_langs: Set[Language],
    ) -> List[BabelSynset]:
        id2docs = self._multi_dictionary_docs(id2license_ids)
        synsets = []

        for ids in id2docs:
            doc = self._merge_dictionary_documents(id2docs[ids])
            lemmas = doc.getFields(BabelNetIndexField.LEMMA.name)
            lemmas_languages = doc.getFields(BabelNetIndexField.LEMMA_LANGUAGE.name)
            lemmas_source = doc.getFields(BabelNetIndexField.LEMMA_SOURCE.name)
            id_sense = doc.getFields(BabelNetIndexField.SENSE_ID.name)

            language = id_.language
            for i in range(len(lemmas)):
                source = BabelSenseSource[lemmas_source[i].stringValue()]

                value = lemmas[i].stringValue().replace(" ", "_")

                # FIMXE: Wikipedia and Wikiquote now have id
                if len(id_sense) > 0 and source is not BabelSenseSource.WIKI:
                    value = (
                        id_sense[i].stringValue().replace(" ", "_")
                        if id_sense[i].stringValue()
                        else lemmas[i].stringValue().replace(" ", "_")
                    )

                if (
                    target_langs is not None
                    and Language[lemmas_languages[i].stringValue()] not in target_langs
                ):
                    continue

                if (
                    language is None
                    or Language[lemmas_languages[i].stringValue()] is language
                ) and value == id_.id.replace(" ", "_"):
                    bss = id_.source

                    if (
                        bss is source
                        or (bss.is_from_wikipedia and source.is_from_wikipedia)
                        or (bss.is_from_wikiquote and source.is_from_wikiquote)
                    ):
                        synsets.append(
                            self._synset_from_all_documents(
                                doc, id2license_ids[ids], None
                            )
                        )
        return synsets

    def successors(self, id_: BabelSynsetID) -> Optional[List[str]]:
        """Given a Babel id, collects the successors of the concept denoted by the id.

        @param id_:  A concept identifier (babel synset ID)
        @type id_: BabelSynsetID

        @return: Return a stringified representation of the edges departing from the Babel synset denoted by the input id.
        @rtype: Optional[List[str]]
        """
        q = TermQuery(Term(BabelNetIndexField.ID.name, id_.id))
        successors = ""
        try:
            docs = self._graph.search(q, 1)

            if len(docs.scoreDocs) > 0:
                doc = self._graph.doc(docs.scoreDocs[0].doc)
                successors += doc.get(BabelNetIndexField.RELATION.name)
        except lucene.JavaError as e:
            print(e)
        if not successors:
            return None
        return _utils.java_split(successors, "\t")

    def _documents_from_glosses_and_examples(
        self,
        *ids: _InternalBabelSynsetID,
    ) -> List[Document]:
        """Search in index Glosses and Examples for the given InternalBabelSynsetIDs.

        @param ids: the _InternalBabelSynsetID to use for the search
        @type ids: _InternalBabelSynsetID

        @return: THe list of Document(s) associated to the given ids
        @rtype: List[Document]

        @raises JavaError:
        """
        docs_through_other_license = []
        for license_id in ids:
            q = TermQuery(Term(BabelNetIndexField.ID.name, license_id.id))
            bl = license_id.babel_license

            if bl not in self._license_to_glosses_and_examples:
                continue
            gloss_license = self._license_to_glosses_and_examples[bl]

            docs = gloss_license.search(q, _MAXIMUM_NUMBER_OF_SYNSETS)
            for score_doc in docs.scoreDocs:
                docs_through_other_license.append(gloss_license.doc(score_doc.doc))
        return docs_through_other_license

    def glosses(
        self,
        id_: BabelSynsetID,
        filter_langs: Set[Language],
        *ids: _InternalBabelSynsetID,
    ) -> List[BabelGloss]:
        """Get the glosses of a specific BabelSynset, given a BabelSynsetID and a collection of InternalBabelSynsetID.

        @param id_: the BabelSynsetID form where to get the glosses
        @type id_: BabelSynsetID
        @param filter_langs: the languages to filter
        @type filter_langs: Set[Language]
        @param ids: the _InternalBabelSynsetID(s)
        @type ids: _InternalBabelSynsetID

        @return: the list of BabelGloss(es) retrieved from the BabelSynset
        @rtype: List[BabelGloss]
        """
        gloss_list = []

        try:
            docs_through_other_license: list = (
                self._documents_from_glosses_and_examples(*ids)
            )
            if not docs_through_other_license:
                return gloss_list
            doc = self._merge_definitions_document(
                id_,
                docs_through_other_license.pop(0),
                docs_through_other_license,
                BabelNetIndexField.GLOSS,
            )
            stored_glosses = doc.getValues(BabelNetIndexField.GLOSS.name)
            for i in range(len(stored_glosses)):
                stored_gloss = stored_glosses[i]
                split = _utils.java_split(stored_gloss, "\t")

                language = Language[split[0]]
                if filter_langs is not None and language not in filter_langs:
                    continue
                sense_source = BabelSenseSource[split[1]]
                sense = split[2]
                gloss = split[3]

                if len(split) > 4:
                    tokens = set()
                    for k in range(4, len(split)):
                        token = split[k]
                        if not token:
                            continue
                        try:
                            pox_start = int(token.split("#")[1])
                            pox_end = int(token.split("#")[2])
                        except Exception:
                            continue
                        try:
                            id_t = BabelSynsetID(token.split("#")[0])
                            word = gloss[pox_start : pox_end + 1]
                            tokens.add(BabelTokenId(pox_start, pox_end, id_t, word))
                        except InvalidSynsetIDError:
                            traceback.print_exc()
                        except IndexError:
                            _log.error("DEBUG: " + gloss)
                    b_gloss = BabelGloss(sense_source, sense, language, gloss, tokens)
                else:
                    b_gloss = BabelGloss(sense_source, sense, language, gloss)
                gloss_list.append(b_gloss)

        except lucene.JavaError as e:
            print(e)
        return gloss_list

    def examples(
        self,
        id_: BabelSynsetID,
        filter_langs: Set[Language],
        *ids: _InternalBabelSynsetID,
    ) -> List[BabelExample]:
        """
        Get the examples of a specific Babel synset, given a BabelSynsetID
        and a collection of InternalBabelSynsetID.

        @param id_: the BabelSynsetID form where to get the examples
        @type id_: BabelSynsetID
        @param filter_langs: the languages to filter
        @type filter_langs: Set[Language]
        @param ids: the _InternalBabelSynsetID(s)
        @type ids: _InternalBabelSynsetID

        @return: the list of BabelExample(es) retrieved from the BabelSynset
        @rtype: List[BabelExample]

        @raise RuntimeError:
        """
        example_list = []

        try:
            docs_through_other_license: list = (
                self._documents_from_glosses_and_examples(*ids)
            )
            if not docs_through_other_license:
                return example_list
            doc = self._merge_definitions_document(
                id_,
                docs_through_other_license.pop(0),
                docs_through_other_license,
                BabelNetIndexField.EXAMPLE,
            )
            stored_example_cryp = doc.getValues(BabelNetIndexField.EXAMPLE.name)
            for i in range(len(stored_example_cryp)):
                stored_example = stored_example_cryp[i]
                split = _utils.java_split(stored_example, "\t")

                if len(split) < 4:
                    raise RuntimeError("Invalid example: " + stored_example)

                language = Language[split[0]]
                if filter_langs is not None and language not in filter_langs:
                    continue
                sense_source = BabelSenseSource[split[1]]
                sense = split[2]
                example = split[3]

                if len(split) > 4:
                    tokens = set()
                    for k in range(4, len(split)):
                        token = split[k]
                        # if not token:
                        #    continue
                        pox_start = int(token.split("#")[1])
                        pox_end = int(token.split("#")[2])
                        lemma = token.split("#")[0]
                        tokens.add(BabelTokenWord(pox_start, pox_end, lemma))
                    b_example = BabelExample(
                        sense_source, sense, language, example, tokens
                    )
                else:
                    b_example = BabelExample(sense_source, sense, language, example)
                example_list.append(b_example)

        except lucene.JavaError as e:
            print(e)
        return example_list

    def _merge_definitions_document(
        self,
        id_: BabelSynsetID,
        doc: Document,
        docs: List[Document],
        type_definition: BabelNetIndexField,
    ) -> Document:
        """Merge glosses or examples.

        @raises JavaError:
        """
        doc.removeField(BabelNetIndexField.ID.name)
        doc.add(StringField(BabelNetIndexField.ID.name, id_.id, Field.Store.YES))
        for doc_license in docs:
            stored = doc_license.getValues(type_definition.name)
            for k in range(len(stored)):
                doc.add(StoredField(type_definition.name, stored[k].encode("utf-8")))
        return doc

    def images(self, id_: BabelSynsetID) -> List[BabelImage]:
        """Get the images of a specific Babel synset, given a BabelSynsetID.

        @param id_: the BabelSynsetID form where to get the images
        @type id_: BabelSynsetID

        @return: the list of BabelImage(es) retrieved from the BabelSynset
        @rtype: List[BabelImage]
        """
        image_list = []
        if self._image is None:
            return image_list
        try:
            q_builder = BooleanQuery.Builder()
            q_builder.add(
                BooleanClause(
                    TermQuery(Term(BabelNetIndexImageField.ID.name, id_.id)),
                    BooleanClause.Occur.MUST,
                )
            )
            docs = self._image.search(q_builder.build(), _MAXIMUM_NUMBER_OF_IMAGES)
            if not docs.scoreDocs:
                return image_list
            # results
            for score_doc in docs.scoreDocs:
                doc = self._image.doc(score_doc.doc)
                title = doc.get(BabelNetIndexImageField.TITLE.name)
                language = doc.get(BabelNetIndexImageField.LANGUAGE.name)
                url = doc.get(BabelNetIndexImageField.URL.name)
                thumb_url = doc.get(BabelNetIndexImageField.URL_THUMBNAIL.name)
                source = doc.get(BabelNetIndexImageField.SOURCE.name)
                license_ = doc.get(BabelNetIndexImageField.LICENSE.name)
                bad = doc.get(BabelNetIndexImageField.BADIMAGE.name).lower() == "true"
                babel_image = BabelImage(
                    title, language, url, thumb_url, source, license_, bad
                )
                if not _USE_IMAGE_FILTER or not babel_image.id_bad:
                    try:
                        pox = image_list.index(babel_image)
                        image_list[pox].add_language(Language[language])
                    except ValueError:
                        image_list.append(babel_image)
        except lucene.JavaError as e:
            print(e)

        image_list.sort(key=BabelImageComparator())
        return image_list

    def _merge_dictionary_documents(self, docs_coll: Set[Document]) -> Document:
        """Merge all Lucene dictionary Documents, that have the same id. The ids
        are replaced initially with the final BabelNet id."""
        docs = list(docs_coll)
        doc = docs.pop(0)
        id_ = doc.get(BabelNetIndexField.ID.name)
        type_ = doc.get(BabelNetIndexField.TYPE.name)
        if type_ is None:
            doc.removeFields(BabelNetIndexField.TYPE.name)

        # controllo se esistono e sono validi i wordnet offset. Altrimenti
        # rimuovo il campo WordNet_offset da primo documento

        wn_offsets = doc.getValues(BabelNetIndexField.WORDNET_OFFSET.name)
        if len(wn_offsets) == 1 and wn_offsets[0].lower() == "-":
            doc.removeFields(BabelNetIndexField.WORDNET_OFFSET.name)

        for doc_license in docs:
            if type_ is None:
                type_ = doc_license.get(BabelNetIndexField.TYPE.name)
                doc.add(
                    StringField(BabelNetIndexField.TYPE.name, type_, Field.Store.YES)
                )

            license_id = doc_license.get(BabelNetIndexField.ID.name)
            # we combine only the docs with same id
            if license_id != id_:
                continue
            lemma_source = doc_license.get(BabelNetIndexField.LEMMA_SOURCE.name)
            if lemma_source is not None:  # esiste almeno un lemma
                wn_offsets = doc_license.getValues(
                    BabelNetIndexField.WORDNET_OFFSET.name
                )
                if len(wn_offsets) == 1 and wn_offsets[0].lower() == "-":
                    doc_license.removeFields(BabelNetIndexField.WORDNET_OFFSET.name)
                else:
                    for wordnet_offset in wn_offsets:
                        doc.add(
                            StringField(
                                BabelNetIndexField.WORDNET_OFFSET.name,
                                wordnet_offset,
                                Field.Store.YES,
                            )
                        )

                # numero sense in doc
                # num_senses = 0
                # lemmas_in_doc = doc.getValues(BabelNetIndexField.LEMMA.name)
                # if lemmas_in_doc is not None:
                #    num_senses = len(lemmas_in_doc)
                # newPositionTraslated.put(bl.getShortName(),numSenses);

                # merge sense
                language_lemmas = doc_license.getValues(
                    BabelNetIndexField.LANGUAGE_LEMMA.name
                )
                lemmas = doc_license.getValues(BabelNetIndexField.LEMMA.name)
                lemma_sources = doc_license.getValues(
                    BabelNetIndexField.LEMMA_SOURCE.name
                )
                lemma_languages = doc_license.getValues(
                    BabelNetIndexField.LEMMA_LANGUAGE.name
                )
                lemma_weights = doc_license.getValues(
                    BabelNetIndexField.LEMMA_WEIGHT.name
                )
                lemma_sensekeys = doc_license.getValues(
                    BabelNetIndexField.LEMMA_SENSEKEY.name
                )
                lemma_sense_frequence = doc_license.getValues(
                    BabelNetIndexField.LEMMA_FREQUENCE.name
                )
                id_senses = doc_license.getValues(BabelNetIndexField.ID_SENSE.name)
                idkeys = doc_license.getValues(BabelNetIndexField.SENSE_ID.name)

                for k in range(len(lemmas)):
                    doc.add(
                        StringField(
                            BabelNetIndexField.LANGUAGE_LEMMA.name,
                            language_lemmas[k],
                            Field.Store.YES,
                        )
                    )
                    # LEMMA => indexed
                    doc.add(
                        StringField(
                            BabelNetIndexField.LEMMA.name,
                            lemmas[k].replace(" ", "_"),
                            Field.Store.YES,
                        )
                    )
                    # LEMMA LANG => indexed
                    doc.add(
                        StringField(
                            BabelNetIndexField.LEMMA_LANGUAGE.name,
                            lemma_languages[k].replace(" ", "_"),
                            Field.Store.YES,
                        )
                    )
                    # LEMMA WEIGHT => not indexed
                    doc.add(
                        StringField(
                            BabelNetIndexField.LEMMA_WEIGHT.name,
                            lemma_weights[k],
                            Field.Store.YES,
                        )
                    )
                    # SENSE SOURCE => not indexed
                    doc.add(
                        StoredField(
                            BabelNetIndexField.LEMMA_SOURCE.name,
                            lemma_sources[k].encode("utf-8"),
                        )
                    )
                    # WN SENSEKEY => indexed
                    doc.add(
                        StringField(
                            BabelNetIndexField.LEMMA_SENSEKEY.name,
                            lemma_sensekeys[k],
                            Field.Store.YES,
                        )
                    )
                    # retro-compatibilita'
                    if len(idkeys) > 0:
                        doc.add(
                            StringField(
                                BabelNetIndexField.SENSE_ID.name,
                                idkeys[k],
                                Field.Store.YES,
                            )
                        )
                    if len(lemma_sense_frequence) > 0:
                        doc.add(
                            StringField(
                                BabelNetIndexField.LEMMA_FREQUENCE.name,
                                lemma_sense_frequence[k],
                                Field.Store.YES,
                            )
                        )
                    doc.add(
                        StoredField(
                            BabelNetIndexField.ID_SENSE.name,
                            id_senses[k].encode("utf-8"),
                        )
                    )

                translations = doc_license.getValues(
                    BabelNetIndexField.TRANSLATION_MAPPING.name
                )
                for tr in translations:
                    doc.add(
                        StoredField(
                            BabelNetIndexField.TRANSLATION_MAPPING.name,
                            tr.encode("utf-8"),
                        )
                    )

            # merge image
            image_names_license = doc_license.getValues(BabelNetIndexField.IMAGE.name)
            for image in image_names_license:
                # IMAGE => not indexed
                doc.add(
                    StoredField(BabelNetIndexField.IMAGE.name, image.encode("utf-8"))
                )
            # merge categories
            category_names_license = doc_license.getValues(
                BabelNetIndexField.CATEGORY.name
            )

            for category in category_names_license:
                # CATEGORY => not indexed
                doc.add(
                    StoredField(
                        BabelNetIndexField.CATEGORY.name, category.encode("utf-8")
                    )
                )
            # audio => not indexed
            audios = doc_license.getValues(BabelNetIndexField.PRONU_AUDIO.name)
            for audio in audios:
                doc.add(
                    StoredField(
                        BabelNetIndexField.PRONU_AUDIO.name, audio.encode("utf-8")
                    )
                )
            # pronunctions => not indexed
            ipas = doc_license.getValues(BabelNetIndexField.PRONU_TRANSC.name)
            for ipa in ipas:
                doc.add(StoredField(BabelNetIndexField.PRONU_TRANSC.name, ipa))
            # yago => not indexed
            yagos = doc_license.getValues(BabelNetIndexField.YAGO_URL.name)
            for yago in yagos:
                doc.add(
                    StoredField(BabelNetIndexField.YAGO_URL.name, yago.encode("utf-8"))
                )

        # se il type non e' stato trovato imposta il tipo sconosciuto
        if type is None:
            doc.add(
                StringField(
                    BabelNetIndexField.TYPE.name,
                    SynsetType.UNKNOWN.name.upper(),
                    Field.Store.YES,
                )
            )
        return doc

    def synset_iterator(
        self,
    ) -> BabelSynsetIterator:
        return BabelSynsetIteratorImpl(self._graph)

    def offset_iterator(
        self,
    ) -> BabelOffsetIterator:
        return BabelOffsetIteratorImpl(self._graph)

    def lexicon_iterator(
        self,
    ) -> BabelLexiconIterator:
        return BabelLexiconIteratorImpl(self._lexicon)

    def wordnet_iterator(
        self,
    ) -> WordNetSynsetIterator:
        return WordNetSynsetIteratorImpl(self._graph)

    def search_old_offset_wordnet(self, offset: str) -> Dict[WordNetVersion, List[str]]:
        """ Search the old offset of wordnet"""
        mapping = {}
        try:
            if self._wordnet_info is None:
                return mapping
            q = TermQuery(Term(BabelNetIndexField.WORDNET_OFFSET.name, offset))

            # interroga l'indice e restituisce il synset se esiste
            docs_found = self._wordnet_info.search(q, _MAXIMUM_NUMBER_OF_SYNSETS)

            # nessun doc trovato
            if docs_found.totalHits == 0:
                return mapping

            # results
            app = defaultdict(OrderedSet)
            for score_doc in docs_found.scoreDocs:
                doc = self._wordnet_info.doc(score_doc.doc)
                version_wn = WordNetVersion.from_string(
                    doc.get(BabelNetIndexField.VERSION.name)
                )
                old_offset = doc.get(BabelNetIndexField.OLD_WORDNET_OFFSET.name)
                app[version_wn].add(old_offset)

            for ver in app:
                mapping[ver] = list(app[ver])
            app_list = [offset]
            mapping[WordNetVersion.WN_30] = app_list

        except lucene.JavaError as e:
            print(e)
        return mapping

    # lol
    def _synset_from_all_documents(
        self,
        doc: Document,
        license_ids: List[_InternalBabelSynsetID],
        target_langs: Set[Language],
    ) -> Optional[BabelSynset]:
        """Get a full-fledged BabelSynset from a Document.

        @param doc: A Lucene Document record for a certain Babel synset.
        @type doc: Document
        @param license_ids: A list of licence IDs.
        @type license_ids: List[_InternalBabelSynsetID]
        @param target_langs: The languages to use for search.
        @type target_langs: Set[Language]

        @return: An instance of a BabelSynset from a Document.
        @rtype: Optional[BabelSynset]
        """
        try:
            id_ = BabelSynsetID(doc.get(BabelNetIndexField.ID.name))
        except InvalidSynsetIDError:
            traceback.print_exc()
            return None

        info_doc = None
        try:
            info_doc = self._info_document(id_.id)
        except lucene.JavaError as e:
            print(e)
        # recupero il dominio del synset
        domains = OrderedDict()

        domains_synset = info_doc.getFields(BabelNetIndexField.DOMAIN.name)
        weights = info_doc.getFields(BabelNetIndexField.DOMAIN_WEIGHT.name)
        for i in range(len(domains_synset)):
            domains[
                BabelDomain.from_position(int(domains_synset[i].stringValue()))
            ] = float(weights[i].stringValue())

        # key concept
        key_senses = set()
        lemma_key = info_doc.getFields(BabelNetIndexField.KEY_CONCEPT.name)

        for i in range(len(lemma_key)):
            key_senses.add(lemma_key[i].stringValue().lower())

        synset_type = SynsetType[doc.get(BabelNetIndexField.TYPE.name)]

        pos = id_.pos
        wordnet_synset_ids: List[WordNetSynsetID] = []
        wordnet_synset_id_cache: Set[str] = set()

        # NOTE: con WN2020 non sappiamo più l'offset a chi appartiene e non abbiamo riferimenti alla risorsa
        # NOTE: Swapped with the new handling below
        #
        # wn_offsets = doc.getValues(BabelNetIndexField.WORDNET_OFFSET.name)
        # wordnet_synset_id_list = []
        # try:
        #     for wn in wn_offsets:
        #         if wn:
        #             wordnet_synset_id = WordNetSynsetID('wn:' + wn)
        #             wordnet_synset_id.version_mapping = search_old_offset_wordnet(wn)
        #             wordnet_synset_id_list.append(wordnet_synset_id)
        #
        # except InvalidSynsetIDError:
        #     traceback.print_exc()

        translation_mappings = list(
            doc.getValues(BabelNetIndexField.TRANSLATION_MAPPING.name)
        )

        images = None
        category_names = doc.getValues(BabelNetIndexField.CATEGORY.name)
        categories = []

        for category_name in category_names:
            cate = BabelCategory.from_string(category_name)
            if target_langs is None or cate.language in target_langs:
                categories.append(cate)

        # AUDIO
        audio_titles = doc.getValues(BabelNetIndexField.PRONU_AUDIO.name)
        audios = defaultdict(set)
        # WIKT:PL:haga:Pl-Haga.ogg
        for audio_name in audio_titles:
            tmp: Tuple[str, BabelAudio] = phonetics.audio_from_string(audio_name)
            if target_langs is None or tmp[1].language in target_langs:
                audios[tmp[0]].add(tmp[1])

        # TRANSCRIPT
        transcript_titles = doc.getValues(BabelNetIndexField.PRONU_TRANSC.name)

        transcripts = defaultdict(set)
        for trasc_name in transcript_titles:
            tmp = phonetics.transcription_from_string(trasc_name)
            transcripts[tmp[0]].add(tmp[1])

        lemma_to_urlyago = set()
        url_yago = doc.getValues(BabelNetIndexField.YAGO_URL.name)
        for ln_url in url_yago:
            if target_langs is None or Language.EN in target_langs:
                lemma_to_urlyago.add(ln_url)

        lemmas = doc.getValues(BabelNetIndexField.LEMMA.name)
        lemma_sources = doc.getValues(BabelNetIndexField.LEMMA_SOURCE.name)
        language_normalized_lemma = doc.getValues(BabelNetIndexField.LANGUAGE_LEMMA.name)
        lemma_languages = doc.getValues(BabelNetIndexField.LEMMA_LANGUAGE.name)
        lemma_sensekeys = doc.getValues(BabelNetIndexField.LEMMA_SENSEKEY.name)
        sense_id = doc.getValues(BabelNetIndexField.SENSE_ID.name)
        id_senses = doc.getValues(BabelNetIndexField.ID_SENSE.name)

        synset_senses = []

        # Mappa ciascun sense al suo id_sense
        translations_index = {}
        # wordnet_synset_ids here is empty. It will be populated later. References "magic"
        babel_synset = _OfflineBabelSynset(
            id_,
            wordnet_synset_ids,
            synset_senses,
            translation_mappings,
            images,
            categories,
            synset_type,
            domains,
            license_ids,
            target_langs if target_langs else None,
            len(key_senses) > 0,
            translations_index
        )
        old_to_new_position = {}
        newpox = 0
        simple_multi_lemma = {}
        language_to_normalized = set()
        for k in range(len(lemmas)):
            lemma = lemmas[k]
            if len(lemma) == 0:
                continue
            lemma_language = Language[lemma_languages[k]]

            if (
                target_langs is not None
                and lemma_language not in target_langs
                and lemma_language is not Language.MUL
            ):
                continue

            old_to_new_position[str(k)] = str(newpox)
            newpox += 1

            lemma_source = BabelSenseSource[lemma_sources[k]]
            lemma_senseskey = lemma_sensekeys[k]

            if len(sense_id) > 0:
                if lemma_sensekeys[k] == sense_id[k]:
                    lemma_senseskey = sense_id[k]
                elif lemma_sensekeys[k]:
                    lemma_senseskey = (
                        sense_id[k] + "#" + lemma_sensekeys[k]
                        if sense_id[k]
                        else lemma_sensekeys[k]
                    )
                else:
                    lemma_senseskey = sense_id[k] if sense_id[k] else lemma_sensekeys[k]

            sense_offset_triples = _utils.java_split(lemma_senseskey, "\t")

            normalized_lemma = _utils.normalized_lemma_to_string(lemma)
            audio = audios.get(str(lemma_language) + ":" + normalized_lemma.lower(), None)

            transcription = transcripts.get(
                str(lemma_language) + ":" + normalized_lemma.lower(), None
            )

            for j in range(0, len(sense_offset_triples), 3):
                lemma_sensekey = sense_offset_triples[j]
                lemma_sense_numbers = lemma_sensekey.split("##")
                sensenumber = 0
                # number sense di wordnet
                if len(lemma_sense_numbers) == 2 and lemma_sense_numbers[1].strip():
                    sensenumber = int(lemma_sense_numbers[1])
                    lemma_sensekey = lemma_sense_numbers[0]

                wordnet_offset = (
                    None
                    if len(sense_offset_triples) == 1
                    else sense_offset_triples[j + 1]
                )

                yagourl = None
                if (
                    lemma_to_urlyago
                    and lemma_source is BabelSenseSource.WIKI
                    and lemma_language is Language.EN
                ):
                    if lemma.lower() in lemma_to_urlyago:
                        yagourl = lemma[:1].upper() + lemma[1:]

                # note: position within the WordNet synset is in hex format
                # (always '1' for Wikipedia titles)
                position = (
                    "1"
                    if len(sense_offset_triples) == 1
                    else sense_offset_triples[j + 2]
                )
                b_key_sense = False
                if lemma_language is Language.EN:
                    b_key_sense = normalized_lemma.lower().strip() in key_senses

                #            cls = WordNetSense if wordnet_offset is not None else BabelSense

                if wordnet_offset is not None:
                    # We first need the prefix to check in the cache
                    wn_prefix = _WN30_PREFIX
                    if lemma_source == BabelSenseSource.WN2020: wn_prefix = _WN2020_PREFIX
                    elif lemma_source == BabelSenseSource.OEWN: wn_prefix = _OEWN_PREFIX

                    prefixed_offset: str = f"{wn_prefix}{wordnet_offset}"
                    if prefixed_offset not in wordnet_synset_id_cache:
                        # NOTE: costruiamo la lista degli offset globale rispetto al synset (elimina il bug dopo l'introduzione di WN2020)
                        wordnet_synset_id: WordNetSynsetID = WordNetSynsetID(
                            prefixed_offset
                        )
                        wordnet_synset_id.version_mapping = (
                            self.search_old_offset_wordnet(wordnet_offset)
                        )
                        wordnet_synset_ids.append(wordnet_synset_id)
                        wordnet_synset_id_cache.add(prefixed_offset)

                    sense = WordNetSense(
                        lemma=lemma,
                        language=lemma_language,
                        pos=pos,
                        sensekey=lemma_sensekey,
                        synset=babel_synset,
                        key_sense=b_key_sense,
                        yago_url=yagourl,
                        phonetics=BabelSensePhonetics(audio, transcription),
                        wordnet_offset=wordnet_offset,
                        wordnet_synset_position=int(str(position), 16),
                        wordnet_sense_number=sensenumber,
                        source=lemma_source,
                    )
                else:
                    sense = BabelSense(
                        lemma,
                        lemma_language,
                        pos,
                        lemma_source,
                        lemma_sensekey,
                        babel_synset,
                        key_sense=b_key_sense,
                        yago_url=yagourl,
                        phonetics=BabelSensePhonetics(audio, transcription),
                    )

                # Associamo l'id sense recuperato da lucene al senso specifico
                sense.id = int(id_senses[k])

                # Ora è possibile disattivare la conversione dei MUL dal file di configurazione di BabelNet
                if (
                    sense.language is Language.MUL
                    and _config.IS_MUL_CONVERSION_FILTER_ACTIVE
                ):
                    lemma_normalized = _utils.flatten_to_ascii_light(lemma)
                    if not lemma_normalized:
                        lemma_normalized = lemma

                    lemma_normalized = lemma_normalized.lower()
                    if lemma_normalized not in simple_multi_lemma:
                        simple_multi_lemma[lemma_normalized] = sense
                else:
                    synset_senses.append(sense)
                    translations_index[sense.id] = sense
                    ln = language_normalized_lemma[k].split(":")[0]
                    lem = language_normalized_lemma[k].replace(ln + ":", "").strip()
                    lemma_normalized = _utils.flatten_to_ascii_light(lem)
                    if lemma_normalized:
                        language_to_normalized.add(ln + ":" + lemma_normalized.lower())
                    else:
                        language_to_normalized.add(ln + ":" + lem.lower())

        # per ogni multilingua lemma trovato
        for multi_lemma in simple_multi_lemma:

            multi_sense = simple_multi_lemma[multi_lemma]
            # per ogni lingua vedi se gia' esiste
            lemma = multi_sense.full_lemma
            source = BabelSenseSource.BABELNET
            sensekey = multi_sense.sensekey
            wordnet_offset = (
                multi_sense.wordnet_offset
                if isinstance(multi_sense, WordNetSense)
                else None
            )
            position = (
                multi_sense.position if isinstance(multi_sense, WordNetSense) else None
            )
            sense_number = (
                multi_sense.sense_number
                if isinstance(multi_sense, WordNetSense)
                else None
            )

            babel_sense_phonetics = BabelSensePhonetics(None, None)
            for lang in self._languages:
                if lang is Language.MUL:
                    continue
                if target_langs and lang not in target_langs:
                    continue
                if str(lang) + ":" + multi_lemma in language_to_normalized:
                    continue

                if wordnet_offset is not None:
                    sense = WordNetSense(
                        lemma=lemma,
                        language=lang,
                        pos=pos,
                        sensekey=sensekey,
                        synset=babel_synset,
                        phonetics=babel_sense_phonetics,
                        wordnet_offset=wordnet_offset,
                        wordnet_synset_position=position,
                        wordnet_sense_number=sense_number,
                        source=source,
                    )
                else:
                    sense = BabelSense(
                        lemma,
                        lang,
                        pos,
                        source,
                        sensekey,
                        babel_synset,
                        phonetics=babel_sense_phonetics,
                    )

                synset_senses.append(sense)

        # FrameID
        frame_id: Optional[str] = doc.get(BabelNetIndexField.VERBATLAS_FRAME_ID.name)
        if frame_id is not None:
            babel_synset._frame_id = VerbAtlasFrameID(frame_id)

        # synsetDegree
        synset_degree = doc.get(BabelNetIndexField.SYNSET_DEGREE.name)
        if synset_degree is None:
            synset_degree = -1
        babel_synset._synset_degree = int(synset_degree)

        # QcodeID
        qCodeIDs = doc.getValues(BabelNetIndexField.QCODE_ID.name)
        if qCodeIDs is not None:
            qCodeIDList: QcodeID = list(map(lambda x: QcodeID(x), qCodeIDs))
            babel_synset._qcode_ids = qCodeIDList

        # Tag

        # tags related to senses
        sense_tags: Dict[int, List[Tag]] = {}

        # tags only related to synset
        synset_tags: Set[Tag] = set()
        label_tags: Set[LabelTag] = set()

        tag_list: Sequence[str] = doc.getValues(BabelNetIndexField.LOCALE_TAG.name)
        for full_tag in tag_list:
            id_sense, ts = full_tag.split("\t")
            id_sense: int = int(id_sense)
            tags: Sequence[str] = ts.split("@")

            for str_tag in tags:
                try:
                    value, classpath = str_tag.split("#")
                    tag: Tag = get_tag(classpath, value)
                    if id_sense != -1:
                        sense_tags.setdefault(id_sense, []).append(tag)
                    if type(tag) == LabelTag:
                        label_tags.add(tag)
                    else:
                        synset_tags.add(tag)
                except Exception as e:
                    _log.warning(
                        "You are using an index with an old version of the tags. Update the index to use full tags support!"
                    )

        babel_synset._tags.setdefault(StringTag, []).extend(synset_tags)
        babel_synset._tags.setdefault(LabelTag, []).extend(label_tags)
        return babel_synset

    def synset_from_empty_document(
        self, id_: BabelSynsetID, id_licenses: List[_InternalBabelSynsetID]
    ) -> BabelSynset:
        """Return an instance of an empty BabelSynset.

        @param id_: the id to search
        @type id_: BabelSynsetID
        @param id_licenses: the list of _InternalBabelSynsetID
        @type id_licenses: List[_InternalBabelSynsetID]

        @return: a BabelSynset
        @rtype: BabelSynset
        """
        wn_offsets = []
        translation_mappings = []
        images = []
        categories = []
        synset_senses = []
        domains_to_weights = OrderedDict()
        # noinspection PyTypeChecker
        babel_synset = _OfflineBabelSynset(
            id_,
            wn_offsets,
            synset_senses,
            translation_mappings,
            images,
            categories,
            SynsetType.UNKNOWN,
            domains_to_weights,
            id_licenses,
            None,
            False,
            {},
        )
        return babel_synset

    def _load(self):
        mapping_file = _config.MAPPING_INDEX_DIR
        dictionary_file = _config.DICT_INDEX_DIR
        gloss_file = _config.GLOSS_INDEX_DIR
        graph_file = _config.GRAPH_INDEX_DIR
        lexicon_file = _config.LEXICON_INDEX_DIR
        image_file = _config.IMAGE_INDEX_DIR
        info_file = _config.INFO_INDEX_DIR
        wn_info_file = _config.WN_INFO_INDEX_DIR

        if (
            os.path.exists(dictionary_file)
            and os.path.exists(gloss_file)
            and os.path.exists(lexicon_file)
            and os.path.exists(graph_file)
        ):

            # TODO: controllare che funziona su tutti gli OS
            # controllo di compatibilità 64 bit
            is_64_bit = platform.architecture()[0] == "64bit"
            if is_64_bit:
                mapping_dir = MMapDirectory(Paths.get(mapping_file))
                lexicon_dir = MMapDirectory(Paths.get(lexicon_file))
                dictionary_dir = MMapDirectory(Paths.get(dictionary_file))
                gloss_dir = MMapDirectory(Paths.get(gloss_file))
                graph_dir = MMapDirectory(Paths.get(graph_file))
                image_dir = MMapDirectory(Paths.get(image_file))
                info_dir = MMapDirectory(Paths.get(info_file))
                if os.path.exists(wn_info_file):
                    wn_info_dir = MMapDirectory(Paths.get(wn_info_file))
            else:
                mapping_dir = SimpleFSDirectory(Paths.get(mapping_file))
                lexicon_dir = SimpleFSDirectory(Paths.get(lexicon_file))
                dictionary_dir = SimpleFSDirectory(Paths.get(dictionary_file))
                gloss_dir = SimpleFSDirectory(Paths.get(gloss_file))
                graph_dir = SimpleFSDirectory(Paths.get(graph_file))
                image_dir = SimpleFSDirectory(Paths.get(image_file))
                info_dir = SimpleFSDirectory(Paths.get(info_file))
                if os.path.exists(wn_info_file):
                    wn_info_dir = SimpleFSDirectory(Paths.get(wn_info_file))

            # open the unrestricted indices and keep them open
            _log.info("Opening dict index: " + dictionary_file)
            reader = DirectoryReader.open(dictionary_dir)
            self._dictionary = IndexSearcher(reader)
            self._license_to_dictionaries[BabelLicense.UNRESTRICTED] = self._dictionary

            _log.info("Opening gloss index: " + gloss_file)
            reader = DirectoryReader.open(gloss_dir)
            glosses = IndexSearcher(reader)
            self._license_to_glosses_and_examples[BabelLicense.UNRESTRICTED] = glosses

            the_lexicon_index_list = []

            # add UNRESTRICTED lexicon index

            _log.info("Opening lexicon index: " + lexicon_file)

            the_lexicon_index_list.append(DirectoryReader.open(lexicon_dir))

            for bl in BabelLicense:
                if bl is BabelLicense.UNRESTRICTED:
                    continue

                if os.path.exists(lexicon_file + "_" + str(bl)):
                    _log.info("Opening lexicon index: " + lexicon_file + "_" + str(bl))
                    if is_64_bit:
                        dir_license = MMapDirectory(
                            Paths.get(lexicon_file + "_" + str(bl))
                        )
                    else:
                        dir_license = SimpleFSDirectory(
                            Paths.get(lexicon_file + "_" + str(bl))
                        )
                    reader = DirectoryReader.open(dir_license)
                    index_license = IndexSearcher(reader)
                    the_lexicon_index_list.append(index_license.getIndexReader())

                if os.path.exists(dictionary_file + "_" + str(bl)):
                    _log.info("Opening dict index: " + dictionary_file + "_" + str(bl))
                    if is_64_bit:
                        dictionary_dir_license = MMapDirectory(
                            Paths.get(dictionary_file + "_" + str(bl))
                        )
                    else:
                        dictionary_dir_license = SimpleFSDirectory(
                            Paths.get(dictionary_file + "_" + str(bl))
                        )
                    reader = DirectoryReader.open(dictionary_dir_license)
                    dictionary_license = IndexSearcher(reader)
                    self._license_to_dictionaries[bl] = dictionary_license

                if os.path.exists(gloss_file + "_" + str(bl)):
                    _log.info("Opening gloss index: " + gloss_file + "_" + str(bl))
                    if is_64_bit:
                        dir_license = MMapDirectory(
                            Paths.get(gloss_file + "_" + str(bl))
                        )
                    else:
                        dir_license = SimpleFSDirectory(
                            Paths.get(gloss_file + "_" + str(bl))
                        )
                    reader = DirectoryReader.open(dir_license)
                    index_license = IndexSearcher(reader)
                    self._license_to_glosses_and_examples[bl] = index_license
            index_version: BabelVersion = self.version()
            config_version: BabelVersion = _config.VERSION
            base_version: BabelVersion = _config.BASE_VERSION
            assert (
                config_version.ordinal >= index_version.ordinal >= base_version.ordinal
            ), f"The version of the loaded index ({index_version}) is not a supported version ({base_version} - {config_version})"
            _log.info("Using BabelNet v" + str(self.version()))

            # if the mapping index does not exist
            if os.path.exists(mapping_file):
                _log.info("Opening mapping index")
                reader = DirectoryReader.open(mapping_dir)
                self._babelnet = IndexSearcher(reader)
            else:
                self._babelnet = None

            # merge lexicon index
            multi_reader = MultiReader(the_lexicon_index_list)

            self._lexicon = IndexSearcher(multi_reader)

            _log.info("Opening graph index: " + graph_file)
            reader = DirectoryReader.open(graph_dir)
            self._graph = IndexSearcher(reader)
            _USE_IMAGE_FILTER = _config.IS_BAD_IMAGE_FILTER_ACTIVE

            if os.path.exists(image_file):
                _log.info("Opening image index: " + image_file)
                reader = DirectoryReader.open(image_dir)
                self._image = IndexSearcher(reader)
            else:
                self._image = None
                _log.info("Opening image index: false")

            if os.path.exists(info_file):
                _log.info("Opening info index: " + info_file)
                reader = DirectoryReader.open(info_dir)
                self._info = IndexSearcher(reader)
            else:
                self._info = None
                _log.info("Opening info index: false")

            if os.path.exists(wn_info_file):
                _log.info("Opening wordNet info index: " + wn_info_file)
                reader = DirectoryReader.open(wn_info_dir)
                self._wordnet_info = IndexSearcher(reader)
            else:
                self._wordnet_info = None
                _log.info("Opening wordNet info index: false")


def instance():
    """Get an instance of the OfflineIndex
    @return: the instance
    @rtype: OfflineIndex
    """
    return OfflineIndex()
