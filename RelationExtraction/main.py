"""build a entity linker to neo4j database
the data will be coming from model 
entity 
relation between entity

built a program that can take entity and relation, and put it to neo4j

create two nodes and add realtion between them"""

import spacy
from spacy_experimental.coref.coref_component import DEFAULT_COREF_MODEL
from spacy_experimental.coref.coref_util import DEFAULT_CLUSTER_PREFIX
from rebel import RelationExtractor
from graph import GraphBuilder
from linker import EntityDisambiguator
import os
import pickle
import pprint
class KnowledgeGraph:
    def __init__(self) -> None:
        self.relex = RelationExtractor()
        self.graph = GraphBuilder("bolt://localhost:7687","neo4j","12345678")
        self.linker = EntityDisambiguator()
        if os.path.exists("article_done.pkl"):
            with open("article_done.pkl", "rb") as a:
                self.article_done = pickle.load(a)
        else:
            self.article_done = dict()

    def preprocess_text(self, text):
        # Add coreference resolution model
        print("first\n", text)
        nlp = spacy.load("en_coreference_web_trf")
        doc = nlp(text=text)
        print("second",doc)
        words = text.strip().split()
        print("words: ", words)
        cluster = 1
        period = False
        for i, word in enumerate(words):
            current_cluster = f"coref_clusters_{str(cluster)}"
            if word[-1] == ".":
                    word = word[:-1]
                    period = True
            if word == str(doc.spans[current_cluster][1]):
                words[i] = f"{str(doc.spans[current_cluster][0])}{'.' if period else ''}"
                period = False
        text = f"{' '.join(words)}"
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        print(f"preprocess\n{doc}")
        # coref = nlp.add_pipe("experimental_coref", config=config)
        # clusters = coref.predict([doc])
        # coref.set_annotations([doc], clusters)
        return doc

    def info_extraction(self, text):
        extracted_triplets = []
        for sent in text.sents:
            print("sent", sent)
            tokenized_sent = self.relex.tokenize_text(str(sent))[0]
            print(f"tokenized: {tokenized_sent}")
            entity_info = self.linker.link_entites(str(sent))
            print(f"entity_info: {entity_info}")
            triplet_list = self.relex.extract_triplets(tokenized_sent, entity_info)
            extracted_triplets.extend(triplet_list)
        return extracted_triplets

    def start(self):
        def make_graph(
                article_id,
                # user_id,
                of_type, 
                og_text
                ):


            text = self.preprocess_text(text = og_text)

            print(f"processed{text}")
            triplets = self.info_extraction(text = text)
            self.graph.new_entity(
                triplets,
                of_type, 
                # user_id,
                article_id,
                text = og_text
                )
        num_of_articles = 10
        print("hi")
        articles = self.graph.generate_text(num_of_articles)
        pprint.pp(articles)
        print("hi")
        for a in articles:
            print("hi")
            # article_id, user_id, of_type, text = a['article_id'], a['user_id'], a['of_type'], a['text']
            article_id, of_type, text = a['article_id'], a['of_type'], a['text']
            if article_id in self.article_done:
                if of_type in self.article_done[article_id]:
                    continue
                else:
                    make_graph(
                        article_id,
                        # user_id, 
                        of_type,
                        text
                        )
                    self.article_done[article_id][of_type] = True
            else:
                make_graph(
                    article_id, 
                    # user_id, 
                    of_type, 
                    text
                    )
                self.article_done[article_id] = {of_type:True}

    def end(self):
        with open("article_done.pkl", 'wb') as file:
            # Serialize and write the dictionary to the file
            pickle.dump(self.article_done, file)
        self.graph.close()

if __name__ == "__main__":
    news = KnowledgeGraph()
    news.start()
    news.end()