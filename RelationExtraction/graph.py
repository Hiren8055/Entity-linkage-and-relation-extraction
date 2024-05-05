"""build a entity linker to neo4j database
the data will be coming from model 
entity 
relation between entity

built a program that can take entity and relation, and put it to neo4j

create two nodes and add realtion between them"""

from neo4j import GraphDatabase
from uuid import uuid4
import re


class GraphBuilder:
    def __init__(self,uri, user,password) -> None:
        self.driver = GraphDatabase.driver(uri,auth=(user,password))

    def close(self):
        self.driver.close()

    # '''MATCH (a:Article)
    # OPTIONAL MATCH (c:Comment)-[:COMMENT_ON]->(a)
    # RETURN a.article_id AS article_id, c.comment_text AS comment_text, a.full_text AS full_text
    # LIMIT'''
    def generate_text(self, num_of_articles):
        print("hi gen")
        read_text = f"""
            MATCH (a:Article)
            OPTIONAL MATCH (c:Comment)-[:COMMENT_ON]->(a)
            RETURN a.article_id AS article_id, c.comment_text AS comment_text, a.full_text AS full_text
            LIMIT {str(num_of_articles)}
            """
        with self.driver.session() as session:
            print("hi before")
            comments_and_articles = session.run(query=read_text)
            print("hi after")
            data_list = []  # Initialize an empty list to accumulate dictionaries
            for record in comments_and_articles:
                if record["comment_text"] is not None:
                    data_list.append({
                        'article_id': record['article_id'],
                        # 'user_id': record['user_id'], 
                        'of_type': 'comment', 
                        'text': record['comment_text']
                        })

                data_list.append({
                    'article_id': record["article_id"],
                    # 'user_id': record["user_id"],
                    'of_type': "article", 
                    'text': record["full_text"]
                    })
            return data_list
                
    def new_entity(
            self,
            triplets, 
            of_type, 
            # user_id,
            article_id,
            text
            ):
        with self.driver.session() as session:
            def remove_special_characters(input_string):
                # Define the regex pattern to match non-alphanumeric characters and underscores
                pattern = r'[^a-zA-Z0-9_]'
                # Replace all non-matching characters with an empty string
                result = re.sub(pattern, '', input_string)
                return result
            for i in triplets:
                message = {
                    "head": i["head"]["name"],
                    "tail": i["tail"]["name"],
                    "tail_id": f"{i['tail']['id'] if 'id' in i['tail'] else str(uuid4())}",
                    "head_id": f"{i['head']['id'] if 'id' in i['head'] else str(uuid4())}",
                    "head_label": i["head"]["label"],
                    "tail_label": i["tail"]["label"],
                    "relation": i["type"],
                    "of_type": of_type,
                    # "user_id": user_id,
                    "article_id": article_id,
                    "text": text
                    }
                for key, val in message.items():
                    message[key] = f"{remove_special_characters(str(val).replace(' ','_')) if val else ''}"
                session.execute_write(self.merge_entities, message)
    
    @staticmethod
    def merge_entities(tx, params):
        query = (
            f"""
            MERGE (n1{{id:"{params['head_id']}"}})
            ON CREATE SET n1.label = "{params['head_label']}", n1.name = "{params['head']}"
            MERGE (n2{{id:"{params['tail_id']}"}})
            ON CREATE SET n2.label = "{params['tail_label']}", n2.name = "{params['tail']}"
            MERGE (n1)-[:{params['relation']}]->(n2)
            WITH n1, n2
            
            {
                f'''
                MATCH (c:Comment{{comment_text:"{params['text']}"}})
                WITH n1, n2, c
                MERGE (c)-[:HAS_ENTITY]->(n1)
                MERGE (c)-[:HAS_ENTITY]->(n2)
                ''' if params['of_type'] == 'comment'
                else
                f'''
                MATCH (a:Article{{article_id:{params['article_id']}}})
                WITH n1, n2, a
                MERGE (a)-[:HAS_ENTITY]->(n1)
                MERGE (a)-[:HAS_ENTITY]->(n2)
                '''
                }
            """
            # f"""
            # MERGE (n1{{id:{params['head_id']}}})
            # ON CREATE SET n1.label = {params['label']}
            # MERGE (n2{{id:{params['tail_id']}}})
            # ON CREATE SET n2.label = {params['label']}
            # MERGE (n1)-[:{params["relation"]}]->(n2)
            # {
            #     f"MATCH (u:User {{user_id:{params['user_id']}}})-[:COMMENTED]->(c:Comment)"
            #     if params['user_id'] is not None
            #     else
            #     f"MATCH (a:Article{{article_id={params['article_id']}}})"
            #     }
            # {
            #     '''
            #     MERGE (c)-[:HAS_ENTITY]->(n1) 
            #     MERGE (c)-[:HAS_ENTITY]->(n2)
            #     ''' if params['of_type'] is 'comment' 
            #     else 
            #     '''
            #     MERGE (a)-[:HAS_ENTITY]->(n1)
            #     MERGE (a)-[:HAS_ENTITY]->(n2)
            #     '''
            #     }
            # """
        )
        tx.run(query)
        print(f"Article:{params['article_id']}, is_type: {params['of_type']}")

if __name__ == "__main__":
    triplets = [{'head': {'name': 'annual league championship', 'label': '', 'id': 'a4e41793-4429-472f-ab87-15750fc90b18'}, 'type': 'participating team', 'tail': {'name': 'Kansas City Chiefs', 'label': '', 'id': '560ace3e-aac4-4514-a027-77769ac09d48'}}, {'head': {'name': 'The Guardian', 'label': '', 'id': '82044213-6a7a-456a-8fab-09a00c8f8f00'}, 'type': 'owned by', 'tail': {'name': 'Bakish', 'label': '', 'id': '8e153880-e8f3-4ed8-bef4-0553b1771bcc'}}, {'head': {'name': 'John F. Kennedy School of Government', 'label': '', 'id': 'cabdb511-c610-4d5a-a9d5-3558cef29fcb'}, 'type': 'part of', 'tail': {'name': 'Harvard University', 'label': '', 'id': '320cc249-eb0e-4fa0-b19f-5962bdfa5c85'}}, {'head': {'name': 'Paramount Pictures', 'label': 'ORG', 'id': 'Q159846'}, 'type': 'owned by', 'tail': {'name': 'Paramount Global', 'label': 'ORG', 'id': '00ec4e82-d0b9-438c-88b2-b78b02aaa823'}}, {'head': {'name': 'Paramount Global', 'label': 'ORG', 'id': '00ec4e82-d0b9-438c-88b2-b78b02aaa823'}, 'type': 'owner of', 'tail': {'name': 'Paramount Pictures', 'label': 'ORG', 'id': 'Q159846'}}, {'head': {'name': 'Paramount Global', 'label': 'ORG', 'id': '00ec4e82-d0b9-438c-88b2-b78b02aaa823'}, 'type': 'owner of', 'tail': {'name': 'MTV', 'label': 'ORG', 'id': 'Q43359'}}, {'head': {'name': 'Paramount Global', 'label': 'ORG', 'id': '00ec4e82-d0b9-438c-88b2-b78b02aaa823'}, 'type': 'owner of', 'tail': {'name': 'Comedy Central', 'label': 'PERSON', 'id': 'Q131439'}}, {'head': {'name': 'Paramount Global', 'label': 'ORG', 'id': '00ec4e82-d0b9-438c-88b2-b78b02aaa823'}, 'type': 'owner of', 'tail': {'name': 'Nickelodeon', 'label': 'ORG', 'id': 'Q154958'}}, {'head': {'name': 'Paramount Global', 'label': 'ORG', 'id': '00ec4e82-d0b9-438c-88b2-b78b02aaa823'}, 'type': 'owner of', 'tail': {'name': 'Channel 5', 'label': 'ORG', 'id': 'Q1062280'}}, {'head': {'name': 'Paramount Global', 'label': 'ORG', 'id': '00ec4e82-d0b9-438c-88b2-b78b02aaa823'}, 'type': 'owner of', 'tail': {'name': 'Paramount+', 'label': '', 'id': '4cf38e3e-dc46-4a53-8fde-3538772a1cc5'}}, {'head': {'name': 'Paramount Global', 'label': 'ORG', 'id': '00ec4e82-d0b9-438c-88b2-b78b02aaa823'}, 'type': 'owner of', 'tail': {'name': 'Pluto', 'label': '', 'id': '88913c66-a7f0-4e28-aaf0-fc90ba8f288f'}}, {'head': {'name': 'MTV', 'label': 'ORG', 'id': 'Q43359'}, 'type': 'owned by', 'tail': {'name': 'Paramount Global', 'label': 'ORG', 'id': '00ec4e82-d0b9-438c-88b2-b78b02aaa823'}}, {'head': {'name': 'Comedy Central', 'label': 'PERSON', 'id': 'Q131439'}, 'type': 'owned by', 'tail': {'name': 'Paramount Global', 'label': 'ORG', 'id': '00ec4e82-d0b9-438c-88b2-b78b02aaa823'}}, {'head': {'name': 'Nickelodeon', 'label': 'ORG', 'id': 'Q154958'}, 'type': 'owned by', 'tail': {'name': 'Paramount Global', 'label': 'ORG', 'id': '00ec4e82-d0b9-438c-88b2-b78b02aaa823'}}, {'head': {'name': 'Channel 5', 'label': 'ORG', 'id': 'Q1062280'}, 'type': 'owned by', 'tail': {'name': 'Paramount Global', 'label': 'ORG', 'id': '00ec4e82-d0b9-438c-88b2-b78b02aaa823'}}]
    graph = GraphBuilder("bolt://localhost:7687","neo4j","12345678")
    graph.new_entity(triplets)
    graph.close()
