# %%
# Trying out pgvector with sentence embeddings
# https://github.com/pgvector/pgvector-python/blob/master/examples/sentence_transformers/example.py

# %%
from pgvector.psycopg import register_vector
import psycopg
from sentence_transformers import SentenceTransformer
from mattmcd.io import pg_engine
import sqlalchemy as sa

# %%
# Replace psycopg usage with SQLAlchemy - first pass just uses the psycopg conn
# but obtained from the SQLAlchemy engine.  Next iteration modify statements to use sa
# conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

engine = pg_engine()
sa_conn = engine.connect()

conn = sa_conn.connection.dbapi_connection
conn.autocommit = True

# %%
# with engine.connect() as conn:
#     # This reaches through the SQLAlchemy wrapper to the actual psycopg object
#     raw_conn = conn.connection.dbapi_connection
#     register_vector(raw_conn)

# %%
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

# %%
conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(1024))')

# %%
# model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

input = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]

# %%
# embeddings = model.encode(input)

task = "retrieval.query"
embeddings = model.encode(
    input,
    task=task,
    prompt_name=task,
)

# %%
for content, embedding in zip(input, embeddings):
    conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, embedding))

# %%
conn.commit()

# %%
query = 'forest'
query = 'dogs purr'
query_embedding = model.encode(query)

# %%
result = conn.execute(
    """SELECT content 
       , embedding <=> %s AS cosine_distance
       FROM documents ORDER BY embedding <=> %s LIMIT 5""",
    (query_embedding, query_embedding)
).fetchall()
for row in result:
    print(f'content: {row[0]}, distance: {row[1]:0.3f}')
