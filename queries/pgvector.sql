-- Trying out pgvector, preinstalled with postgres.app on Mac.
CREATE EXTENSION vector;
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));

INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');

SELECT *
     , embedding <-> '[3,1,2]' as l2_distance
    , embedding <#> '[3,1,2]' as neg_inner_product
    , embedding <=> '[3,1,2]' as cosine_distance
FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;

select *

from documents;