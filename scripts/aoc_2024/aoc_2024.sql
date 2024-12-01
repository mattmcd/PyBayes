-- create schema aoc_2024;

-- Day 01 part 1
with left_col as (select row_number() over (order by left_col) as row
                       , left_col                              as loc
                  from aoc_2024.day01
                  order by left_col),
     right_col as (select row_number() over (order by right_col) as row
                        , right_col                              as loc
                   from aoc_2024.day01
                   order by right_col),
     dist as (select l.loc, r.loc, abs(l.loc - r.loc) as dist
              from left_col l
                       join right_col r
                            on l.row = r.row)
select sum(dist)
from dist
;

-- Day 01 part 2
with similarity as (select l.left_col as loc
                         , count(r.right_col) as match
                    from aoc_2024.day01 l
                             left join aoc_2024.day01 r
                                       on l.left_col = r.right_col
                    group by l.left_col)
select sum(loc * match)
from similarity;

