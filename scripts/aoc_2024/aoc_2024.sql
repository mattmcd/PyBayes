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
with similarity as (select l.left_col         as loc
                         , count(r.right_col) as match
                    from aoc_2024.day01 l
                             left join aoc_2024.day01 r
                                       on l.left_col = r.right_col
                    group by l.left_col)
select sum(loc * match)
from similarity;

-- Day 02 part 1
with inc as (select record,
                    level - lag(level) over (partition by record order by position) as diff
             from aoc_2024.day02),
     check_diff as (select record
                         , diff
                         , diff > 0               as increasing
                         , diff between 1 and 3   as valid_step_up
                         , diff < 0               as decreasing
                         , diff between -3 and -1 as valid_step_down
                    from inc
                    where diff is not null),
     check_record as (select record
                           , (bool_and(increasing) and bool_and(valid_step_up))
             or (bool_and(decreasing) and bool_and(valid_step_down)) as valid
                      from check_diff
                      group by record)
select count(*)
from check_record
where valid
;

-- Day 02 part 2
with inc as (select record,
                    position,
                    level - lag(level) over (partition by record order by position) as diff
             from aoc_2024.day02),
     check_diff as (select record
                         , position
                         , diff
                         , diff > 0               as increasing
                         , diff between 1 and 3   as valid_step_up
                         , diff < 0               as decreasing
                         , diff between -3 and -1 as valid_step_down
                    from inc
                    where diff is not null),
     -- Problem Damper logic: add column flagging error levels so they can be removed
     -- Could have been part of previous query
     errors as (select record
                     , position
                     , diff
                     , increasing
                     , valid_step_up
                     , decreasing
                     , valid_step_down
                     , sum(
                       case
                           when (not (increasing and valid_step_up)) and (not (decreasing and valid_step_down))
                               then 1
                           else 0 end) over (partition by record order by position) as invalid_count
                from check_diff),
     check_record as (select record
                           -- Initial attempt at the Problem Damper logic, fails because steps can become too big
                           -- Kept the new logic anyway
                           , sum(case when (not (increasing and valid_step_up)) then 1 else 0 end)   as invalid_up
                           , sum(case when (not (decreasing and valid_step_down)) then 1 else 0 end) as invalid_down
                      from errors
                      where invalid_count != 1 -- Remove first invalid record
                      group by record)
select count(*)
from check_record
where (invalid_down = 0)
   or (invalid_up = 0)
;

