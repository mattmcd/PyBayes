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

-- Day 03 Part 1
with muls as (select regexp_matches(string_agg(program, ''), 'mul\((\d+,\d+)\)', 'g') as numbers
              from aoc_2024.day03)
select sum(split_part(numbers[1], ',', 1)::integer * split_part(numbers[1], ',', 2)::integer)
from muls
;

-- Day 03 Part 2
with matches as (select regexp_matches(
                                string_agg(program, '')
                            , '((do)\(\)|mul\((\d+,\d+)\)|(don''t)\(\))', 'g'
                        ) as token
                 from aoc_2024.day03),
     numbers as (select row_number() over () as ind,
                        case
                            when token[3] is not null then
                                (split_part(token[3], ',', 1)::integer * split_part(token[3], ',', 2)::integer)
                            end              as mul,
                        case
                            when token[2] is not null then 1
                            when token[4] is not null then 0
                            when row_number() over () = 1 then 1
                            end              as enable_flag
                 from matches),
     flagged_muls as (select mul,
                             first_value(enable_flag) over (partition by flag_partition order by ind) as flag,
                             mul *
                             first_value(enable_flag) over (partition by flag_partition order by ind) as flagged_mul
                      from (select ind
                                 , mul
                                 , enable_flag
                                 , sum(case when enable_flag is not null then 1 else 0 end)
                                   over (order by ind) as flag_partition
                            from numbers) o)
select sum(mul) as part_1, sum(flagged_mul) as part_2
from flagged_muls
;

-- Day 4 Part 1
with hv as (select x.row as xr, x.col as xc, s.row as sr, s.col as sc
            from (select row, col from aoc_2024.day04 where letter = 'X') x
                     join (select row, col from aoc_2024.day04 where letter = 'M') m
                          on abs(x.row - m.row) + abs(x.col - m.col) = 1
                     join (select row, col from aoc_2024.day04 where letter = 'A') a
                          on abs(m.row - a.row) + abs(m.col - a.col) = 1
                     join (select row, col from aoc_2024.day04 where letter = 'S') s
                          on abs(a.row - s.row) + abs(a.col - s.col) = 1
            where abs(x.row - a.row) + abs(x.col - a.col) = 2
              and abs(m.row - s.row) + abs(m.col - s.col) = 2
              and abs(x.row - s.row) + abs(x.col - s.col) = 3
              and ((x.row = s.row) or (x.col = s.col))),
     diag as (select x.row as xr, x.col as xc, s.row as sr, s.col as sc
              from (select row, col from aoc_2024.day04 where letter = 'X') x
                       join (select row, col from aoc_2024.day04 where letter = 'M') m
                            on abs(x.row - m.row) + abs(x.col - m.col) = 2
                       join (select row, col from aoc_2024.day04 where letter = 'A') a
                            on abs(m.row - a.row) + abs(m.col - a.col) = 2
                       join (select row, col from aoc_2024.day04 where letter = 'S') s
                            on abs(a.row - s.row) + abs(a.col - s.col) = 2
              where abs(x.row - a.row) + abs(x.col - a.col) = 4
                and abs(m.row - s.row) + abs(m.col - s.col) = 4
                and abs(x.row - s.row) + abs(x.col - s.col) = 6
                and (abs(x.row - m.row) = abs(x.col - m.col))
                and (abs(m.row - a.row) = abs(m.col - a.col))
                and (abs(a.row - s.row) = abs(a.col - s.col))
                and (abs(x.row - s.row) = abs(x.col - s.col))),
     combined as (select distinct xr, xc, sr, sc
                  from (select distinct xr, xc, sr, sc
                        from hv
                        union all
                        select distinct xr, xc, sr, sc
                        from diag) c
                  order by xr, xc, sr, sc)
select count(*)
from combined
;

-- Day 04 Part 2
with diag as (select m1.row as m1r, m1.col as m1c, s1.row as s1r, s1.col as s1c
              from (select row, col from aoc_2024.day04_test where letter = 'A') a
                       join (select row, col from aoc_2024.day04_test where letter = 'M') m1
                            on abs(m1.row - a.row) + abs(m1.col - a.col) = 2
                       join (select row, col from aoc_2024.day04_test where letter = 'M') m2
                            on abs(m2.row - a.row) + abs(m2.col - a.col) = 2
                       join (select row, col from aoc_2024.day04_test where letter = 'S') s1
                            on abs(s1.row - a.row) + abs(s1.col - a.col) = 2
                       join (select row, col from aoc_2024.day04_test where letter = 'S') s2
                            on abs(s2.row - a.row) + abs(s2.col - a.col) = 2

              where abs(m1.row - s1.row) + abs(m1.col - s1.col) = 4
                and abs(m2.row - s2.row) + abs(m2.col - s2.col) = 4
                and (abs(m1.row - s1.row) = abs(m1.col - s1.col))
                and (abs(m2.row - s2.row) = abs(m2.col - s2.col))
--                 and (m1.row != m2.row and m1.col != m2.col)

    )
select count(*)
from diag
;