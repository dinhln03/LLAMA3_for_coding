FRANCHISES = """
select t1.aliases, overall, firsts, seconds, third, y1,y2, unique_a, unique_1, unique_12
from 
	(select Count(A."PlayerID") as overall,T."Aliases" as aliases, MAX(A."year") as y1, MIN(A."year") as y2, Count (distinct A."PlayerID") as unique_a
	from public."all-nba-teams_list" A, public.teams T
	where A."TeamID"=any(T."Aliases")
	group by T."Aliases"
	order by T."Aliases"
	) as t1 
join 
	(
	select Count(A."PlayerID") as firsts,T."Aliases" as aliases, Count (distinct A."PlayerID") as unique_1
	from public."all-nba-teams_list" A, public.teams T
	where A."TeamID"=any(T."Aliases") and A."type"=1
	group by T."Aliases"
	order by T."Aliases"
	) as t2 on t1.aliases=t2.aliases
join 
	(
	select Count(A."PlayerID") as seconds,T."Aliases" as aliases
	from public."all-nba-teams_list" A, public.teams T
	where A."TeamID"=any(T."Aliases") and A."type"=2
	group by T."Aliases"
	order by T."Aliases"
	) as t3 on t1.aliases=t3.aliases
join 
	(
	select Count(A."PlayerID") as third,T."Aliases" as aliases
	from public."all-nba-teams_list" A, public.teams T
	where A."TeamID"=any(T."Aliases") and A."type"=3
	group by T."Aliases"
	order by T."Aliases"
	) as t4 on t1.aliases=t4.aliases
join 
	(
	select Count (distinct A."PlayerID") as unique_12, T."Aliases" as aliases
	from public."all-nba-teams_list" A, public.teams T
	where A."TeamID"=any(T."Aliases") and A."type"in(1,2)
	group by T."Aliases"
	order by T."Aliases"
	) as t5 on t1.aliases=t5.aliases
"""