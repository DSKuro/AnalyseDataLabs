from sqlalchemy import func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.expression import case, cast
from sqlalchemy.sql.sqltypes import Numeric

from models import Champ, Match, Participant, Stats1, TeamBan, engine

Session = sessionmaker(bind=engine)
session = Session()

kda = (
    session.query(
        Champ.name.label('champion'),
        func.count().label('games'),
        func.round(
            func.avg(
                cast(
                    (Stats1.kills + Stats1.assists),
                    Numeric
                )
                / func.nullif(Stats1.deaths, 0)
            ),
            2
        ).label('avg_kda')
    )
    .join(Participant, Champ.id == Participant.championid)
    .join(Stats1, Participant.id == Stats1.id)   # PK ↔ PK
    .group_by(Champ.name)
    .having(func.count() > 100)
    .order_by(func.round(
        func.avg(
            cast(
                (Stats1.kills + Stats1.assists),
                Numeric
            )
            / func.nullif(Stats1.deaths, 0)
        ),
        2
    ).desc())
    .limit(10)
    .all()
)

for row in kda:
    print(
        f"Champion: {row.champion:<12} | "
        f"Games: {row.games:>5} | "
        f"Avg KDA: {row.avg_kda}"
    )

winrate_expr = func.round(
    func.avg(
        case(
            (Stats1.win.is_(True), 1),
            else_=0
        )
    ) * 100,
    2
)

win = (
    session.query(
        Champ.name.label('champion'),
        Participant.position.label('position'),
        func.count().label('games'),
        winrate_expr.label('winrate')
    )
    .join(Participant, Participant.championid == Champ.id)
    .join(Stats1, Participant.id == Stats1.id)  # PK = PK
    .group_by(Champ.name, Participant.position)
    .having(func.count() > 50)
    .order_by(winrate_expr.desc())
    .all()
)

print(f"{'Champion':<14} {'Pos':<8} {'Games':>6} {'Winrate %':>10}")
print("-" * 42)

for row in win:
    print(
        f"{row.champion:<14} "
        f"{row.position:<8} "
        f"{row.games:>6} "
        f"{row.winrate:>10.2f}"
    )

jax = (
    session.query(
        Champ.name.label('champion'),
        func.count().label('games'),
        func.round(
            func.avg(
                case(
                    (Stats1.win == True, 1),
                    else_=0
                )
            ) * 100, 2
        ).label('winrate'),
        func.round(func.avg(Stats1.kills), 2).label('avg_kills'),
        func.round(func.avg(Stats1.deaths), 2).label('avg_deaths'),
        func.round(func.avg(Stats1.assists), 2).label('avg_assists'),
        func.round(func.avg(Stats1.goldearned), 0).label('avg_gold'),
        func.round(func.avg(Stats1.totdmgtochamp), 0).label('avg_damage')
    )
    .join(Participant, Champ.id == Participant.championid)
    .join(Participant.stats1)
    .filter(Champ.name == 'Jax')
    .group_by(Champ.name)
    .all()
)

for row in jax:
    print(f"Champion: {row.champion}, Games: {row.games}, Winrate: {row.winrate}%, "
          f"Avg Kills: {row.avg_kills}, Avg Deaths: {row.avg_deaths}, Avg Assists: {row.avg_assists}, "
          f"Avg Gold: {row.avg_gold}, Avg Damage: {row.avg_damage}")

total_participants = (
    session.query(func.count(Participant.id).label('total_participants'))
    .join(Match, Participant.matchid == Match.id)
    .scalar()
)

print(total_participants)

damage = (
    session.query(
        Champ.name.label('champion'),
        func.round(func.avg(Stats1.totdmgtochamp), 0).label('avg_damage'),
        func.count().label('games')
    )
    .join(Participant, Participant.championid == Champ.id)
    .join(Stats1, Participant.id == Stats1.id)  # связь по PK = PK
    .group_by(Champ.name)
    .having(func.count() > 100)
    .order_by(func.avg(Stats1.totdmgtochamp).desc())
    .limit(10)
    .all()
)

print(f"{'Champion':<15} {'Avg damage':>12} {'Games':>8}")
print("-" * 38)

for row in damage:
    print(f"{row.champion:<15} {int(row.avg_damage):>12} {row.games:>8}")


av_time = (
    session.query(
        Match.version,
        func.round(func.avg(Match.duration) / 60, 2).label('avg_duration_minutes'),
        func.count().label('matches')
    )
    .group_by(Match.version)
    .order_by(func.round(func.avg(Match.duration) / 60, 2).desc())
    .all()
)

for row in av_time:
    print(f"Version: {row.version}, Avg Duration (min): {row.avg_duration_minutes}, Matches: {row.matches}")

bans = (
    session.query(
        Champ.name.label('champion'),
        func.count().label('bans')
    )
    .join(TeamBan, TeamBan.championid == Champ.id)
    .group_by(Champ.name)
    .order_by(func.count().desc())
    .limit(10)
    .all()
)

print(f"{'Champion':<15} {'Bans':>6}")
print("-" * 22)

for row in bans:
    print(f"{row.champion:<15} {row.bans:>6}")


session.close()
