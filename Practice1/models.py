import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Boolean, BigInteger, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

load_dotenv()

DB_USER = "postgres"
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = os.getenv('DB_NAME')

class Champ(Base):
    __tablename__ = 'champs'

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)

    # Relationship to participants and teambans
    participants = relationship('Participant', back_populates='champ')
    teambans = relationship('TeamBan', back_populates='champ')


class Match(Base):
    __tablename__ = 'matches'

    id = Column(BigInteger, primary_key=True)
    gameid = Column(BigInteger)
    platformid = Column(String(10))
    queueid = Column(Integer)
    seasonid = Column(Integer)
    duration = Column(Integer)
    creation = Column(BigInteger)
    version = Column(String(20))

    # Relationships
    participants = relationship('Participant', back_populates='match')
    teamstats = relationship('TeamStat', back_populates='match')


class Participant(Base):
    __tablename__ = 'participants'

    id = Column(Integer, primary_key=True)
    matchid = Column(BigInteger, ForeignKey('matches.id'), nullable=False)
    player = Column(String(50))
    championid = Column(Integer, ForeignKey('champs.id'), nullable=False)
    ss1 = Column(Integer)
    ss2 = Column(Integer)
    role = Column(String(20))
    position = Column(String(20))

    # Relationships
    match = relationship('Match', back_populates='participants')
    champ = relationship('Champ', back_populates='participants')
    stats1 = relationship(
        'Stats1',
        primaryjoin='Participant.id == foreign(Stats1.id)',
        back_populates='participant',
        uselist=False
    )
    stats2 = relationship('Stats2', back_populates='participant')


class Stats1(Base):
    __tablename__ = 'stats1'

    id = Column(Integer, primary_key=True)
    win = Column(Boolean)
    item1 = Column(Integer)
    item2 = Column(Integer)
    item3 = Column(Integer)
    item4 = Column(Integer)
    item5 = Column(Integer)
    item6 = Column(Integer)
    trinket = Column(Integer)
    kills = Column(Integer)
    deaths = Column(Integer)
    assists = Column(Integer)
    largestkillingspree = Column(Integer)
    largestmultikill = Column(Integer)
    killingsprees = Column(Integer)
    longesttimespentliving = Column(Integer)
    doublekills = Column(Integer)
    triplekills = Column(Integer)
    quadrakills = Column(Integer)
    pentakills = Column(Integer)
    legendarykills = Column(Integer)
    totdmgdealt = Column(Integer)
    magicdmgdealt = Column(Integer)
    physicaldmgdealt = Column(Integer)
    truedmgdealt = Column(Integer)
    largestcrit = Column(Integer)
    totdmgtochamp = Column(Integer)
    magicdmgtochamp = Column(Integer)
    physdmgtochamp = Column(Integer)
    truedmgtochamp = Column(Integer)
    totheal = Column(Integer)
    totunitshealed = Column(Integer)
    dmgselfmit = Column(Integer)
    dmgtoobj = Column(Integer)
    dmgtoturrets = Column(Integer)
    visionscore = Column(Integer)
    timecc = Column(Integer)
    totdmgtaken = Column(Integer)
    magicdmgtaken = Column(Integer)
    physdmgtaken = Column(Integer)
    truedmgtaken = Column(Integer)
    goldearned = Column(Integer)
    goldspent = Column(Integer)
    turretkills = Column(Integer)
    inhibkills = Column(Integer)
    totminionskilled = Column(Integer)
    neutralminionskilled = Column(Integer)
    ownjunglekills = Column(Integer)
    enemyjunglekills = Column(Integer)
    totcctimedealt = Column(Integer)
    champlvl = Column(Integer)
    pinksbought = Column(Integer)
    wardsbought = Column(Integer)
    wardsplaced = Column(Integer)
    wardskilled = Column(Integer)
    firstblood = Column(Boolean)

    participant = relationship(
        'Participant',
        primaryjoin='Participant.id == foreign(Stats1.id)',
        back_populates='stats1',
        uselist=False
    )


class Stats2(Base):
    __tablename__ = 'stats2'

    id = Column(Integer, primary_key=True)
    win = Column(Boolean)
    item1 = Column(Integer)
    item2 = Column(Integer)
    item3 = Column(Integer)
    item4 = Column(Integer)
    item5 = Column(Integer)
    item6 = Column(Integer)
    trinket = Column(Integer)
    kills = Column(Integer)
    deaths = Column(Integer)
    assists = Column(Integer)
    largestkillingspree = Column(Integer)
    largestmultikill = Column(Integer)
    killingsprees = Column(Integer)
    longesttimespentliving = Column(Integer)
    doublekills = Column(Integer)
    triplekills = Column(Integer)
    quadrakills = Column(Integer)
    pentakills = Column(Integer)
    legendarykills = Column(Integer)
    totdmgdealt = Column(Integer)
    magicdmgdealt = Column(Integer)
    physicaldmgdealt = Column(Integer)
    truedmgdealt = Column(Integer)
    largestcrit = Column(Integer)
    totdmgtochamp = Column(Integer)
    magicdmgtochamp = Column(Integer)
    physdmgtochamp = Column(Integer)
    truedmgtochamp = Column(Integer)
    totheal = Column(Integer)
    totunitshealed = Column(Integer)
    dmgselfmit = Column(Integer)
    dmgtoobj = Column(Integer)
    dmgtoturrets = Column(Integer)
    visionscore = Column(Integer)
    timecc = Column(Integer)
    totdmgtaken = Column(Integer)
    magicdmgtaken = Column(Integer)
    physdmgtaken = Column(Integer)
    truedmgtaken = Column(Integer)
    goldearned = Column(Integer)
    goldspent = Column(Integer)
    turretkills = Column(Integer)
    inhibkills = Column(Integer)
    totminionskilled = Column(Integer)
    neutralminionskilled = Column(Integer)
    ownjunglekills = Column(Integer)
    enemyjunglekills = Column(Integer)
    totcctimedealt = Column(Integer)
    champlvl = Column(Integer)
    pinksbought = Column(Integer)
    wardsbought = Column(Integer)
    wardsplaced = Column(Integer)
    wardskilled = Column(Integer)
    firstblood = Column(Boolean)

    participant_id = Column(Integer, ForeignKey('participants.id'))
    participant = relationship('Participant', back_populates='stats2')


class TeamStat(Base):
    __tablename__ = 'teamstats'

    matchid = Column(BigInteger, ForeignKey('matches.id'), primary_key=True)
    teamid = Column(Integer, primary_key=True)
    firstblood = Column(Boolean)
    firsttower = Column(Boolean)
    firstinhib = Column(Boolean)
    firstbaron = Column(Boolean)
    firstdragon = Column(Boolean)
    firstharry = Column(Boolean)
    towerkills = Column(Integer)
    inhibkills = Column(Integer)
    baronkills = Column(Integer)
    dragonkills = Column(Integer)
    harrykills = Column(Integer)

    match = relationship('Match', back_populates='teamstats')


class TeamBan(Base):
    __tablename__ = 'teambans'

    matchid = Column(BigInteger, ForeignKey('matches.id'), primary_key=True)
    teamid = Column(Integer, primary_key=True)
    championid = Column(Integer, ForeignKey('champs.id'), nullable=False)
    banturn = Column(Integer)

    champ = relationship('Champ', back_populates='teambans')

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
Base.metadata.create_all(engine)
