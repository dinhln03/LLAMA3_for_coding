'''

'''
import hashlib
import uuid
import logging
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy
from subprocess import PIPE, Popen
from kazoo.client import KazooClient
from multiprocessing import Process


class CassandraAware:

    def init__(self):
        self.cluster = Cluster("", protocol_version=4,
                               load_balancing_policy=DCAwareRoundRobinPolicy(local_dc='datacenter1'))
        self.session = self.cluster.connect("")


class KazooAware:

    def __init__(self):
        zk = KazooClient(hosts='')
        zk.start()
        self.ver = zk.Counter('/ver', default=0x821)


class blocks_view(CassandraAware, KazooAware):

    def __init__(self):
        logging.info('initialize blocks_view')
        self.updateBlock = super.session.prepare('''
        insert into blocks_view(sha256_id, block_id, ver, raw, sha256, nonce, consent_algorithm, predecessor, counter, setup)
        values(?, ?, ?, ?, ?, ?, ?, ?, ? toTimestamp(now()))''')
        self.updateStatusStatement = super.session.prepare('update candidate set processed=true where sha256_id = ?')
        self.genesis()

    def runLoop(self):
        while True:
            self.run()

    def publicMining(self, raw):
        ver = super.ver
        super.ver += 1
        predecessor = self.block_id
        m = hashlib.sha256()
        m.update(bytes(raw, 'utf-8'))
        sha256 = m.hexdigest()
        self.block_ptr = sha256
        block_id = ''  # fxxx
        nonce = ''
        while True:
            if block_id.startswith('f'):
                self.block_id = block_id
                self.nonce = nonce
                break
            nonce = uuid.uuid4().hex
            with Popen(('./mining', sha256, nonce), stdout=PIPE) as p:
                block_id = p.stdout.read()
        m = hashlib.sha256()
        m.update(bytes(block_id, 'utf-8'))
        sha256_id = m.hexdigest()
        self.counter += 1
        super.session.execute(self.updateBlock, [sha256_id, block_id, ver, raw, sha256, nonce,
                                                 'Mersenne15Mersenne14', predecessor, self.counter])

    def genesis(self):
        self.counter = 0
        self.block_id = None
        raw = '''
        In God We Trust
        '''
        self.publicMining(raw)

    def run(self):
        rows = super.session.execute('''select sha256_id, pq, proposal, verdict, target, raw_statement, block_id
        from candidate where ready = true and processed = false''')
        candidates = []
        ids = []
        for row in rows:
            [sha256_id, pq, proposal, verdict, target, raw_statement, block_id] = row
            # verify the transaction sanity
            candidates.append('||'.join([pq, proposal, verdict, target, raw_statement, block_id]))
            ids.append(sha256_id)
        candidates.sort()
        candidate_transactions = '@@'.join(candidates)
        predecessor = self.block_id
        raw = '<{0}/{1}/{2}>{3}'.format(self.block_ptr, self.nonce, predecessor, candidate_transactions)
        self.publicMining(raw)

        for shaId in ids:
            super.session.execute(self.updateStatusStatement, [shaId])


'''
create table
player3(sha256_id text primary key, symbol text, ver bigint,
pq0 text, d0 text, f0 text, pq1 text ,d1 text, f1 text, setup timestamp);
'''


class player3(CassandraAware, KazooAware):

    def __init__(self):
        logging.info('initialize player3')
        self.newPlayer = super.session.prepare('''
        insert into player3(sha256_id, symbol, ver, pq0, d0, f0, pq1, d1, f1, setup)
        values(?, ?, ?, ?, ?, ?, ?, ?, ?, toTimestamp(now()))
        ''')

    def new(self, symbol):
        ver = super.ver
        super.ver += 1
        m = hashlib.sha256()
        m.update(bytes(symbol, 'utf-8'))
        sha256_id = m.hexdigest()
        numbers = []
        with Popen('./openssl genrsa 2048 {0}'.format(sha256_id).split(' '), stdout=PIPE) as p:
            output = str(p.stdout.read(), 'utf-8')
            for row in output.split('INTEGER'):
                numbers.extend(list(filter(lambda x: x.startswith('           :'), row.splitlines())))
        pqKey = ''.join(reversed(numbers[1])).lower().replace(':', '')
        dKey = ''.join(reversed(numbers[3])).lower().replace(':', '')
        jgKey = ''.join(reversed(numbers[-1])).lower().replace(':', '')
        pq0 = pqKey.strip()
        d0 = dKey.strip()
        f0 = jgKey.strip()

        with Popen('./openssl genrsa 2048 {0}'.format(sha256_id).split(' '), stdout=PIPE) as p:
            output = str(p.stdout.read(), 'utf-8')
            for row in output.split('INTEGER'):
                numbers.extend(list(filter(lambda x: x.startswith('           :'), row.splitlines())))
        pqKey = ''.join(reversed(numbers[1])).lower().replace(':', '')
        dKey = ''.join(reversed(numbers[3])).lower().replace(':', '')
        jgKey = ''.join(reversed(numbers[-1])).lower().replace(':', '')
        pq1 = pqKey.strip()
        d1 = dKey.strip()
        f1 = jgKey.strip()
        super.session.execute(self.newPlayer, [sha256_id, symbol, ver, pq0, d0, f0, pq1, d1, f1])


'''
create table draft(sha256_id text primary key, note_id text, target text, ver bigint,
symbol text, quantity bigint, refer text, processed boolean, setup timestamp);

create table symbol_chain(sha256_id primary key, symbol text, ver bigint,
block_counter bigint, updated timestamp);
'''


class draft(CassandraAware, KazooAware):

    def __init__(self):
        logging.info('initialize draft')
        self.newDraft = super.session.prepare('''
        insert into draft(sha256_id, note_id, target, ver, symbol, quantity, refer, type, processed, setup)
        values(?, ?, ?, ?, ?, ?, ?, ?, false, toTimestamp(now())
        ''')
        self.updateSymbolChain = super.session.prepare('''
        update symbol_chain set  symbol =? , ver= ?, block_counter= ?, updated = toTimestamp(now()
        where sha256_id = ?
        ''')

    def issue(self, symbol, quantity):
        logging.info('going to issue with symbol:{}'.format(symbol))
        ver = super.ver
        super.ver += 1
        m = hashlib.sha256()
        m.update(bytes(symbol, 'utf-8'))
        sha256_id = m.hexdigest()
        result = super.session.execute('select block_counter from symbol_chain where sha256_id = {0}'.format(sha256_id)).one()
        if not result:
            counter = 0
        else:
            counter = int(result.block_counter)
        counter += 1
        super.session.execute(self.updateSymbolChain, [symbol, ver, counter, sha256_id])
        (block_id) = super.session.execute('select sha256_id from blocks_view where counter = {}'.format(counter)).one()
        note_id = '{}||{}||{}'.format(symbol, block_id[:16], quantity)
        super.session.execute(self.newDraft, [m.hexdigest(), note_id[:32], sha256_id, ver, quantity, block_id, 'issue'])

    def transfer(self, note_id, target, quantity, refer):
        logging.info('going to transfer {} to {}'.format(note_id, target))
        ver = super.ver
        super.ver += 1
        m = hashlib.sha256()
        m.update(bytes(note_id, 'utf-8'))
        m.update(bytes(ver, 'utf-8'))
        sha256_id = m.hexdigest()
        super.session.execute(self.newDraft, [sha256_id, note_id, target, ver, quantity, refer, 'transfer'])


class proposal(CassandraAware, KazooAware):

    def __init__(self):
        logging.info('initialize proposal')

    def runLoop(self):
        while True:
            self.process()

    def process(self):
        result = super.session.execute('''
        select sha256_id, note_id, target, symbol, quantity, refer from draft where processed=false
        ''')
        for row in result:
            [sha256_id, note_id, target, symbol, quantity, refer, type] = row
            if type == 'issue':
                self.processIssue()
            if type == 'transfer':
                self.processTransfer()
            super.seesion.execute('''update draft set processed=true where sha256_id = {0}
            '''.format(sha256_id))

    def processIssue(self):
        # insert into candidate
        pass

    def processTransfer(self):
        # insert into candidate
        pass


if __name__ == '__main__':
    b = blocks_view()
    player = player3()
    d = draft()
    prop = proposal()
    Process(target=b.runLoop).start()
    Process(target=player.new('ABCDEFG')).start()
    Process(target=d.issue('ABCDEFG', '1')).start()
    Process(target=prop.runLoop).start()
