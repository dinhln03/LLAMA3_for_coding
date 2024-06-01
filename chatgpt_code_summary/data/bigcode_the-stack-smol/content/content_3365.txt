import datetime
from datetime import datetime, timedelta
from time import sleep

from app.search import add_to_index, delete_index, create_index, query_index

from app import db
from app.models import Post, User
from tests.BaseDbTest import BaseDbTest


class SearchTest(BaseDbTest):
    index_name = "test_index"

    def setUp(self):
        super(SearchTest, self).setUp()
        create_index(SearchTest.index_name)

    def tearDown(self):
        super(SearchTest, self).tearDown()
        delete_index(SearchTest.index_name)


    def test_index_posts(self):
        # create four users
        u1 = User(username='john', email='john@example.com')
        u2 = User(username='susan', email='susan@example.com')

        db.session.add_all([u1, u2])

        # create four posts
        now = datetime.utcnow()
        p1 = Post(body="post post1 from john", author=u1,
                  timestamp=now + timedelta(seconds=1))
        p2 = Post(body="post post2 from susan", author=u2,
                  timestamp=now + timedelta(seconds=4))
        p3 = Post(body="post post3 from john", author=u1,
                  timestamp=now + timedelta(seconds=3))
        p4 = Post(body="post post4 from john", author=u1,
                  timestamp=now + timedelta(seconds=2))
        db.session.add_all([p1, p2, p3, p4])
        db.session.commit()

        add_to_index(SearchTest.index_name, p1)
        add_to_index(SearchTest.index_name, p2)
        add_to_index(SearchTest.index_name, p3)
        add_to_index(SearchTest.index_name, p4)

        sleep(1)

        ids, total = query_index(SearchTest.index_name, "post1", 1, 20)
        self.assertEqual(1, total)
        self.assertEqual(p1.id, ids[0])

        ids, total = query_index(SearchTest.index_name, "post", 1, 20)
        self.assertEqual(4, total)