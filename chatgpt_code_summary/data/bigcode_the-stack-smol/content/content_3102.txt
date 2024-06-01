from tests import create_rand


def prepare_database_with_table(name: str, rows: list):
    from peewee import IntegerField, Proxy, CharField, Model
    from playhouse.sqlite_ext import CSqliteExtDatabase

    db = Proxy()
    db.initialize(CSqliteExtDatabase(':memory:', bloomfilter=True))
    NameModel = type(name, (Model,), {
        'id_': IntegerField(primary_key=True, column_name='id'),
        'name': CharField(column_name='name')
    })
    table: Model = NameModel()
    table.bind(db)
    db.create_tables([NameModel])
    for row in rows:
        table.insert(row).execute()
    return db


def test_ds_list():
    from rand.providers.ds import RandDatasetBaseProvider, ListDatasetTarget

    db = {
        'names': [{'name': 'test1'}, {'name': 'test1'}],
        'cities': [{'name': 'test2'}, {'name': 'test2'}],
    }
    ds = RandDatasetBaseProvider(prefix='ds', target=ListDatasetTarget(db=db))
    rand = create_rand()
    rand.register_provider(ds)
    assert rand.gen('(:ds_get:)', ['names']) == ['test1']
    assert rand.gen('(:ds_get_names:)-(:ds_get_cities:)') == ['test1-test2']


def test_ds_db():
    from rand.providers.ds import RandDatasetBaseProvider, DBDatasetTarget

    rows = [{'name': 'test'}, {'name': 'test'}]
    db = prepare_database_with_table('names', rows)
    ds = RandDatasetBaseProvider(prefix='ds', target=DBDatasetTarget(db=db))
    rand = create_rand()
    rand.register_provider(ds)
    assert rand.gen('(:ds_get:)', ['names']) == ['test']
