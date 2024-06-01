import os
import json
import logging
logger = logging.getLogger(__name__)

from aiohttp import web, ClientSession

async def index(request):
    logger.debug('Accessing index')
    client = request.app['arango']
    sys_db = client.db('_system', username='root', password=os.environ['MULTINET_ROOT_PASSWORD'])
    dbs = sys_db.databases()
    logger.info('Response: %s' % dbs)
    return web.Response(text=json.dumps(dbs, indent=4))

async def addDB(request):
    logger.debug('Adding DB')
    client = request.app['arango']
    sys_db = client.db('_system', username='root', password=os.environ['MULTINET_ROOT_PASSWORD'])

    name = request.match_info['name']
    if not sys_db.has_database(name):
        sys_db.create_database(name)
    else:
        logger.info('Request to add db {} is a no-op because database is already present'.format(name))

    return web.Response(text=name)

async def getDB(request):
    logger.debug('Getting DB')
    client = request.app['arango']
    db = client.db(request.match_info['name'], username='root', password=os.environ['MULTINET_ROOT_PASSWORD'])
    graphs = [coll for coll in db.graphs() if not coll['name'].startswith('_')]
    return web.Response(text=json.dumps(graphs, indent=4))

async def getGraph(request):
    logger.debug('Getting Graph')
    client = request.app['arango']
    db = client.db(request.match_info['db_name'], username='root', password=os.environ['MULTINET_ROOT_PASSWORD'])
    graph = db.graph(request.match_info['name'])
    vertex_collections = graph.vertex_collections()
    edge_definitions = graph.edge_definitions()
    return web.Response(text=json.dumps(
        {
            "vertex_collections": vertex_collections,
            "edge_definitions": edge_definitions
        },
        indent=4
    ))

async def addGraph(request):
    logger.debug('Adding Graph')
    client = request.app['arango']
    db = client.db(request.match_info['db_name'], username='root', password=os.environ['MULTINET_ROOT_PASSWORD'])
    name = request.match_info['name']
    graph = db.graph(name) if db.has_graph(name) else db.create_graph(name)

    return web.Response(text=graph.name)

async def addVertices(request):
    logger.debug('Adding Vertices')
    client = request.app['arango']
    db = client.db(request.match_info['db_name'], username='root', password=os.environ['MULTINET_ROOT_PASSWORD'])
    graph = db.graph(request.match_info['graph_name'])
    name = request.match_info['name']
    collection = graph.vertex_collection(name) if graph.has_vertex_collection(name) else graph.create_vertex_collection(name)

    reader = await request.multipart()
    import_file = await reader.next()
    logger.info(import_file.filename)
    filedata = await import_file.text()

    fileschema = [key.strip('"') for key in filedata.splitlines()[0].split(',')]
    logger.info(fileschema)
    filelines = filedata.splitlines()[1:]
    for line in filelines:
        values = [value.strip('"') for value in line.split(',')]
        doc = {key:value for key, value in zip(fileschema, values)}
        try:
            collection.insert(doc)
        except Exception as e:
            logger.info(e)

    return web.Response(text=collection.name)

async def getVertices(request):
    logger.debug('Getting Vertices')
    client = request.app['arango']
    db = client.db(request.match_info['db_name'], username='root', password=os.environ['MULTINET_ROOT_PASSWORD'])
    graph = db.graph(request.match_info['graph_name'])
    collection = db.collection(request.match_info['name'])
    cursor = collection.all()
    documents = [doc for doc in cursor]

    return web.Response(text=json.dumps(documents[0:5], indent=4))

async def addEdges(request):
    logger.debug('Adding Edges')
    client = request.app['arango']
    db = client.db(request.match_info['db_name'], username='root', password=os.environ['MULTINET_ROOT_PASSWORD'])
    graph = db.graph(request.match_info['graph_name'])
    name = request.match_info['name']

    reader = await request.multipart()
    field = await reader.next()
    text = await field.text()
    from_collections = text.split(',')

    field = await reader.next()
    text = await field.text()
    to_collections = text.split(',')

    if graph.has_edge_definition(name):
        collection = graph.edge_collection(name)
    else:
        collection = graph.create_edge_definition(
            edge_collection=name,
            from_vertex_collections=from_collections,
            to_vertex_collections=to_collections)

    import_file = await reader.next()
    filedata = await import_file.text()

    fileschema = [key.strip('"') for key in filedata.splitlines()[0].split(',')]
    filelines = filedata.splitlines()[1:]
    for line in filelines:
        values = [value.strip('"') for value in line.split(',')]
        doc = {key:value for key, value in zip(fileschema, values)}
        try:
            collection.insert(doc)
        except Exception as e:
            logger.info(e)

    return web.Response(text=collection.name)

async def getEdges(request):
    logger.debug('Getting Edges')
    client = request.app['arango']
    db = client.db(request.match_info['db_name'], username='root', password=os.environ['MULTINET_ROOT_PASSWORD'])
    graph = db.graph(request.match_info['graph_name'])
    collection = graph.edge_collection(request.match_info['name'])
    cursor = collection.all()
    documents = [doc for doc in cursor]

    return web.Response(text=json.dumps(documents[0:5], indent=4))
