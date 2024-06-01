# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date:
# @note:
from ddlf.cluster import *

async def main():
    cluster = Cluster()
    await cluster.connect()
    await cluster.show_data()
    await cluster.clean()
    await cluster.show_data()
    await cluster.close()


asyncio.run(main())
