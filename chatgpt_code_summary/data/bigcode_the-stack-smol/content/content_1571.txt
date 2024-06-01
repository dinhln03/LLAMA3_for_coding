class Doer(object):

    def __init__(self, frontend):
        self.__frontend = frontend

    async def do(self, action):
        return await self.__frontend.do(action)