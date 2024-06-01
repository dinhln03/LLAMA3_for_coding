from rate_limiter.Limit import Limit
from rate_limiter.Parser import Parser
from rate_limiter.LimitProcessor import LimitProcessor
from rate_limiter.LogLine import LogLine
from typing import Dict, List
from datetime import datetime


class IpRateLimiter:
    # used to store unban time. Also used to maintain what is currently banned
    ipToUnbanTimeMap: List
    logLineList: List[LogLine]
    limitProcessorList: List[LimitProcessor]

    def __init__(self, logLineList, limitProcessorList) -> None:
        self.logLineList = logLineList
        self.limitProcessorList = limitProcessorList
        self.ipToUnbanTimeMap = {}

    def processNewBan(self, newIpToUnbanTimeMap, currTime):
        for ip in newIpToUnbanTimeMap:
            if ip not in self.ipToUnbanTimeMap:
                # new ban. Need to print
                print("{0},BAN,{1}".format(int(currTime.timestamp()), ip))
                self.ipToUnbanTimeMap[ip] = newIpToUnbanTimeMap[ip]
            else:
                self.ipToUnbanTimeMap[ip] = max(
                    self.ipToUnbanTimeMap[ip], newIpToUnbanTimeMap[ip]
                )

    def needsToBeUnbanned(self, ip: str, currTime: datetime):
        toUnban = False
        if currTime >= self.ipToUnbanTimeMap[ip]:
            toUnban = True
        return toUnban

    def unbanPassedIPs(self, currTime: datetime):
        toUnban = []
        for ip in self.ipToUnbanTimeMap:
            if self.needsToBeUnbanned(ip, currTime):
                toUnban.append(ip)

        for ip in toUnban:
            print("{0},UNBAN,{1}".format(int(currTime.timestamp()), ip))
            # print("{0},UNBAN,{1}".format(self.ipToUnbanTimeMap[ip].timestamp(), ip))
            self.ipToUnbanTimeMap.pop(ip)

    def run(self):
        for line in self.logLineList:

            currTime = line.getTime()

            # evict expired entries from each processor window
            for limitProcessor in self.limitProcessorList:
                limitProcessor.evictExpired(currTime)

            # check all banned ips if they need to be unbanned
            self.unbanPassedIPs(currTime)

            # process new request in limit processors
            for limitProcessor in self.limitProcessorList:
                newBanMap = limitProcessor.processNewRequest(line)
                self.processNewBan(newBanMap, currTime)

        if self.logLineList and self.ipToUnbanTimeMap:
            for ip in self.ipToUnbanTimeMap:
                print(
                    "{0},UNBAN,{1}".format(
                        int(self.ipToUnbanTimeMap[ip].timestamp()), ip
                    )
                )


class IpRateLimiterBuilder:
    filePath = ""
    fileParser: Parser = None
    limitList: List[Limit] = []

    def addFile(self, parser: Parser, filePath: str):
        self.filePath = filePath
        self.fileParser = parser
        return self

    def addLimit(self, limit: Limit):
        self.limitList.append(limit)
        return self

    def build(self):
        logLineList = self.fileParser.parse(self.filePath)
        limitProcessorList: List[LimitProcessor] = []
        for limit in self.limitList:
            limitProcessorList.append(LimitProcessor(limit))
        return IpRateLimiter(logLineList, limitProcessorList)
