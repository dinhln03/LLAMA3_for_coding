# Copyright 2020 XEBIALABS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# https://stackoverflow.com/questions/16910955/programmatically-configure-logback-appender?noredirect=1
#
import ch.qos.logback.core.Appender as LogAppender
import ch.qos.logback.core.util.COWArrayList as COWArrayList
import ch.qos.logback.classic.encoder.PatternLayoutEncoder as PatternLayoutEncoder
import ch.qos.logback.core.FileAppender as FileAppender

import org.slf4j.LoggerFactory as LoggerFactory
import ch.qos.logback.classic.Level as logLevels
import json

def getLogAppenders( loggerName="console" ):
    loggerMap = []
    myLogger = LoggerFactory.getLogger("logmanager")
    loggerContext = LoggerFactory.getILoggerFactory()

    myLogger.error("===================")
    appenderMap = {}
    for logger in loggerContext.getLoggerList():
        appenderList = logger.iteratorForAppenders()
        while appenderList.hasNext():
            appender = appenderList.next()
            logger.error("Logger %s" % appender.getName())
            if appender.getName() not in appenderMap.keys():
                loggerMap.append({"name": appender.getName(), "appender": "NA"})
                myLogger.error("Appender %s: %s" % (appender.getName(), "NA"))
    myLogger.error("===================")
    return loggerMap

def createLogAppender( name, file ):
    lc = LoggerFactory.getILoggerFactory()
    ple = PatternLayoutEncoder()
    ple.setPattern("%date %level [%thread] %logger{10} [%file:%line] %msg%n")
    ple.setContext(lc)
    ple.start()
    fileAppender = FileAppender()
    fileAppender.setFile(file)
    fileAppender.setEncoder(ple)
    fileAppender.setContext(lc)
    fileAppender.start()

    logger = LoggerFactory.getLogger(string)
    logger.addAppender(fileAppender)
    #logger.setLevel(logLevels.DEBUG)
    # set to true if root should log too
    logger.setAdditive(True)
    return logger


myLogger = LoggerFactory.getLogger("logmanager")
verb = "GET"

if (request):
    if (request.query):
        if (request.query['verb']):
            verb = request.query['verb']

if( verb == "create"):
    string = request.query['string']
    file = request.query['file']
    myLogger.info("Setting %s to %s" % (string, file))
    createLogAppender(string, file)

loggerMap = getLogAppenders()
myLogger.error("%s" % json.dumps(loggerMap, indent=4, sort_keys=True))

response.entity = {"status": "OK", "data":loggerMap }
