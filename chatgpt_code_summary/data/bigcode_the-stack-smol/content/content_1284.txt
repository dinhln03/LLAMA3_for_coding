from argparse import ArgumentParser
import datetime
import dateutil
import sys, re
from os import path

def parseArgs():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("-a", "--action", help="Please select an option out of <discover, manage, settings>", type=str, required=True)
    parser.add_argument("-f", "--file", help="Please specify absolute path to initial dataset", type=str)
    args = parser.parse_args()
    
    # for debugging TODO: remove later
    args.file = r"C:\Users\flietz\OneDrive - TU Wien\!Studium\1_MSc\!Diplomarbeit\code\pipeline\resources\dataset\Mail_ApplicationDummy.csv"
    if args.action is None or args.action not in ("discover", "manage", "settings"):
        sys.exit('Please specify an action out of <"discover", "manager", "settings">')

    if args.action == "discover" and (args.file is None or not path.exists(args.file)):
        sys.exit("The input file could not be found in the filesystem.")

    arguments = {"file": args.file}
    return args.action, arguments

class DataCleaner:
    def __init__(self, removeURLs, removeMultWhitespace, lowercasing, dateFormat):
        self.removeURLs = removeURLs
        self.removeMultWhitespace = removeMultWhitespace
        self.lowercasing = lowercasing
        self.dateFormat = dateFormat

    def apply(self, inputDf):
        def removeUrl(content):
            return re.sub(r'https?://\S+', '', content)
        def removeMultWhitespace(content):
            return re.sub(r' +', ' ', content)
        # Remove URLs
        if self.removeURLs:
            inputDf["Content"] = inputDf.apply(lambda row: removeUrl(row["Content"]), axis=1)
        # Remove Multi-Whitespaces
        if self.removeMultWhitespace:
            inputDf["Content"] = inputDf.apply(lambda row: removeMultWhitespace(row["Content"]), axis=1)
        if self.lowercasing:
            inputDf["Content"] = inputDf.apply(lambda row: row["Content"].lower(), axis=1)
        # Not-Empty-Constraints
        if inputDf["Content"].isnull().values.any() or \
            inputDf["Datetime"].isnull().values.any() or \
            inputDf["From"].isnull().values.any() or \
            inputDf["To"].isnull().values.any():
            raise AttributeError("Content, Datetime, From and To field cannot be empty. Please check your input dataset.") 
        # Unify Date format - reformat to %Y-%m-%d %H:%M:%S
        def reformatDate(datestring, dateformat):
            try:
                newDate = dateutil.parser.parse(datestring, dayfirst=True)
                return newDate.strftime(dateformat)
            except ValueError as e:
                raise ValueError("Make sure that all datetime columns are well-formatted "
                "and that they contain dates that are within the possible bounds.") from e
        
        inputDf["Datetime"] = inputDf.apply(lambda row: reformatDate(row["Datetime"], self.dateFormat), axis=1)
        # clean signatures, clauses
        def stripEndClauses(content, clauses):
            clauseIndex = 0
            index = 0
            # Find lowest greetings or end clause index and strip off everything that comes after it
            for item in clauses:
                # needle and haystack both in lowercase to ignore case
                index = content.lower().find(item.lower())
                if index > -1 and (index < clauseIndex or clauseIndex == 0):
                    clauseIndex = index
            if clauseIndex > 0:
                return content[:clauseIndex]
            else:
                return content
        
        def stripStartClauses(content, clauses):
            clauseIndex = 0
            index = 0
            # Find lowest greetings or end clause index and strip off everything that comes after it
            for item in clauses:
                # needle and haystack both in lowercase to ignore case
                index = content.lower().find(item.lower())
                if index > -1 and (index > clauseIndex or clauseIndex == 0):
                    clauseIndex = index
            if clauseIndex > 0:
                return content[clauseIndex:]
            else:
                return content
        
        startClausesList = []
        
        endGreetingsList = ["Yours sincerely", "Sincerely", "Sincerely yours", "Take care", "Regards",
                 "Warm regards", "Best regards", "Kind regards", "Warmest regards", "Yours truly", "Yours,",
                 "Warmly,", "Warm wishes", "Best,", "Best Wishes", "Thanks in advance", "Thank you in advance",
                 "Thanks in advance"]

        confList = ["The information contained in this communication",
                    "The content of this email is confidential", "The content of this e-mail", "This email and attachments (if any) is intended",
                    "This email is intended solely", "This e-mail is intended solely"]

        endClausesList = endGreetingsList+confList

        inputDf["Content"] = inputDf.apply(lambda row: stripEndClauses(row["Content"], endClausesList), axis=1)    
        inputDf["Content"] = inputDf.apply(lambda row: stripStartClauses(row["Content"], startClausesList), axis=1)    

        # Reduce multiple new-lines to one
        inputDf["Content"] = inputDf.apply(lambda row: re.sub(r'\n+', '\n', row["Content"]), axis=1)
        # Replace new-lines with whitespaces
        inputDf["Content"] = inputDf.apply(lambda row: re.sub(r'\n', ' ', row["Content"]), axis=1)

def convertDateString(datestring):
    try:
        return datetime.datetime.strptime(datestring, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.datetime.strptime(datestring, "%Y-%m-%d %H:%M:%S")