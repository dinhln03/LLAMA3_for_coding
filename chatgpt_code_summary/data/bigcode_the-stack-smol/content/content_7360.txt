#-*- coding: utf-8 -*-


from DBP.models import Base, session
from DBP.models.user import User
from sqlalchemy.orm import class_mapper
from sqlalchemy.inspection import inspect
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import INTEGER,VARCHAR, DATETIME
from datetime import datetime
import csv
import io

from openpyxl import Workbook
from openpyxl import load_workbook

def utf_8_encoder(unicode_csv_data):
	for line in unicode_csv_data:
		yield line.encode('utf-8')

class OriginalData (object):

	


	def __init__(self, length, name, mappinginfo):
		self.length = length
		self.name  = name
		cols = inspect(self.__class__).columns
		if len(mappinginfo) != len(cols) -3:
			raise TypeError

		for col in mappinginfo:
			setattr(self,str( u"sch_"+col["label"]["name"]),int(col["col"]))


	def dict(self):
		data = {
			"id" : self.id,
			"length" : self.length,
			"name" : self.name,
			"mapinfo" : self.mapList()
		}
		return data

	def getInfo(self):
		data = self.dict()
		data["parsednum"] = len(self.parseds)
		data["tasknum"] = sum(map(lambda x: len(x.tasks),self.parseds))
		
		return data

	def mapList(self):
		maplist = list()
		for col in filter(lambda x: x.name[:3] == u"sch", inspect(self.__class__).columns ):
			maplist.append(getattr(self,col.name))
		return maplist

	def getSchema(self):
		return filter(lambda x: x.name[:3] == u"sch", inspect(self.__class__).columns )


	def loadcsv(self,submitter,csvread,nth,duration_start,duration_end):
		reader = csv.reader(csvread, delimiter=',', quotechar="'")
		csvwrite = io.BytesIO()
		writer =  csv.writer(csvwrite, delimiter=',', quotechar="'")
		maplist = self.mapList()
		counter = 0
		dupset = set()
		dupcounter = 0
		nullcount = dict()
		schema = self.getSchema()
		for col in schema:
			nullcount[col.name] = 0


		for rrow in reader:
			crow = list()
			for mapnum, col in zip(maplist, schema):
				crow.append(rrow[mapnum])

				if rrow[mapnum] == "":
					nullcount[col.name] +=1



			dupset.add(unicode(crow))
			writer.writerow(crow)
			counter += 1


		evaluator = User.randomEvaluator()
		
		parsedmodel =  self.parsedclass(nth,duration_start,duration_end,csvwrite,counter, counter - len(dupset))
		parsedmodel.submitterid = submitter.id
		parsedmodel.evaluatorid = evaluator.id
		self.taskrow.addUser(evaluator)
		for col in schema :
			setattr(parsedmodel,"null_" + col.name[4:] , nullcount[col.name] / (counter*1.0) )


		self.parseds.append(parsedmodel)

		session.commit()

		return parsedmodel

	def loadxlsx(self,submitter,xlsxread,nth,duration_start,duration_end):
		wb = load_workbook(xlsxread)
		ws = wb.active
		csvwrite = io.BytesIO()
		writer =  csv.writer(csvwrite, delimiter=',', quotechar="'")
		maplist = self.mapList()
		counter = 0
		dupset = set()
		dupcounter = 0
		nullcount = dict()
		schema = self.getSchema()
		for col in schema:
			nullcount[col.name] = 0


		for rrow in ws.rows:
			crow = list()
			for mapnum, col in zip(maplist, schema):
				if type(rrow[mapnum].value) == datetime:
					crow.append(rrow[mapnum].value.strftime("%Y-%m-%d %H:%M"))
				else :
					crow.append(rrow[mapnum].value)

				if rrow[mapnum].value == "":
					nullcount[col.name] +=1



			dupset.add(unicode(crow))

			utfrow = list ()
			for x in crow:
				if type(x) == unicode :
					utfrow.append(x.encode("utf8"))

				else :
					utfrow.append(x)
			writer.writerow(utfrow)
			counter += 1


		evaluator = User.randomEvaluator()
		
		parsedmodel =  self.parsedclass(nth,duration_start,duration_end,csvwrite,counter, counter - len(dupset))
		parsedmodel.submitterid = submitter.id
		parsedmodel.evaluatorid = evaluator.id

		self.taskrow.addUser(evaluator)
		

		for col in schema :
			setattr(parsedmodel,"null_" + col.name[4:] , nullcount[col.name] / (counter*1.0) )


		self.parseds.append(parsedmodel)

		session.commit()

		return parsedmodel

	def getInfoByUser(self,user):
		data = self.dict()
		data["nth"] = self.getNextnth (user)


		return data


	def getNextnth(self,user):
		nth =  session.query( func.max(self.parsedclass.nth)).filter(self.parsedclass.originalid == self.id).filter(self.parsedclass.submitterid == user.id).first()
		if nth[0]:
			return nth[0] +1
		else :
			return 1

	



class ParsedData (object):

	def __init__(self,nth,duration_start,duration_end, csvfile, tuplenum,duplicatetuplenum):
		self.nth = nth
		self.duration_start = duration_start
		self.duration_end = duration_end
		self.file = csvfile.getvalue()
		self.tuplenum = tuplenum
		self.duplicatetuplenum = duplicatetuplenum


	def parsecsv(self):
		csvread = io.StringIO(self.file.decode("utf8"))
		reader = csv.reader(utf_8_encoder(csvread), delimiter=',', quotechar="'")
		parsedlist = list()
		for row in reader:
			tsmodel = self.taskclass(User.getUser(self.submitterid).name, self.id)
			for (column, data) in zip(filter(lambda x: x.name[:3] == u"sch", inspect(self.taskclass).columns ), row):

				if type(column.type) == INTEGER:
					try :
						setattr(tsmodel,column.name, int(data))
					except :
						setattr(tsmodel,column.name, None)

				elif type(column.type) == DATETIME:
					try :
						setattr(tsmodel,column.name, datetime.strptime( data, "%Y-%m-%d %H:%M"))
					except :
						setattr(tsmodel,column.name, None)
	
				else :
					setattr(tsmodel,column.name, data)
			parsedlist.append(tsmodel)

		return parsedlist


	def insertcsv(self):
		if self.pnp != "Pass":
			return False

		session.bulk_save_objects(self.parsecsv())
		session.commit()
		return True


	def dict(self):

		return {
			"id" : self.id,
			"nth" : self.nth,
			"tuplenum" : self.tuplenum,
			"duplicatetuplenum" : self.duplicatetuplenum,
			"duration_start" : self.duration_start.isoformat(),
			"duration_end" : self.duration_end.isoformat(),
			"status" : self.status,
			"score" : self.score,
			"pnp" : self.pnp,
			"submitter" : User.getUser(self.submitterid).name,
			"original" : self.original.name,
			"evaluator": User.getUser(self.evaluatorid).name,
			"nullratio" : self.nullInfo()

		}


	def evaluate(self, score,pnp):
		self.status = "Evaluated"
		self.score = 5 * score + 25 *( 1.0 - self.duplicatetuplenum/(self.tuplenum * 1.0) ) + 25 * (1.0 - sum(map(lambda x : x['ratio'] ,self.nullInfo()))/(len(self.nullInfo())*1.0))
		self.pnp = pnp

		
		session.commit()

	def nullInfo(self):
		nulllist = list()
		for col in filter(lambda x: x.name[:4] == u"null", inspect(self.__class__).columns ):
			nulllist.append(dict(ratio=getattr(self,col.name) ,name =  col.name[5:] ))

		return nulllist

class TaskData (object):
	
	def __init__ (self,submittername, parsedid):
		self.submittername = submittername
		self.parsedid = parsedid


