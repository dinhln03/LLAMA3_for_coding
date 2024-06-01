from krgram.tl.core_types.native import TL_string
from krgram.tl.base import *


class getConfig(TLFunction):
	ID = 0xc4f9186b


TLRegister.register(getConfig)


class getNearestDc(TLFunction):
	ID = 0x1fb33026


TLRegister.register(getNearestDc)


class getAppUpdate(TLFunction):
	ID = 0xc812ac7e

	def get_structure(self):
		return ("device_model", TL_string()), ("system_version", TL_string()), \
			   ("app_version", TL_string()), ("lang_code", TL_string()),


TLRegister.register(getAppUpdate)


class saveAppLog(TLFunction):
	ID = 0x6f02f748

	def get_structure(self):
		return ("events", Vector()),


TLRegister.register(saveAppLog)


class getInviteText(TLFunction):
	ID = 0xa4a95186

	def get_structure(self):
		return ("lang_code", TL_string()),


TLRegister.register(getInviteText)


class getSupport(TLFunction):
	ID = 0x9cdf08cd


TLRegister.register(getSupport)
