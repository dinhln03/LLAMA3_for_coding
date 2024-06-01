from __future__ import unicode_literals
import dataent
from dataent.model.rename_doc import rename_doc

def execute():
	if dataent.db.table_exists("Email Alert Recipient") and not dataent.db.table_exists("Notification Recipient"):
		rename_doc('DocType', 'Email Alert Recipient', 'Notification Recipient')
		dataent.reload_doc('email', 'doctype', 'notification_recipient')

	if dataent.db.table_exists("Email Alert") and not dataent.db.table_exists("Notification"):
		rename_doc('DocType', 'Email Alert', 'Notification')
		dataent.reload_doc('email', 'doctype', 'notification')
