# Copyright (c) 2013, RC and contributors
# For license information, please see license.txt

import frappe
from frappe import _

def execute(filters=None):
	columns = get_columns()
	data = get_data(filters)
	return columns, data

def get_columns():
	return [
		{
			"fieldname": "po_number",
			"fieldtype": "Data",
			"label": "Po Number",
			"width": 120
		},
		{
			"fieldname": "ordered_qty",
			"fieldtype": "Float",
			"label": "Ordered Qty",
			"width": 150
		},
		{
			"fieldname": "received_qty",
			"fieldtype": "Float",
			"label": "Received Qty",
			"width": 150
		},
		{
			"fieldname": "pending_qty",
			"fieldtype": "Float",
			"label": "Pending Qty",
			"width": 150
		}
	]

def get_data(filters):
	if not filters.get('company'):
		frappe.throw(_("Select Company!"))

	if not filters.get('from_date'):
		frappe.throw(_("Select From Date!"))

	if not filters.get('to_date'):
		frappe.throw(_("Select To Date!"))

	query = """select po_number, sum(cust_total_box) as order_qty from `tabPurchase Order` 
				where company = '{0}' and transaction_date between '{1}' and '{2}' 
				and po_number is not null and po_number != 'PENDING' 
				and docstatus = 1""".format(filters.get('company'),filters.get('from_date'),filters.get('to_date'))

	if filters.get('supplier'):
		query += " and supplier = '{0}'".format(filters.get('supplier'))

	query += " group by po_number"

	po = frappe.db.sql(query, as_dict=True)
	data = []
	for res in po:
		query1 = """select sum(boxes) from `tabPurchase Invoice` as pi 
					inner join `tabPurchase Invoice Item` as pii on pii.parent = pi.name 
					where company = '{0}' and pi.posting_date between '{1}' and '{2}' 
					and pi.po_number = '{3}' 
					and pi.docstatus = 1""".format(filters.get('company'), filters.get('from_date'),
											filters.get('to_date'), res.po_number)

		if filters.get('supplier'):
			query1 += " and pi.supplier = '{0}'".format(filters.get('supplier'))

		pi = float(frappe.db.sql(query1)[0][0] or 0)
		data.append(frappe._dict({
			"po_number": res.po_number,
			"ordered_qty": res.order_qty,
			"received_qty": pi,
			"pending_qty": res.order_qty - pi
		}))
	return data
