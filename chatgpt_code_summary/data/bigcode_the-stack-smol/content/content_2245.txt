import frappe, re
from renovation_service_provider_manager import invoke_mediator

@frappe.whitelist(allow_guest=True)
def get_service_provider_client_id(provider):
  k = f"client_id_{re.sub('[^0-9a-zA-Z]+', '_', provider.lower())}"
  client_id = frappe.cache().get_value(k)
  if client_id:
    return client_id

  client_id = get_client_id_from_mediator(provider)
  frappe.cache().set_value(k, client_id, expires_in_sec=18000) # 5hr
  return client_id

def get_client_id_from_mediator(provider):
  try:
    r = invoke_mediator("/api/method/renovation_mediator.api.get_service_provider_client_id", {"provider": provider})
    r.raise_for_status()

    r = r.json()
    return r["message"]
  except:
    frappe.throw(r.text)