from ctypes import cdll, c_int, c_ulonglong, c_char_p

import time
import json

import thread

def runList(ll, interpID, inputList):
    strList = json.dumps(inputList)
    listID = ll.elconn_list_from_json(strList.encode())
    resultID = ll.elconn_call(interpID, listID)
    return resultID

# === load library
ll = cdll.LoadLibrary("../sharedlib/elconn.so")

# === set return types
ll.elconn_get_type.restype = c_int
ll.elconn_init.restype = c_ulonglong
ll.elconn_list_from_json.restype = c_ulonglong
ll.elconn_make_interpreter.restype = c_ulonglong
ll.elconn_call.restype = c_ulonglong
ll.elconn_connect_remote.restype = c_ulonglong
ll.elconn_list_strfirst.restype = c_char_p
ll.elconn_list_to_json.restype = c_char_p

# === set argument types
ll.elconn_list_from_json.argtypes = [c_char_p]
ll.elconn_serve_remote.argtypes = [c_char_p, c_ulonglong]


# == Manual Test 1 == Using the interpreter
initMsg = ll.elconn_init(0)
ll.elconn_display_info(initMsg)

testList = json.dumps(["format", "Hello, %s!", "World"])
listID = ll.elconn_list_from_json(testList.encode())
ll.elconn_list_print(listID)

interpID = ll.elconn_make_interpreter()
resultID = ll.elconn_call(interpID, listID)
ll.elconn_list_print(resultID)

# == Manual Test 2 == Connecting to remote interpreter
ll.elconn_serve_remote(b":3003", interpID)
time.sleep(1)
remoteID = ll.elconn_connect_remote(b"http://localhost:3003")
rResultID = ll.elconn_call(remoteID, listID)
ll.elconn_list_print(rResultID)

# == Manual Test 3 == Value on server
someList = json.dumps([":", "test-var", ["store", "hello", 1]])
listID = ll.elconn_list_from_json(someList.encode())
resultID = ll.elconn_call(interpID, listID)

someList = json.dumps(["format", "%s there %f", ["test-var"]])
listID = ll.elconn_list_from_json(someList.encode())
resultID = ll.elconn_call(remoteID, listID)

rResultID = ll.elconn_call(remoteID, listID)
ll.elconn_list_print(rResultID)

# == Manual Test 3 == Directory with value on server
runList(ll, interpID, [":", "test-map", ["@", "directory"]])
runList(ll, interpID, ["test-map", ":", "a", ["store", "test value 2"]])

resID = runList(ll, remoteID, ["test-map", "a"])
ll.elconn_list_print(resID)

# == Manual Test 5 == Request queue
runList(ll, interpID, ["test-map", ":", "b", ["@", "requests"]])
runList(ll, interpID, ["test-map", "b", "enque", "[\"some_json\"]"])
resID = runList(ll, remoteID, ["test-map", "b", "block"])
ll.elconn_list_print(resID)

# -- schedule something to be enqueued later
def do_the_thing(ll, interpID, item, delay):
    time.sleep(delay)
    runList(ll, interpID, ["test-map", "b", "enque", item])
thread.start_new_thread(do_the_thing, (ll, interpID, "test-thread", 4))

print("Wait 4 seconds...")
resID = runList(ll, remoteID, ["test-map", "b", "block"])
ll.elconn_list_print(resID)

# == Maual Test 6 == Getting values
someList = json.dumps(["format", "%s there %f", ["test-var"]])
listID = ll.elconn_list_from_json(someList.encode())
firstStr = ll.elconn_list_strfirst(listID)
print("firstStr = %s" % firstStr)
asJSON   = ll.elconn_list_to_json(listID)
print("asJSON = %s" % asJSON)
