# third-party
from flask import render_template, url_for, request, jsonify

# locals
from . import warehouse

@warehouse.route('/element_types', methods=['GET'])
def index():
    return render_template("warehouse/element_types.html")

@warehouse.route('/element_type', methods=['POST'])
def create_new_element_type():
    print(request.__dict__)
    print(request.data)
    print(request.get_json())
    return jsonify({
        "success": True
    })

# @warehouse.route('/element_type', methods=['GET'])
# @warehouse.route('/element_type/<element_type_id>', methods=['GET'])
# def element_type(element_type_id=None):
#     pass

# @warehouse.route('/element_type', methods=['POST'])
# def new_element_type()
