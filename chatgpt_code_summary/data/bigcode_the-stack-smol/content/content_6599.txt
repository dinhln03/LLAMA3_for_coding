from flask import (Blueprint, Response, request, render_template)
import json
from flask_test import db
from flask_test.schema import FORM_SCHEMA
from flask_test.utils import row_as_json, list_as_json

bp = Blueprint('list', __name__)

@bp.route('/resource/<doc_type>', methods=['GET'])
def get(doc_type):
    data = db.get_list(doc_type)

    if 'application/json' in request.headers.get('accept'):
        response = json.dumps(list_as_json(data))
        return Response(response, mimetype="application/json")

    return render_template(
        'list.html',
        items=data,
        doc_type=doc_type,
        fields=FORM_SCHEMA[doc_type],
        get_options=db.get_list
    )
