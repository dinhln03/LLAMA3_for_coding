# -*- coding: utf-8 -*-
import six
from flask import Blueprint, jsonify, current_app

from ..utils import MountTree
from .utils import is_testing

api_bp = Blueprint('api', __name__.rsplit('.')[1])


if is_testing():
    @api_bp.route('/_hello/')
    def api_hello():
        return jsonify('api hello')


@api_bp.route('/all')
def all_storage():
    """Get all storage in JSON."""
    trees = current_app.trees
    mounts = MountTree()
    for prefix, tree in six.iteritems(trees):
        for path, storage in tree.iter_storage():
            mounts.mount(prefix + '/' + path, storage)

    # get a compressed representation of the tree
    def dfs(node):
        children = node.children
        if children:
            ret = []
            for name in sorted(six.iterkeys(children)):
                child = children[name]
                child_ret = dfs(child)
                if child_ret:
                    ret.append((name, child_ret))
            if ret:
                return ret
        data = node.data
        if data:
            return data.to_dict()

    return jsonify(dfs(mounts.root) or [])
