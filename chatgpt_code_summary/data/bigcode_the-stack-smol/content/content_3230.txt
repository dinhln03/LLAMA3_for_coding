from unittest.mock import call

from sls.completion.complete import Completion
from sls.completion.context import CompletionContext
from sls.document import Document
import sls.sentry as sentry


def test_complete(magic, patch):
    patch.init(Document)
    patch.many(Document, ["line_to_cursor", "word_to_cursor"])
    patch.many(CompletionContext, ["_blocks"])
    cache = magic()
    c = Completion(plugins=[], context_cache=cache)
    doc = Document()
    ws = magic()
    pos = magic()
    result = c.complete(ws, doc, pos)
    assert isinstance(cache.update.call_args[0][0], CompletionContext)
    assert result == {
        "isIncomplete": False,
        "items": [],
    }


def test_complete_plugin(magic, patch):
    patch.init(Document)
    patch.many(Document, ["line_to_cursor", "word_to_cursor"])
    patch.many(CompletionContext, ["_blocks"])
    my_plugin = magic()
    i1 = {"label": "i1"}
    i2 = {"label": "i2"}
    cache = magic()
    my_plugin.complete.return_value = [i1, i2]
    c = Completion(plugins=[my_plugin], context_cache=cache)
    doc = Document()
    ws = magic()
    pos = magic()
    result = c.complete(ws, doc, pos)
    assert isinstance(my_plugin.complete.call_args[0][0], CompletionContext)
    assert result == {
        "isIncomplete": False,
        "items": [i1, i2],
    }


def test_complete_exec(magic, patch):
    patch.init(Document)
    patch.many(Document, ["line_to_cursor", "word_to_cursor"])
    patch.many(CompletionContext, ["_blocks"])
    patch.object(sentry, "handle_exception")
    cache = magic()
    plugin = magic()
    ex = Exception("e")
    plugin.complete.side_effect = ex
    c = Completion(plugins=[plugin], context_cache=cache)
    doc = Document()
    ws = magic()
    pos = magic()
    result = c.complete(ws, doc, pos)
    assert isinstance(cache.update.call_args[0][0], CompletionContext)
    assert result == {
        "isIncomplete": False,
        "items": [],
    }
    assert sentry.handle_exception.call_args == call(ex)
