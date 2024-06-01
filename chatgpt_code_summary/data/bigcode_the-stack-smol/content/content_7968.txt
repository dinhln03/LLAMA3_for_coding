from typing import List

from scripter.backend.note_text import NoteText
from scripter.io.writer_base import WriterBase

class FormatWriter(WriterBase):
    def __init__(self, *, format:str=None, **kwargs):
        super().__init__()
        if format is None:
            self.format = 'P.{page}\n{text}\n'

    def dump(self, fp, texts: List[NoteText]):
        for text in texts:
            fp.write(self.format.format(
                page=text.page,
                text=text.text
            ))
