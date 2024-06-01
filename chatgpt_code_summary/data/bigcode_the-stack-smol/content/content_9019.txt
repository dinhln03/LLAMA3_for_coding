from fastapi import APIRouter

from modules.core.converter import Converter
from modules.schemas.converter import ConverterSchema

converter = APIRouter()


@converter.post('/convert', tags=['converter'])
def convert_document(convert: ConverterSchema):
    result = Converter.convert(convert.source_format, convert.target_format, convert.content)
    return dict(status="converted", result_format=convert.target_format, result=result)
