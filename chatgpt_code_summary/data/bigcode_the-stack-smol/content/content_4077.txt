import os
import logging
import json
from typing import Union, Dict, List

from documentstore_migracao.utils.isis2json import isis2json

logger = logging.getLogger(__name__)


class OutputContainer:
    """Classe que mimetiza a escrita de arquivos para a escrita em uma estrutura
    de lista. Cada linha em um arquivo representa uma entrada na lista."""

    def __init__(self):
        self._lines = []

    def write(self, string: str) -> None:
        try:
            _string = json.loads(string)
        except Exception:
            pass
        else:
            self._lines.append(_string)

    def close(self):
        pass

    @property
    def lines(self):
        return self._lines


def create_output_dir(path):
    output_dir = "/".join(path.split("/")[:-1])

    if not os.path.exists(output_dir):
        logger.debug("Creating folder: %s", output_dir)
        os.makedirs(output_dir)


def run(path: str, output_file: str = "", mongo=False) -> Union[None, List[dict]]:
    """Invoca o utilitário `isis2json` com os parâmetros adaptados para a
    leitura de arquivos MST de acordo com as definições padrões utilizadas
    pelo __main__ da ferramenta `isis2json`.

    O resultado de saída pode ser escrito diretamente para um arquivo em disco
    ou retornará uma lista contento as linhas passíveis de conversão para
    JSON.

    Exemplo:
    >>> run("file.mst")
    >>> [{"mfn": 1}, {"mfn": 2}]

    >>> run("file.mst", output_file="/tmp/output.json")
    >>> None
    """

    if not os.path.exists(path):
        raise FileNotFoundError("File '%s' does not exist.")

    if len(output_file) > 0:
        output_file = open(output_file, "wb")
    else:
        output_file = OutputContainer()

    isis2json.writeJsonArray(
        iterRecords=isis2json.iterMstRecords,
        file_name=path,
        output=output_file,
        qty=isis2json.DEFAULT_QTY,
        skip=0,
        id_tag=0,
        gen_uuid=False,
        mongo=mongo,
        mfn=True,
        isis_json_type=3,
        prefix="v",
        constant="",
    )

    output_file.close()

    if isinstance(output_file, OutputContainer):
        return output_file.lines
