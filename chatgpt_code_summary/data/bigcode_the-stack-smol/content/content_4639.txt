from typing import Any, Dict, Mapping, Optional, Set

from pydantic import validator

from transformer.transformers.abstract import ExtraHashableModel, Transformer
from transformer.transformers.flatters import Flatter, FlatterConfig, Unflatter


class ReportMissingData(Exception):
    def __init__(self, keys: Set[str]):
        self.keys = keys
        self.message = f"The keys f{self.keys} are missing in the payload."


class MapKeysConfig(ExtraHashableModel):
    """
    This is the configuration for the MapKeys transformer.
    In order to call this transformer pass the name "map-keys" and a mapping dict.
    """

    mapping: Mapping[str, str]
    preserve_unmapped: bool = True
    ignore_missing_data: bool = True
    level_separator: str = "."
    return_plain: bool = False

    @validator("mapping")
    def backwards_compatibility(cls, mapping: Mapping[str, str]):
        return {
            key.replace(".$[", "["): value.replace(".$[", "[")
            for key, value in mapping.items()
        }


class MapKeys(Transformer[MapKeysConfig]):
    """
    The MapKeys is a complete dict re-designer.
    It lets you rename the keys and also restructure the entire dict. Creating new nested data where there wasn't
    and also flattening data that was previously nested is possible, all that preserving the data from the input
    dictionary.
    """

    def __init__(self, config: MapKeysConfig) -> None:
        super().__init__(config)
        self.__flatters_config = FlatterConfig(level_separator=config.level_separator)
        self.__flatter = Flatter(self.__flatters_config)
        self.__unflatter = Unflatter(self.__flatters_config)

    def transform(
        self, payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ):
        """
        The mapping is done in 4 major steps:

        1. Flattens the data.
        2. Metadata Replacers:
            Some key mapping parameters are specified in the metadata. Keys that have placeholders like
            ${metadata_key} will be substituted by values on the specified metadata key.
        3. Map Data.
                In this moment the keys of the mapping inside config match the keys of the flat payload. That is, the
            payload and self._config.mapping have matching keys. Maybe not all keys in payload are in
            self._config.mapping, in which case we choose what to do with those extra keys with the config
            self._config.preserve_unmapped. If the opposite happens, the self._config.mapping have keys not present
            in the payload, the configuration self._config.ignore_missing_data chooses what should be done.
        4. Unflattens the data.
        :return: transformed and restructured data.
        """
        flat_data = self.__flatter.transform(payload)
        translated_dict: Dict = {}

        map_keys_set = set(self._config.mapping.keys())
        for map_key in map_keys_set.intersection(flat_data.keys()):
            map_value = self._config.mapping[map_key]

            if metadata is not None:
                for meta_key, meta_value in metadata.items():
                    map_key = map_key.replace("@{" + meta_key + "}", str(meta_value))
                    map_value = map_value.replace(
                        "@{" + meta_key + "}", str(meta_value)
                    )

            translated_dict[map_value] = flat_data[map_key]

        if not self._config.ignore_missing_data:
            missing_keys = map_keys_set - flat_data.keys()
            if missing_keys:
                raise ReportMissingData(missing_keys)

        if self._config.preserve_unmapped:
            for unmapped_key in flat_data.keys() - self._config.mapping.keys():
                translated_dict[unmapped_key] = flat_data[unmapped_key]

        if self._config.return_plain:
            return translated_dict, metadata

        if metadata is None:
            return self.__unflatter.transform(translated_dict)

        return self.__unflatter.transform(translated_dict, metadata)
