import warnings

warnings.simplefilter("ignore", category=FutureWarning)
from pmaf.biome.essentials._metakit import EssentialFeatureMetabase
from pmaf.biome.essentials._base import EssentialBackboneBase
from pmaf.internal._constants import (
    AVAIL_TAXONOMY_NOTATIONS,
    jRegexGG,
    jRegexQIIME,
    BIOM_TAXONOMY_NAMES,
    VALID_RANKS,
)
from pmaf.internal._shared import (
    generate_lineages_from_taxa,
    get_rank_upto,
    indentify_taxon_notation,
    validate_ranks,
    extract_valid_ranks,
    cols2ranks,
)
from collections import defaultdict
from os import path
import pandas as pd
import numpy as np
import biom
from typing import Union, Sequence, Tuple, Any, Optional
from pmaf.internal._typing import AnyGenericIdentifier, Mapper


class RepTaxonomy(EssentialBackboneBase, EssentialFeatureMetabase):
    """An `essential` class for handling taxonomy data."""

    def __init__(
        self,
        taxonomy: Union[pd.DataFrame, pd.Series, str],
        taxonomy_columns: Union[str, int, Sequence[Union[int, str]]] = None,
        **kwargs: Any
    ) -> None:
        """Constructor for :class:`.RepTaxonomy`

        Parameters
        ----------
        taxonomy
            Data containing feature taxonomy
        taxonomy_columns
            Column(s) containing taxonomy data
        kwargs
             Passed to :func:`~pandas.read_csv` or :mod:`biome` loader.
        """
        tmp_metadata = kwargs.pop("metadata", {})
        self.__avail_ranks = []
        self.__internal_taxonomy = None
        if isinstance(taxonomy, pd.DataFrame):
            if taxonomy.shape[0] > 0:
                if taxonomy.shape[1] > 1:
                    if validate_ranks(list(taxonomy.columns.values), VALID_RANKS):
                        tmp_taxonomy = taxonomy
                    else:
                        raise ValueError(
                            "Provided `taxonomy` Datafame has invalid ranks."
                        )
                else:
                    tmp_taxonomy = taxonomy.iloc[:, 0]
            else:
                raise ValueError("Provided `taxonomy` Datafame is invalid.")
        elif isinstance(taxonomy, pd.Series):
            if taxonomy.shape[0] > 0:
                tmp_taxonomy = taxonomy
            else:
                raise ValueError("Provided `taxonomy` Series is invalid.")
        elif isinstance(taxonomy, str):
            if path.isfile(taxonomy):
                file_extension = path.splitext(taxonomy)[-1].lower()
                if file_extension in [".csv", ".tsv"]:
                    if taxonomy_columns is None:
                        tmp_taxonomy = pd.read_csv(
                            taxonomy,
                            sep=kwargs.pop("sep", ","),
                            header=kwargs.pop("header", "infer"),
                            index_col=kwargs.pop("index_col", None),
                        )
                    else:
                        if isinstance(taxonomy_columns, int):
                            tmp_taxonomy = pd.read_csv(
                                taxonomy,
                                sep=kwargs.pop("sep", ","),
                                header=kwargs.pop("header", "infer"),
                                index_col=kwargs.pop("index_col", None),
                            ).iloc[:, taxonomy_columns]
                        else:
                            tmp_taxonomy = pd.read_csv(
                                taxonomy,
                                sep=kwargs.pop("sep", ","),
                                header=kwargs.pop("header", "infer"),
                                index_col=kwargs.pop("index_col", None),
                            ).loc[:, taxonomy_columns]
                elif file_extension in [".biom", ".biome"]:
                    tmp_taxonomy, new_metadata = self.__load_biom(taxonomy, **kwargs)
                    tmp_metadata.update({"biom": new_metadata})
                else:
                    raise NotImplementedError("File type is not supported.")
            else:
                raise FileNotFoundError("Provided `taxonomy` file path is invalid.")
        else:
            raise TypeError("Provided `taxonomy` has invalid type.")
        self.__init_internal_taxonomy(tmp_taxonomy, **kwargs)
        super().__init__(metadata=tmp_metadata, **kwargs)

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        taxonomy_columns: Union[str, int, Sequence[Union[int, str]]] = None,
        **kwargs: Any
    ) -> "RepTaxonomy":
        """Factory method to construct a :class:`.RepTaxonomy` from CSV file.

        Parameters
        ----------
        filepath
            Path to .csv File
        taxonomy_columns
            Column(s) containing taxonomy data
        kwargs
            Passed to the constructor.
        filepath:

        Returns
        -------
        Instance of
            class:`.RepTaxonomy`
        """
        if taxonomy_columns is None:
            tmp_taxonomy = pd.read_csv(filepath, **kwargs)
        else:
            if isinstance(taxonomy_columns, int):
                tmp_taxonomy = pd.read_csv(filepath, **kwargs).iloc[:, taxonomy_columns]
            else:
                tmp_taxonomy = pd.read_csv(filepath, **kwargs).loc[:, taxonomy_columns]
        tmp_metadata = kwargs.pop("metadata", {})
        tmp_metadata.update({"filepath": path.abspath(filepath)})
        return cls(taxonomy=tmp_taxonomy, metadata=tmp_metadata, **kwargs)

    @classmethod
    def from_biom(cls, filepath: str, **kwargs: Any) -> "RepTaxonomy":
        """Factory method to construct a :class:`.RepTaxonomy` from :mod:`biom`
        file.

        Parameters
        ----------
        filepath
            :mod:`biom` file path.
        kwargs
            Passed to the constructor.

        Returns
        -------
        Instance of
            class:`.RepTaxonomy`
        """
        taxonomy_frame, new_metadata = cls.__load_biom(filepath, **kwargs)
        tmp_metadata = kwargs.pop("metadata", {})
        tmp_metadata.update({"biom": new_metadata})
        return cls(taxonomy=taxonomy_frame, metadata=tmp_metadata, **kwargs)

    @classmethod
    def __load_biom(cls, filepath: str, **kwargs: Any) -> Tuple[pd.DataFrame, dict]:
        """Actual private method to process :mod:`biom` file.

        Parameters
        ----------
        filepath
            :mod:`biom` file path.
        kwargs
            Compatibility
        """
        biom_file = biom.load_table(filepath)
        if biom_file.metadata(axis="observation") is not None:
            obs_data = biom_file.metadata_to_dataframe("observation")
            col_names = list(obs_data.columns.values)
            col_names_low = [col.lower() for col in col_names]
            avail_col_names = [
                colname
                for tax_name in BIOM_TAXONOMY_NAMES
                for colname in col_names_low
                if colname[::-1].find(tax_name[::-1]) < 3
                and colname[::-1].find(tax_name[::-1]) > -1
            ]
            metadata_cols = [
                col for col in col_names if col.lower() not in avail_col_names
            ]
            if len(avail_col_names) == 1:
                tmp_col_index = col_names_low.index(avail_col_names[0])
                taxonomy_frame = obs_data[col_names[tmp_col_index]]
            else:
                taxonomy_frame = obs_data
            tmp_metadata = obs_data.loc[:, metadata_cols].to_dict()
            return taxonomy_frame, tmp_metadata
        else:
            raise ValueError("Biom file does not contain observation metadata.")

    def _remove_features_by_id(
        self, ids: AnyGenericIdentifier, **kwargs: Any
    ) -> Optional[AnyGenericIdentifier]:
        """Remove features by features ids and ratify action.

        Parameters
        ----------
        ids
            Feature identifiers
        kwargs
            Compatibility
        """
        tmp_ids = np.asarray(ids, dtype=self.__internal_taxonomy.index.dtype)
        if len(tmp_ids) > 0:
            self.__internal_taxonomy.drop(tmp_ids, inplace=True)
        return self._ratify_action("_remove_features_by_id", ids, **kwargs)

    def _merge_features_by_map(
        self, map_dict: Mapper, done: bool = False, **kwargs: Any
    ) -> Optional[Mapper]:
        """Merge features and ratify action.

        Parameters
        ----------
        map_dict
            Map to use for merging
        done
            Whether merging was completed or not. Compatibility.
        kwargs
            Compatibility
        """
        if not done:
            raise NotImplementedError
        if map_dict:
            return self._ratify_action(
                "_merge_features_by_map",
                map_dict,
                _annotations=self.__internal_taxonomy.loc[:, "lineage"].to_dict(),
                **kwargs
            )

    def drop_feature_by_id(
        self, ids: AnyGenericIdentifier, **kwargs: Any
    ) -> Optional[AnyGenericIdentifier]:
        """Remove features by feature `ids`.

        Parameters
        ----------
        ids
            Feature identifiers
        kwargs
            Compatibility
        """
        target_ids = np.asarray(ids)
        if self.xrid.isin(target_ids).sum() == len(target_ids):
            return self._remove_features_by_id(target_ids, **kwargs)
        else:
            raise ValueError("Invalid feature ids are provided.")

    def get_taxonomy_by_id(
        self, ids: Optional[AnyGenericIdentifier] = None
    ) -> pd.DataFrame:
        """Get taxonomy :class:`~pandas.DataFrame` by feature `ids`.

        Parameters
        ----------
        ids
            Either feature indices or None for all.

        Returns
        -------
            class:`pandas.DataFrame` with taxonomy data
        """
        if ids is None:
            target_ids = self.xrid
        else:
            target_ids = np.asarray(ids)
        if self.xrid.isin(target_ids).sum() <= len(target_ids):
            return self.__internal_taxonomy.loc[target_ids, self.__avail_ranks]
        else:
            raise ValueError("Invalid feature ids are provided.")

    def get_lineage_by_id(
        self,
        ids: Optional[AnyGenericIdentifier] = None,
        missing_rank: bool = False,
        desired_ranks: Union[bool, Sequence[str]] = False,
        drop_ranks: Union[bool, Sequence[str]] = False,
        **kwargs: Any
    ) -> pd.Series:
        """Get taxonomy lineages by feature `ids`.

        Parameters
        ----------
        ids
            Either feature indices or None for all.
        missing_rank
            If True will generate prefix like `s__` or `d__`
        desired_ranks
            List of desired ranks to generate.
            If False then will generate all main ranks
        drop_ranks
            List of ranks to drop from desired ranks.
            This parameter only useful if `missing_rank` is True
        kwargs
            Compatibility.

        Returns
        -------
            class:`pandas.Series` with consensus lineages and corresponding IDs
        """
        if ids is None:
            target_ids = self.xrid
        else:
            target_ids = np.asarray(ids)
        tmp_desired_ranks = VALID_RANKS if desired_ranks is False else desired_ranks
        total_valid_rids = self.xrid.isin(target_ids).sum()
        if total_valid_rids == len(target_ids):
            return generate_lineages_from_taxa(
                self.__internal_taxonomy.loc[target_ids],
                missing_rank,
                tmp_desired_ranks,
                drop_ranks,
            )
        elif total_valid_rids < len(target_ids):
            return generate_lineages_from_taxa(
                self.__internal_taxonomy.loc[np.unique(target_ids)],
                missing_rank,
                tmp_desired_ranks,
                drop_ranks,
            )
        else:
            raise ValueError("Invalid feature ids are provided.")

    def find_features_by_pattern(
        self, pattern_str: str, case_sensitive: bool = False, regex: bool = False
    ) -> np.ndarray:
        """Searches for features with taxa that matches `pattern_str`

        Parameters
        ----------
        pattern_str
            Pattern to search for
        case_sensitive
            Case sensitive mode
        regex
            Use regular expressions


        Returns
        -------
            class:`~numpy.ndarray` with indices
        """
        return self.__internal_taxonomy[
            self.__internal_taxonomy.loc[:, "lineage"].str.contains(
                pattern_str, case=case_sensitive, regex=regex
            )
        ].index.values

    def drop_features_without_taxa(
        self, **kwargs: Any
    ) -> Optional[AnyGenericIdentifier]:
        """Remove features that do not contain taxonomy.

        Parameters
        ----------
        kwargs
            Compatibility
        """
        ids_to_drop = self.find_features_without_taxa()
        return self._remove_features_by_id(ids_to_drop, **kwargs)

    def drop_features_without_ranks(
        self, ranks: Sequence[str], any: bool = False, **kwargs: Any
    ) -> Optional[AnyGenericIdentifier]:  # Done
        """Remove features that do not contain `ranks`

        Parameters
        ----------
        ranks
            Ranks to look for
        any
            If True removes feature with single occurrence of missing rank.
            If False all `ranks` must be missing.
        kwargs
            Compatibility
        """
        target_ranks = np.asarray(ranks)
        if self.__internal_taxonomy.columns.isin(target_ranks).sum() == len(
            target_ranks
        ):
            no_rank_mask = self.__internal_taxonomy.loc[:, ranks].isna()
            no_rank_mask_adjusted = (
                no_rank_mask.any(axis=1) if any else no_rank_mask.all(axis=1)
            )
            ids_to_drop = self.__internal_taxonomy.loc[no_rank_mask_adjusted].index
            return self._remove_features_by_id(ids_to_drop, **kwargs)
        else:
            raise ValueError("Invalid ranks are provided.")

    def merge_duplicated_features(self, **kwargs: Any) -> Optional[Mapper]:
        """Merge features with duplicated taxonomy.

        Parameters
        ----------
        kwargs
            Compatibility
        """
        ret = {}
        groupby = self.__internal_taxonomy.groupby("lineage")
        if any([len(group) > 1 for group in groupby.groups.values()]):
            tmp_feature_lineage = []
            tmp_groups = []
            group_indices = list(range(len(groupby.groups)))
            for lineage, feature_ids in groupby.groups.items():
                tmp_feature_lineage.append(lineage)
                tmp_groups.append(list(feature_ids))
            self.__init_internal_taxonomy(
                pd.Series(data=tmp_feature_lineage, index=group_indices)
            )
            ret = dict(zip(group_indices, tmp_groups))
        return self._merge_features_by_map(ret, True, **kwargs)

    def merge_features_by_rank(self, level: str, **kwargs: Any) -> Optional[Mapper]:
        """Merge features by taxonomic rank/level.

        Parameters
        ----------
        level
            Taxonomic rank/level to use for merging.
        kwargs
            Compatibility
        """
        ret = {}
        if not isinstance(level, str):
            raise TypeError("`rank` must have str type.")
        if level in self.__avail_ranks:
            target_ranks = get_rank_upto(self.avail_ranks, level, True)
            if target_ranks:
                tmp_lineages = generate_lineages_from_taxa(
                    self.__internal_taxonomy, False, target_ranks, False
                )
                groups = tmp_lineages.groupby(tmp_lineages)
                if len(groups.groups) > 1:
                    tmp_feature_lineage = []
                    tmp_groups = []
                    group_indices = list(range(len(groups.groups)))
                    for lineage, feature_ids in groups.groups.items():
                        tmp_feature_lineage.append(lineage)
                        tmp_groups.append(list(feature_ids))
                    self.__init_internal_taxonomy(
                        pd.Series(data=tmp_feature_lineage, index=group_indices)
                    )
                    ret = dict(zip(group_indices, tmp_groups))
        else:
            raise ValueError("Invalid rank are provided.")
        return self._merge_features_by_map(ret, True, **kwargs)

    def find_features_without_taxa(self) -> np.ndarray:
        """Find features without taxa.

        Returns
        -------
            class:`~numpy.ndarray` with feature indices.
        """
        return self.__internal_taxonomy.loc[
            self.__internal_taxonomy.loc[:, VALID_RANKS].agg(
                lambda rank: len("".join(map(lambda x: (str(x or "")), rank))), axis=1
            )
            < 1
        ].index.values

    def get_subset(
        self, rids: Optional[AnyGenericIdentifier] = None, *args, **kwargs: Any
    ) -> "RepTaxonomy":
        """Get subset of the :class:`.RepTaxonomy`.

        Parameters
        ----------
        rids
            Feature identifiers.
        args
            Compatibility
        kwargs
            Compatibility

        Returns
        -------
            class:`.RepTaxonomy`
        """
        if rids is None:
            target_rids = self.xrid
        else:
            target_rids = np.asarray(rids).astype(self.__internal_taxonomy.index.dtype)
        if not self.xrid.isin(target_rids).sum() == len(target_rids):
            raise ValueError("Invalid feature ids are provided.")
        return type(self)(
            taxonomy=self.__internal_taxonomy.loc[target_rids, "lineage"],
            metadata=self.metadata,
            name=self.name,
        )

    def _export(
        self, taxlike: str = "lineage", ascending: bool = True, **kwargs: Any
    ) -> Tuple[pd.Series, dict]:
        """Creates taxonomy for export.

        Parameters
        ----------
        taxlike
            Generate taxonomy in format(currently only `lineage` is supported.)
        ascending
            Sorting
        kwargs
            Compatibility
        """
        if taxlike == "lineage":
            return (
                self.get_lineage_by_id(**kwargs).sort_values(ascending=ascending),
                kwargs,
            )
        else:
            raise NotImplemented

    def export(
        self,
        output_fp: str,
        *args,
        _add_ext: bool = False,
        sep: str = ",",
        **kwargs: Any
    ) -> None:
        """Exports the taxonomy into the specified file.

        Parameters
        ----------
        output_fp
            Export filepath
        args
            Compatibility
        _add_ext
            Add file extension or not.
        sep
            Delimiter
        kwargs
            Compatibility
        """
        tmp_export, rkwarg = self._export(*args, **kwargs)
        if _add_ext:
            tmp_export.to_csv("{}.csv".format(output_fp), sep=sep)
        else:
            tmp_export.to_csv(output_fp, sep=sep)

    def copy(self) -> "RepTaxonomy":
        """Copy of the instance."""
        return type(self)(
            taxonomy=self.__internal_taxonomy.loc[:, "lineage"],
            metadata=self.metadata,
            name=self.name,
        )

    def __fix_taxon_names(self) -> None:
        """Fix invalid taxon names."""

        def taxon_fixer(taxon):
            if taxon is not None and pd.notna(taxon):
                tmp_taxon_trimmed = taxon.lower().strip()
                if len(tmp_taxon_trimmed) > 0:
                    if tmp_taxon_trimmed[0] == "[":
                        tmp_taxon_trimmed = tmp_taxon_trimmed[1:]
                    if tmp_taxon_trimmed[-1] == "]":
                        tmp_taxon_trimmed = tmp_taxon_trimmed[:-1]
                    return tmp_taxon_trimmed.capitalize()
                else:
                    return None
            else:
                return None

        self.__internal_taxonomy.loc[:, VALID_RANKS] = self.__internal_taxonomy.loc[
            :, VALID_RANKS
        ].applymap(taxon_fixer)

    def __reconstruct_internal_lineages(self) -> None:
        """Reconstruct the internal lineages."""
        self.__internal_taxonomy.loc[:, "lineage"] = generate_lineages_from_taxa(
            self.__internal_taxonomy, True, self.__avail_ranks, False
        )

    def __init_internal_taxonomy(
        self,
        taxonomy_data: Union[pd.Series, pd.DataFrame],
        taxonomy_notation: Optional[str] = "greengenes",
        order_ranks: Optional[Sequence[str]] = None,
        **kwargs: Any
    ) -> None:
        """Main method to initialize taxonomy.

        Parameters
        ----------
        taxonomy_data
            Incoming parsed taxonomy data
        taxonomy_notation
            Taxonomy lineage notation style. Can be one of
                 :const:`pmaf.internals._constants.AVAIL_TAXONOMY_NOTATIONS`
        order_ranks
            List with the target rank order. Default is set to None.
                The 'silva' notation require `order_ranks`.
        kwargs
            Compatibility
        """
        if isinstance(taxonomy_data, pd.Series):
            new_taxonomy = self.__init_taxonomy_from_lineages(
                taxonomy_data, taxonomy_notation, order_ranks
            )
        elif isinstance(taxonomy_data, pd.DataFrame):
            if taxonomy_data.shape[1] == 1:
                taxonomy_data_series = pd.Series(
                    data=taxonomy_data.iloc[:, 0], index=taxonomy_data.index
                )
                new_taxonomy = self.__init_taxonomy_from_lineages(
                    taxonomy_data_series, taxonomy_notation, order_ranks
                )
            else:
                new_taxonomy = self.__init_taxonomy_from_frame(
                    taxonomy_data, taxonomy_notation, order_ranks
                )
        else:
            raise RuntimeError(
                "`taxonomy_data` must be either pd.Series or pd.Dataframe"
            )

        if new_taxonomy is None:
            raise ValueError("Provided taxonomy is invalid.")

        # Assign newly constructed taxonomy to the self.__internal_taxonomy
        self.__internal_taxonomy = new_taxonomy
        self.__fix_taxon_names()  # Fix incorrect taxa
        tmp_avail_ranks = [rank for rank in VALID_RANKS if rank in new_taxonomy.columns]
        self.__avail_ranks = [
            rank for rank in tmp_avail_ranks if new_taxonomy.loc[:, rank].notna().any()
        ]
        # Reconstruct internal lineages for default greengenes notation
        self.__reconstruct_internal_lineages()
        self._init_state = True

    def __init_taxonomy_from_lineages(
        self,
        taxonomy_series: pd.Series,
        taxonomy_notation: Optional[str],
        order_ranks: Optional[Sequence[str]],
    ) -> pd.DataFrame:  # Done
        """Main method that produces taxonomy dataframe from lineages.

        Parameters
        ----------
        taxonomy_series
            :class:`pandas.Series` with taxonomy lineages
        taxonomy_notation
            Taxonomy lineage notation style. Can be one of :const:`pmaf.internals._constants.AVAIL_TAXONOMY_NOTATIONS`
        order_ranks
            List with the target rank order. Default is set to None. The 'silva' notation require `order_ranks`.
        """
        # Check if taxonomy is known and is available for parsing. Otherwise indentify_taxon_notation() will try to identify notation
        if taxonomy_notation in AVAIL_TAXONOMY_NOTATIONS:
            notation = taxonomy_notation
        else:
            # Get first lineage _sample for notation testing assuming the rest have the the same notations
            sample_taxon = taxonomy_series.iloc[0]
            # Identify notation of the lineage string
            notation = indentify_taxon_notation(sample_taxon)
        if order_ranks is not None:
            if all([rank in VALID_RANKS for rank in order_ranks]):
                target_order_ranks = order_ranks
            else:
                raise NotImplementedError
        else:
            target_order_ranks = VALID_RANKS
        if notation == "greengenes":
            lineages = taxonomy_series.reset_index().values.tolist()
            ordered_taxa_list = []
            ordered_indices_list = [elem[0] for elem in lineages]
            for lineage in lineages:
                tmp_lineage = jRegexGG.findall(lineage[1])
                tmp_taxa_dict = {
                    elem[0]: elem[1] for elem in tmp_lineage if elem[0] in VALID_RANKS
                }
                for rank in VALID_RANKS:
                    if rank not in tmp_taxa_dict.keys():
                        tmp_taxa_dict.update({rank: None})
                tmp_taxa_ordered = [tmp_taxa_dict[rank] for rank in VALID_RANKS]
                ordered_taxa_list.append([None] + tmp_taxa_ordered)
            taxonomy = pd.DataFrame(
                index=ordered_indices_list,
                data=ordered_taxa_list,
                columns=["lineage"] + VALID_RANKS,
            )
            return taxonomy
        elif notation == "qiime":
            lineages = taxonomy_series.reset_index().values.tolist()
            tmp_taxa_dict_list = []
            tmp_ranks = set()
            for lineage in lineages:
                tmp_lineage = jRegexQIIME.findall(lineage[1])
                tmp_lineage.sort(key=lambda x: x[0])
                tmp_taxa_dict = defaultdict(None)
                tmp_taxa_dict[None] = lineage[0]
                for rank, taxon in tmp_lineage:
                    tmp_taxa_dict[rank] = taxon
                    tmp_ranks.add(rank)
                tmp_taxa_dict_list.append(dict(tmp_taxa_dict))
            tmp_taxonomy_df = pd.DataFrame.from_records(tmp_taxa_dict_list)
            tmp_taxonomy_df.set_index(None, inplace=True)
            tmp_taxonomy_df = tmp_taxonomy_df.loc[:, sorted(list(tmp_ranks))]
            tmp_taxonomy_df.columns = [
                rank for rank in target_order_ranks[::-1][: len(tmp_ranks)]
            ][::-1]
            for rank in VALID_RANKS:
                if rank not in tmp_taxonomy_df.columns:
                    tmp_taxonomy_df.loc[:, rank] = None
            return tmp_taxonomy_df
        elif notation == "silva":
            lineages = taxonomy_series.reset_index().values.tolist()
            tmp_taxa_dict_list = []
            tmp_ranks = set()
            for lineage in lineages:
                tmp_lineage = lineage[1].split(";")
                tmp_taxa_dict = defaultdict(None)
                tmp_taxa_dict[None] = lineage[0]
                for rank_i, taxon in enumerate(tmp_lineage):
                    rank = target_order_ranks[rank_i]
                    tmp_taxa_dict[rank] = taxon
                    tmp_ranks.add(rank)
                tmp_taxa_dict_list.append(dict(tmp_taxa_dict))
            tmp_taxonomy_df = pd.DataFrame.from_records(tmp_taxa_dict_list)
            tmp_taxonomy_df.set_index(None, inplace=True)
            tmp_rank_ordered = [
                rank for rank in target_order_ranks if rank in VALID_RANKS
            ]
            tmp_taxonomy_df = tmp_taxonomy_df.loc[:, tmp_rank_ordered]
            tmp_taxonomy_df.columns = [
                rank for rank in target_order_ranks[::-1][: len(tmp_ranks)]
            ][::-1]
            for rank in VALID_RANKS:
                if rank not in tmp_taxonomy_df.columns:
                    tmp_taxonomy_df.loc[:, rank] = None
            return tmp_taxonomy_df

        else:
            raise NotImplementedError

    def __init_taxonomy_from_frame(
        self,
        taxonomy_dataframe: pd.DataFrame,
        taxonomy_notation: Optional[str],
        order_ranks: Optional[Sequence[str]],
    ) -> pd.DataFrame:  # Done # For now only pass to _init_taxonomy_from_series
        """Main method that produces taxonomy sheet from dataframe.

        Parameters
        ----------
        taxonomy_dataframe
            :class:`~pandas.DataFrame` with taxa split by ranks.
        taxonomy_notation
            Taxonomy lineage notation style. Can be one of :const:`pmaf.internals._constants.AVAIL_TAXONOMY_NOTATIONS`
        order_ranks
            List with the target rank order. Default is set to None. The 'silva' notation require `order_ranks`.

        Returns
        -------
            :class:`~pandas.DataFrame`
        """
        valid_ranks = extract_valid_ranks(taxonomy_dataframe.columns, VALID_RANKS)
        if valid_ranks is not None:
            if len(valid_ranks) > 0:
                return pd.concat(
                    [
                        taxonomy_dataframe,
                        pd.DataFrame(
                            data="",
                            index=taxonomy_dataframe.index,
                            columns=[
                                rank for rank in VALID_RANKS if rank not in valid_ranks
                            ],
                        ),
                    ],
                    axis=1,
                )
            else:
                taxonomy_series = taxonomy_dataframe.apply(
                    lambda taxa: ";".join(taxa.values.tolist()), axis=1
                )
                return self.__init_taxonomy_from_lineages(
                    taxonomy_series, taxonomy_notation, order_ranks
                )
        else:
            valid_ranks = cols2ranks(taxonomy_dataframe.columns)
            taxonomy_dataframe.columns = valid_ranks
            taxonomy_series = taxonomy_dataframe.apply(
                lambda taxa: ";".join([(t if isinstance(t,str) else '') for t in taxa.values]), axis=1
            )
            return self.__init_taxonomy_from_lineages(
                taxonomy_series, taxonomy_notation, order_ranks
            )

    @property
    def avail_ranks(self) -> Sequence[str]:
        """List of available taxonomic ranks."""
        return self.__avail_ranks

    @property
    def duplicated(self) -> pd.Index:
        """List of duplicated feature indices."""
        return self.__internal_taxonomy.index[
            self.__internal_taxonomy["lineage"].duplicated(keep=False)
        ]

    @property
    def data(self) -> pd.DataFrame:
        """Actual data representation as pd.DataFrame."""
        return self.__internal_taxonomy

    @property
    def xrid(self) -> pd.Index:
        """Feature indices as pd.Index."""
        return self.__internal_taxonomy.index
