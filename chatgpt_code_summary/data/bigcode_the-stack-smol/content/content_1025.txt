# coding: utf-8
#
# Copyright 2018 Moriyoshi Koizumi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import six
import pandas
from .column import _DataFrameColumn, _Function, _Literal, eval_column, infer_data_type, compile_to_raf, resolve_alias
from .functions import SimpleAggregationFunctionSpec
from .group import GroupedData
from .types import StructType, StructField


class _Raw(object):
    def __init__(self, pdf):
        self.pdf = pdf

    def __call__(self, df):
        return self.pdf


class _Filter(object):
    def __init__(self, df, expr):
        self.df = df
        self.expr = expr

    def __call__(self, df):
        raf = compile_to_raf(df, self.expr)
        pdf = self.df._yield_pdf()
        return pdf.loc[raf]


class _Aggregation(object):
    def __init__(self, grouped_data, agg_cols):
        self.grouped_data = grouped_data
        self.agg_cols = agg_cols

    def __call__(self, df):
        pdf = self.grouped_data.df._yield_pdf()
        agg_fn_cols = []
        agg_variations = set()
        const_cols = []
        resolved_cols = []
        for col in self.agg_cols:
            col = resolve_alias(col)
            resolved_cols.append(col)
            if isinstance(col, _Function):
                agg_fn_cols.append(col)
                if isinstance(col.spec, SimpleAggregationFunctionSpec):
                    agg_variations.add(col.spec.fn)
                else:
                    raise TypeError()
            elif isinstance(col, _Literal):
                const_cols.append(col)
            else:
                raise TypeError(col.__class__)

        if len(self.grouped_data.cols) > 0:
            pg = pdf.groupby(
                by=[pdf.iloc[:,col.index] for col in self.grouped_data.cols]
            )
            agg_result = pg.aggregate(list(agg_variations))
            agg_result = pandas.concat([agg_result.index.to_frame(), agg_result], axis=1)
            # convert columns to a set of series
            agg_result_index = agg_result.index.to_frame()
            series_set = [
                agg_result_index[col].rename(i)
                for i, col in enumerate(agg_result_index.columns)
            ]
            for col in resolved_cols:
                if isinstance(col, _Function):
                    if isinstance(col.spec, SimpleAggregationFunctionSpec):
                        series = agg_result[col.operands[0].index, col.spec.fn].rename(len(series_set))
                    else:
                        # should never get here; already validated in the above loop
                        assert False
                elif isinstance(col, _Literal):
                    series = pandas.Series([col.value], name=len(series_set))
                else:
                    # should never get here; already validated in the above loop
                    assert False
                series_set.append(series)
        else:
            agg_result = pdf.aggregate(list(agg_variations))
            # convert columns to a set of series
            series_set = []
            for col in self.agg_cols:
                if isinstance(col, _Function):
                    if isinstance(col.spec, SimpleAggregationFunctionSpec):
                        series = pandas.Series([agg_result[col.operands[0].index][col.spec.fn]], name=len(series_set))
                    else:
                        # should never get here; already validated in the above loop
                        assert False
                elif isinstance(col, _Literal):
                    series = pandas.Series([col.value], name=len(series_set))
                else:
                    # should never get here; already validated in the above loop
                    assert False
                series_set.append(series)

        return pandas.concat(series_set, axis=1)


class _WithColumns(object):
    def __init__(self, df, name_col_pairs):
        self.df = df
        self.name_col_pairs = name_col_pairs

    def __call__(self, df):
        extra_fields = df.schema.fields[len(self.df.schema.fields):]
        lhs = self.df._yield_pdf()
        return pandas.concat(
            [lhs] + [
                eval_column(df, lhs, col).rename(i)
                for i, (_, col) in enumerate(self.name_col_pairs, len(self.df.columns))
            ],
            axis=1
        )


class _Union(object):
    def __init__(self, df, following):
        self.df = df
        self.following = following

    def __call__(self, df):
        return pandas.concat([self.df._yield_pdf(), self.following._yield_pdf()], axis=0)


class _OrderBy(object):
    def __init__(self, df, cols, ascending=None):
        self.df = df
        self.cols = cols
        self.ascending = ascending

    def __call__(self, df):
        assert all(isinstance(col, _DataFrameColumn) for col in self.cols)
        return self.df._yield_pdf().sort_values(by=[col.index for col in self.cols], ascending=self.ascending)


class Row(object):
    def __init__(self, pdf, schema, i, name_to_column_map):
        self.pdf = pdf
        self.schema = schema
        self.i = i
        self.name_to_column_map = name_to_column_map

    def __str__(self):
        return str(self.pdf.iloc[self.i])

    def __getitem__(self, i):
        if isinstance(i, six.string_types):
            return self.pdf.iloc[self.i][self.name_to_column_map[i].index]
        else:
            return self.pdf.iloc[self.i][i]


class DataFrame(object):
    def __init__(self, sql_ctx, schema, modifier=None):
        self.sql_ctx = sql_ctx
        self.schema = schema
        self.modifier = modifier
        self._columns = [
            _DataFrameColumn(self, f, i)
            for i, f in enumerate(schema.fields)
        ]
        self._name_to_column_map = {
            f.name: c
            for f, c in zip(schema.fields, self._columns)
        }

    def __getitem__(self, i):
        if isinstance(i, six.string_types):
            return self._name_to_column_map[i]
        elif isinstance(i, (int, long)):
            return self._columns[i]
        else:
            raise TypeError()

    def filter(self, cond):
        return DataFrame(
            self.sql_ctx,
            self.schema,
            _Filter(self, cond)
        )

    def groupBy(self, *cols):
        return GroupedData(self, cols)

    def agg(self, *exprs):
        return self.groupBy().agg(*exprs)

    def withColumn(self, name, col):
        return self._with_columns([(name, col)])

    def unionAll(self, following):
        return DataFrame(
            self.sql_ctx,
            self.schema,
            _Union(self, following)
        )

    def orderBy(self, *cols, **kwargs):
        ascending = kwargs.pop('ascending', None)
        return DataFrame(
            self.sql_ctx,
            self.schema,
            _OrderBy(self, cols, ascending)
        )

    @property
    def columns(self):
        return [col.field.name for col in self._columns]

    def _with_columns(self, name_col_pairs):
        return DataFrame(
            self.sql_ctx,
            StructType(
                fields=self.schema.fields + [
                    StructField(
                        name,
                        infer_data_type(col)
                    )
                    for name, col in name_col_pairs
                ]
            ),
            _WithColumns(self, name_col_pairs)
        )

    def _yield_pdf(self):
        return self.modifier(self)

    def collect(self):
        pdf = self._yield_pdf()
        return [
            Row(pdf, self.schema, i, self._name_to_column_map)
            for i in range(0, len(pdf))
        ]
