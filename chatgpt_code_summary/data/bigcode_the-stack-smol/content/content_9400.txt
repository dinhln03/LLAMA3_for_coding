from typing import List
import argparse
import chart_studio.plotly as py
import plotly.express as px
import pandas as pd

class TokyoCovid19Stat:
    """Holds Tokyo Covid-19 stat data."""

    def __init__(self, csv_file_path: str = None):
        self.csv_file_path = csv_file_path
        self._df = None
        self.area_list = []

    def update(self) -> None:
        df = pd.read_csv(self.csv_file_path,
                         parse_dates=['Date'])
        for area in df['Area']:
            if area in self.area_list:
                break
            self.area_list.append(area)
        df = df.pivot(index='Date', columns='Area', values='New Cases')
        self._df = df[self.area_list]

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.update()
        return self._df

    @property
    def cases_by_area(self) -> pd.DataFrame:
        return self.df

    @property
    def cases(self) -> pd.DataFrame:
        return pd.DataFrame({'Cases': self.cases_by_area.sum(axis=1)})


def sma(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    return df.rolling(days).mean()


def with_date(orig_df: pd.DataFrame) -> pd.DataFrame:
    df = orig_df.copy()
    df['Date'] = df.index.to_list()
    return df


def melt(orig_df: pd.DataFrame,
         value_columns: List[str],
         var_name: str,
         value_name: str = 'Cases') -> pd.DataFrame:
    """Unpivot the given DataFrame to be used with Plotly."""
    df = with_date(orig_df)
    df = df[['Date'] + value_columns]
    return df.melt(id_vars=['Date'],
                   value_vars=value_columns,
                   var_name=var_name,
                   value_name=value_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file_path')
    args = parser.parse_args()

    if args.csv_file_path is None:
        return

    st = TokyoCovid19Stat(args.csv_file_path)

    cases_by_area = melt(st.cases_by_area,
                        value_columns=st.area_list,
                        var_name='Area')

    sma_by_area = melt(sma(st.cases_by_area),
                       value_columns=st.area_list,
                      var_name='Area')

    # title = 'Tokyo Covid-19 New Cases By Area'
    # fig = px.area(cases_by_area, x='Date', y='Cases', color='Area', title=title)
    # py.plot(fig, filename=title, auto_open=False)

    title = '[TEST] Tokyo Covid-19 New Cases 7-day Moving Average By Area'
    fig = px.line(sma_by_area, x='Date', y='Cases', color='Area', title=title)
    fig.add_bar(x=st.cases.index,
                y=st.cases['Cases'],
                name='Raw Total',
                marker=dict(color='#dddddd'))
    py.plot(fig, filename=title, auto_open=False)



if __name__ == '__main__':
    main()
