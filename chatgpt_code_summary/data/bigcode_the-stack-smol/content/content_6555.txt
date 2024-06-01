import dfstools as dt
import sys

if __name__ == "__main__":
    print(sys.version)
    print(sys.executable)

    print("---pecanCookies Demo---")

    print("Loading into dataframe from csv...", '\n')
    data_list = dt.load_csv_to_df()

    print("identifing relationships by column content....", '\n')
    relationship_list = dt.find_related_cols_by_content(data_list)

    print("printing relationships....", '\n')
    print(relationship_list)