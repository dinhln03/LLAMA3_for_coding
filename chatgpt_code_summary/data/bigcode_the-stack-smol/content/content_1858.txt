## Calculate feature importance, but focus on "meta-features" which are categorized by
## rules from different perspectives: orders, directions, powers.

## for "comprehensive methods"

from util_relaimpo import *
from util_ca import *
from util import loadNpy


def mainCA(x_name, y_name, divided_by = "", feature_names = []):
    X = loadNpy(['data', 'X', x_name])
    Y = loadNpy(['data', 'Y', y_name])
    # INFO
    print("Dataset", x_name, y_name)
    print("Method: ", "CA")
    print("Divided by", divided_by)
    # make dataframe
    if feature_names: xdf = pd.DataFrame(data=X, columns=feature_names)
    else: xdf = pd.DataFrame(data=X)
    # divide X
    x_list, feature_names = dvdX(xdf, divided_by=divided_by)
    # if power, only use the first four terms
    if divided_by=='power': x_list, feature_names = x_list[0:4], feature_names[0:4]
    print("bootstrapping ...")
    coef_boot, comb_feature = bootstrappingCA(x_list, Y)
    result_df = caResultDf(coef_boot, comb_feature)
    printBootResultCA(result_df)

def mainDA(x_name, y_name, divided_by = "", feature_names = []):
    X = loadNpy(['data', 'X', x_name])
    Y = loadNpy(['data', 'Y', y_name])
    # INFO
    print("Dataset", x_name, y_name)
    print("Method: ", "DA")
    print("Divided by", divided_by)
    # make dataframe
    if feature_names:
        xdf = pd.DataFrame(data=X, columns=feature_names)
    else:
        xdf = pd.DataFrame(data=X)
    # divide X
    x_list, feature_names = dvdX(xdf, divided_by=divided_by)
    # if power, only use the first four terms
    if divided_by=='power': x_list, feature_names = x_list[0:4], feature_names[0:4]
    print("bootstrapping ...")
    coef_boot, comb_feature, r2_mean, r2_ci, da_data, ave_data = bootstrappingDA(x_list, Y)
    da_df = daResultDf(da_data, ave_data, r2_mean, comb_feature, feature_name=feature_names)
    printBootResultCA(da_df)

if __name__ == '__main__':
    # da or ca
    x_prefix = ["HM", "MMA"]
    y_suffix = ["MPS95", "MPSCC95", "CSDM"]
    x_main = "{}_X_ang_vel.npy"
    y_main = "{}_{}.npy"
    divided_list = ["order", "direction", "power"]
    for ys in y_suffix:
        for xp in x_prefix:
            for divide in divided_list:
                x_name = x_main.format(xp)
                y_name = y_main.format(xp, ys)
                mainCA(x_name,y_name,divide,feature_names)
                mainDA(x_name,y_name,divide,feature_names)