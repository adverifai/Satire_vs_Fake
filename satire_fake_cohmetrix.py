import my_utils
import copy
import pandas as pd
import numpy as np
from pre_processing import text_clean
from my_utils import drop_constant_columns


def create_cohmetrix_input():
    # reading Fake and Satire data
    data = my_utils.read_fake_satire_dataset("data/FakeNewsData/StoryText 2/")

    for index, row in data.iterrows():
        with open('data/cohmetrix/input/d' + str(index) + '_' + str(row[2]) + '.txt', 'w+') as text_file:
            text_file.write(text_clean(row[0], True, True, False, 1))
            text_file.close()

    print("Created the input files for Coh-Metrix successfully.")


def create_regresssion_input():
    """
    creating a single excel file as inout to regression analysis using the coh-metrix output
    :return:
    """
    coh_data = pd.read_csv("data/cohmetrix/output/satirefake.csv")
    x_columns = list(coh_data.iloc[:, 1:])
    x_reg = []
    y_reg = []

    for item in coh_data.iterrows():
        tmp = item[1][0].split('\\')
        tmp = tmp[len(tmp)-1].split('.')[0].split('_')
        doc_label = int(tmp[1])

        x_reg.append(item[1][1:])
        y_reg.append(doc_label)

    # y_reg = np.array(y_reg)

    # converting list to dataframe
    x_reg = pd.DataFrame(np.array(x_reg).reshape(len(x_reg), len(x_columns)), columns=x_columns)
    # dropping constant value columns
    # x_lin = np.array(x_lin)
    x_reg = drop_constant_columns(x_reg)

    # scaling the data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # x_lin = scaler.fit_transform(x_lin)

    x_full = copy.deepcopy(x_reg)
    x_full["label"] = y_reg
    x_full = x_full.fillna(0)
    writer = pd.ExcelWriter('data/satire_fake_full.xlsx')
    x_full.to_excel(writer, 'Sheet1')
    writer.save()

    print("Created regression input files successfully.")

