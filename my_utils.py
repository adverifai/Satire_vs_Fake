import warnings
import codecs
import pandas as pd
warnings.filterwarnings("ignore", category=DeprecationWarning)


def read_fake_satire_data(base_dir):
    """
    reading the fake satire data
    :param base_dir:
    :return:
    """
    headlines_data = []
    text_data = []
    target = []
    files_list = open(base_dir+"list2", "r")
    for f in files_list:
        try:
            curr_file = codecs.open(base_dir+f.strip(), "r", encoding="ISO-8859-1")
            lines = curr_file.readlines()
            headlines_data.append(lines[0].strip())
            text = lines[0].strip() + " " + lines[2].strip()
            text_data.append(text)
            if "Fake" in f:
                target.append(-1)
            else:
                target.append(1)
            curr_file.close()
        except Exception as e:
            print(e)
    files_list.close()
    return [headlines_data, text_data, target]


def load_data(base_dir, label):
    data = {"document": [], "label": []}
    files_list = open(base_dir + "list2", "r")
    for f in files_list:
        try:
            curr_file = codecs.open(base_dir + f.strip(), "r", encoding="ISO-8859-1")
            lines = curr_file.readlines()
            # this is the headline text
            # text = lines[0].strip()
            # this is the main body text
            text = lines[2].strip()
            # this is headline AND main body text
            # text = lines[0].strip() + ". " + lines[2].strip()
            if label in f:
                data["document"].append(text)
                data["label"].append(label)
            curr_file.close()
        except Exception as e:
            print(e)
    files_list.close()
    return pd.DataFrame.from_dict(data)


def read_fake_satire_dataset(base_dir):
    fake_df = load_data(base_dir, "Fake")
    satire_df = load_data(base_dir, "Satire")
    fake_df["bin_label"] = 0
    satire_df["bin_label"] = 1
    return pd.concat([fake_df, satire_df]).sample(frac=1).reset_index(drop=True)


def drop_constant_columns(df):
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    return df.drop(cols_to_drop, axis=1)
