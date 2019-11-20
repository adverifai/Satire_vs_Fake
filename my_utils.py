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
    data_columns = ['doc_id', 'document', 'label', 'bin_label']
    data = pd.DataFrame(columns=data_columns)
    files_list = open(base_dir + "list2", "r")
    for f in files_list:
        try:
            with open(base_dir + f.strip(), 'r', encoding="utf-8") as curr_file:
                lines = curr_file.readlines()

            final_lines = []
            for i in range(len(lines)):
                if lines[i] != '\n':
                    final_lines.append(lines[i])

            # final_lines format:
            # title = final_lines[0].strip()
            # url = final_lines[1].strip()  # if there's any http://
            # text = final_lines[2:]

            if "http" in final_lines[1]:
                text = "\n".join(final_lines[2:])
            else:
                text = "\n".join(final_lines[1:])

            if label in f and text != "":
                f = f.replace(".txt\n", "").split('/')
                doc_id = f[len(f)-1]

                if "fake" in f[0].lower():
                    bin_label = 0
                else:
                    bin_label = 1

                data = data.append(pd.DataFrame(
                    [[doc_id, text, label, bin_label]],
                    columns=data_columns
                ), ignore_index=True)

            curr_file.close()
        except Exception as e:
            print(e)
    files_list.close()
    return data


def read_fake_satire_dataset(base_dir):
    # fake: 0, satire: 1
    fake_df = load_data(base_dir, "Fake")
    satire_df = load_data(base_dir, "Satire")
    return pd.concat([fake_df, satire_df]).sample(frac=1).reset_index(drop=True)


def drop_constant_columns(df):
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    return df.drop(cols_to_drop, axis=1)
