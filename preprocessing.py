import pandas
import numpy as np
import sys

def standardize(df):
    print("inside tandardize")
    cols = df.columns
    print(cols)
    [row, col] = df.shape
    for i in range(len(cols)):
        if cols[i] != 'class':
            print("inside for loop", cols[i])
            mean= np.mean(df[cols[i]])
            std= np.std(df[cols[i]])
            for x in range(0, row):
                df.iloc[x, df.columns.get_loc(cols[i])] = (df.iloc[x, df.columns.get_loc(cols[i])] - mean) / std
        else:
            for y in range(0, row):

                df.iloc[y, df.columns.get_loc(cols[i])] = 2**df.iloc[y][cols[i]]
    print(df.head(n=7))
    return df

def remove_unknown(df):
    cols = df.columns
    print("inside remove unknown----------")
    for i in range(len(cols)):
       df = df[~df[cols[i]].astype(str).str.contains("\?")]
    return df




def category_to_num(df):
    obj_df = df.select_dtypes(include=['object']).copy()
    obj_df.head(n=5)
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat_col = list(set(cols) - set(num_cols))
    for i in range(len(cat_col)):
        print(cat_col[i])
        if cat_col[i] != 'class':
            obj_df[cat_col[i]] = obj_df[cat_col[i]].astype('category')
            obj_df[cat_col[i]] = 2 ** obj_df[cat_col[i]].cat.codes
        else:
            obj_df[cat_col[i]] = obj_df[cat_col[i]].astype('category')
            obj_df[cat_col[i]] = obj_df[cat_col[i]].cat.codes
    print("obj_head=============")
    obj_df.head(n=3)
    for i in range(len(cat_col)):
        df[cat_col[i]] = obj_df[cat_col[i]]
    print(" data frame head text to num=============")
    print(df.head(n=3))
    return df

url = sys.argv[1]
output = sys.argv[2]
if (url == "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"):
    names=['length','width','petal_length','petal_width','class']
elif (url =="https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"):
    names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
elif  (url == "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"):
    names = ['age', 'workclass', 'fnlwgt', 'education', 'edu_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hrs_per_week', 'C', 'class']
else:
    print("enter correct url")
    sys.exit()
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
dataframe.head(n=5)
dataframe.to_csv(output+'original_iris.csv',  index = False)
#dataframe.to_csv(output)
df_filtered= remove_unknown(dataframe)
df_num= category_to_num(df_filtered)
print("standardised  data------------")
df_std=  standardize(df_num)
df_std.to_csv(output+'standardized_dataframe.csv',  index = False)







