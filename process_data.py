import numpy as np
from sklearn.preprocessing import LabelEncoder


def preprocess(df, train_labels, inference=False):
    features = df.drop(['customer_ID'], axis=1).columns.to_list()
    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68"
    ]

    binary_cols = [
      'R_1', 
      'B_8', 
      'D_54', 
      'R_27', 
      'D_112', 
      'D_128', 
      'D_130'
    ]

    noncat_features = [col for col in features if col not in cat_features]

    # Add agg features
    result_df = agg_features(df, cat_features, noncat_features)

    # Add lag features
    result_df = lag_features(result_df)

    # Round 2 decimals
    result_df = round2(result_df)

    result_df = round_to_cat(result_df, cat_features, binary_cols)

    # Add labels to train df
    if not inference:
        result_df = result_df.merge(
            train_labels, how='inner', on='customer_ID')

    return result_df


def agg_features(df, cat_features, noncat_features):
    # Generate aggregation features (numerical features)
    df_num_agg = df.groupby("customer_ID")[noncat_features].agg(
        ['mean', 'std', 'min', 'max', 'first', 'last'])
    df_num_agg.columns = ['_'.join(x) for x in df_num_agg.columns]
    df_num_agg.reset_index(inplace=True)

    # Transform categorical features from object to numerical labels
    df['D_63'] = LabelEncoder().fit_transform(df['D_63'])
    df['D_64'] = LabelEncoder().fit_transform(df['D_64'])

    # Generate aggregation features (categorical features)
    df_cat_agg = df.groupby("customer_ID")[cat_features].agg(
        ['count', 'last', 'nunique'])
    df_cat_agg.columns = ['_'.join(x) for x in df_cat_agg.columns]
    df_cat_agg.reset_index(inplace=True)

    cat_features = [f"{cf}_last" for cf in cat_features]

    # Convert floats to ints; Nans to strings
    for cat_feature in cat_features:
        if not df_cat_agg[cat_feature].isnull().values.any():
            df_cat_agg[cat_feature] = df_cat_agg[cat_feature].apply(lambda x: int(x))
        else:
            df_cat_agg[cat_feature] = LabelEncoder().fit_transform(df_cat_agg[cat_feature])
            df_cat_agg[cat_feature] = df_cat_agg[cat_feature].apply(lambda x: int(x))

    # Merge numerical and categorical features to same df
    result_df = df_num_agg.merge(df_cat_agg, how='inner', on='customer_ID')

    return result_df

def lag_features(df):
    num_cols = [x for x in df.columns if 'first' in x]
    num_cols = [x[:-6] for x in num_cols]
    last_cols = [x + '_last' for x in num_cols]
    first_cols = [x + '_first' for x in num_cols]

    for num_col, last_col, first_col in zip(num_cols, last_cols, first_cols):
        df[f'{num_col}_lag_sub'] = df[last_col] - df[first_col]
        df[f'{num_col}_lag_div'] = df[last_col] / df[first_col]

    return df

def round2(df):
    num_cols = list(df.dtypes[(df.dtypes == 'float32') | (df.dtypes == 'float64')].index)
    last_cols = [col for col in num_cols if 'last' in col]
    first_cols = [col for col in num_cols if 'first' in col]

    for col in last_cols:
        df[col + '_round2'] = df[col].round(2)
    
    for col in first_cols:
        df[col + '_round2'] = df[col].round(2)

    return df

def round_to_cat(df, cat_features, binary_cols):
    df['R_1_last_round2'] = (df['R_1_last_round2'] * 4).round(0).astype(int)
    df['R_1_first_round2'] = (df['R_1_first_round2'] * 4).round(0).astype(int)

    df['B_8_first_round2'] = df['B_8_first_round2'].round(0)
    df['D_54_first_round2'] = df['D_54_first_round2'].round(0)
    df['R_27_first_round2'] = df['R_27_first_round2'].round(0)
    df['D_112_first_round2'] = df['D_112_first_round2'].round(0)
    df['D_128_first_round2'] = df['D_128_first_round2'].round(0)
    df['D_130_first_round2'] = df['D_130_first_round2'].round(0)

    df['B_8_last_round2'] = df['B_8_last_round2'].round(0)
    df['D_54_last_round2'] = df['D_54_last_round2'].round(0)
    df['R_27_last_round2'] = df['R_27_last_round2'].round(0)
    df['D_112_last_round2'] = df['D_112_last_round2'].round(0)
    df['D_128_last_round2'] = df['D_128_last_round2'].round(0)
    df['D_130_last_round2'] = df['D_130_last_round2'].round(0)

    for cat_col in binary_cols:
        first_col = cat_col + '_first_round2'
        last_col = cat_col + '_last_round2'
        df[first_col] = LabelEncoder().fit_transform(df[first_col])
        df[last_col] = LabelEncoder().fit_transform(df[last_col])

    return df

def last_mean_diff(df):
    num_cols = [col for col in df.columns if 'last' in col]
    num_cols = [col[:-5] for col in num_cols if 'round' not in col]

    for col in num_cols:
        try:
            df[f'{col}_last_mean_diff'] = df[f'{col}_last'] - df[f'{col}_mean']
        except:
            pass
    
    num_cols = list(df.dtypes[(df.dtypes == 'float32') | (df.dtypes == 'float64')].index)

    for col in num_cols:
        df[col] = df[col].astype(np.float16)
