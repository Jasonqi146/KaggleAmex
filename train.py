from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import gc
import joblib
import lightgbm as lgb
import config as CFG
import os
import pandas as pd


def oof(train, model, model_name, metric, pseudo_df=None):
    kfold = StratifiedKFold(n_splits=CFG.n_folds,
                            shuffle=True, random_state=CFG.seed)
    oof_predictions = np.zeros(len(train))

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

    cat_features = [f"{cf}_last" for cf in cat_features]

    # new_cat_features = [
    #   'R_1', 
    #   'B_8', 
    #   'D_54', 
    #   'R_27', 
    #   'D_112', 
    #   'D_128', 
    #   'D_130'
    # ]

    # cat_features.extend([x + '_first_round2' for x in new_cat_features])
    # cat_features.extend([x + '_last_round2' for x in new_cat_features])
    
    features = [col for col in train.columns if col not in ['customer_ID', CFG.target]]
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train, train[CFG.target])):
        print('-'*50)
        print(f'Training fold {fold} with {len(features)} features...')
        
        save_dir = f'{CFG.MODEL_PATH}/{model_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = f'{save_dir}/fold{fold}_seed{CFG.seed}.pkl'

        if os.path.exists(save_name):
            print(f'Exists {save_name}')
            continue

        x_train, x_val = train[features].iloc[train_idx], train[features].iloc[val_idx]
        y_train, y_val = train[CFG.target].iloc[train_idx], train[CFG.target].iloc[val_idx]

        len_original = len(x_train)
        if pseudo_df is not None:
            x_train = pd.concat([x_train, pseudo_df[features]])
            y_train = pd.concat([y_train, pseudo_df[CFG.target]])

        cv_model = model().train(x_train, y_train, x_val, y_val, cat_features)

        # Save best model
        cv_model.save_model(file_name=save_name)
        
        # Predict validation
        val_pred = cv_model.predict(x_val)
        
        # Add to out of folds array
        oof_predictions[val_idx] = val_pred
        
        # Compute fold metric
        score = metric(y_val, val_pred)
        print(f'Our fold {fold} CV score is {score}')
        del x_train, x_val, y_train, y_val
        gc.collect()
    
    # Output oof results
    oof_df = train[['customer_ID']]
    oof_df['Predictions'] = oof_predictions
    oof_df.to_csv(f'{save_dir}/oof_pred.csv')

    return oof_df


def inference_cv(model, test, model_name):
    test_predictions = np.zeros(len(test))
    features = [col for col in test.columns if col not in ['customer_ID', CFG.target]]

    for fold in range(CFG.n_folds):
        cv_model = model().load_model(file_name=f'{CFG.MODEL_PATH}/{model_name}/fold{fold}_seed{CFG.seed}.pkl')
        test_pred = cv_model.predict(test[features])
        test_predictions += test_pred / CFG.n_folds
        del cv_model, test_pred
        gc.collect()
    
    return test_predictions


def split_inference(model, test_df, model_name, split_num=5):
    submission_df = pd.DataFrame(columns=['customer_ID', 'prediction'])
    submission_df['customer_ID'] = test_df['customer_ID'] 

    for i in range(split_num):
        print(i)
        length = len(test_df) // split_num
        start, end = i * length, len(test_df) if i == split_num - 1 else (i + 1) * length
        submission_df.loc[start:end - 1, 'prediction'] = inference_cv(model, test_df[start:end], model_name=model_name)
    
    return submission_df


def ensemble_inference(pred_csvs, weights, test_df):
    submission_df = pd.DataFrame(columns=['customer_ID', 'prediction'])
    submission_df['customer_ID'] = test_df['customer_ID']
    submission_df['prediction'] = 0

    for i, pred_csv in enumerate(pred_csvs):
        pred_df = pd.read_csv(pred_csv)
        submission_df['prediction'] += pred_df['prediction'] * weights[i]
    
    return submission_df

