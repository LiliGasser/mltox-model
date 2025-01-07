import pandas as pd


def filter_and_merge_predictions(df_p_gp_all, 
                                 df_p_lasso_all,
                                 df_p_rf_all,
                                 df_p_xgboost_all,
                                 challenge,
                                 groupsplit,
                                 conctype,
                                 tax_pdm):
    
    # set column for concentration
    if conctype == 'molar':
        col_conc = 'result_conc1_mean_mol_log'
    elif conctype == 'mass':
        col_conc = 'result_conc1_mean_log'

    # get sub data for all models
    if len(df_p_gp_all) > 0:
        df_p_gp = df_p_gp_all[#(df_p_gp_all['chem_fp'] == chem_fp)
                              (df_p_gp_all['challenge'] == challenge)
                              & (df_p_gp_all['groupsplit'] == groupsplit)
                              & (df_p_gp_all['conctype'] == conctype)
                              & (df_p_gp_all['tax_pdm'] == tax_pdm)].copy()
    else:
        df_p_gp = pd.DataFrame()
    df_p_lasso = df_p_lasso_all[#(df_p_lasso_all['chem_fp'] == chem_fp)
                                (df_p_lasso_all['challenge'] == challenge)
                                & (df_p_lasso_all['conctype'] == conctype)
                                & (df_p_lasso_all['groupsplit'] == groupsplit)].copy()
    df_p_rf = df_p_rf_all[#(df_p_rf_all['chem_fp'] == chem_fp)
                          (df_p_rf_all['challenge'] == challenge)
                          & (df_p_rf_all['conctype'] == conctype)
                          & (df_p_rf_all['groupsplit'] == groupsplit)].copy()
    df_p_xgboost = df_p_xgboost_all[#(df_p_xgboost_all['chem_fp'] == chem_fp)
                                    (df_p_xgboost_all['challenge'] == challenge)
                                    & (df_p_xgboost_all['conctype'] == conctype)
                                    & (df_p_xgboost_all['groupsplit'] == groupsplit)].copy()

    # rename concentration columns
    df_p_lasso = df_p_lasso.rename(columns={col_conc: 'true',
                                      'conc_pred': 'lasso_pred',
                                      })
    if len(df_p_gp) > 0:
        df_p_gp = df_p_gp.rename(columns={'conc_pred': 'gp_pred'})
    df_p_rf = df_p_rf.rename(columns={'conc_pred': 'rf_pred'})
    df_p_xgboost = df_p_xgboost.rename(columns={'conc_pred': 'xgboost_pred'})

    # add other predictions to LASSO table
    if len(df_p_gp) > 0:
        df_p = pd.merge(df_p_lasso, 
                        df_p_gp[['chem_fp', 'result_id', 'gp_pred']],
                        left_on=['chem_fp', 'result_id'],
                        right_on=['chem_fp', 'result_id'],
                        how='left')
    else:
        df_p = df_p_lasso.copy()
    df_p = pd.merge(df_p, 
                    df_p_rf[['chem_fp', 'result_id', 'rf_pred']],
                    left_on=['chem_fp', 'result_id'],
                    right_on=['chem_fp', 'result_id'],
                    how='left')
    df_p = pd.merge(df_p, 
                    df_p_xgboost[['chem_fp', 'result_id', 'xgboost_pred']],
                    left_on=['chem_fp', 'result_id'],
                    right_on=['chem_fp', 'result_id'],
                    how='left')
    
    return df_p