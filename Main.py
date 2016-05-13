"""
Primary program for Style Analysis
"""

from scipy import optimize as opt
from sklearn import preprocessing as pp
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
import time
sys.path.append('D:\GD\Python\TextualAnalysis\StyleAnalysis')  # Modify to identify path for custom modules
import GetMacroData as GMD
import Read_CRSPMF_Data as RCD


# ==========================================+ PARAMETERS +==============================================+

PARM_BGNYR = 1991
PARM_TM1 = 5
PARM_ENDYR = 1991
PARM_TESTPERIOD_YRS = 1
PARM_LOGFILE = open('D:/Temp/StyleAnalysisMain_Logfile_{0}.txt'.format(time.strftime('%Y%m%d')), 'w')
PARM_TESTMODE = True
pd.options.mode.chained_assignment = None  # default = 'warn
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)

# ======================================================================================================+


def main():

    df_macro, mf_data, mfheader_dict = start_me_up(PARM_BGNYR, update=True)

    # Loop thru years
    for year in range(PARM_BGNYR, PARM_BGNYR + 1):
        parms = setup_pointers(year, True)
        df_x, df_xp, x_labels = load_x(df_macro, parms)

        # Loop thru each mutual fund
        for mf in mf_data:
            df_y, df_yp = load_y(mf_data[mf], parms)
            if (len(df_y.index) == PARM_TM1 * 12) & (len(df_yp.index) == PARM_TESTPERIOD_YRS * 12):
                df_y['mf_ret'] = pp.scale(df_y['mf_ret'])
                df_yp['mf_ret'] = pp.scale(df_yp['mf_ret'])
                run_stats(df_x, df_xp, df_y, df_yp, x_labels)
                break
    return


def func_OLS(x, df_y, df_x, labels):
    sse = np.sum((df_y.mf_ret - (np.dot(x, df_x[labels].T))) ** 2)
    return sse


def func_lasso(x, df_y, df_x, labels):
    sse = np.sum((df_y.mf_ret - (np.dot(x, df_x[labels].T))) ** 2)
    # penalty = np.sum(np.absolute(x))
    # penalty = np.sum(x ** 2)
    penalty = np.sum(np.absolute(x[0:8])) - np.sum(x[0:8] ** 2)
    xlambda = 0.0
    func = ((1 / (2 * len(df_y))) * sse) + (xlambda * penalty)
    return func


def func_lasso_g(x, df_y, df_x, labels):
    sse = np.sum((df_y.mf_ret - (np.dot(x[0:8], df_x[labels].T))) ** 2)
    penalty = np.sum(np.absolute(x[0:8])) - np.sum(x[0:8] ** 2)
    xlambda = x[8]
    func = (1 / (2 * len(df_y))) * sse + xlambda * penalty
    return func

def f_eq(x, df_y, df_x, labels):
    c = np.sum(x) - 1
    return c


def run_stats(df_x, df_xp, df_y, df_yp, x_labels):
    # Run correlations
    corr_df = run_CORR(df_x)

    # Run sp500 benchmark regression
    ols_sp500 = run_OLS(df_x, df_y, x_labels[0:1])
    ols_sp500_R2os = get_R2os(ols_sp500.params, df_xp, df_yp, x_labels[0:1])

    # Run OLS full xp regression
    ols_xp = run_OLS(df_x, df_y, x_labels[1:])
    ols_xp_R2os = get_R2os(ols_xp.params, df_xp, df_yp, x_labels[1:])

    # Run OLS using optimization
    ols_opt = opt.minimize(func_OLS, [0.1] * 8, args=(df_y, df_x, x_labels[1:],),
              method='BFGS', options={'disp': False})
    ols_opt_R2os = get_R2os(ols_opt.x, df_xp, df_yp, x_labels[1:])
    if PARM_TESTMODE:
        print(ols_opt.x)
        print('ols_opt_R2os = {0}'.format(ols_opt_R2os))

    # Run traditional Style Analysis
    xbounds = [(0, 1)] * 8
    ols_sa = opt.fmin_slsqp(func_OLS, [0.1] * 8, acc=1e-9, args=(df_y, df_x, x_labels[1:],),
                            disp=0, bounds=xbounds, full_output=False, f_eqcons=f_eq)
    ols_sa_str = np.array_repr(ols_sa).replace('\n', '')
    ols_sa_R2os = get_R2os(ols_sa, df_xp, df_yp, x_labels[1:])
    if PARM_TESTMODE:
        print(ols_sa)
        print('ols_sa_R2os = {0}'.format(ols_sa_R2os))

    # Run lasso
    xbounds = [(-1, 1)] * 8
    # xbounds.append((0, 10))
    lasso_sa = opt.fmin_slsqp(func_lasso, [0.1] * 8, acc=1e-9, args=(df_y, df_x, x_labels[1:],),
                              disp=0, bounds=xbounds, full_output=False, f_eqcons=f_eq)
    lasso_sa_str = np.array_repr(lasso_sa).replace('\n', '')
    lasso_sa_R2os = get_R2os(lasso_sa[0:8], df_xp, df_yp, x_labels[1:])
    if PARM_TESTMODE:
        print(lasso_sa)
        print('lasso_sa_R2os = {0}'.format(lasso_sa_R2os))
    return


def get_R2os(bvec, df_xp, df_yp, labels):
    bxp = np.dot(bvec, df_xp[labels].T)
    corr = 1 - np.sum((df_yp.mf_ret - bxp)**2) / len(df_yp)
    return corr


def start_me_up(bgn_year, update=False):
    # Load SP500 + FF data, mf_data and mf_header
    df_macro = GMD.get_macro_data(bgn_year, update)  # Flag is for live update
    mf_data = RCD.read_crsp_mf_datadict('D:/GD/Data/MutualFund/DataDict.csv', PARM_BGNYR, True)
    mfheader_dict = RCD.read_mfheader()
    # Log startup
    PARM_LOGFILE.write('\n{0}\nPROGRAM: ...Python/StyleAnalysis/Main.py\n\n'.format(time.strftime('%c')))
    PARM_LOGFILE.write('  df_macro loaded into DataFrame: shape = {0}\n'.format(df_macro.shape))
    PARM_LOGFILE.write('  df_macro descriptive statistics:\n\n')
    df_macro.describe().transpose().to_csv(PARM_LOGFILE)
    _corr_df, _corr_results = run_CORR(df_macro)
    PARM_LOGFILE.write('\n  df_macro Pearson correlations: \n\n')
    _corr_df.head(len(df_macro)).to_csv(PARM_LOGFILE)
    PARM_LOGFILE.write('\n  mf_data loaded into data dictionary: len(mf_data) = {0:,}\n'.format(len(mf_data)))
    return df_macro, mf_data, mfheader_dict


def load_x(_df_macro, _parms):
        # Load x and xp matrix
        _df_x = _df_macro[(_df_macro.index >= _parms.bgn_sample) & (_df_macro.index <= _parms.end_sample)]
        _df_xp = _df_macro[(_df_macro.index >= _parms.bgn_testperiod) &
                           (_df_macro.index <= _parms.end_testperiod)]
        x_labels = _df_x.columns.values.tolist()
        for label in x_labels:
            _df_x[label] = pp.scale(_df_x[label])
            _df_xp[label] = pp.scale(_df_xp[label])
        if PARM_TESTMODE:
            _df_x.to_csv('D:/Temp/DF.csv')
            _df_xp.to_csv('D:/Temp/DF.csv')
        return _df_x, _df_xp, x_labels


def load_y(_mf, parms):
    _df = pd.DataFrame(list(_mf.items()))
    _df.columns = ['yymm', 'mf_ret']
    _df = _df.sort_values(by='yymm')
    _df_y = _df[(_df.yymm >= parms.bgn_sample) & (_df.yymm <= parms.end_sample)]
    _df_y.set_index(['yymm'], inplace=True)
    _df_y.index.rename('yymm', inplace=True)
    _df_yp = _df[(_df.yymm >= parms.bgn_testperiod) & (_df.yymm <= parms.end_testperiod)]
    _df_yp.set_index(['yymm'], inplace=True)
    _df_yp.index.rename('yymm', inplace=True)
    return _df_y, _df_yp


def run_CORR(df_x, log=False):
    corr_df = df_x.corr(method='pearson')
    corr_results = corr_class(corr_df)
    if log:
        print(corr_df.head(len(df_x)))
    return corr_df, corr_results


def run_OLS(df_x, df_y, x_list):
    # X = sm.add_constant(X)
    results = sm.OLS(df_y, df_x[x_list], hasconst=False).fit()
    if PARM_TESTMODE: print(results.summary())
    return results


def setup_pointers(_year, log=False):
    _parms = p_class(_year)
    if log:
        print('bgn_sample = {0} : end_sample = {1} : bgn_testperiod = {2} : end_testperiod = {3}\n'.format(
            _parms.bgn_sample, _parms.end_sample, _parms.bgn_testperiod, _parms.end_testperiod))
    return _parms


class p_class:
    def __init__(self, _year):
        self.bgn_sample = _year * 100 + 1
        self.end_sample = (_year + PARM_TM1 - 1) * 100 + 12
        tyr = (_year + PARM_TM1 - 1) + PARM_TESTPERIOD_YRS
        self.bgn_testperiod = tyr * 100 + 1
        self.end_testperiod = (tyr + (PARM_TESTPERIOD_YRS - 1)) * 100 + 12
        return


class corr_class:
    def __init__(self, df):
        self.avg_corr = df.values[np.triu_indices_from(df.values, 1)].mean()
        self.min_corr = df.values[np.triu_indices_from(df.values, 1)].min()
        self.max_corr = df.values[np.triu_indices_from(df.values, 1)].max()
        

if __name__ == '__main__':
    print('\n' + time.strftime('%c') + '\nStyleAnalysis/Main')
    main()
    print('\n' + time.strftime('%c') + '\nNormal termination.')
