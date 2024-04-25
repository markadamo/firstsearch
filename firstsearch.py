import numpy as np
import pandas as pd
import weightedstats as ws
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from sklearn.isotonic import IsotonicRegression
from cir_model import CenteredIsotonicRegression

def sigmoid(x, L, x0, k):
    return L / (1 + np.exp(-k * (x-x0)))

def monoexp(x, m, t, b):
    return m * np.exp(-t * x) + b

class Alignment:
    model_mapping = {'centered_isotonic': CenteredIsotonicRegression,
                     'isotonic': IsotonicRegression}
    model_kwargs = {'centered_isotonic': {'out_of_bounds':'clip'},
                    'isotonic': {'out_of_bounds':'clip'}}
    func_mapping = {'sigmoid': sigmoid,
                    #'monoexp': monoexp,
                    }
    p0_func_mapping = {'sigmoid': lambda ys: [max(ys), np.median(ys), 1],
                       #'monoexp': lambda ys: [max(ys), 1, 1],
                       }
    
    def __init__(self, *args, model='centered_isotonic', model_kwargs=None, func=None, **kwargs):
        self.model = None
        self.func = None
        self.p0_func = None
        self.predictor = None
        self.stats = {}
        if func is None:
            try:
                self.model = self.model_mapping[model](**model_kwargs if model_kwargs is not None else self.model_kwargs[model])
            except KeyError:
                raise KeyError(f'Invalid model specified. Model must be one of {list(self.model_mapping.keys())}')
        else:
            try:
                self.func = self.func_mapping[func]
                self.p0_func = self.p0_func_mapping[func] 
            except KeyError:
                raise KeyError(f'Invalid function specified. Function must be one of {list(self.func_mapping.keys())}')
        
    def fit_from_csv(self, csv, spec_id='spec_id', lib_rt='library_rt', obs_rt='rt', score='hyperscore', bin_ax='y', bin_width=0.75, std_tol_factor=0.5, csv_kwargs={}, plot=False):
        #read csv (file path or buffer) into pandas dataframe
        df = pd.read_csv(csv, **csv_kwargs)

        #select binning variable based on provided axis
        if bin_ax == 'y': bin_col = obs_rt
        elif bin_ax == 'x': bin_col = lib_rt
        else: KeyError("Invalid bin_ax given. Values may be 'x' or 'y'.")

        #define bin breakpoints based on provided bin width (seconds)
        rt_bins = np.arange(min(df[bin_col])-bin_width,
                            max(df[bin_col])+bin_width,
                            bin_width)

        #assign bin indexes to dataframe
        df['bin'] = pd.cut(df[bin_col], rt_bins)

        #calculate processed x (lib_rt) and y (obs_rt) values from binned data
        #x and y coordinates are weighted medians, weighted by score
        #observed=True excludes empty bins 
        y = df.groupby('bin', observed=True).apply(lambda a: ws.numpy_weighted_median(a[obs_rt], weights=a[score]**2))
        x = df.groupby('bin', observed=True).apply(lambda a: ws.numpy_weighted_median(a[lib_rt], weights=a[score]**2))

        #calculate aggregate weighting values for each point (used to refine model fitting)
        w = df.groupby('bin', observed=True)[score].aggregate('sum')

        #fit model to binned data
        if self.func: #use scipy.optimize.curve_fit
            p0 = self.p0_func(y)
            popt, pcov = curve_fit(self.func, x, y, p0, sigma=(1/w), maxfev=1000000)
            self.predictor = lambda a: self.func(a, *popt) 
        else: #use sklearn regression model
            self.predictor = self.model.fit(x, y, sample_weight=w**2).predict

        #compute corrected library RTs and distances from y=x line
        df['corrected_lib_rts'] = self.predictor(df[lib_rt])
        df['obs_dist'] = df['corrected_lib_rts'] - df[obs_rt]

        #filtered data by score (top decile)
        topdf = df[df[score] > df[score].quantile(0.90)]
        std = np.std(topdf['obs_dist']) #std of y distances from y=x line
        self.stats['rt_tol'] = std * std_tol_factor

        #filtered data by rt within computed std tolerance
        toldf = df[abs(df['obs_dist']) < std*std_tol_factor]

        #kernel density estimation on rt-tolerance-filtered data
        mz_kernel = gaussian_kde(toldf['mz_error'], weights=toldf[score]**2)
        mz_space = np.linspace(min(toldf['mz_error']), max(toldf['mz_error']), 200)
        mz_peak = mz_space[np.argmax(mz_kernel(mz_space))]
        self.stats['mz_error'] = mz_peak
        
        if plot:
            plt.figure(1, figsize=[10,9])
            plt.subplot(221)
            plt.title('Fit')
            plt.scatter(df[lib_rt], df[obs_rt], s=df[score], alpha=0.2, label='raw data')
            plt.scatter(x, y, label='binned data')
            plt.plot(sorted(x), self.predictor(sorted(x)), 'orange', label='model fit')
            plt.xlabel('library RT')
            plt.ylabel('observed RT')
            plt.legend()

            plt.subplot(222)
            plt.title('Corrected data (top-scoring decile)')
            plt.scatter(topdf['corrected_lib_rts'], topdf[obs_rt], s=topdf[score], alpha=0.75)
            plt.plot(topdf['corrected_lib_rts'], topdf['corrected_lib_rts'])
            plt.fill_between(sorted(topdf['corrected_lib_rts']),
                             sorted(topdf['corrected_lib_rts']+std*std_tol_factor),
                             sorted(topdf['corrected_lib_rts']-std*std_tol_factor), label=f'+-{std*std_tol_factor}', alpha=0.25)
            plt.xlabel('corrected library RT')
            plt.ylabel('observed RT')

            plt.subplot(223)
            plt.title('Corrected data')
            plt.scatter(df['corrected_lib_rts'], df[obs_rt], s=df[score], alpha=0.2)
            plt.plot(df['corrected_lib_rts'], df['corrected_lib_rts'])
            plt.fill_between(sorted(df['corrected_lib_rts']),
                             sorted(df['corrected_lib_rts']+std*std_tol_factor),
                             sorted(df['corrected_lib_rts']-std*std_tol_factor), label=f'+-{std*std_tol_factor}', alpha=0.25)
            plt.xlabel('corrected library RT')
            plt.ylabel('observed RT')

            plt.subplot(224)
            plt.title('M/z KDE')
            mz_space = np.linspace(min(toldf['mz_error']), max(toldf['mz_error']), 200)
            plt.plot(mz_space, mz_kernel(mz_space))
            plt.axvline(self.stats['mz_error'])
            plt.xlabel('M/z error')
            plt.show()

        return self.stats

    def predict(self, a):
        if self.predictor is None:
            raise RuntimeError('Values cannot be predicted until model is fitted. Call fit_from_csv first.')

        return self.predictor(a)
        
