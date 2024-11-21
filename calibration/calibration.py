"""
Define the calibration class
"""
import os
import numpy as np
import sciris as sc
import optuna as op
import matplotlib.pyplot as plt

# Local imports ... should tidy up later
thisdir = sc.thispath(__file__)
rabm = sc.importbypath(thisdir.parent / 'rotaABM.py')
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')


__all__ = ['Calibration', 'compute_gof']


def compute_gof(actual, predicted, normalize=True, use_frac=True, use_squared=False, eps=1e-9):
    """
    Calculate the goodness of fit. By default use normalized absolute error, but
    highly customizable. For example, mean squared error is equivalent to
    setting normalize=False, use_squared=True, as_scalar='mean'.

    Args:
        actual      (arr):   array of actual (data) points
        predicted   (arr):   corresponding array of predicted (model) points
        normalize   (bool):  whether to divide the values by the largest value in either series
        use_frac    (bool):  convert to fractional mismatches rather than absolute
        use_squared (bool):  square the mismatches
        eps         (float): to avoid divide-by-zero

    Returns:
        gofs (arr): array of goodness-of-fit values, or a single value if as_scalar is True

    **Examples**::

        x1 = np.cumsum(np.random.random(100))
        x2 = np.cumsum(np.random.random(100))

        e1 = compute_gof(x1, x2) # Default, normalized absolute error
        e2 = compute_gof(x1, x2, normalize=False, use_frac=False) # Fractional error
        e3 = compute_gof(x1, x2, normalize=False, use_squared=True, as_scalar='mean') # Mean squared error
    """

    # Handle inputs
    actual    = np.array(sc.dcp(actual), dtype=float)
    predicted = np.array(sc.dcp(predicted), dtype=float)

    # Key step -- calculate the mismatch!
    gofs = abs(np.array(actual) - np.array(predicted))

    if normalize and not use_frac:
        actual_max = abs(actual).max()
        if actual_max > 0:
            gofs /= actual_max

    if use_frac:
        if (actual<0).any() or (predicted<0).any():
            print('Warning: Calculating fractional errors for non-positive quantities is ill-advised!')
        else:
            maxvals = np.maximum(actual, predicted) + eps
            gofs /= maxvals

    return gofs


class Calibration(sc.prettyobj):
    """
    A class to handle calibration of RotaABM simulations. Uses the Optuna hyperparameter
    optimization library (optuna.org).

    Args:
        sim          (RotaABM): the simulation to calibrate
        data         (df)   : dataframe of the data to calibrate to
        calib_pars   (dict) : a dictionary of the parameters to calibrate of the format dict(key1=[best, low, high])
        n_trials     (int)  : the number of trials per worker
        n_workers    (int)  : the number of parallel workers (default: maximum number of available CPUs)
        total_trials (int)  : if n_trials is not supplied, calculate by dividing this number by n_workers
        weights      (dict) : the relative weights of each data source
        debug        (bool) : if True, do not run in parallel
        verbose      (bool) : whether to print details of the calibration

    Returns:
        A Calibration object
    """
    def __init__(self, sim, data, calib_pars, n_trials=None, n_workers=None, total_trials=100,
                 die=False, debug=False, verbose=True):

        # Handle run arguments
        name = 'rota'
        db_name = f'{name}.db'
        storage = f'sqlite:///{db_name}'
        if n_workers is None:
            n_workers = sc.cpu_count()
        if total_trials is not None:
            n_trials = int(np.ceil(total_trials/n_workers))
        kw = dict(n_trials=int(n_trials), n_workers=int(n_workers), debug=debug,
                  name=name, db_name=db_name, storage=storage)
        self.run_args = sc.objdict(kw)

        # Store calibration settings
        self.known_pars = ['reassortment_rate', 'rel_beta']

        # Handle other inputs
        self.sim        = sim
        self.data       = data
        self.calib_pars = calib_pars
        self.die        = die
        self.verbose    = verbose
        self.before_sim = None
        self.after_sim  = None
        self.calibrated = False
        return

    def run_sim(self, calib_pars=None, sim_pars=None, trial=None):
        """ Create and run a simulation """

        # Convert calib_pars (best, low, high) to a guess
        if calib_pars is not None:
            sim_pars = self.trial_to_sim_pars(calib_pars=calib_pars, trial=trial)

        # Update sim with new pars
        sim = self.translate_pars(sim_pars=sim_pars)

        # Run sim
        sim.run()
        return sim

    def calib_to_sim_pars(self):
        """ Pull out "best" from the list of calibration pars """
        sim_pars = sc.objdict()
        for par,(best,low,high) in self.calib_pars.items():
            sim_pars[par] = best
        return sim_pars

    def trial_to_sim_pars(self, calib_pars, trial):
        """ Take in an optuna trial and sample from pars """
        calib_pars = sc.mergedicts(calib_pars) # To allow None
        sim_pars = sc.objdict()
        for par, (best,low,high) in calib_pars.items():
            val = trial.suggest_float(par, low, high)
            sim_pars[par] = val
        return sim_pars

    def translate_pars(self, sim_pars):
        """ Take the nested dict of calibration pars and modify the sim """
        sim_pars = sc.mergedicts(sim_pars) # To allow None
        sim = sc.dcp(self.sim)
        pars = list(sim_pars.keys())
        for par in pars:
            val = sim_pars.pop(par)
            if par in self.known_pars:
                setattr(sim, par, val) # Set the new value
            else:
                errormsg = f'Do not know how to handle parameter "{par}"'
                raise NotImplementedError(errormsg)
        return sim

    def compute_fit(self, sim, full=False):
        """ Compute goodness-of-fit """
        sim_df = self.sim_to_df(sim)
        actual = self.data.inci.values
        expected = sim_df.inci.values
        gofs = compute_gof(actual, expected)
        fit = gofs.sum()
        if full:
            return fit, gofs
        else:
            return fit

    @staticmethod
    def sim_to_df(sim):
        """ Convert the sim output to a data-like dataframe """
        df = process_incidence.process_model(sim.df)
        return df

    def run_trial(self, trial):
        """ Define the objective for Optuna """
        sim = self.run_sim(calib_pars=self.calib_pars, trial=trial)
        fit = self.compute_fit(sim)
        return fit

    def load_study(self):
        """ Load a study from disk """
        args = self.run_args
        study = op.load_study( storage=args.storage, study_name=args.name)
        return study

    def worker(self):
        """ Run a single worker """
        if self.verbose:
            op.logging.set_verbosity(op.logging.DEBUG)
        else:
            op.logging.set_verbosity(op.logging.ERROR)
        study = self.load_study()
        output = study.optimize(self.run_trial, n_trials=self.run_args.n_trials)
        return output

    def run_workers(self):
        """ Run multiple workers in parallel """
        if self.run_args.n_workers > 1 and not self.run_args.debug: # Normal use case: run in parallel
            output = sc.parallelize(self.worker, iterarg=self.run_args.n_workers)
        else: # Special case: just run one
            output = [self.worker()]
        return output

    def remove_db(self):
        """ Remove the database file """
        if os.path.exists(self.run_args.db_name):
            os.remove(self.run_args.db_name)
            print(f'Removed existing calibration file {self.run_args.db_name}')
        return

    def make_study(self):
        """ Make a study, deleting one if it already exists """
        self.remove_db()
        output = op.create_study(storage=self.run_args.storage, study_name=self.run_args.name)
        return output

    def calibrate(self, **kwargs):
        """
        Perform calibration.

        Args:
            kwargs (dict): if supplied, overwrite stored run_args (n_trials, n_workers, etc.)
        """
        # Load and validate calibration parameters
        self.run_args.update(kwargs) # Update optuna settings

        # Run the optimization
        self.T = sc.timer()
        self.make_study()
        self.run_workers() # Actually run!

        # Load and parse results
        study = self.load_study()
        self.best_pars = sc.objdict(study.best_params)
        self.parse_study(study)
        if self.verbose:
            print('Best pars:', self.best_pars)

        # Tidy up
        self.remove_db()
        self.calibrated = True
        self.T.toc()
        return self

    def check_fit(self, verbose=True):
        """ Run before and after simulations to validate the fit """
        if verbose: print('Checking fit...')
        before_pars = self.calib_to_sim_pars()
        self.before_sim = self.run_sim(sim_pars=before_pars)
        self.after_sim  = self.run_sim(sim_pars=self.best_pars)
        self.before_df = self.sim_to_df(self.before_sim)
        self.after_df = self.sim_to_df(self.after_sim)
        self.before_fit, self.before_gofs = self.compute_fit(self.before_sim, full=True)
        self.after_fit, self.after_gofs = self.compute_fit(self.after_sim, full=True)

        if verbose:
            print(f'Fit with original pars: {self.before_fit:n}')
            print(f'Fit with best-fit pars: {self.after_fit:n}')
            if self.after_fit <= self.before_fit:
                print('✓ Calibration improved fit')
            else:
                print('✗ Calibration did not improve fit')
        return self.before_fit, self.after_fit

    def parse_study(self, study):
        """Parse the study into a data frame -- called automatically """
        best = study.best_params
        self.best_pars = best

        if self.verbose: print('Making results structure...')
        results = []
        n_trials = len(study.trials)
        failed_trials = []
        for trial in study.trials:
            data = {'index':trial.number, 'mismatch': trial.value}
            for key,val in trial.params.items():
                data[key] = val
            if data['mismatch'] is None:
                failed_trials.append(data['index'])
            else:
                results.append(data)
        if self.verbose: print(f'Processed {n_trials} trials; {len(failed_trials)} failed')

        keys = ['index', 'mismatch'] + list(best.keys())
        data = sc.objdict().make(keys=keys, vals=[])
        for i,r in enumerate(results):
            for key in keys:
                if key not in r:
                    warnmsg = f'Key {key} is missing from trial {i}, replacing with default'
                    print(warnmsg)
                    r[key] = best[key]
                data[key].append(r[key])
        self.study_data = data
        self.df = sc.dataframe.from_dict(data)
        self.df = self.df.sort_values(by=['mismatch']) # Sort
        return

    def to_json(self, filename=None, indent=2, **kwargs):
        """ Convert the results to JSON """
        order = np.argsort(self.df['mismatch'])
        json = []
        for o in order:
            row = self.df.iloc[o,:].to_dict()
            rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'), pars={})
            for key,val in row.items():
                rowdict['pars'][key] = val
            json.append(rowdict)
        self.json = json
        if filename:
            return sc.savejson(filename, json, indent=indent, **kwargs)
        else:
            return json

    def plot_sims(self, **kwargs):
        """ Plot sims, before and after calibration """
        data = self.data
        if not hasattr(self, 'after_fit'):
            self.check_fit(verbose=False)

        fig = plt.figure()

        # Plot raw values
        plt.subplot(3,1,1)
        for label,df in dict(Data=data, Before=self.before_df, After=self.after_df).items():
            plt.scatter(df.ages, df.inci, label=label)
        plt.ylim(bottom=0)
        plt.xlabel('Age')
        plt.ylabel('Incidence')
        plt.legend()

        # Plot goodness-of-fit
        plt.subplot(3,1,2)
        for i,label,gofs in sc.odict(Before=self.before_gofs, After=self.after_gofs).enumitems():
            x = np.arange(len(gofs))
            dx = 0.3
            plt.bar(x+i*dx, gofs, label=label, width=dx)
        plt.ylim(bottom=0)
        plt.xlabel('Age')
        plt.ylabel('Goodness-of-fit')
        plt.legend()

        # Plot fit
        plt.subplot(3,1,3)
        plt.barh(y=[1,0], width=[self.before_fit, self.after_fit])
        plt.yticks([1,0], ['Before', 'After'])
        plt.xlabel('Total mismatch')
        return fig

    def plot_trend(self, best_thresh=None, fig_kw=None):
        """ Plot the trend in best mismatch over trials """
        df = self.df.sort_values('index') # Make a copy of the dataframe, sorted by trial number
        mismatch = sc.dcp(df['mismatch'].values)
        best_mismatch = np.zeros(len(mismatch))
        for i in range(len(mismatch)):
            best_mismatch[i] = mismatch[:i+1].min()
        smoothed_mismatch = sc.smooth(mismatch)
        fig = plt.figure(**sc.mergedicts(fig_kw))

        ax1 = plt.subplot(2,1,1)
        plt.plot(mismatch, alpha=0.2, label='Original')
        plt.plot(smoothed_mismatch, lw=3, label='Smoothed')
        plt.plot(best_mismatch, lw=3, label='Best')

        ax2 = plt.subplot(2,1,2)
        max_mismatch = mismatch.min()*best_thresh if best_thresh is not None else np.inf
        inds = sc.findinds(mismatch<=max_mismatch)
        plt.plot(best_mismatch, lw=3, label='Best')
        plt.scatter(inds, mismatch[inds], c=mismatch[inds], label='Trials')
        for ax in [ax1, ax2]:
            plt.sca(ax)
            plt.grid(True)
            plt.legend()
            sc.setylim()
            sc.setxlim()
            plt.xlabel('Trial number')
            plt.ylabel('Mismatch')

        sc.figlayout()
        return fig


if __name__ == '__main__':

    # Run in debug mode (serial)
    debug = False
    total_trials = 20

    # Create the base sim
    sim = rabm.RotaABM(
        N = 10_000,
        timelimit = 2,
        to_csv = False,
        verbose = False,
    )

    # Convert the data
    data = process_incidence.process_data()

    # Specify the calibration parameters
    calib_pars = sc.objdict(
        rel_beta = [1.0, 0.5, 1.0],
        reassortment_rate = [0.10, 0.09, 0.11]
    )

    # Run the calibration
    calib = Calibration(
        sim = sim,
        data = data,
        calib_pars = calib_pars,
        total_trials = total_trials,
        debug = debug,
    )
    calib.calibrate()
    calib.check_fit()
    calib.plot_sims()