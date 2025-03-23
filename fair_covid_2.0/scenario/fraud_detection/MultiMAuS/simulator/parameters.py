from datetime import datetime
from pytz import timezone
from os.path import join
import numpy as np
from scenario.fraud_detection.MultiMAuS.data import utils_data
import pandas as pd


def get_default_parameters():

    aggregated_data = pd.read_csv(join(utils_data.FOLDER_SIMULATOR_INPUT, 'aggregated_data.csv'), index_col=0)

    params = {

        # seed for random number generator of current simulation
        'seed': 666,

        # start and end date of simulation
        'start_date': datetime(2016, 1, 1).replace(tzinfo=timezone('US/Pacific')),
        'end_date': datetime(2016, 12, 31).replace(tzinfo=timezone('US/Pacific')),

        # how much noise we use in the simulation (use more for fraudulent than genuine customers)
        'noise_level': 0.1,

        # number of customers and fraudsters at beginning of simulation
        # (note that this doesn't really influence the total amount of transactions;
        #  for that change the probability of making transactions)
        'num_customers': 3333,
        'num_fraudsters': 55,

        # the initial satisfaction of customers
        'init_satisfaction': 1.,

        # number of fraud cards also used in genuine cards
        'fraud_cards_in_genuine': float(aggregated_data.loc['fraud cards in genuine', 'fraud']),

        # number of merchants at the beginning of simulation
        'num_merchants': int(aggregated_data.loc['num merchants', 'all']),

        # total number of transactions we want in one year
        'trans_per_year': np.array(aggregated_data.loc['transactions'].values, dtype=np.float64)[1:],

        # transactions per day in a month
        'frac_monthday': np.load(join(utils_data.FOLDER_SIMULATOR_INPUT, 'monthday_frac.npy')),
        # transactions per day in a week
        'frac_weekday': np.load(join(utils_data.FOLDER_SIMULATOR_INPUT, 'weekday_frac.npy')),
        # transactions per month in a year
        'frac_month': np.load(join(utils_data.FOLDER_SIMULATOR_INPUT, 'month_frac.npy')),
        # transactions hour in a day
        'frac_hour': np.load(join(utils_data.FOLDER_SIMULATOR_INPUT, 'hour_frac.npy')),

        # countries
        'country_frac': pd.read_csv(join(utils_data.FOLDER_SIMULATOR_INPUT, 'country_frac.csv'), index_col=0),
        # currencies per country
        'currency_per_country': [pd.read_csv(join(utils_data.FOLDER_SIMULATOR_INPUT, 'currency_per_country0.csv'), index_col=[0, 1], header=None),
                                 pd.read_csv(join(utils_data.FOLDER_SIMULATOR_INPUT, 'currency_per_country1.csv'), index_col=[0, 1], header=None)],

        # merchant per currency
        'merchant_per_currency': [
            pd.read_csv(join(utils_data.FOLDER_SIMULATOR_INPUT, 'merchant_per_currency0.csv'), index_col=[0, 1], header=None),
            pd.read_csv(join(utils_data.FOLDER_SIMULATOR_INPUT, 'merchant_per_currency1.csv'), index_col=[0, 1], header=None)],

        # amount per merchant
        'merchant_amount_distr': np.load(join(utils_data.FOLDER_SIMULATOR_INPUT, 'merchant_amount_distr.npy')),

        # probability of doing another transaction
        'stay_prob': np.load(join(utils_data.FOLDER_SIMULATOR_INPUT, 'prob_stay.npy')),
        'stay_after_fraud': np.load(join(utils_data.FOLDER_SIMULATOR_INPUT, 'prob_stay_after_fraud.npy'))
    }

    return params
