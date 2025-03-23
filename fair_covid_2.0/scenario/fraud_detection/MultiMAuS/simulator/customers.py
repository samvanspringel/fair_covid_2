import numpy as np
from pytz import timezone, country_timezones
from scenario.fraud_detection.MultiMAuS.simulator.customer_abstract import AbstractCustomer


class BaseCustomer(AbstractCustomer):
    def __init__(self, transaction_model, fraudster):
        """
        Base class for customers/fraudsters that support uni-modal authentication.
        :param transaction_model: 
        :param fraudster: 
        """

        unique_id = transaction_model.get_next_customer_id(fraudster)
        super().__init__(unique_id, transaction_model, fraudster)

        # initialise probability of making a transaction per month/hour/...
        self.noise_level = self.params['noise_level']

        # average number of transaction per hour in general; varies per customer
        self.avg_trans_per_hour = self.initialise_avg_trans_per_hour()

        # initialise transaction probabilities per month/monthday/weekday/hour
        self.trans_prob_month, self.trans_prob_monthday, self.trans_prob_weekday, self.trans_prob_hour = self.initialise_transaction_probabilities()

        # whether the current transaction was cancelled by the customer
        self.curr_trans_cancelled = False

    def decide_making_transaction(self):
        # reset that the current transaction was not cancelled
        self.curr_trans_cancelled = False
        if self.stay:
            make_transaction = self.get_transaction_prob() > self.random_state.uniform(0, 1)
        else:
            make_transaction = False
        return make_transaction

    def post_process_transaction(self):
        # decide whether to stay
        self.stay = self.stay_after_transaction()

    def get_transaction_prob(self):

        # get the current local time
        self.local_datetime = self.get_local_datetime()

        # get the average transactions per hour
        trans_prob = self.avg_trans_per_hour

        # now weigh by probabilities of transactions per month/week/...
        trans_prob *= 12 * self.trans_prob_month[self.local_datetime.month - 1]
        trans_prob *= 24 * self.trans_prob_hour[self.local_datetime.hour]
        trans_prob *= 30.5 * self.trans_prob_monthday[self.local_datetime.day - 1]
        trans_prob *= 7 * self.trans_prob_weekday[self.local_datetime.weekday()]

        return trans_prob

    def get_local_datetime(self):
        # convert global to local date (first add global timezone info, then convert to local)
        local_datetime = self.model.curr_global_date
        local_datetime = local_datetime.astimezone(timezone(country_timezones(self.country)[0]))
        return local_datetime

    def get_curr_merchant(self):
        """
        Can be called at each transaction; will select a merchant to buy from.
        :return:    merchant ID
        """
        merchant_prob = self.params['merchant_per_currency'][self.fraudster]
        merchant_prob = merchant_prob.loc[self.currency]
        merchant_ID = self.random_state.choice(merchant_prob.index.values, p=merchant_prob.values.flatten())
        return next(m for m in self.model.merchants if m.unique_id == merchant_ID)

    def get_curr_amount(self):
        return self.curr_merchant.get_amount(self)

    def stay_after_transaction(self):
        return self.get_staying_prob() > self.random_state.uniform(0, 1)

    def get_staying_prob(self):
        return self.params['stay_prob'][self.fraudster]

    def initialise_country(self):
        country_frac = self.params['country_frac']
        return self.random_state.choice(country_frac.index.values, p=country_frac.iloc[:, self.fraudster].values)

    def initialise_currency(self):
        currency_prob = self.params['currency_per_country'][self.fraudster]
        currency_prob = currency_prob.loc[self.country]
        return self.random_state.choice(currency_prob.index.values, p=currency_prob.values.flatten())

    def initialise_card_id(self):
        return self.model.get_next_card_id()

    def initialise_transaction_probabilities(self):
        # transaction probability per month
        trans_prob_month = self.params['frac_month'][:, self.fraudster]
        trans_prob_month = self.random_state.multivariate_normal(trans_prob_month, np.eye(12) * self.noise_level / 1200)
        trans_prob_month[trans_prob_month < 0] = 0

        # transaction probability per day in month
        trans_prob_monthday = self.params['frac_monthday'][:, self.fraudster]
        trans_prob_monthday = self.random_state.multivariate_normal(trans_prob_monthday, np.eye(31) * self.noise_level / 305)
        trans_prob_monthday[trans_prob_monthday < 0] = 0

        # transaction probability per weekday (we assume this differs per individual)
        trans_prob_weekday = self.params['frac_weekday'][:, self.fraudster]
        trans_prob_weekday = self.random_state.multivariate_normal(trans_prob_weekday, np.eye(7) * self.noise_level / 70)
        trans_prob_weekday[trans_prob_weekday < 0] = 0

        # transaction probability per hour (we assume this differs per individual)
        trans_prob_hour = self.params['frac_hour'][:, self.fraudster]
        trans_prob_hour = self.random_state.multivariate_normal(trans_prob_hour, np.eye(24) * self.noise_level / 240)
        trans_prob_hour[trans_prob_hour < 0] = 0

        return trans_prob_month, trans_prob_monthday, trans_prob_weekday, trans_prob_hour

    def initialise_avg_trans_per_hour(self):
        trans_per_year = self.params['trans_per_year'][self.fraudster]
        rand_addition = self.random_state.normal(0, self.noise_level * trans_per_year)
        if trans_per_year + rand_addition > 0:
            trans_per_year += rand_addition
        avg_trans_per_hour = trans_per_year / 366. / 24.
        avg_trans_per_hour *= self.params['transaction_motivation'][self.fraudster]
        return avg_trans_per_hour


class GenuineCustomer(BaseCustomer):
    def __init__(self, transaction_model, satisfaction=1):

        super().__init__(transaction_model, fraudster=False)

        # add field for whether the credit card was corrupted by a fraudster
        self.card_corrupted = False

        # field whether current transaction was authorised or not
        self.curr_auth_step = 0

        # initialise the customer's patience (optimistically)
        self.patience = self.random_state.beta(10, 2)

        # instantiate the customer's satisfaction
        self.satisfaction = satisfaction

    def stay_after_transaction(self):
        stay_prob = self.satisfaction * self.params['stay_prob'][self.fraudster]
        return (1-stay_prob) <= self.random_state.uniform(0, 1)

    def card_got_corrupted(self):
        self.card_corrupted = True

    def get_transaction_prob(self):
        return self.satisfaction * super().get_transaction_prob()

    def decide_making_transaction(self):
        """
        For a genuine customer, we add the option of leaving
        when the customer's card was subject to fraud
        :return:
        """
        # reset authentication step count
        self.curr_auth_step = 0

        # if the card was corrupted, the user is more likely to leave
        if self.card_corrupted:
            if self.params['stay_after_fraud'] < self.random_state.uniform(0, 1):
                self.stay = False

                # can skip the entire super().decide_making_transaction() computation
                return False

        return super().decide_making_transaction()

    def post_process_transaction(self):
        self.update_satisfaction()
        super().post_process_transaction()

    def update_satisfaction(self):
        """
        Adjust the satisfaction of the user after a transaction was made.
        :return: 
        """
        # if the customer cancelled the transaction, the satisfaction goes down by 5%
        if self.curr_trans_cancelled:
            self.satisfaction *= 0.95
        else:
            # if no authentication was done, the satisfaction goes up by 0.01
            if self.curr_auth_step == 0:
                self.satisfaction *= 1.01
            # otherwise, it goes down by 1%
            else:
                self.satisfaction *= 0.99
        self.satisfaction = min([1, self.satisfaction])
        self.satisfaction = max([0, self.satisfaction])

    def give_authentication(self):
        """
        Authenticate self; this can be called several times per transaction.
        Returns the authentication quality.
        :return:
        """
        curr_patience = 0.8 * self.patience + 0.2 * self.curr_amount/self.curr_merchant.max_amount
        if curr_patience > self.random_state.uniform(0, 1):
            auth_quality = 1
        else:
            # cancel the transaction
            self.curr_trans_cancelled = True
            auth_quality = None

        self.curr_auth_step += 1

        return auth_quality


class FraudulentCustomer(BaseCustomer):
    def __init__(self, transaction_model):
        super().__init__(transaction_model, fraudster=True)

    def initialise_card_id(self):
        """
        Pick a card either by using a card from an existing user,
        or a completely new one (i.e., from customers unnknown to the processing platform)
        :return: 
        """
        if self.params['fraud_cards_in_genuine'] > self.random_state.uniform(0, 1):
            # the fraudster picks a customer...
            # ... (1) from a familiar country
            fraudster_countries = self.params['country_frac'].index[self.params['country_frac']['fraud'] !=0].values
            # ... (2) from a familiar currency
            fraudster_currencies = self.params['currency_per_country'][1].index.get_level_values(1).unique()
            # ... (3) that has already made a transaction
            customers_with_active_cards = [c for c in self.model.customers if c.card_id is not None]
            # now pick the fraud target (if there are no targets get own credit card)
            try:
                customer = self.random_state.choice([c for c in customers_with_active_cards if (c.country in fraudster_countries) and (c.currency in fraudster_currencies)])
                # get the information from the target
                card = customer.card_id
                self.country = customer.country
                self.currency = customer.currency
            except ValueError:
                card = super().initialise_card_id()
        else:
            card = super().initialise_card_id()
        return card

    def give_authentication(self):
        """
        Authenticate self; this can be called several times per transaction.
        Returns the authentication quality.
        :return:
        """
        # we assume that the fraudster cannot provide a second authentication
        self.curr_trans_cancelled = True
        return None
