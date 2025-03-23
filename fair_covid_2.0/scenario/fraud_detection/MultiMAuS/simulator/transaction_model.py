from scenario.fraud_detection.MultiMAuS.simulator.merchant import Merchant
#from mesa.time import RandomActivation
from scenario.fraud_detection.MultiMAuS.simulator.log_collector import LogCollector
from scenario.fraud_detection.MultiMAuS.simulator import parameters
from mesa import Model
from scenario.fraud_detection.MultiMAuS.authenticators.simple_authenticators import NeverSecondAuthenticator
from scenario.fraud_detection.MultiMAuS.simulator.customers import GenuineCustomer, FraudulentCustomer
from datetime import timedelta
import numpy as np
import pycountry_convert as pc


class TransactionModel(Model):
    def __init__(self, model_parameters, authenticator=NeverSecondAuthenticator(), scheduler=None, seed=123):
        super().__init__(seed=seed)
        self.reset_randomizer(seed)

        # load parameters
        if model_parameters is None:
            model_parameters = parameters.get_default_parameters()
        self.parameters = model_parameters

        # calculate the intrinsic transaction motivation per customer (proportional to number of customers/fraudsters)
        # we keep this separate because then we can play around with the number of customers/fraudsters,
        # but individual behaviour doesn't change
        self.parameters['transaction_motivation'] = np.array([1./self.parameters['num_customers'], 1./self.parameters['num_fraudsters']])

        df = self.parameters['currency_per_country']
        tuples = df[0].index.unique()
        self.countries = {}
        self.countries_r = {}
        self.currencies = {}
        self.currencies_r = {}
        self.continents_countries = {}
        self.continents = {}
        self.continents_r = {}
        c1 = 0
        c2 = 0
        c3 = 0
        for country, currency in tuples:
            if self.countries.get(country) is None:
                self.countries[country] = c1
                self.countries_r[c1] = country
                c1 += 1
                # Store continent of country
                try:
                    continent = pc.country_alpha2_to_continent_code(country)
                except KeyError as e:
                    # print(e)
                    mapping = {
                        'UM': 'OC',  # US Minor Outlying Islands
                    }
                    continent = mapping[country]

                self.continents_countries[country] = continent
                if self.continents.get(continent) is None:
                    self.continents[continent] = c3
                    self.continents_r[c3] = continent
                    c3 += 1

            if self.currencies.get(currency) is None:
                self.currencies[currency] = c2
                self.currencies_r[c2] = currency
                c2 += 1
        # print(self.countries, "\n", self.currencies)
        # print(self.countries_r, "\n", self.currencies_r)
        # print(self.continents_countries)
        # print(self.continents, "\n", self.continents_r)

        # save authenticator for checking transactions
        self.authenticator = authenticator

        # random internal state
        self.random_state = np.random.RandomState(self.parameters["seed"])

        # current date
        self.curr_global_date = self.parameters['start_date']

        # set termination status
        self.terminated = False

        # create merchants, customers and fraudsters
        self.next_customer_id = 0
        self.next_fraudster_id = 0
        self.next_card_id = 0
        self.merchants = self.initialise_merchants()
        self.customers = self.initialise_customers()
        self.fraudsters = self.initialise_fraudsters()

        # set up a scheduler
        self.schedule = scheduler if scheduler is not None else RandomActivation(self)

        # we add to the log collector whether transaction was successful
        self.log_collector = self.initialise_log_collector()

        #
        self.revenue = 0
        self.genuine_transactions = 0
        self.fraudulent_transactions = 0
        self.lost_customers = 0

    @staticmethod
    def initialise_log_collector():
        return LogCollector(
            agent_reporters={"Global_Date": lambda c: c.model.curr_global_date.replace(tzinfo=None),
                             "Local_Date": lambda c: c.local_datetime.replace(tzinfo=None),
                             "CardID": lambda c: c.card_id,
                             "MerchantID": lambda c: c.curr_merchant.unique_id,
                             "Amount": lambda c: c.curr_amount,
                             "Currency": lambda c: c.currency,
                             "Country": lambda c: c.country,
                             "Target": lambda c: c.fraudster,
                             "AuthSteps": lambda c: c.curr_auth_step,
                             "TransactionCancelled": lambda c: c.curr_trans_cancelled,
                             "TransactionSuccessful": lambda c: not c.curr_trans_cancelled},
            model_reporters={
                "Satisfaction": lambda m: sum((customer.satisfaction for customer in m.customers)) / len(m.customers)})

    def inform_attacked_customers(self):
        fraud_card_ids = [f.card_id for f in self.fraudsters if f.active and f.curr_trans_success]
        for card_id in fraud_card_ids:
            customer = next((c for c in self.customers if c.card_id == card_id), None)
            if customer is not None:
                customer.card_got_corrupted()

    def step(self):

        self.pre_step()

        self.schedule.step()

        self.post_step()

    def pre_step(self):  # TODO: added
        # print some logs every month
        if self.curr_global_date.month != (self.curr_global_date - timedelta(hours=1)).month:
            print(self.curr_global_date.date())
            print('num customers:', len(self.customers))
            print('num fraudsters:', len(self.fraudsters))
            print('')

        # this calls the step function of each agent in the schedule (customer, fraudster)
        # self.schedule.agents = []
        # self.schedule.agents.extend(self.customers)
        # self.schedule.agents.extend(self.fraudsters)
        for agent in self.schedule.agent_buffer():
            # print("Agent:", agent.unique_id, agent)
            self.schedule.remove(agent)
        for agent in self.customers:
            # print("Add customer:", agent.unique_id, agent)
            self.schedule.add(agent)
        for agent in self.fraudsters:
            # print("Add fraudster:", agent.unique_id, agent)
            self.schedule.add(agent)

    def post_step(self):  # TODO: added
        # inform the customers whose card got corrupted
        self.inform_attacked_customers()

        # write new transactions to log
        self.log_collector.collect(self)

        # migration of customers/fraudsters
        self.customer_migration()

        # update time
        self.curr_global_date = self.curr_global_date + timedelta(hours=1)

        # check if termination criterion met
        if self.curr_global_date.date() > self.parameters['end_date'].date():
            self.terminated = True

    def process_transaction(self, customer):
        self.authenticator.authorise_transaction(customer)

    def customer_migration(self):

        current_n = len(self.customers)  # TODO: + len(self.fraudsters) ?

        # emigration
        self.customers = [c for c in self.customers if c.stay]
        self.fraudsters = [f for f in self.fraudsters if f.stay]

        self.lost_customers += len(self.customers) - current_n

        # immigration
        self.immigration_customers()
        self.immigration_fraudsters()

    def immigration_customers(self):

        fraudster = 0

        # estimate how many genuine transactions there were
        num_transactions = self.parameters['trans_per_year'][fraudster] / 366 / 24

        # scale by current month
        num_trans_month = num_transactions * 12 * self.parameters['frac_month'][self.curr_global_date.month - 1, fraudster]
        num_transactions = (1 - self.parameters['noise_level']) * num_trans_month + \
                           self.parameters['noise_level'] * num_transactions

        # estimate how many customers on avg left; this many we will add
        num_new_customers = num_transactions * (1 - self.parameters['stay_prob'][fraudster])

        # weigh by mean satisfaction
        social_satisfaction = np.mean([c.satisfaction for c in self.customers])
        num_new_customers *= social_satisfaction

        if num_new_customers > 1:
            num_new_customers += self.random_state.normal(0, 1)
            num_new_customers = int(round(num_new_customers, 0))
            num_new_customers = max([0, num_new_customers])
        else:
            if num_new_customers > self.random_state.uniform(0, 1):
                num_new_customers = 1
            else:
                num_new_customers = 0

        # add as many customers as we think that left
        self.customers.extend([GenuineCustomer(self) for _ in range(num_new_customers)])

    def immigration_fraudsters(self):

        fraudster = 1
        # estimate how many fraudulent transactions there were
        num_transactions = self.parameters['trans_per_year'][fraudster] / 366 / 24
        # scale by current month
        num_trans_month = num_transactions * 12 * self.parameters['frac_month'][self.curr_global_date.month - 1, fraudster]
        num_transactions = (1 - self.parameters['noise_level']) * num_trans_month + \
                           self.parameters['noise_level'] * num_transactions

        # estimate how many fraudsters on avg left
        num_fraudsters_left = num_transactions * (1 - self.parameters['stay_prob'][fraudster])

        if num_fraudsters_left > 1:
            num_fraudsters_left += self.random_state.normal(0, 1)
            num_fraudsters_left = int(round(num_fraudsters_left, 0))
            num_fraudsters_left = max([0, num_fraudsters_left])
        else:
            if num_fraudsters_left > self.random_state.uniform(0, 1):
                num_fraudsters_left = 1
            else:
                num_fraudsters_left = 0

        # add as many fraudsters as we think that left
        self.add_fraudsters(num_fraudsters_left)

    def add_fraudsters(self, num_fraudsters):
        """
        Adds n new fraudsters to the simulation

        :param num_fraudsters:
            The number n of new fraudsters to add
        """
        self.fraudsters.extend([FraudulentCustomer(self) for _ in range(num_fraudsters)])

    def initialise_merchants(self):
        return [Merchant(i, self) for i in range(self.parameters["num_merchants"])]

    def initialise_customers(self):
        return [GenuineCustomer(self) for _ in range(self.parameters['num_customers'])]

    def initialise_fraudsters(self):
        return [FraudulentCustomer(self) for _ in range(self.parameters["num_fraudsters"])]

    def get_next_customer_id(self, fraudster):
        # if not fraudster:
        #     next_id = self.next_customer_id
        #     self.next_customer_id += 1
        # else:
        #     next_id = self.next_fraudster_id
        #     self.next_fraudster_id += 1
        next_id = self.next_customer_id
        self.next_customer_id += 1
        return next_id

    def get_next_card_id(self):
        next_id = self.next_card_id
        self.next_card_id += 1
        return next_id
