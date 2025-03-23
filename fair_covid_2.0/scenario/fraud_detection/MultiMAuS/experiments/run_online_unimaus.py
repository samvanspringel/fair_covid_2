"""
This module provides an online API for the Unimaus Simulator. This can be useful in cases where
we want to have other code interacting with the simulator online, and don't necessarily need to
store the generated data in a file.

For a simple example of usage, see __main__ code at the bottom of this module.

@author Dennis Soemers (only the online API: Luisa Zintgraf developed the original simulator)
"""

from datetime import datetime
from scenario.fraud_detection.MultiMAuS.data.features.aggregate_features import AggregateFeatures
from scenario.fraud_detection.MultiMAuS.data.features.apate_graph_features import ApateGraphFeatures
from mesa.time import BaseScheduler
from scenario.fraud_detection.MultiMAuS.simulator import parameters
from scenario.fraud_detection.MultiMAuS.simulator.transaction_model import TransactionModel


class OnlineUnimaus:

    def __init__(self, seed=123, stay_prob_genuine=0.9, stay_prob_fraud=0.99,
                 end_date=datetime(2999, 12, 31), params=None, random_schedule=False):
        """
        Creates an object that can be used to run the simulator online / interactively. This means
        that we can have it generate a bit of data, do something with the data, generate a bit more
        data, do something again, etc. (as opposed to, generating one large batch of data, storing it
        in a file, and then using it in a different program).

        :param end_date:
            Final possible date in the simulation. By default set to 31st December 2999, which allows for
            a sufficiently long simulation run. If set to anything other than None, will override the
            end_date as specified in params
        :param params:
            Parameters passed on to the UniMausTransactionModel. Will use the default parameters if None
        :param random_schedule:
            False by default. If set to True, we use a RandomActivation schedule to shuffle the order in
            which agents are updated every step.
        """
        if params is None:
            params = parameters.get_default_parameters()

        if end_date is not None:
            params['end_date'] = end_date

        if stay_prob_genuine is not None:
            params['stay_prob'][0] = stay_prob_genuine

        if stay_prob_fraud is not None:
            params['stay_prob'][1] = stay_prob_fraud

        if seed is not None:
            params['seed'] = seed

        if random_schedule:
            self.model = TransactionModel(params)
        else:
            self.model = TransactionModel(params, scheduler=BaseScheduler(None))

        self.params = params

        self.aggregate_feature_constructor = None
        self.apate_graph_feature_constructor = None

    def block_cards(self, card_ids, replace_fraudsters=True):
        """
        Blocks the given list of Card IDs (removing all genuine and fraudulent customers with matching
        Card IDs from the simulation).

        NOTE: This function is only intended to be called using Card IDs that are 100% known to have
        been involved in fraudulent transactions. If the list contains more than a single Card ID,
        and the Card ID has not been used in any fraudulent transactions, the function may not be able
        to find the matching customer (due to an optimization in the implementation)

        :param card_ids:
            List of one or more Card IDs to block
        :param replace_fraudsters:
            If True, also replaces the banned fraudsters by an equal number of new fraudsters. True by default
        """
        n = len(card_ids)

        '''
        print("block_cards called!")

        if n == 0:
            print("Not blocking anything")

        for card_id in card_ids:
            print("Blocking ", card_id)
        '''

        if n == 0:
            # nothing to do
            return

        num_banned_fraudsters = 0

        if n == 1:
            # most efficient implementation in this case is simply to loop once through all customers (fraudulent
            # as well as genuine) and compare to our single blocked card ID
            blocked_card_id = card_ids[0]

            for customer in self.model.customers:
                if customer.card_id == blocked_card_id:
                    customer.stay = False

                    # should not be any more customers with same card ID, so can break
                    break

            for fraudster in self.model.fraudsters:
                if fraudster.card_id == blocked_card_id:
                    fraudster.stay = False
                    num_banned_fraudsters += 1

                    # should not be any more fraudsters with same card ID, so can break
                    break
        else:
            # with naive implementation, we'd have n loops through the entire list of customers, which may be expensive
            # instead, we loop through it once to collect only those customers with corrupted cards. Then, we follow
            # up with n loops through that much smaller list of customers with corrupted cards
            compromised_customers = [c for c in self.model.customers if c.card_corrupted]

            for blocked_card_id in card_ids:
                for customer in compromised_customers:
                    if customer.card_id == blocked_card_id:
                        customer.stay = False

                        # should not be any more customers with same card ID, so can break
                        break

                for fraudster in self.model.fraudsters:
                    if fraudster.card_id == blocked_card_id:
                        fraudster.stay = False
                        num_banned_fraudsters += 1

                        # should not be any more fraudsters with same card ID, so can break
                        break

        if replace_fraudsters:
            self.model.add_fraudsters(num_banned_fraudsters)

    def clear_log(self):
        """
        Clears all transactions generated so far from memory
        """
        agent_vars = self.model.log_collector.agent_vars
        for reporter_name in agent_vars:
            agent_vars[reporter_name] = []

    def get_log(self, clear_after=True):
        """
        Returns a log (in the form of a pandas dataframe) of the transactions generated so far.

        :param clear_after:
            If True, will clear the transactions from memory. This means that subsequent calls to get_log()
            will no longer include the transactions that have already been returned in a previous call.
        :return:
            The logged transactions. Returns None if no transactions were logged
        """
        log = self.model.log_collector.get_agent_vars_dataframe()

        if log is None:
            return None

        log.reset_index(drop=True, inplace=True)

        if clear_after:
            self.clear_log()

        return log

    def get_params_string(self):
        """
        Returns a single string describing all of our param values.

        :return:
        """
        output = ""

        for key in self.params:
            output += str(key) + ":" + str(self.params[key]) + "-"

        return output

    def get_seed_str(self):
        return str(self.params['seed'])

    def get_stay_prob_genuine_str(self):
        return str(self.params['stay_prob'][0])

    def get_stay_prob_fraud_str(self):
        return str(self.params['stay_prob'][1])

    def step_simulator(self, num_steps=1):
        """
        Runs num_steps steps of the simulator (simulates num_steps hours of transactions)

        :param num_steps:
            The number of steps to run. 1 by default.
        :return:
            True if we successfully simulated a step, false otherwise
        """
        for step in range(num_steps):
            if self.model.terminated:
                print("WARNING: cannot step simulator because model is already terminated. ",
                      "Specify a later end_date in params to allow for a longer simulation.")
                return False

            self.model.step()

        return True

    def prepare_feature_constructors(self, data):
        """
        Prepares feature constructors (objects which can compute new features for us) using
        a given set of ''training data''. The training data passed into this function should
        NOT be re-used when training predictive models using the new features, because the new
        features will likely be unrealistically accurate on this data (and therefore models
        trained on this data would learn to rely on the new features too much)

        :param data:
            Data used to ''learn'' features
        """
        self.aggregate_feature_constructor = AggregateFeatures(data)
        self.apate_graph_feature_constructor = ApateGraphFeatures(data)

    def print_debug_info(self, data):
        """
        Useful to call from Java, so that we can observe a dataset we want to debug through Python and easily
        print info about it

        :param data:
            Dataset we want to know more about
        """
        print("----- print_debug_info called! -----")

        if data is None:
            print("data is None")
        else:
            print("data is not None")
            print(data.head())
            print("num fraudulent transactions in data = ", data.loc[data["Target"] == 1].shape[0])

    def process_data(self, data):
        """
        Processes the given data, so that it will be ready for use in Machine Learning models. New features
        are added by the feature constructors, features which are no longer necessary afterwards are removed,
        and the Target feature is moved to the back of the dataframe

        NOTE: processing is done in-place

        :param data:
            Data to process
        :return:
            Processed dataframe
        """
        self.apate_graph_feature_constructor.add_graph_features(data)
        self.aggregate_feature_constructor.add_aggregate_features(data)

        # remove non-numeric columns / columns we don't need after adding features
        data.drop(["Global_Date", "Local_Date", "MerchantID", "Currency", "Country",
                   "AuthSteps", "TransactionCancelled", "TransactionSuccessful"],
                  inplace=True, axis=1)

        # move CardID and Target columns to the end
        data = data[[col for col in data if col != "Target" and col != "CardID"] + ["CardID", "Target"]]

        return data

    def update_feature_constructors_unlabeled(self, data):
        """
        Performs an update of existing feature constructors, treating the given new data
        as being unlabeled.

        :param data:
            (unlabeled) new data (should NOT have been passed into prepare_feature_constructors() previously)
        """
        self.aggregate_feature_constructor.update_unlabeled(data)


class DataLogWrapper:

    def __init__(self, dataframe):
        """
        Constructs a wrapper for a data log (in a dataframe). Provides some useful functions to make
        it easier to access this data from Java through jpy. This class is probably not very useful in
        pure Python.

        :param dataframe:
            The dataframe to wrap in an object
        """
        self.dataframe = dataframe

    def get_column_names(self):
        """
        Returns a list of column names

        :return:
            List of column names
        """
        return self.dataframe.columns

    def get_data_list(self):
        """
        Returns a flat list representation of the dataframe

        :return:
        """
        return [item for sublist in self.dataframe.as_matrix().tolist() for item in sublist]

    def get_num_cols(self):
        """
        Returns the number of columns in the dataframe

        :return:
            The number of columns in the dataframe
        """
        return self.dataframe.shape[1]

    def get_num_rows(self):
        """
        Returns the number of rows in the dataframe

        :return:
            The number of rows in the dataframe
        """
        return self.dataframe.shape[0]

if __name__ == '__main__':
    # construct our online simulator
    simulator = OnlineUnimaus()

    # change this value to change how often we run code inside the loop.
    # with n_steps = 1, we run code after every hour of transactions.
    # with n_steps = 2 for example, we would only run code every 2 steps
    n_steps = 1

    # if this is set to False, our simulator will not clear logged transactions after returning them from get_log.
    # This would mean that subsequent get_log calls would also return transactions that we've already seen earlier
    clear_logs_after_return = True

    # if this is set to True, we block card IDs as soon as we observe them being involved in fraudulent transactions
    # (we cheat a bit here by simply observing all true labels, this is just an example usage of API)
    block_fraudsters = True

    # keep running until we fail (which will be after 1 year due to end_date in default parameters)
    while simulator.step_simulator(n_steps):
        # get all transactions generated by the last n_steps (or all steps if clear_logs_after_return == False)
        data_log = simulator.get_log(clear_after=clear_logs_after_return)

        if data_log is not None:
            #print(data_log)

            if block_fraudsters:
                simulator.block_cards(
                    [transaction.CardID for transaction in data_log.itertuples() if transaction.Target == 1])
