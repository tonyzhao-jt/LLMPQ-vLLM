import os
import numpy as np
import pickle
import pandas as pd
import math
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from . import mops
from .comm.utils import get_comm_payload_size


def fit_cost_model(profiled_result_folder, cost_model_store_path, target_device=None):
    from sklearn.model_selection import train_test_split

    assert os.path.exists(
        profiled_result_folder
    ), f"Folder {profiled_result_folder} does not exist."
    assert target_device, "target_device cannot be None"
    # read the profiled result from folder
    # list the dir to find the deviced related profile result
    profiled_result_files = os.listdir(profiled_result_folder)
    # filter the file with the target device
    profiled_result_files = [
        os.path.join(profiled_result_folder, f)
        for f in profiled_result_files
        if target_device in f
    ]
    # read all the profiled result and form a big df
    df = pd.concat([pd.read_csv(f) for f in profiled_result_files])
    # average result with same column value for : shard,h1,h2,bit,batch_size,input_seq_length,past_seq_length,
    df = (
        df.groupby(
            [
                "shard",
                "h1",
                "h2",
                "bit",
                "batch_size",
                "input_seq_length",
                "past_seq_length",
            ]
        )
        .mean()
        .reset_index()
    )
    profile_df = df

    # first get shard = 0: ATTENTION MODELS
    # then get shard = 1: FFN MODELS
    for target_shard in [0, 1]:
        df = profile_df[profile_df["shard"] == target_shard]
        df = df[df["lat_avg"] < 99998]

        if target_shard == 0:
            df[
                [
                    "weight_size",
                    "qkv_act_size",
                    "kv_concat_size",
                    "bmm_act_size",
                    "layer_norm_size",
                    "dequant_size",
                ]
            ] = df.apply(
                lambda row: mops.SELF_ATTN_MOPS_PARAMS(
                    row["batch_size"],
                    row["h1"],
                    row["input_seq_length"] + row["past_seq_length"],
                    row["bit"],
                ),
                axis=1,
                result_type="expand",
            )
            X = df[
                [
                    "weight_size",
                    "qkv_act_size",
                    "kv_concat_size",
                    "bmm_act_size",
                    "layer_norm_size",
                    "dequant_size",
                ]
            ]
        else:
            df[["weight_size", "act_size", "layer_norm_size", "dequant_size"]] = (
                df.apply(
                    lambda row: mops.FFN_MOPS_PARAMS(
                        row["batch_size"], row["h1"], row["h2"], row["bit"]
                    ),
                    axis=1,
                    result_type="expand",
                )
            )
            X = df[["weight_size", "act_size", "layer_norm_size", "dequant_size"]]
        X = sm.add_constant(X)
        y = df["lat_avg"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit an OLS regression model on the training data
        X_train = sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train).fit()

        # Print the model summary
        print(model.summary())

        # Generate predictions for the testing data
        X_test = sm.add_constant(X_test)
        y_pred = model.predict(X_test)

        # Compute the prediction errors (residuals) for the testing data
        residuals = y_test - y_pred

        # Print the mean squared error (MSE) and the mean absolute error (MAE) for the testing data
        mse = np.mean(residuals**2)
        mae = np.mean(abs(residuals))
        print("Mean squared error (MSE) for the testing data:", mse)
        print("Mean absolute error (MAE) for the testing data:", mae)

        with open(
            f"{cost_model_store_path}/{target_device}_{target_shard}_lat_model.pkl",
            "wb",
        ) as f:
            pickle.dump(model, f)


def find_pairs(n):
    pairs = []
    for i in range(1, n + 1):
        if n % i == 0:
            pairs.append((i, n // i))
    return pairs


class NonNegativeLinearRegression(LinearRegression):

    def predict(self, X):
        predictions = super().predict(X)
        return predictions.clip(min=0)


class LatCostModel:
    def __init__(
        self, device_names=[], cost_model_store_path: str = "./leanred_cost_model"
    ) -> None:
        self.device_names = device_names
        self.has_hypers = False
        self.has_profiled = False
        self.has_fit = False
        self.profiled_data = {}
        self.regression_models = {}

        self.profiled_prepost_data = {}
        self.has_fit_prepost = False

        self.eval_metric_name = "mse"

        # check if the path exists
        if not os.path.exists(cost_model_store_path):
            os.makedirs(cost_model_store_path)
        self.cost_model_store_path = cost_model_store_path

    def device_in_cost_model(self, device_name):
        return device_name in self.device_names

    def register_hyper_params(self, b, s, i, h1, h2):
        self.b = b
        self.s = s  # prompt length
        self.i = i
        self.h1 = h1
        self.h2 = h2
        self.has_hypers = True

    def get_available_chunks(self):
        # get all prime factorization of b
        assert self.has_hypers, "Hyper params not registered."
        available_paris = find_pairs(
            self.b
        )  # return micro batch size and number of chunks
        return available_paris

    def change_bs(self, b):
        assert self.has_hypers, "Hyper params not registered."
        self.b = b

    def fetch_lat(self, device_name, shard, b, s, i, h1, h2, bit):
        profiled_data_device = self.profiled_data[device_name]
        # columns: shard,h1,h2,bit,batch_size,input_seq_length,past_seq_length,lat_avg,mem_weight,mem_kv,mem_embedding,mem_all
        # fetch data with hyper params
        profiled_data_device = profiled_data_device[
            (profiled_data_device["shard"] == shard)
            & (profiled_data_device["h1"] == h1)
            & (profiled_data_device["h2"] == h2)
            & (profiled_data_device["bit"] == str(bit))
            & (profiled_data_device["batch_size"] == b)
            & (profiled_data_device["input_seq_length"] == s)
            & (profiled_data_device["past_seq_length"] == i)
        ]
        if len(profiled_data_device) == 0:
            return None
        # lat_avg
        lat_avg = profiled_data_device["lat_avg"].values[0]
        return lat_avg

    # pure profiler
    # fatch prefill result
    # s is the prompt length
    def fetch_prefill(self, s, shard, device_name, bit):
        # input_seq = s
        # past_seq = 0
        i = 0
        return self.fetch_lat(device_name, shard, self.b, s, i, self.h1, self.h2, bit)

    def fetch_prefill_use_hyper_s(self, shard, device_name, bit):
        s = self.s
        return self.fetch_prefill(s, shard, device_name, bit)

    # fetch decoding result
    # i is the past sequence length
    def fetch_decode(self, i, shard, device_name, bit):
        # input_seq = s
        # past_seq = s
        s = 1  # one token
        return self.fetch_lat(device_name, shard, self.b, s, i, self.h1, self.h2, bit)

    def fetch_decode_use_hyper_i(self, shard, device_name, bit):
        i = self.i
        return self.fetch_decode(i, shard, device_name, bit)

    # following are pure analytical model
    def update_profiled_result(self, profiled_folder):
        # list file under the folder
        for file in os.listdir(profiled_folder):
            # end with .csv
            if file.endswith(".csv"):
                # get device name
                target_device = None
                for device_name in self.device_names:
                    if device_name in file:
                        target_device = device_name
                        break
                if target_device is None:
                    continue
                if target_device not in self.profiled_data:
                    self.profiled_data[target_device] = pd.read_csv(
                        os.path.join(profiled_folder, file)
                    )
                else:
                    # update the pandas array
                    self.profiled_data[target_device] = self.profiled_data[
                        target_device
                    ]._append(pd.read_csv(os.path.join(profiled_folder, file)))
                # drop the row with lat_avg > 99998
                self.profiled_data[target_device] = self.profiled_data[target_device][
                    self.profiled_data[target_device]["lat_avg"] < 99998
                ]
                self.profiled_data[target_device]["bit"] = self.profiled_data[
                    target_device
                ]["bit"].astype(str)

        # check whether each device has one
        for device_name in self.device_names:
            if device_name not in self.profiled_data:
                print(f"Cannot find profiling result for {device_name}, pls add later")

        # read profiled data
        if not self.has_profiled:
            self.has_profiled = True

        # following are pure analytical model

    def update_profiled_prepost_result(self, profiled_folder):
        # list file under the folder
        for file in os.listdir(profiled_folder):
            # end with .csv
            if file.endswith(".csv"):
                # get device name
                target_device = None
                for device_name in self.device_names:
                    if device_name in file:
                        target_device = device_name
                        break
                if target_device is None:
                    continue
                if target_device not in self.profiled_prepost_data:
                    self.profiled_prepost_data[target_device] = pd.read_csv(
                        os.path.join(profiled_folder, file)
                    )
                else:
                    # update the pandas array
                    self.profiled_prepost_data[target_device] = (
                        self.profiled_prepost_data[target_device]._append(
                            pd.read_csv(os.path.join(profiled_folder, file))
                        )
                    )
                # drop the row with lat_avg > 99998
                self.profiled_prepost_data[target_device] = self.profiled_prepost_data[
                    target_device
                ][self.profiled_prepost_data[target_device]["time"] < 99998]

        # check whether each device has one
        for device_name in self.device_names:
            if device_name not in self.profiled_prepost_data:
                print(f"Cannot find profiling result for {device_name}, pls add later")

        # read profiled data
        if not self.has_fit_prepost:
            self.has_fit_prepost = True

    def fetch_prepost_lat(self, device_name, stage, batch_size, prompt_length):
        # model_name,model_size,h1,h2,batch_size,prompt_length,stage,time
        # input_seq = s
        # past_seq = 0
        h1, h2 = self.h1, self.h2
        profiled_data = self.profiled_prepost_data[device_name]
        profiled_data_device = profiled_data[
            (profiled_data["prompt_length"] == prompt_length)
            & (profiled_data["batch_size"] == batch_size)
            & (profiled_data["stage"] == stage)
            & (profiled_data["h1"] == h1)
            & (profiled_data["h2"] == h2)
        ]
        # if profiled data is 0 and stage is 1: decode,
        # then find the closest prompt length
        if len(profiled_data_device) == 0 and stage == 1:
            print(
                "Cannot find profiled data for this prompt length{}, find the closest one".format(
                    prompt_length
                )
            )
            profiled_data = profiled_data[
                (profiled_data["batch_size"] == batch_size)
                & (profiled_data["stage"] == stage)
                & (profiled_data["h1"] == h1)
                & (profiled_data["h2"] == h2)
            ]
            profiled_data_device = profiled_data.iloc[
                (profiled_data["prompt_length"] - prompt_length).abs().argsort()[:1]
            ]

        # lat_avg
        lat_avg = profiled_data_device["time"].values[0]
        return lat_avg

    def verbose_regression_names(self):
        assert len(self.regression_models) > 0, "Please load regression models first."
        for device_name in self.device_names:
            print(
                f"Device {device_name} has {len(self.regression_models[device_name])} regression models."
            )
            # print(f"Regression models are: {self.regression_models[device_name].keys()}")

    # control the variables used in the cost model
    def fetch_prefill_variables_in_data(self, profiled_data_device):
        # Extract batch size, input sequence length, and past sequence length from profiled_data_device
        b = profiled_data_device["batch_size"].values
        s = profiled_data_device["input_seq_length"].values
        i = profiled_data_device["past_seq_length"].values
        bs = b * s
        bs_sq = b * s**2
        bt = b * (s + i)

        prefill_index = profiled_data_device["past_seq_length"] == 0
        X = np.column_stack(
            (
                b[prefill_index],
                s[prefill_index],
                bs[prefill_index],
                bs_sq[prefill_index],
            )
        )
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return X

    def fetch_prefill_variables_in_inf(self, b, s, i, h1, h2, bit):
        # Extract batch size, input sequence length, and past sequence length from profiled_data_device
        bs = b * s
        bs_sq = b * s**2
        bt = b * (s + i)

        X = np.column_stack((b, s, bs, bs_sq))
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return X

    def fetch_decode_variables_in_data(self, profiled_data_device):
        # Extract batch size, input sequence length, and past sequence length from profiled_data_device
        b = profiled_data_device["batch_size"].values
        s = profiled_data_device["input_seq_length"].values
        i = profiled_data_device["past_seq_length"].values
        bs = b * s
        bs_sq = b * s**2
        bt = b * (s + i)

        prefill_index = profiled_data_device["past_seq_length"] == 0
        X = np.column_stack((bt[~prefill_index], b[~prefill_index], i[~prefill_index]))
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return X

    def fetch_decode_variables_in_inf(self, b, s, i, h1, h2, bit):
        # Extract batch size, input sequence length, and past sequence length from profiled_data_device
        bs = b * s
        bs_sq = b * s**2
        bt = b * (s + i)

        X = np.column_stack((bt, b, i))
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return X

    def fit_metric(self, model, X, y, phase="prefill"):
        if self.eval_metric_name.lower() == "mse":
            y_pred = model.predict(X)
            mse_prefill = mean_squared_error(y, y_pred)
            print(f"{phase} MSE: {mse_prefill:.4f}", end="|")
            return mse_prefill
        elif self.eval_metric_name.lower() == "r2":
            fit_value = model.score(X, y)
            print(f"{phase} R^2: {fit_value:.4f}", end="|")
            return fit_value

    def load_regression_cost_model(self):
        assert len(self.profiled_data) > 0, "Please update profiled result first."
        all_err = []
        for device_name in self.device_names:
            self.regression_models[device_name] = {}
            device_profile_data = self.profiled_data[device_name]
            # generate a fitted result for each shard, each h1 h2, bit. leaving batch_size and past_seq_length as variables
            # columns: shard,h1,h2,bit,batch_size,input_seq_length,past_seq_length,lat_avg,mem_weight,mem_kv,mem_embedding,mem_all
            # for each shard, h1, h2, bit, we fit a model
            shards = device_profile_data["shard"].unique()
            h_pairs = device_profile_data[["h1", "h2"]].drop_duplicates()
            bits = device_profile_data["bit"].unique()
            for shard in shards:
                for h1, h2 in h_pairs.values:
                    for bit in bits:
                        model_name = f"{device_name}_{shard}_{h1}_{h2}_{bit}.pkl"
                        print("Loading model: ", model_name)
                        # load model if exists
                        if os.path.exists(
                            os.path.join(self.cost_model_store_path, model_name)
                        ):
                            self.regression_models[device_name][model_name] = model = (
                                joblib.load(
                                    os.path.join(self.cost_model_store_path, model_name)
                                )
                            )
                            # verify the model if has data
                            profiled_data_device = device_profile_data[
                                (device_profile_data["shard"] == shard)
                                & (device_profile_data["h1"] == h1)
                                & (device_profile_data["h2"] == h2)
                                & (device_profile_data["bit"] == str(bit))
                            ]

                            model_prefill = model["prefill"]
                            model_decode = model["decode"]
                            # prefill stage
                            # get value with past_seq_length = 0
                            prefill_index = profiled_data_device["past_seq_length"] == 0
                            profiled_data_device_prefill = profiled_data_device[
                                prefill_index
                            ]
                            profiled_data_device_decode = profiled_data_device[
                                ~prefill_index
                            ]

                            X = self.fetch_prefill_variables_in_data(
                                profiled_data_device_prefill
                            )
                            if len(X) == 0:
                                print(
                                    f"Cannot find profiled data for {model_name}, skip"
                                )
                                continue
                            y = profiled_data_device_prefill["lat_avg"].values
                            model_prefill = LinearRegression().fit(X, y)
                            err = self.fit_metric(model_prefill, X, y, phase="prefill")
                            all_err.append(err)

                            # decode stage
                            X = self.fetch_decode_variables_in_data(
                                profiled_data_device_decode
                            )
                            if len(X) == 0:
                                print(
                                    f"Cannot find profiled data for {model_name}, skip"
                                )
                                continue
                            y = profiled_data_device_decode["lat_avg"].values
                            model_decode = LinearRegression().fit(X, y)
                            err_dec = self.fit_metric(
                                model_decode, X, y, phase="decode"
                            )
                            all_err.append(err_dec)
                            print()
                            # print(f'MSE: {mse:.3f}')
                            # print(f'Intercept: {model.intercept_}')
                            # print(f'Coefficients: {model.coef_}')
                        else:
                            print(
                                f"Cannot find regression model {model_name} for device {device_name}. pls run fit"
                            )
                            # raise Exception(f"Cannot find regression model {model_name} for device {device_name}. pls run fit")
        print("Verified all regress models, maximum mse: ", max(all_err))
        self.has_fit = True

    def fit_regression_cost_model(self, store=True):
        # start fitting...
        # according our usage, we only care about the past_length's impact on the latency
        # when others are fixed.
        assert len(self.profiled_data) > 0, "Please update profiled result first."

        for device_name in self.device_names:
            device_profile_data = self.profiled_data[device_name]
            # generate a fitted result for each shard, each h1 h2, bit. leaving batch_size and past_seq_length as variables
            # columns: shard,h1,h2,bit,batch_size,input_seq_length,past_seq_length,lat_avg,mem_weight,mem_kv,mem_embedding,mem_all
            # for each shard, h1, h2, bit, we fit a model
            shards = device_profile_data["shard"].unique()
            h_pairs = device_profile_data[["h1", "h2"]].drop_duplicates()
            bits = device_profile_data["bit"].unique()
            for shard in shards:
                for h1, h2 in h_pairs.values:
                    for bit in bits:
                        model_name = f"{device_name}_{shard}_{h1}_{h2}_{bit}.pkl"
                        print("Est model: ", model_name)
                        # fetch data with hyper params
                        profiled_data_device = device_profile_data[
                            (device_profile_data["shard"] == shard)
                            & (device_profile_data["h1"] == h1)
                            & (device_profile_data["h2"] == h2)
                            & (device_profile_data["bit"] == str(bit))
                        ]
                        if len(profiled_data_device) == 0:
                            print(f"Cannot find profiled data for {model_name}, skip")
                            continue

                        # fit a model for prefill stage and another model for decode stage
                        # columns: shard,h1,h2,bit,batch_size,input_seq_length,past_seq_length,lat_avg,mem_weight,mem_kv,mem_embedding,mem_all
                        # prefill stage
                        # get value with past_seq_length = 0
                        prefill_index = profiled_data_device["past_seq_length"] == 0
                        profiled_data_device_prefill = profiled_data_device[
                            prefill_index
                        ]
                        profiled_data_device_decode = profiled_data_device[
                            ~prefill_index
                        ]
                        # add b and past_sequence length for check?

                        X = self.fetch_prefill_variables_in_data(
                            profiled_data_device_prefill
                        )
                        if len(X) == 0:
                            print(f"Cannot find profiled data for {model_name}, skip")
                            continue
                        y = profiled_data_device_prefill["lat_avg"].values
                        model_prefill = LinearRegression().fit(X, y)
                        self.fit_metric(model_prefill, X, y, phase="prefill")

                        # decode stage
                        X = self.fetch_decode_variables_in_data(
                            profiled_data_device_decode
                        )
                        if len(X) == 0:
                            print(f"Cannot find profiled data for {model_name}, skip")
                            continue
                        y = profiled_data_device_decode["lat_avg"].values
                        model_decode = LinearRegression().fit(X, y)
                        self.fit_metric(model_decode, X, y, phase="decode")
                        print()

                        model = {"prefill": model_prefill, "decode": model_decode}

                        # store the model
                        if device_name not in self.regression_models:
                            self.regression_models[device_name] = {}
                        self.regression_models[device_name][model_name] = model

                        if store:
                            joblib.dump(
                                model,
                                os.path.join(
                                    self.cost_model_store_path, f"{model_name}"
                                ),
                            )

        self.has_fit = True

    def predict(self, device_name, shard, b, s, i, h1, h2, bit):
        assert self.has_fit, "Cost model is not fitted."
        assert (
            device_name in self.regression_models
        ), f"Cannot find regression model for {device_name}"
        model_name = f"{device_name}_{shard}_{h1}_{h2}_{bit}.pkl"
        if model_name not in self.regression_models[device_name]:
            # print(f"Cannot find regression model for {model_name}")
            return None
        bs = b * s
        bs_sq = b * s**2
        bt = b * (s + i)
        model = self.regression_models[device_name][model_name]
        if i == 0:
            # prefill
            model = model["prefill"]
            X = self.fetch_prefill_variables_in_inf(b, s, i, h1, h2, bit)
            return model.predict(X)[0]
        else:
            # decode
            model = model["decode"]
            X = self.fetch_decode_variables_in_inf(b, s, i, h1, h2, bit)
            return model.predict(X)[0]

    def predict_same_bit(self, device_name, b, s, i, h1, h2, bit):
        assert self.has_fit, "Cost model is not fitted."
        assert (
            device_name in self.regression_models
        ), f"Cannot find regression model for {device_name}"
        shard = 2
        model_name = f"{device_name}_{shard}_{h1}_{h2}_{bit}.pkl"
        if model_name not in self.regression_models[device_name]:
            # print(f"Cannot find regression model for {model_name}")
            return None
        bs = b * s
        bs_sq = b * s**2
        bt = b * (s + i)
        model = self.regression_models[device_name][model_name]
        if i == 0:
            # prefill
            model = model["prefill"]
            X = self.fetch_prefill_variables_in_inf(b, s, i, h1, h2, bit)
            return model.predict(X)[0]
        else:
            # decode
            model = model["decode"]
            X = self.fetch_decode_variables_in_inf(b, s, i, h1, h2, bit)
            return model.predict(X)[0]

    def predict_same_bit_with_b_s_i_bit(self, device_name, b, s, i, bit):
        return self.predict_same_bit(device_name, b, s, i, self.h1, self.h2, bit)

    def predict_same_bit_by_profiled_with_b_s_i_bit(self, device_name, b, s, i, bit):
        shard = 2
        return self.fetch_lat(device_name, shard, b, s, i, self.h1, self.h2, bit)

    def predict_by_model_with_b_s_i_bit(self, device_name, shard, b, s, i, bit):
        return self.predict(device_name, shard, b, s, i, self.h1, self.h2, bit)

    def predict_by_profiled_with_b_s_i_bit(self, device_name, shard, b, s, i, bit):
        return self.fetch_lat(device_name, shard, b, s, i, self.h1, self.h2, bit)

    # only use hyper to predict.
    def predict_with_hyper(self, device_name, shard, bit):
        assert self.has_hypers, "Hyper parameters are not registered."
        return self.predict(device_name, shard, self.b, self.i, self.h1, self.h2, bit)

    def predict_with_profiled(self, device_name, shard, bit):
        assert self.has_profiled, "Profiled data is not registered."
        return self.fetch_lat_with_hyper(shard, device_name, bit)


# produce latency prediction
def lat_prediction(
    lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=True
):
    stage_lat = 0
    if atten_bit == ffn_bit:
        if not use_profiler_prediction:
            lat = lat_cost_model.predict_same_bit_with_b_s_i_bit(
                D_name, b, s, i, atten_bit
            )
        else:
            lat = lat_cost_model.predict_same_bit_by_profiled_with_b_s_i_bit(
                D_name, b, s, i, atten_bit
            )
        if lat is None:
            stage_lat = float("inf")
        else:
            stage_lat = lat
    else:
        if not use_profiler_prediction:
            atten_lat = lat_cost_model.predict_by_model_with_b_s_i_bit(
                D_name, 0, b, s, i, atten_bit
            )
            ffn_lat = lat_cost_model.predict_by_model_with_b_s_i_bit(
                D_name, 1, b, s, i, ffn_bit
            )
        else:
            atten_lat = lat_cost_model.predict_by_profiled_with_b_s_i_bit(
                D_name, 0, b, s, i, atten_bit
            )
            ffn_lat = lat_cost_model.predict_by_profiled_with_b_s_i_bit(
                D_name, 1, b, s, i, ffn_bit
            )

        if atten_lat is None:
            atten_lat = float("inf")
        if ffn_lat is None:
            ffn_lat = float("inf")
        stage_lat = atten_lat + ffn_lat
    return stage_lat


def stage_pure_exe_latency(
    D_name, bits, lat_cost_model, b, s, i, use_profiler_prediction=False
):
    stage_lat = 0
    for _, bit in enumerate(bits):
        atten_bit, ffn_bit = bit
        stage_lat += lat_prediction(
            lat_cost_model,
            D_name,
            b,
            s,
            i,
            atten_bit,
            ffn_bit,
            use_profiler_prediction=use_profiler_prediction,
        )
        # assert stage_lat > 0, "stage_lat should be positive, but got {}.".format(stage_lat)
    return stage_lat


def calculate_max_stage_lat(
    D,
    use_plan,
    cost_model_pack,
    b,
    s=1,
    i=1,
    use_profiler_prediction=False,
    comm_size=0,
):
    lat_cost_model, comm_cost_model = cost_model_pack

    minmax_lat = 0
    stage_sum = 0

    stage_lat_list = []
    comm_lat_list = []
    for device_rank, shard_strategy in use_plan.items():
        D_name = D[device_rank]
        bits = [layer_spec["bits"] for layer_idx, layer_spec in shard_strategy.items()]
        stage_lat = stage_pure_exe_latency(
            D_name,
            bits,
            lat_cost_model,
            b,
            s,
            i,
            use_profiler_prediction=use_profiler_prediction,
        )
        # next stage
        next_stage = (device_rank + 1) % len(D)
        t_comm = comm_cost_model.predict_comm_time(device_rank, next_stage, comm_size)
        # minmax throughput
        minmax_lat = max(minmax_lat, stage_lat, t_comm)
        stage_sum += stage_lat
        stage_lat_list.append(stage_lat)
        comm_lat_list.append(t_comm)

    return (minmax_lat, stage_sum), (stage_lat_list, comm_lat_list)


def run_simu(
    gen_config,
    sol,
    lat_cost_model,
    comm_cost_model,
    use_profiler_prediction,
    mu_n,
    comm_multiplier,
):
    D = sol["D"]
    use_plan = sol["use_plan"]
    prefill_bz = sol["prefill_bz"]
    bz_decode_max = sol["bz_decode_max"]
    maps = sol["maps"]
    if maps is not None:
        comm_cost_model.set_device_rank_map(maps)
    global_bz = gen_config.global_bz
    data_pack = (prefill_bz, bz_decode_max)
    cost_model_pack = (lat_cost_model, comm_cost_model)
    s = gen_config.s
    n = gen_config.n

    comm_size_prefill, comm_size_decode = get_comm_payload_size(
        lat_cost_model, s, prefill_bz, bz_decode_max, comm_multiplier
    )
    # comm_size_prefill = lat_cost_model.h1 * s * prefill_bz * 2 / 1024 / 1024 * comm_multiplier
    # comm_size_decode = lat_cost_model.h1 * 1 * bz_decode_max * 2 / 1024 / 1024 * comm_multiplier

    sol_name = sol["name"]
    # if sol_name == 'llm_pq':
    #     import pdb; pdb.set_trace()
    # average throughput should equals to
    (prefill_result, prefill_sum), (prefill_lat_list, prefill_comm_lat_list) = (
        calculate_max_stage_lat(
            D,
            use_plan,
            cost_model_pack,
            prefill_bz,
            s,
            0,
            use_profiler_prediction,
            comm_size_prefill,
        )
    )
    (decode_result, decode_sum), (decode_lat_list, decode_comm_lat_list) = (
        calculate_max_stage_lat(
            D,
            use_plan,
            cost_model_pack,
            bz_decode_max,
            1,
            s + int(mu_n / 2),
            use_profiler_prediction,
            comm_size_decode,
        )
    )

    # print("Prefill cost stage", prefill_lat_list, prefill_comm_lat_list)
    # print("Decode cost stage", decode_lat_list, decode_comm_lat_list)
    for cost in prefill_lat_list + decode_lat_list:
        if cost < 0:
            import pdb

            pdb.set_trace()
    print("Prefill cost stage", prefill_lat_list)
    print("Decode cost stage", decode_lat_list)
    # print(global_bz / prefill_bz,  prefill_bz)
    prefill_micro_bs_num = math.ceil(global_bz / prefill_bz)
    decode_micro_bs_num = math.ceil(global_bz / bz_decode_max)
    prefill_time = prefill_sum + prefill_result * (prefill_micro_bs_num - 1)
    decode_time = decode_sum + decode_result * (decode_micro_bs_num - 1) * (mu_n - 1)
    # latency equals
    e2e_lat = prefill_time + decode_time
    print(
        "Prefill Time {:.2f}ms, Decode Time {:.2f}ms, E2E Latency {:.2f}ms".format(
            prefill_time, decode_time, e2e_lat
        )
    )
    # remove maps
    if maps is not None:
        comm_cost_model.clear_device_rank_map()
    return e2e_lat


def get_latency_with_layer_device_bit_pair(
    current_D, bit_pairs, lat_cost_model, b, s, i, use_profiler_prediction=True
):
    device_names = list(current_D.values())
    dtypes = set(device_names)
    device_bit_res = {}

    for device_name in dtypes:
        for idx, bit_pair in enumerate(bit_pairs):
            attn_bit, ffn_bit = bit_pair
            device_bit_res[(device_name, bit_pair)] = 0
            lat = lat_prediction(
                lat_cost_model,
                device_name,
                b,
                s,
                i,
                attn_bit,
                ffn_bit,
                use_profiler_prediction,
            )
            device_bit_res[(device_name, bit_pair)] = lat
    # create latency matrix
    lat_device_bits_matrix = np.zeros((len(current_D), len(bit_pairs)))
    for i, device_name in enumerate(device_names):
        for j, bit_pair in enumerate(bit_pairs):
            lat_device_bits_matrix[i, j] = device_bit_res[(device_name, bit_pair)]
    return lat_device_bits_matrix
