
import sys
import getopt
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
import math
from math import sqrt
import scipy.stats as stats
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=14)  # Linux环境下显示中文



dataroot = "./2巴塘1998-2016.CSV"
pred_label = 'runoff'

history_data = "./历史数据98-14.CSV"
incre_learning_data = "./增量数据15-16.CSV"


class replay:

    def __init__(self, df, alpha_s, alpha_k, alpha_d, w, N_release):
        self.df = df
        self.df_splits = [df.iloc[i:i+w] for i in range(0, len(df), w)]
        self.alpha_s = alpha_s
        self.alpha_k = alpha_k
        self.alpha_d = alpha_d
        self.w = w
        self.N_release = N_release
        self.n = len(self.df_splits)
        self.stats_dict = {}

    def replay(self):

        if self.N_release <= self.n:

            # 计算总数据集偏度和峰度
            tw_skew = stats.skew(self.df[pred_label])
            tw_kurt = stats.kurtosis(self.df[pred_label])

            # 计算子数据集与总数据集的偏度差和峰度差
            for key in range(self.n):
                skew = stats.skew(self.df_splits[key][pred_label])
                kurt = stats.kurtosis(self.df_splits[key][pred_label])
                self.stats_dict[key] = {
                    'delta_Skew': abs(skew - tw_skew),
                    'delta_Kurt': abs(kurt - tw_kurt),
                }

            # 计算子数据集各分量方差
            row_dict = {i: [] for i in range(self.w)}
            for tidx, mat in enumerate(self.df_splits):
                for i, row in enumerate(mat[pred_label].values):
                    if i < w:
                        row_dict[i].append(row)

            variance = pd.DataFrame(np.var(row_dict[i]) for i in range(self.w))

            # 计算各子数据集间平均标准化欧式距离
            de = {}
            for tidx, mat in enumerate(self.df_splits):
                tsde = []
                for ttk, omat in enumerate(self.df_splits):
                    if ttk != tidx:
                        min_len = min(len(mat), len(omat))
                        mat_slice = np.array(mat.iloc[:min_len][pred_label])
                        omat_slice = np.array(omat.iloc[:min_len][pred_label])
                        variance_slice = np.array(variance.iloc[:min_len])
                        sde = math.sqrt(((mat_slice - omat_slice) ** 2 / variance_slice).sum())
                        tsde.append(sde)
                de[tidx] = np.mean(tsde)

            for i in range(self.n):
                self.stats_dict[i].update({
                    'DE': de[i]
                })

            # 计算子数据集重放得分
            for i in range(self.n):
                self.stats_dict[i].update({
                    'S_replay': self.alpha_s / self.stats_dict[i]['delta_Skew'] + self.alpha_k / self.stats_dict[i]['delta_Kurt'] + self.alpha_d / self.stats_dict[i]['DE']
                })

            # 根据重放得分对子数据集进行排序
            target_key = 'S_replay'
            sorted_items = sorted(self.stats_dict.items(), key=lambda item: item[1][target_key], reverse=True)
            # print(sorted_items)

            # 根据N_release确定重放集
            top_N = [key for key, _ in sorted_items[:self.N_release]]
            print('子数据集中重放得分最高的前' + str(self.N_release) + '项: ', ', '.join(str(item) for item in top_N))

            # for key, value in enumerate(sorted_items):
            #     if key + 1 <= self.N_release:
            #         print(key + 1, value)


            return sorted_items


        else:

            print('N_release设置过大，超过了子数据集个数。')


class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, T,
                 input_size,
                 encoder_num_hidden,
                 parallel=False):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers = 1
        )

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T - 1,
            out_features=1
        )

    def forward(self, X):
        """forward.

        Args:
            X: input data

        """
        X_tilde = Variable(X.data.new(
            X.size(0), self.T - 1, self.input_size).zero_())
        X_encoded = Variable(X.data.new(
            X.size(0), self.T - 1, self.encoder_num_hidden).zero_())

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)

        # h_n, s_n: initial states with dimention hidden_size
        h_n = self._init_states(X)
        s_n = self._init_states(X)

        for t in range(self.T - 1):
            # batch_size * input_size * (2 * hidden_size + T - 1) cat拼接
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)

            x = self.encoder_attn(
                x.view(-1, 2 * self.encoder_num_hidden + self.T - 1))

            # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size), dim=-1)

            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :])

            # Fix the warning about non-contiguous memory
            # https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.encoder_lstm.flatten_parameters()

            # encoder LSTM
            _, final_state = self.encoder_lstm(x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n

        return X_tilde, X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X

        Returns:
            initial_hidden_states
        """
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())


class Decoder(nn.Module):
    """decoder in DA_LSTM."""

    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        """Initialize a decoder in DA_LSTM."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_num_hidden + encoder_num_hidden, encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(encoder_num_hidden, 1)
        )
        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=decoder_num_hidden
        )
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)

        self.fc.weight.data.normal_()

    def forward(self, X_encoded, y_prev):
        """forward."""
        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)

        for t in range(self.T - 1):

            x = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           X_encoded), dim=2)

            beta = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T - 1), dim=-1)

            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoded)[:, 0, :]
            if t < self.T - 1:
                # Eqn. 15
                # batch_size * 1
                y_tilde = self.fc(
                    torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))

                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))

                d_n = final_states[0]  # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]  # 1 * batch_size * decoder_num_hidden

        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))

        return y_pred

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())


class Attention_LSTM(nn.Module):
    """da_rnn."""

    def __init__(self, X, y, T,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 batch_size,
                 learning_rate,
                 epochs,
                 parallel=False):
        """da_rnn initialization."""
        super(Attention_LSTM, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parallel = parallel
        self.shuffle = False
        self.epochs = epochs
        self.T = T
        self.X = X
        self.y = y

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        self.Encoder = Encoder(input_size=X.shape[1],
                               encoder_num_hidden=encoder_num_hidden,
                               T=T).to(self.device)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               T=T).to(self.device)

        # Loss function
        self.criterion = nn.MSELoss()

        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=self.learning_rate)

        # Training set
        self.train_timesteps = int(self.X.shape[0] * 15/17)
        self.y = self.y
        self.input_size = self.X.shape[1]

        self.fisher_info = None

    def train_model(self):
        """training process."""
        iter_per_epoch = int(np.ceil(self.train_timesteps * 1. / self.batch_size))
        self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)

        n_iter = 0

        for epoch in range(self.epochs):
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - self.T)
            else:
                ref_idx = np.array(range(self.train_timesteps - self.T))

            idx = 0

            while idx < self.train_timesteps:
                # get the indices of X_train
                indices = ref_idx[idx:(idx + self.batch_size)]
                # x = np.zeros((self.T - 1, len(indices), self.input_size))
                x = np.zeros((len(indices), self.T - 1, self.input_size))
                y_prev = np.zeros((len(indices), self.T - 1))
                y_gt = self.y[indices + self.T]

                # format x into 3D tensor
                for bs in range(len(indices)):
                    x[bs, :, :] = self.X[indices[bs]:(indices[bs] + self.T - 1), :]
                    y_prev[bs, :] = self.y[indices[bs]: (indices[bs] + self.T - 1)]

                loss = self.forward(x, y_prev, y_gt)
                self.iter_losses[int(epoch * iter_per_epoch + idx / self.batch_size)] = loss

                idx += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

                self.epoch_losses[epoch] = np.mean(self.iter_losses[range(
                    epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])

            if epoch % 10 == 0:
                print("Epochs: ", epoch, " Iterations: ", n_iter,
                      " Loss: ", self.epoch_losses[epoch])

            # if epoch % 50 == 0:
            #     y_train_pred = self.test(on_train=True)
            #     y_test_pred = self.test(on_train=False)
            #     y_pred = np.concatenate((y_train_pred, y_test_pred))
            #     plt.ioff()
            #     plt.figure()
            #     plt.plot(range(1, 1 + len(self.y)), self.y, label="batangTrue")
            #     plt.plot(range(self.T, len(y_train_pred) + self.T),
            #              y_train_pred, label='batangPred - Train')
            #     plt.plot(range(self.T + len(y_train_pred), len(self.y) + 1),
            #              y_test_pred, label='batangPred - Test')
            #     plt.legend(loc='upper left')
            #     plt.title("batang")
            #     plt.show(block=False)

            # # Save files in last iterations（迭代）
            # if epoch == self.epochs - 1:
            #     np.savetxt('./{ntimestep}batangloss.txt'.format(ntimestep = ntimestep),
            #                np.array(self.epoch_losses), delimiter=',')
            #     np.savetxt('./{ntimestep}batangy_pred.txt'.format(ntimestep = ntimestep),
            #                np.array(y_pred), delimiter=',')
            #     np.savetxt('./{ntimestep}batangy_true.txt'.format(ntimestep = ntimestep),
            #                np.array(self.y), delimiter=',')

    def train_model_with_ewc(self):

        iter_per_epoch = int(np.ceil(self.train_timesteps * 1. / self.batch_size))
        self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)

        n_iter = 0

        for epoch in range(self.epochs):
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - self.T)
            else:
                ref_idx = np.array(range(self.train_timesteps - self.T))

            idx = 0

            while idx < self.train_timesteps:
                # get the indices of X_train
                indices = ref_idx[idx:(idx + self.batch_size)]
                # x = np.zeros((self.T - 1, len(indices), self.input_size))
                x = np.zeros((len(indices), self.T - 1, self.input_size))
                y_prev = np.zeros((len(indices), self.T - 1))
                y_gt = self.y[indices + self.T]

                # format x into 3D tensor
                for bs in range(len(indices)):
                    x[bs, :, :] = self.X[indices[bs]:(indices[bs] + self.T - 1), :]
                    y_prev[bs, :] = self.y[indices[bs]: (indices[bs] + self.T - 1)]

                prev_weights = torch.cat([param.view(-1) for param in model.parameters()]).detach().clone()

                loss = self.forward_with_ewc(x, y_prev, y_gt, prev_weights)

                self.iter_losses[int(epoch * iter_per_epoch + idx / self.batch_size)] = loss

                idx += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

                self.epoch_losses[epoch] = np.mean(self.iter_losses[range(
                    epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])

            if epoch % 10 == 0:
                print("Epochs: ", epoch, " Iterations: ", n_iter,
                      " Loss: ", self.epoch_losses[epoch])

            # if epoch % 50 == 0:
            #     y_train_pred = self.test(on_train=True)
            #     y_test_pred = self.test(on_train=False)
            #     y_pred = np.concatenate((y_train_pred, y_test_pred))
            #     plt.ioff()
            #     plt.figure()
            #     plt.plot(range(1, 1 + len(self.y)), self.y, label="batangTrue")
            #     plt.plot(range(self.T, len(y_train_pred) + self.T),
            #              y_train_pred, label='batangPred - Train')
            #     plt.plot(range(self.T + len(y_train_pred), len(self.y) + 1),
            #              y_test_pred, label='batangPred - Test')
            #     plt.legend(loc='upper left')
            #     plt.title("batang")
            #     plt.show(block=False)

            # # Save files in last iterations（迭代）
            # if epoch == self.epochs - 1:
            #     np.savetxt('./{ntimestep}batangloss.txt'.format(ntimestep = ntimestep),
            #                np.array(self.epoch_losses), delimiter=',')
            #     np.savetxt('./{ntimestep}batangy_pred.txt'.format(ntimestep = ntimestep),
            #                np.array(y_pred), delimiter=',')
            #     np.savetxt('./{ntimestep}batangy_true.txt'.format(ntimestep = ntimestep),
            #                np.array(self.y), delimiter=',')


    def forward(self, X, y_prev, y_gt):
        """
        Forward pass.

        Args:
            X:
            y_prev:
            y_gt: Ground truth label

        """
        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
        y_pred = self.Decoder(input_encoded, Variable(torch.from_numpy(y_prev).type(torch.FloatTensor).to(self.device)))

        y_true = Variable(torch.from_numpy(y_gt).type(torch.FloatTensor).to(self.device))

        y_true = y_true.view(-1, 1)
        loss = self.criterion(y_pred, y_true)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()


    def forward_with_ewc(self, X, y_prev, y_gt, prev_weights):
        # 正常的 forward 过程
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
        y_pred = self.Decoder(input_encoded, Variable(torch.from_numpy(y_prev).type(torch.FloatTensor).to(self.device)))

        y_true = Variable(torch.from_numpy(y_gt).type(torch.FloatTensor).to(self.device))

        y_true = y_true.view(-1, 1)
        loss = self.criterion(y_pred, y_true)

        # 计算 EWC 损失
        current_weights = torch.cat([param.view(-1) for param in model.parameters()])
        ewc_loss = calculate_ewc_loss(self.fisher_info, current_weights, prev_weights, lambda_ewc)
        total_loss = loss + lambda_ewc * ewc_loss

        # 反向传播
        total_loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return total_loss.item()


    def test(self, on_train=False):
        """test."""

        if on_train:
            y_pred = np.zeros(self.train_timesteps - self.T + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_timesteps)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i: (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))

            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j], batch_idx[j] + self.T - 1)]
                else:
                    X[j, :, :] = self.X[range(batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps - 1)]

            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).to(self.device))
            _, input_encoded = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
            y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
            i += self.batch_size

        return y_pred


def read_data(input_path, debug=True):
    """Read data.

    Args:
        input_path (str): directory to  dataset.

    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.

    """
    df = pd.read_csv(input_path, nrows=250 if debug else None)

    # 平滑操作
    df[pred_label] = df[pred_label].apply(np.log1p)

    # X = df.iloc[:, 0:-1].values
    # X = df.loc[:, [x for x in df.columns.tolist() if x != 'NDX']].as_matrix() # AttributeError: 'DataFrame' object has no attribute 'as_matrix'
    X = df.loc[:, [x for x in df.columns.tolist() if x != 'DT']].iloc[:, :].values
    # y = df.iloc[:, -1].values
    y = np.array(df.runoff)

    # 0-1标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    ynew = []
    for yy in y:
        yy = float(yy - np.min(y)) / (np.max(y) - np.min(y))
        ynew.append(yy)
    ynew = np.array(ynew)
    ymin = np.min(y)
    ymax = np.max(y)
    # print(f'ynew:{ynew}')

    return X, ynew, ymin, ymax


def deal_with_ildata(df):

    df[pred_label] = df[pred_label].apply(np.log1p)

    X = df.loc[:, [x for x in df.columns.tolist() if x != 'DT']].iloc[:, :].values
    y = np.array(df.runoff)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    ynew = []
    for yy in y:
        yy = float(yy - np.min(y)) / (np.max(y) - np.min(y))
        ynew.append(yy)
    ynew = np.array(ynew)
    ymin = np.min(y)
    ymax = np.max(y)

    return X, ynew, ymin, ymax


def calculate_fisher(model, X, y, T, criterion):
    fisher_info = []
    model.train()

    train_timesteps = int(X.shape[0] * 15 / 17)
    batch_size = model.batch_size

    ref_idx = np.array(range(train_timesteps - T))

    idx = 0
    while idx < train_timesteps:
        indices = ref_idx[idx:(idx + batch_size)]

        x_batch = np.zeros((len(indices), T - 1, X.shape[1]))
        y_prev_batch = np.zeros((len(indices), T - 1))
        y_gt_batch = y[indices + T]

        for bs in range(len(indices)):
            x_batch[bs, :, :] = X[indices[bs]:(indices[bs] + T - 1), :]
            y_prev_batch[bs, :] = y[indices[bs]: (indices[bs] + T - 1)]

        x_batch = torch.from_numpy(x_batch).float().to(model.device)
        y_prev_batch = torch.from_numpy(y_prev_batch).float().to(model.device)
        y_gt_batch = torch.from_numpy(y_gt_batch).float().to(model.device).view(-1, 1)

        input_weighted, input_encoded = model.Encoder(x_batch)
        outputs = model.Decoder(input_encoded, y_prev_batch)

        loss = criterion(outputs, y_gt_batch)

        model.zero_grad()

        loss.backward()

        gradients_encoder = [param.grad.flatten().detach().cpu().numpy() for param in model.Encoder.parameters()]
        gradients_decoder = [param.grad.flatten().detach().cpu().numpy() for param in model.Decoder.parameters()]

        gradients = gradients_encoder + gradients_decoder
        fisher_info.append(np.square(np.concatenate(gradients)) / len(X))

        idx += batch_size

    fisher_info = np.mean(fisher_info, axis=0)

    return fisher_info


def calculate_ewc_loss(fisher_information, weight, weight_old, lambda_):
    return lambda_ / 2 * torch.sum(torch.tensor(fisher_information).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')) * (weight - weight_old) ** 2)


def benchmark(true, pred):

    rmse = sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    pearson, p_value = pearsonr(true, pred)
    nse = 1 - sum(np.square(true - pred)) / sum(np.square(true - np.mean(true)))

    print(f'MAE:{mae}\n', f'RMSE:{rmse}\n', f'pearsonr:{pearson}, p_value={p_value}\n', f'nse:{nse}')
    print(f'pred长度:{len(pred)}')

    return rmse, pearson, p_value, nse


def Curve_Fitting(x, y, deg, ntimestep, name, num):
    parameter = np.polyfit(x, y, deg)  # 拟合deg次多项式
    p = np.poly1d(parameter)
    # 方程拼接
    aa = ''
    for i in range(deg+1):
        bb = round(parameter[i], 2)
        if bb > 0:
            if i == 0:
                bb = str(bb)
            else:
                bb = '+' + str(bb)
        else:
            bb = str(bb)
        if deg == i:
            aa = aa + bb
        else:
            aa = aa + bb + 'x^' + str(deg - i)
    fig = plt.figure()
    plt.scatter(x, y, color='red', label='true--pred')  # 原始数据散点图
    plt.plot(x, p(x), 'b--')  # 画拟合曲线
    # plt.text(-1, 0, aa, fontdict={'size':'10', 'color':'b'})
    # plt.legend(round(np.corrcoef(y, p(x))[0, 1] ** 2, 2))  # 拼接好的方程和R方放到图例
    plt.title('true--pred batang_{name}结果{num}'.format(name=name, num=num))
    plt.xlabel('true')
    plt.ylabel('pred')
    # 去除图边框的顶部刻度和右边刻度
    plt.tick_params(top='off', right='off')
    # 添加图例
    plt.legend(loc='upper left')
    plt.savefig("./{name}结果{num}.jpg".format(name=name, num=num))
    # plt.show(block=False)
    plt.close(fig)
    # print('曲线方程为：', aa)
    # print('    r^2为：', round(np.corrcoef(y, p(x))[0, 1] ** 2, 2))


if __name__ == '__main__':

    # 参数默认值
    alpha_s = 2
    alpha_k = 1
    alpha_d = 1
    w = 90  # 子数据集宽度
    N_release = 20  # 重放集数量
    batchsize = 128
    nhidden_encoder = 128
    nhidden_decoder = 128
    ntimestep = 10
    lr = 0.001
    epochs = 600
    lambda_ewc = 0.8  # 正则项权重

    df = pd.read_csv(dataroot)  # 总数据集
    # N_release = math.ceil(len(df)/w/4)  # N_release另一种设置方法

    # 从命令行获取参数
    opts, args = getopt.getopt(sys.argv[1:], "-b:-r:-e:-w:-n:-l:-i:")
    Hyperparameters = {}
    for i in range(len(opts)):
        Hyperparameters.update({
            opts[i][0]: opts[i][1]
        })

    if '-b' in Hyperparameters.keys():
        batchsize = int(Hyperparameters['-b'])
    if '-r' in Hyperparameters.keys():
        lr = float(Hyperparameters['-r'])
    if '-e' in Hyperparameters.keys():
        epochs = int(Hyperparameters['-e'])
    if '-w' in Hyperparameters.keys():
        w = int(Hyperparameters['-w'])
    if '-n' in Hyperparameters.keys():
        N_release = int(Hyperparameters['-n'])
    if '-l' in Hyperparameters.keys():
        lambda_ewc = float(Hyperparameters['-l'])


    # 读入历史数据
    print("历史数据训练：")
    print("==> 读入历史数据 ...")
    X, y, ymin, ymax = read_data(history_data, debug=False)

    # 初始化模型
    print("==> 初始化模型 ...")
    model = Attention_LSTM(
        X,
        y,
        ntimestep,
        nhidden_encoder,
        nhidden_decoder,
        batchsize,
        lr,
        epochs
    )


    # 历史数据训练
    print("==> 开始训练 ...")
    t1 = time.time()
    model.train_model()
    t2 = time.time()
    print('训练用时：', t2 - t1)


    # 测试
    y_pred = model.test()
    y_pred = y_pred * (ymax - ymin) + ymin
    model.y = model.y * (ymax - ymin) + ymin

    # 对数还原
    y_pred = np.exp(y_pred)
    model.y = np.exp(model.y)


    # 计算评价指标
    print("评估：")
    true = model.y[model.train_timesteps:]
    pred = y_pred

    rmse1, pearson1, p_value1, nse1 = benchmark(true, pred)
    rmse1 = round(rmse1, 4)
    pearson1 = round(pearson1, 4)
    p_value1 = round(p_value1, 4)
    nse1 = round(nse1, 4)


    # 结果展示
    # np.savetxt('./{ntimestep}batangy_pred_finally.txt'.format(ntimestep=ntimestep),
    #            np.array(y_pred), delimiter=',')
    #
    # fig1 = plt.figure()
    # plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    # plt.savefig("./{ntimestep}batang_iter_losses.jpg".format(ntimestep=ntimestep))
    # plt.close(fig1)
    #
    # fig2 = plt.figure()
    # plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    # plt.savefig("./{ntimestep}batang_epoch_losses.jpg".format(ntimestep=ntimestep))
    # plt.close(fig2)
    #
    # np.savetxt('./{ntimestep}batangy_true_finally.txt'.format(ntimestep=ntimestep),
    #            np.array(model.y[model.train_timesteps:]), delimiter=',')


    fig3 = plt.figure()
    plt.plot(model.y[model.train_timesteps:], color='blue', label="True")
    plt.plot(y_pred, color='red', label='Pred')
    plt.plot(model.y[model.train_timesteps:] - y_pred, color='green', label="True-Pred")
    plt.title("batang_训练结果1\nRMSE:{rmse}; Pearson:{pearson}, p={p_value}; NSE:{nse}\n用时：{time}s".format(rmse=rmse1, pearson=pearson1, p_value=p_value1, nse=nse1, time=round(t2 - t1, 2)))
    plt.xlabel('time')
    plt.ylabel('runoff')
    plt.legend(loc='upper center')
    plt.grid()
    # plt.text(plt.xlim()[0]+20, plt.ylim()[1]-600, "RMSE:{rmse}\nPearson:{pearson}, p={p_value}\nNSE:{nse}".format(rmse=rmse1, pearson=pearson1, p_value=p_value1, nse=nse1))
    plt.savefig("./训练结果1.jpg")
    # plt.show(block=False)
    plt.close(fig3)
    print('训练结束')


    Curve_Fitting(true, pred, 1, ntimestep, "训练", "2")


    # 增量训练
    print("\n增量训练：")

    # 计算fisher值
    criterion = nn.MSELoss()
    fisher_info = calculate_fisher(model, X, y, ntimestep, criterion)
    model.fisher_info = fisher_info
    print('模型fisher值：', fisher_info)

    # 重放
    release = replay(df, alpha_s, alpha_k, alpha_d, w, N_release)
    std = release.replay()

    t5 = time.time()

    epochs_incre = 10
    if '-i' in Hyperparameters.keys():
        epochs_incre = int(Hyperparameters['-i'])
    n = 0
    for key, _ in std[:N_release]:
        il_data = pd.DataFrame(release.df_splits[key])

        n += 1
        print("增量训练", n, "：")

        # 读入增量数据
        print("==> 读入增量数据 ...")
        X_incre, y_incre, ymin_incre, ymax_incre = deal_with_ildata(il_data)

        # 在增量数据上训练
        model.X = X_incre
        model.y = y_incre
        model.epochs = epochs_incre

        # 重新计算train_timesteps
        model.train_timesteps = int(model.X.shape[0] * 9 / 10)


        print("==> 开始增量训练 ...")
        t3 = time.time()
        model.train_model_with_ewc()
        t4 = time.time()
        print('用时：', t4 - t3)


    t6 = time.time()

    print("总用时：", t6 - t5)
    print("增量训练结束")


    # 评估
    print("评估：")
    dataroot_total = './2巴塘1998-2016.CSV'
    X_total, y_total, ymin_total, ymax_total = read_data(dataroot_total, debug=False)

    model.X = X_total
    model.y = y_total

    model.train_timesteps = int(model.X.shape[0] * 15 / 17)


    # 测试
    y_pred_total = model.test()

    y_pred_total = y_pred_total * (ymax_total - ymin_total) + ymin_total
    model.y = model.y * (ymax_total - ymin_total) + ymin_total

    # 对数还原
    y_pred_total = np.exp(y_pred_total)
    model.y = np.exp(model.y)

    # 计算评价指标
    print("评估：")
    true = model.y[model.train_timesteps:]
    pred_total = y_pred_total

    rmse2, pearson2, p_value2, nse2 = benchmark(true, pred_total)
    rmse2 = round(rmse2, 4)
    pearson2 = round(pearson2, 4)
    p_value2 = round(p_value2, 4)
    nse2 = round(nse2, 4)


    fig4 = plt.figure()
    plt.plot(model.y[model.train_timesteps:], color='blue', label="True")
    plt.plot(y_pred_total, color='red', label='Pred')
    plt.plot(model.y[model.train_timesteps:] - y_pred_total, color='green', label="True-Pred")
    plt.title("batang_增量结果1\nRMSE:{rmse}; Pearson:{pearson}, p={p_value}; NSE:{nse}\n增量训练用时：{time}s".format(rmse=rmse2, pearson=pearson2, p_value=p_value2, nse=nse2, time=round(t6 - t5, 2)))
    plt.xlabel('time')
    plt.ylabel('runoff')
    plt.legend(loc='upper center')
    plt.grid()
    # plt.text(plt.xlim()[0]+20, plt.ylim()[1]-550, "RMSE:{rmse}\nPearson:{pearson}, p={p_value}\nNSE:{nse}".format(rmse=rmse2, pearson=pearson2, p_value=p_value2, nse=nse2))
    plt.savefig("./增量结果1.jpg")
    # plt.show(block=False)
    plt.close(fig4)


    Curve_Fitting(true, pred_total, 1, ntimestep, "增量", "2")

