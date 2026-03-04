import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Retrieval import RetrievalTool


class Model(nn.Module):
    """
    Retrieval-Augmented Forecasting Transformer (RAFT).
    Paper: https://arxiv.org/pdf/2205.13504.pdf

    Uses periodic decomposition and nearest-neighbour retrieval to inject
    relevant historical segments as dynamic exogenous variables, instead of
    extending the raw context window.
    """

    def __init__(self, configs, individual=False):
        super().__init__()
        if torch.cuda.is_available() and configs.use_gpu:
            self.device = torch.device(f'cuda:{configs.gpu}')
        elif torch.backends.mps.is_available() and configs.use_gpu:
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)

        self.n_period = configs.n_period
        self.topm = configs.topm

        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=self.topm,
        )

        self.period_num = self.rt.period_num[-1 * self.n_period:]

        module_list = [
            nn.Linear(self.pred_len // g, self.pred_len)
            for g in self.period_num
        ]
        self.retrieval_pred = nn.ModuleList(module_list)
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)

    def prepare_dataset(self, train_data, valid_data, test_data):
        """Pre-compute retrieval indices for all splits."""
        self.rt.prepare_dataset(train_data)
        self.retrieval_dict = {}

        print('Performing train retrieval...')
        train_rt = self.rt.retrieve_all(train_data, train=True, device=self.device)

        print('Performing validation retrieval...')
        valid_rt = self.rt.retrieve_all(valid_data, train=False, device=self.device)

        print('Performing test retrieval...')
        test_rt = self.rt.retrieve_all(test_data, train=False, device=self.device)

        del self.rt
        torch.cuda.empty_cache()

        self.retrieval_dict['train'] = train_rt.detach()
        self.retrieval_dict['valid'] = valid_rt.detach()
        self.retrieval_dict['test'] = test_rt.detach()

    def encoder(self, x, index, mode):
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels, \
            f"Expected ({self.seq_len}, {self.channels}), got ({seq_len}, {channels})"

        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset

        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        # Retrieval indices are on CPU after empty_cache() in prepare_dataset
        index_cpu = index.cpu() if index.is_cuda else index
        max_idx = self.retrieval_dict[mode].shape[1] - 1
        index_cpu = torch.clamp(index_cpu, 0, max_idx)

        pred_from_retrieval = self.retrieval_dict[mode][:, index_cpu]
        pred_from_retrieval = pred_from_retrieval.to(self.device)

        retrieval_pred_list = []
        for i, pr in enumerate(pred_from_retrieval):
            assert (bsz, self.pred_len, channels) == pr.shape
            g = self.period_num[i]
            pr = pr.reshape(bsz, self.pred_len // g, g, channels)
            pr = pr[:, :, 0, :]
            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            pr = pr.reshape(bsz, self.pred_len, self.channels)
            retrieval_pred_list.append(pr)

        retrieval_pred_list = torch.stack(retrieval_pred_list, dim=1)
        retrieval_pred_list = retrieval_pred_list.sum(dim=1)

        pred = torch.cat([x_pred_from_x, retrieval_pred_list], dim=1)
        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1)
        pred = pred.reshape(bsz, self.pred_len, self.channels)
        pred = pred + x_offset
        return pred

    def forecast(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def forward(self, x_enc, index, mode='train'):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, index, mode)
            return dec_out[:, -self.pred_len:, :]
        return None
