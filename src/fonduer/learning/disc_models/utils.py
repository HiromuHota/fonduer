import collections
import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from emmental.data import EmmentalDataset
from emmental.modules.rnn_module import RNN
from emmental.modules.sparse_linear_module import SparseLinear
from emmental.scorer import Scorer
from emmental.task import EmmentalTask

from fonduer.learning.disc_models.modules.loss import SoftCrossEntropyLoss
from fonduer.utils.config import get_config

logger = logging.getLogger(__name__)


class SymbolTable(object):
    """Wrapper for dict to encode unknown symbols

    :param starting_symbol: Starting index of symbol.
    :type starting_symbol: int
    :param unknown_symbol: Index of unknown symbol.
    :type unknown_symbol: int
    """

    def __init__(self, starting_symbol=2, unknown_symbol=1):
        self.s = starting_symbol
        self.unknown = unknown_symbol
        self.d = dict()

    def get(self, w):
        if w not in self.d:
            self.d[w] = self.s
            self.s += 1
        return self.d[w]

    def lookup(self, w):
        return self.d.get(w, self.unknown)

    def lookup_strict(self, w):
        return self.d.get(w)

    def len(self):
        return self.s

    def reverse(self):
        return {v: k for k, v in self.d.items()}


def mention_to_tokens(mention, token_type="words", lowercase=False):
    """
    Extract tokens from the mention

    :param mention: mention object.
    :param token_type: token type that wants to extract.
    :type token_type: str
    :param lowercase: use lowercase or not.
    :type lowercase: bool
    :return: The token list.
    :rtype: list
    """

    tokens = getattr(mention.context.sentence, token_type)
    return [w.lower() if lowercase else w for w in tokens]


def mark(l, h, idx):
    """
    Produce markers based on argument positions

    :param l: sentence position of first word in argument.
    :type l: int
    :param h: sentence position of last word in argument.
    :type h: int
    :param idx: argument index (1 or 2).
    :type idx: int
    :return: markers.
    :rtype: list of markers
    """

    return [(l, f"~~[[{idx}"), (h + 1, f"{idx}]]~~")]


def mark_sentence(s, args):
    """Insert markers around relation arguments in word sequence

    :param s: list of tokens in sentence.
    :type s: list
    :param args: list of triples (l, h, idx) as per @_mark(...) corresponding
               to relation arguments
    :type args: list
    :return: The marked sentence.
    :rtype: list

    Example: Then Barack married Michelle.
         ->  Then ~~[[1 Barack 1]]~~ married ~~[[2 Michelle 2]]~~.
    """

    marks = sorted([y for m in args for y in mark(*m)], reverse=True)
    x = list(s)
    for k, v in marks:
        x.insert(k, v)
    return x


def pad_batch(batch, max_len=0, type="int"):
    """Pad the batch into matrix

    :param batch: The data for padding.
    :type batch: list of word index sequences
    :param max_len: Max length of sequence of padding.
    :type max_len: int
    :param type: mask value type.
    :type type: str
    :return: The padded matrix and correspoing mask matrix.
    :rtype: pair of torch.Tensors with shape (batch_size, max_sent_len)
    """

    batch_size = len(batch)
    max_sent_len = int(np.max([len(x) for x in batch]))
    if max_len > 0 and max_len < max_sent_len:
        max_sent_len = max_len
    if type == "float":
        idx_matrix = np.zeros((batch_size, max_sent_len), dtype=np.float32)
    else:
        idx_matrix = np.zeros((batch_size, max_sent_len), dtype=np.int)

    for idx1, i in enumerate(batch):
        for idx2, j in enumerate(i):
            if idx2 >= max_sent_len:
                break
            idx_matrix[idx1, idx2] = j
    idx_matrix = torch.tensor(idx_matrix)
    mask_matrix = torch.tensor(torch.eq(idx_matrix.data, 0))
    return idx_matrix, mask_matrix


def collect_word_counter(candidates):
    """Collect word counter from candidates

    :param candidates: The candidates used to collect word counter.
    :type candidates: list (of list) of candidates
    :return: The word counter.
    :rtype: collections.Counter
    """
    word_counter = collections.Counter()

    if isinstance(candidates[0], list):
        candidates = [cand for candidate in candidates for cand in candidate]
    for candidate in candidates:
        for mention in candidate:
            word_counter.update(mention_to_tokens(mention))
    return word_counter


sce_loss = SoftCrossEntropyLoss()


def loss(module_name, intermediate_output_dict, Y, active):
    if len(Y.size()) == 1:
        label = intermediate_output_dict[module_name][0].new_zeros(
            intermediate_output_dict[module_name][0].size()
        )
        label.scatter_(1, (Y - 1).view(Y.size()[0], 1), 1.0)
    else:
        label = Y

    return sce_loss(intermediate_output_dict[module_name][0][active], label[active])


def output(module_name, intermediate_output_dict):
    return F.softmax(intermediate_output_dict[module_name][0])


class Sum_module(nn.Module):
    def __init__(self, sum_output_keys):
        super().__init__()

        self.sum_output_keys = sum_output_keys

    def forward(self, intermediate_output_dict):
        # print("Sum", intermediate_output_dict["lstm0"])
        # import pdb; pdb.set_trace()
        return torch.stack(
            [intermediate_output_dict[key][0] for key in self.sum_output_keys], dim=0
        ).sum(dim=0)


class FonduerDataset(EmmentalDataset):
    def __init__(self, name, candidates, features, word2id, labels=None, index=None):

        self.name = name
        self.candidates = candidates
        self.features = features
        self.word2id = word2id
        self.labels = labels
        self.index = index

        self._map_to_id()
        self._map_features()
        self._map_labels()

        uids = [f"{self.name}_{idx}" for idx in range(len(self.candidates))]
        self.add_features({"_uids_": uids})

        super().__init__(name, self.X_dict, self.Y_dict, "_uids_")

    def __len__(self):
        try:
            if self.index is not None:
                return len(self.index)
            else:
                return len(next(iter(self.X_dict.values())))
        except StopIteration:
            return 0

    def __getitem__(self, index):
        if self.index is not None:
            index = self.index[index]

        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}
        return x_dict, y_dict

    def _map_to_id(self):
        print(self.candidates[0])
        self.X_dict = dict([(f"m{i}", []) for i in range(len(self.candidates[0]))])

        for candidate in self.candidates:
            for i in range(len(candidate)):
                # Add mark for each mention in the original sentence
                args = [
                    (
                        candidate[i].context.get_word_start_index(),
                        candidate[i].context.get_word_end_index(),
                        i,
                    )
                ]
                s = mark_sentence(mention_to_tokens(candidate[i]), args)
                self.X_dict[f"m{i}"].append(
                    torch.LongTensor(
                        [
                            self.word2id[w]
                            if w in self.word2id
                            else self.word2id["<unk>"]
                            for w in s
                        ]
                    )
                )

    def _map_features(self):
        self.X_dict.update({"feature_index": [], "feature_weight": []})
        for i in range(len(self.candidates)):
            self.X_dict["feature_index"].append(
                torch.LongTensor(
                    self.features.indices[
                        self.features.indptr[i] : self.features.indptr[i + 1]
                    ]
                )
                + 1
            )
            self.X_dict["feature_weight"].append(
                torch.Tensor(
                    self.features.data[
                        self.features.indptr[i] : self.features.indptr[i + 1]
                    ]
                )
            )

    def _map_labels(self):
        if isinstance(self.labels, int):
            self.Y_dict = {
                "labels": torch.from_numpy(
                    np.random.randint(self.labels, size=len(self.candidates)) + 1
                )
            }
        else:
            self.Y_dict = {"labels": torch.Tensor(np.array(self.labels))}


def create_task(
    task_names, n_arites, n_features, n_classes, emb_layer, mode="mtl", model="LSTM"
):

    if not isinstance(task_names, list):
        task_names = [task_names]
        n_arites = [n_arites]
        n_classes = [n_classes]

    tasks = []

    for task_name, n_arity, n_class in zip(task_names, n_arites, n_classes):
        if model == "LSTM":
            config = get_config()["learning"][model]
            logger.info(f"{model} model config: {config}")

            module_pool = nn.ModuleDict(
                {"emb": emb_layer, "feature": SparseLinear(n_features + 1, n_class)}
            )
            for i in range(n_arity):
                module_pool.update(
                    {
                        f"lstm{i}": RNN(
                            num_classes=n_class,
                            emb_size=emb_layer.dim,
                            lstm_hidden=config["hidden_dim"],
                            attention=config["attention"],
                            dropout=config["dropout"],
                            bidirectional=config["bidirectional"],
                        )
                    }
                )
            module_pool.update(
                {
                    f"{task_name}_pred_head": Sum_module(
                        [f"lstm{i}" for i in range(n_arity)] + ["feature"]
                    )
                }
            )

            task_flow = []
            task_flow += [
                {"name": f"emb{i}", "module": "emb", "inputs": [("_input_", f"m{i}")]}
                for i in range(n_arity)
            ]
            task_flow += [
                {
                    "name": f"lstm{i}",
                    "module": f"lstm{i}",
                    "inputs": [(f"emb{i}", 0), ("_input_", f"m{i}_mask")],
                }
                for i in range(n_arity)
            ]
            task_flow += [
                {
                    "name": "feature",
                    "module": "feature",
                    "inputs": [
                        ("_input_", "feature_index"),
                        ("_input_", "feature_weight"),
                    ],
                }
            ]
            task_flow += [
                {
                    "name": f"{task_name}_pred_head",
                    "module": f"{task_name}_pred_head",
                    "inputs": None,
                }
            ]
        elif model == "LogisticRegression":
            config = get_config()["learning"][model]
            logger.info(f"{model} model config: {config}")

            module_pool = nn.ModuleDict(
                {
                    "feature": SparseLinear(n_features + 1, n_class),
                    f"{task_name}_pred_head": Sum_module(["feature"]),
                }
            )

            task_flow = [
                {
                    "name": "feature",
                    "module": "feature",
                    "inputs": [
                        ("_input_", "feature_index"),
                        ("_input_", "feature_weight"),
                    ],
                },
                {
                    "name": f"{task_name}_pred_head",
                    "module": f"{task_name}_pred_head",
                    "inputs": None,
                },
            ]
        else:
            raise ValueError(f"Unrecognized model {model}.")

        tasks.append(
            EmmentalTask(
                name=task_name,
                module_pool=module_pool,
                task_flow=task_flow,
                loss_func=partial(loss, f"{task_name}_pred_head"),
                output_func=partial(output, f"{task_name}_pred_head"),
                scorer=Scorer(metrics=["accuracy", "precision", "recall", "f1"]),
            )
        )

    return tasks
