import logging

import numpy as np
import torch
from emmental.data import EmmentalDataset

from fonduer.learning.utils import mark_sentence, mention_to_tokens

logger = logging.getLogger(__name__)


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
