import logging
from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from emmental.modules.rnn_module import RNN
from emmental.modules.sparse_linear_module import SparseLinear
from emmental.scorer import Scorer
from emmental.task import EmmentalTask

from fonduer.learning.modules.soft_cross_entropy_loss import SoftCrossEntropyLoss
from fonduer.learning.modules.sum_module import Sum_module
from fonduer.utils.config import get_config

logger = logging.getLogger(__name__)


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
