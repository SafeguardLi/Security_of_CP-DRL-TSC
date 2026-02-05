import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.utils import check_and_transform_label_format

import torch

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)

class SaliencyMapMethod(EvasionAttack):
    """
    Implementation of the Jacobian-based Saliency Map Attack (Papernot et al. 2016).
    Restricted to features 10-20 and returns top 2 features.
    """

    attack_params = EvasionAttack.attack_params + ["theta", "gamma", "batch_size", "verbose"]
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        theta: float = 0.01,
        gamma: float = 1,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> None:
        super().__init__(estimator=classifier)
        self.theta = theta
        self.gamma = gamma
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        dims = list(x.shape[1:])
        self._nb_features = np.product(dims)
        x_adv = np.reshape(x.astype(ART_NUMPY_DTYPE), (-1, self._nb_features))
        x_torch = torch.tensor(x).to(torch.float32)
        preds = np.argmax(self.estimator.predict(x_torch, batch_size=self.batch_size), axis=1)

        # Initialize variable to store the final pair of indices
        final_pair = np.array([])

        if y is None:
            from art.utils import random_targets
            targets = np.argmax(random_targets(preds, self.estimator.nb_classes), axis=1)
        else:
            targets = np.argmax(y, axis=1)

        for batch_id in trange(
            int(np.ceil(x_adv.shape[0] / float(self.batch_size))), desc="JSMA", disable=not self.verbose
        ):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]

            # 1. HARD CONSTRAINT: Strictly limit search space to indices 10 to 20
            search_space = np.zeros(batch.shape)
            search_space[:, 10:21] = 1 

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                if self.theta > 0:
                    search_space[batch >= clip_max] = 0
                else: 
                    search_space[batch <= clip_min] = 0

            current_pred = preds[batch_index_1:batch_index_2]
            target = targets[batch_index_1:batch_index_2]
            active_indices = np.where(current_pred != target)[0]
            all_feat = np.zeros_like(batch)

            while active_indices.size != 0:
                # 2. Get top 2 features from restricted search space
                feat_ind = self._saliency_map(
                    np.reshape(batch, [batch.shape[0]] + dims)[active_indices],
                    target[active_indices],
                    search_space[active_indices],
                )

                # Store the most recent pair found
                final_pair = feat_ind[0]

                # Update used features for both selected indices
                all_feat[active_indices, feat_ind[:, 0]] = 1
                all_feat[active_indices, feat_ind[:, 1]] = 1

                # Apply attack
                tmp_batch = batch[active_indices]
                if self.estimator.clip_values is not None:
                    clip_func = np.minimum if self.theta > 0 else np.maximum
                    clip_value = clip_max if self.theta > 0 else clip_min

                    # Update both features
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] = clip_func(
                        clip_value, tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] + self.theta
                    )
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] = clip_func(
                        clip_value, tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] + self.theta
                    )
                    batch[active_indices] = tmp_batch
                    search_space[batch == clip_value] = 0
                else:
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] += self.theta
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] += self.theta
                    batch[active_indices] = tmp_batch

                current_pred = np.argmax(
                    self.estimator.predict(np.reshape(batch, [batch.shape[0]] + dims)),
                    axis=1,
                )

                active_indices = np.where(
                    (current_pred != target)
                    * (np.sum(all_feat, axis=1) / self._nb_features <= self.gamma)
                    * (np.sum(search_space, axis=1) > 0)
                )[0]

            x_adv[batch_index_1:batch_index_2] = batch

        x_adv = np.reshape(x_adv, x.shape)
        
        # Return adversarial example and the pair of top 2 features
        return x_adv, final_pair

    def _saliency_map(self, x: np.ndarray, target: Union[np.ndarray, int], search_space: np.ndarray) -> np.ndarray:
        """
        Compute the saliency map of `x`. Return the top 2 coefficients in `search_space`.
        """
        grads = self.estimator.class_gradient(x, label=target)
        grads = np.reshape(grads, (-1, self._nb_features))

        # 3. Apply mask: Features outside search_space (0-9, 21+) are set to -inf
        used_features = 1 - search_space
        coeff = 2 * int(self.theta > 0) - 1
        grads[used_features == 1] = -np.inf * coeff

        # 4. Partition to get top 2 indices
        if self.theta > 0:
            ind = np.argpartition(grads, -2, axis=1)[:, -2:]
        else:
            ind = np.argpartition(-grads, -2, axis=1)[:, -2:]

        return ind

    def _check_params(self) -> None:
        if self.gamma <= 0 or self.gamma > 1:
            raise ValueError("The total perturbation percentage `gamma` must be between 0 and 1.")
        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")
        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")