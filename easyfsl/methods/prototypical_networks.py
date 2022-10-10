"""
See original implementation (quite far from this one)
at https://github.com/jakesnell/prototypical-networks
"""

import torch
from torch import Tensor

from easyfsl.methods import FewShotClassifier
from easyfsl.utils import compute_prototypes


class PrototypicalNetworks(FewShotClassifier):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their euclidean distance to the prototypes.
    """

    def __init__(self, *args, **kwargs):
        """
        Raises:
            ValueError: if the backbone is not a feature extractor,
            i.e. if its output for a given image is not a 1-dim tensor.
        """
        super().__init__(*args, **kwargs)

        if len(self.backbone_output_shape) != 1:
            raise ValueError(
                "Illegal backbone for Prototypical Networks. "
                "Expected output for an image is a 1-dim tensor."
            )

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract feature vectors from the support set and store class prototypes.

        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        # support feautres에서 support label에 포함 되는 feature 값들을 배치 내에서 뽑아서
        # prototype에 넣어줌. prototype의 shape  => (n_shot, 1280)
        # batch size = n_way * (n_shot + n_query)

        support_features = self.backbone.forward(support_images)

        # # add
        # if self.prototypes is None:
        #     self.prototypes = compute_prototypes(support_features, support_labels).unsqueeze(0)
        # else :
        #     self.prototypes = torch.cat([self.prototypes, compute_prototypes(support_features, support_labels).unsqueeze(0)], dim=0)

        # # original
        self.prototypes = compute_prototypes(support_features, support_labels)

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Predict query labels based on their distance to class prototypes in the feature space.
        Classification scores are the negative of euclidean distances.

        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        # Extract the features of support and query images
        z_query = self.backbone.forward(query_images)

        # Compute the euclidean distance from queries to prototypes
        # dists = self.l2_distance_to_prototypes(query_images)
        dists = torch.cdist(z_query, self.prototypes)

        # Use it to compute classification scores
        scores = -dists

        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False
