from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class TumorHNDataset(BaseSegDataset):
    """
    Custom dataset for Head & Neck Tumor segmentation.
    Classes:
    - bg: Background (Label 0)
    - tumor: Primary Tumor GTVp (Label 1)
    - lyn: Metastatic Lymph Nodes GTVn (Label 2)
    """

    METAINFO = dict(
        classes=('bg', 'tumor', 'lyn'),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0]]  # Black, Red, Green
    )

    def __init__(self, **kwargs):
        """
        Initializes the dataset.
        reduce_zero_label is set to False because the background is actively 
        used as a valid class (Label 0).
        """
        super().__init__(
            img_suffix='.png', 
            seg_map_suffix='.png', 
            reduce_zero_label=False, 
            **kwargs
        )
