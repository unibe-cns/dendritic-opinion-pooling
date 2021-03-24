__version__ = "0.0.1"
__maintainer__ = "Jakob Jordan"
__author__ = "Jakob Jordan"
__license__ = "MIT"
__description__ = "Library for models implementing the dendritic opinion pooling framework."
__url__ = "https://github.com/unibe-cns/dendritic-opinion-pooling"
__doc__ = f"{__description__} <{__url__}>"

from .dynamic_feedforward_cell import DynamicFeedForwardCell
from .feedforward_cell import FeedForwardCell
from .feedforward_current_cell import FeedForwardCurrentCell
