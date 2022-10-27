from . import dataset
from . import models
from . import ops
from . import optims
from . import runner
from . import utils

# version must use ' instead of "
__version__ = '0.1.1.0'
from jittor_utils import LOG
LOG.i("JNeRF version ", __version__)
