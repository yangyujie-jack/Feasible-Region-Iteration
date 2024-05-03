import warnings

import relax.utils.flag

warnings.filterwarnings("ignore", r".*jax\.tree_util.*", FutureWarning, r"haiku")
