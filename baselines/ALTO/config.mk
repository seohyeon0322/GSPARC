#################################################################
#################################################################
# Configuration options                                         #
#################################################################
#################################################################

# Supported: ICC, GCC
COMPILER = GCC
# Supported: MKL, OPENBLAS
BLAS_LIBRARY = OPENBLAS

# either 64 or 128
ALTO_MASK_LENGTH = 64

# List of modes and ranks to specialize code for; use 0 to
# disable specialization.
MODES_SPECIALIZED := 3, 4, 5
RANKS_SPECIALIZED := 8, 16, 100
MAX_NUM_MODES = 10

# use ALTERNATIVE_PEXT if the ISA does not support BMI2 instructions
ALTERNATIVE_PEXT = false
THP_PRE_ALLOCATION = false

MEMTRACE = false
DEBUG = false
