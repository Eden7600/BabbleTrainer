import os
from pathlib import Path

ROOT = Path(__file__).parent


BACKBONE = os.environ.get("BACKBONE", "torchvision")
_BB_TAG = "" if BACKBONE in ("torchvision", "mobilenet_v2") else "__" + BACKBONE.split(".")[0]

SEED = 1234

LAMBDA_RANGE = 0.01  # soft hinge keeping raw linear outputs near [0,1]

REG_LOSS = os.environ.get("REG_LOSS", "mse")
HUBER_BETA = 0.05  # smooth-L1 transition point (only used when REG_LOSS="huber")

DUAL_PER_EYE = ["EyeY", "EyeX", "EyeLid", "EyeWiden", "EyeSquint", "EyeBrow"]
DUAL_NUM_PER_EYE = len(DUAL_PER_EYE)   # 6
GAZE_SLICE = (0, 2)   # [start,end) of gaze channels within the 6 (masked supervision)
EXPR_SLICE = (2, 6)   # [start,end) of expression channels within the 6 (always supervised)

OUTPUT_SIDE_ORDER = ("right", "left")
BLENDSHAPE_NAMES = [side + e for side in OUTPUT_SIDE_ORDER for e in DUAL_PER_EYE]  # 12 names
SWAP_OUTPUT_SIDES = os.environ.get("SWAP_OUTPUT_SIDES", "0") != "0"

SWAP_INPUT_EYES = True
EXPORT_NOFLIP_CH = [1, 3, 5, 7] if SWAP_INPUT_EYES else [0, 2, 4, 6]  # fed un-flipped -> leftEye* output
EXPORT_FLIP_CH = [0, 2, 4, 6] if SWAP_INPUT_EYES else [1, 3, 5, 7]    # hflip -> tower -> rightEye* output
TRAIN_MIRROR_BIN_SIDE = 0 if SWAP_INPUT_EYES else 1   # bin eye mirrored in training (0=left jpeg, 1=right)

TEMPORAL_FRAMES = 4      # current + 3 previous frames per eye -> 4 input channels per tower
DUAL_INPUT_SIZE = 128    # per-eye frame size the frontend sends (grayscale [0,1], no ImageNet norm)
GAZE_RANGE_DEG = 45.0    # per-eye gaze assumed in [-45,+45] deg; norm = clamp((deg+45)/90, 0, 1)

DUAL_CKPT_DIR = ROOT / "ckpt" / ("dual_userdata" + _BB_TAG)   # namespaced per backbone
DUAL_INDEX_PATH = ROOT / "userdata" / "bin_temporal_index.npz"

GAZE_LR = 1e-3               # AdamW lr (ref lrs[0])
GAZE_WEIGHT_DECAY = 1e-2     # AdamW default wd (ref leaves it default)
GAZE_BATCH = 32             # ref make_batch
GAZE_WARMUP = 500          # 5 "epochs" x 100 steps (ref EPOCH_SIZE), linear
GAZE_ETA_MIN = 1e-6        # cosine floor (ref eta_min; NOT 0)
GAZE_STEPS = 3200         # per-user budget (ref GAZE_EPOCHS=32 x EPOCH_SIZE=100)
# Gaze head sees MicroChad's 4x4 grid instead of a global max (~4% lower MAE, benchmarks/gaze_arch_experiment.py).
GAZE_SPATIAL = os.environ.get("GAZE_SPATIAL", "1") != "0"
GAZE_CLOSED_RATIO = 0.20   # sampler: 80% gaze-valid / 20% eyes-closed (ref batcher_gaze f>0.8)
GAZE_CLOSED_LID_THRESH = 0.5  # "eyes closed" = EyeLid (=1-lid) > this
EXPR_STEPS = 4000          # per-user expression-loop budget

EXPR_TEMPORAL_MARGIN = [0.35, 0.10, 0.10, 0.10]
EXPR_LAMBDA_TEMP = 0.1       # weight of the temporal-smoothness loss (ref lambda_temp)
EXPR_LAMBDA_CONS = 1.0       # weight of the Mean-Teacher consistency loss (ref lambda_cons)
EXPR_CONS_RAMPUP = 500       # quadratic ramp-up steps for the consistency loss (ref cons_rampup_steps)

GAZE_LAMBDA_CONS = 0.3
GAZE_LAMBDA_TEMP = 0.1
GAZE_CONS_RAMPUP = 500
GAZE_TEMPORAL_MARGIN = [0.15, 0.15]

GAZE_OCCLUDE_PROB = float(os.environ.get("GAZE_OCCLUDE", "0.0"))
GAZE_OCCLUDE_MAX_FRAC = 0.35  # each lid can cover up to this fraction of the eye height
GAZE_EXPR_RATIO = float(os.environ.get("GAZE_EXPR", "0.6"))
GAZE_EXPR_ACTIVE_THRESH = 0.5   # a gaze-valid frame is "expression-during-gaze" if max(EXPR label) > this

FOOTAGE_UNLABELED_PATH = ROOT / "footage_unlabeled.npz"

EXPR_MIXUP_NEUTRAL_PROB = 0.5    # fraction of expr samples = neutral<->max-expression blends
EXPR_MIXUP_CROSS_PROB = 0.15     # fraction = general cross-frame blends (the rest are plain frames)
EXPR_MIXUP_NEUTRAL_THRESH = 0.2  # a frame is "neutral" if max expr channel < this
EXPR_MIXUP_EXPR_THRESH = 0.8     # a frame is "expression" if max expr channel > this

EXPR_DECONFUSE = [
    # target:  EyeLid  EyeWiden  EyeSquint  EyeBrow
    [0.0, 0.0, 0.5, 0.7],   # driver EyeLid (blink): soft-mask squint + brow (blink != squint/brow)
    [0.0, 0.0, 0.0, 0.0],   # driver EyeWiden: no confound
    [1.0, 0.0, 0.0, 1.0],   # driver EyeSquint: mask lid + brow (they co-occur with a squint)
    [0.7, 0.0, 0.7, 0.0],   # driver EyeBrow (brow-down): soft-mask lid + squint (narrows the eye)
]
EXPR_TUNE_LAMBDA_CONS = float(os.environ.get("EXPR_TUNE_LAMBDA_CONS", "0"))  # 0 = Mean-Teacher off
EXPR_NEUTRAL_GAZE_WEIGHT = float(os.environ.get("EXPR_NEUTRAL_GAZE_W", "1.0"))

EXPR_STAGE_A_STEPS = 2000           # supervised + mixup only (ref STAGE_A total_steps)
EXPR_STAGE_B_STEPS = 2000           # + Mean-Teacher + temporal (ref STAGE_B 3000; trimmed to keep ~EXPR_STEPS)
EXPR_STAGE_A_WARMUP = 200           # head-only warmup, backbone frozen (ref STAGE_A warmup_steps)
EXPR_STAGE_A_LR_HEAD = 5e-4         # ref STAGE_A lr_head
EXPR_STAGE_A_LR_BACKBONE = 1e-4     # ref STAGE_A lr_backbone
EXPR_STAGE_A_LR_WARMUP_HEAD = 1e-3  # ref STAGE_A lr_warmup_head (head LR while the backbone is frozen)
EXPR_STAGE_B_LR_HEAD = 2e-4         # ref STAGE_B lr_head
EXPR_STAGE_B_LR_BACKBONE = 5e-5     # ref STAGE_B lr_backbone

LAMBDA_EVEN = float(os.environ.get("LAMBDA_EVEN", "0.1"))   # legacy uniform scalar (0 = Mechanism C off)
EXPR_EVEN_BATCH = 32                 # paired-eye batch (both eyes of the same real frame) per step
if os.environ.get("EVEN_WEIGHTS"):
    EXPR_EVEN_WEIGHTS = [float(x) for x in os.environ["EVEN_WEIGHTS"].split(",")]
elif os.environ.get("LAMBDA_EVEN"):
    EXPR_EVEN_WEIGHTS = [LAMBDA_EVEN] * 4
else:
    EXPR_EVEN_WEIGHTS = [0.3, 1.0, 0.3, 0.3]

DUAL_CONSENSUS_HEAD = os.environ.get("CONSENSUS", "1") != "0"
EXPR_CONSENSUS_HIDDEN = 64           # ConsensusHead hidden width (ref BabbleTrainer ConsensusHead 212->64)
LAMBDA_CONSENSUS = 1.0               # weight of the consensus-head loss (head-only; features are detached)
CONSENSUS_GROUPS = [[1, 2], [3]]
CONSENSUS_NAMES = ["combinedEyeWiden", "combinedEyeSquint", "combinedEyeBrow"]
BLENDSHAPE_NAMES_CONSENSUS = BLENDSHAPE_NAMES + CONSENSUS_NAMES
# Optional headset-worn probability per eye, appended after every other output.
PRESENCE_NAMES = [side + "EyePresent" for side in OUTPUT_SIDE_ORDER]


def output_names(consensus, presence=False):
    return (BLENDSHAPE_NAMES_CONSENSUS if consensus else BLENDSHAPE_NAMES) + (PRESENCE_NAMES if presence else [])

if "FUSE_CHANNELS" in os.environ:
    EXPR_FUSE_CHANNELS = [int(x) for x in os.environ["FUSE_CHANNELS"].split(",") if x.strip()]
else:
    EXPR_FUSE_CHANNELS = [3]
EXPR_FUSE_SOURCE = os.environ.get("FUSE_SOURCE", "consensus")
