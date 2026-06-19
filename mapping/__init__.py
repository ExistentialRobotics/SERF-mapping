import sys
from pathlib import Path

# Add project root to sys.path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mapping.models.vision_backbone import EvaClipWrapper, DINOv3Wrapper
from mapping.models.mlp import MLP
from mapping.representations.neural_point_map import NeuralPointMap
from mapping.representations.robot_neural_points import RobotNeuralPointMap
from mapping.data.env_dataset import EnvDataset, env_collate_fn
from utils.geometry import unproject_depth_to_world
from utils.optimizer import update_optimizer_param
from utils.visualization import TorchPCA, run_pca_visualization

__all__ = [
    # Models
    "EvaClipWrapper",
    "DINOv3Wrapper",
    "MLP",
    # Representations
    "NeuralPointMap",
    "RobotNeuralPointMap",
    # Data
    "EnvDataset",
    "env_collate_fn",
    # Utils
    "unproject_depth_to_world",
    "update_optimizer_param",
    "TorchPCA",
    "run_pca_visualization",
]
