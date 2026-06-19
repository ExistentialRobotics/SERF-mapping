import sys
from pathlib import Path

# Add project root to sys.path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mapping.representations.neural_point_map import NeuralPointMap
from mapping.representations.robot_neural_points import RobotNeuralPointMap

__all__ = [
    "NeuralPointMap",
    "RobotNeuralPointMap",
]
