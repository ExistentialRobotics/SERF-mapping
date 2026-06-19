"""Training configuration dataclass for robot neural point map training."""

from dataclasses import dataclass


@dataclass
class RobotTrainConfig:
    """Training configuration for robot neural point mapping.

    Mirrors mapping/config/env_train_config.py structure with robot-specific fields.
    """

    # Robot surface sampler
    voxel_size: float

    # Point features
    feature_dim: int
    knn_k: int
    temperature: float

    # Decoder
    decoder_hidden_dim: int
    decoder_num_res_blocks: int
    decoder_dropout: float

    # Training
    learning_rate: float
    batch_size: int
    num_epochs: int
    log_interval: int
    save_interval: int

    # Vision model
    image_size: int
    patch_size: int

    # Decoder training
    train_decoder: bool = True

    # Part-level contrastive loss
    lambda_contrastive: float = 0.1
    contrastive_temperature: float = 0.1
    contrastive_max_samples: int = 16384

    # Visualization
    run_pca: bool = False
    vis_interval: int = 0

    @classmethod
    def from_dict(cls, config: dict) -> "RobotTrainConfig":
        """Create RobotTrainConfig from config dictionary."""
        robot = config.get("robot", {})
        point_map = config["point_map"]
        decoder = config["decoder"]
        training = config["training"]
        dino_model = config["dino_model"]
        vis = config.get("visualization", {})

        return cls(
            voxel_size=robot.get("voxel_size", 0.02),
            feature_dim=point_map["feature_dim"],
            knn_k=point_map["knn_k"],
            temperature=point_map.get("temperature", 0.05),
            decoder_hidden_dim=decoder["hidden_dim"],
            decoder_num_res_blocks=decoder.get("num_res_blocks", 2),
            decoder_dropout=decoder.get("dropout", 0.0),
            train_decoder=decoder.get("train_decoder", True),
            learning_rate=training["optimizer_lr"],
            batch_size=training.get("batch_size", 32),
            num_epochs=training["epochs"],
            log_interval=training.get("log_interval", 100),
            save_interval=training.get("save_interval", 1),
            image_size=dino_model["image_size"],
            patch_size=dino_model["patch_size"],
            lambda_contrastive=training.get("lambda_contrastive", 0.1),
            contrastive_temperature=training.get("contrastive_temperature", 0.1),
            contrastive_max_samples=training.get("contrastive_max_samples", 256),
            run_pca=vis.get("enabled", False),
            vis_interval=vis.get("interval", 0),
        )
