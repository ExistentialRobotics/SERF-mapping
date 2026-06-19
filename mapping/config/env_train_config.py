"""Training configuration dataclass for neural point map training."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class EnvTrainConfig:
    """Training configuration extracted from config dict."""

    # Scene bounds
    scene_min: Tuple[float, float, float]
    scene_max: Tuple[float, float, float]

    # Point map
    voxel_size: float
    knn_k: int
    feature_dim: int
    hash_table_size: int
    num_nei_cells: int
    search_alpha: float

    # Decoder
    decoder_hidden_dim: int

    # Training
    learning_rate: float
    batch_size: int
    num_epochs: int
    log_interval: int
    save_interval: int

    # Model
    model_type: str
    image_size: int
    patch_size: int
    load_dino_features: bool

    # Optional - Decoder
    decoder_num_res_blocks: int = 2
    decoder_dropout: float = 0.0
    train_decoder: bool = True  # Train decoder (false = freeze weights)

    # Optional - Encoder
    encoder_hidden_dim: int = 384
    encoder_num_res_blocks: int = 2
    encoder_dropout: float = 0.0
    use_encoder_init: bool = False  # Use encoder for point initialization
    train_encoder: bool = False     # Train encoder with consistency loss

    # Optional - Visualization
    run_pca: bool = False  # visualization.enabled
    lambda_consistency: float = 1.0
    vis_interval: int = 0  # visualization.interval
    num_images: int = -1
    target_envs: List[str] = None

    # Inter-category contrastive loss (semantic-based)
    lambda_inter: float = 0.1
    inter_temperature: float = 0.1
    inter_max_samples: int = 8192
    cross_scene_inter: bool = False  # True: cross-scene balanced loss; False: single-scene

    # Intra-instance contrastive loss (part-based)
    lambda_intra: float = 0.1
    intra_temperature: float = 0.1
    intra_max_samples: int = 16384

    @classmethod
    def from_dict(cls, config: dict) -> "EnvTrainConfig":
        """Create EnvTrainConfig from config dictionary."""
        point_map = config['point_map']
        training = config['training']
        decoder = config['decoder']
        encoder = config.get('encoder', {})

        model_type = config['model_type']
        if model_type == "dino":
            model_config = config['dino_model']
            load_dino_features = True
        else:
            model_config = config['clip_model']
            load_dino_features = False

        return cls(
            scene_min=tuple(config['scene_min']),
            scene_max=tuple(config['scene_max']),
            voxel_size=point_map['voxel_size'],
            knn_k=point_map['knn_k'],
            feature_dim=point_map['feature_dim'],
            hash_table_size=point_map['hash_table_size'],
            num_nei_cells=point_map['num_nei_cells'],
            search_alpha=point_map['search_alpha'],
            decoder_hidden_dim=decoder['hidden_dim'],
            decoder_num_res_blocks=decoder.get('num_res_blocks', 2),
            decoder_dropout=decoder.get('dropout', 0.0),
            train_decoder=decoder.get('train_decoder', True),
            encoder_hidden_dim=encoder.get('hidden_dim', 384),
            encoder_num_res_blocks=encoder.get('num_res_blocks', 2),
            encoder_dropout=encoder.get('dropout', 0.0),
            use_encoder_init=encoder.get('use_encoder_init', False),
            train_encoder=encoder.get('train_encoder', False),
            lambda_consistency=training.get('lambda_consistency', 1.0),
            learning_rate=training['optimizer_lr'],
            batch_size=training.get('batch_size', 1),
            num_epochs=training['epochs'],
            log_interval=training['log_interval'],
            save_interval=training.get('save_interval', 10),
            model_type=model_type,
            image_size=model_config['image_size'],
            patch_size=model_config['patch_size'],
            load_dino_features=load_dino_features,
            run_pca=config.get('visualization', {}).get('enabled', False),
            vis_interval=config.get('visualization', {}).get('interval', 0),
            num_images=config.get('num_images', -1),
            target_envs=config.get('target_envs', []),
            # Inter-category contrastive loss parameters
            lambda_inter=training.get('lambda_inter', 0.1),
            inter_temperature=training.get('inter_temperature', 0.1),
            inter_max_samples=training.get('inter_max_samples', 256),
            cross_scene_inter=training.get('cross_scene_inter', False),
            # Intra-instance contrastive loss parameters
            lambda_intra=training.get('lambda_intra', 0.1),
            intra_temperature=training.get('intra_temperature', 0.1),
            intra_max_samples=training.get('intra_max_samples', 256),
        )
