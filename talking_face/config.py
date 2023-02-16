from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
def generate_absolute_path_string(relative_path: str) -> str:
    return str(PROJECT_ROOT / relative_path)

OPENFACE_POSE_EXTRACTOR_PATH = generate_absolute_path_string('OpenFace/FeatureExtraction')
GENERATOR_CKPT = generate_absolute_path_string("checkpoints/generator.ckpt")
AUDIO2POSE_CKPT = generate_absolute_path_string("checkpoints/audio2pose.ckpt")
