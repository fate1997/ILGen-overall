import os
import pathlib

from moima.pipeline.vae_pipe import VAEPipe, VAEPipeConfig

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


if __name__ == '__main__':
    config_file = str(REPO_DIR / 'step2_vae_generation' / 'vae_config.yaml')
    output_path = str(REPO_DIR / 'step2_vae_generation' / 'output')
    config = VAEPipeConfig.from_file(config_file)
    pipe = VAEPipe(config)
    pipe.train()
