"""Training and evaluation"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import lib.diffusion.trainer as trainer
import lib.diffusion.evaler as evaler

"""Example command
[Training]

[Inference]
(Unconditional generation)
1. 
python main_diffusion.py --config=/root/dev/MeshDiffusion/configs/res128.py --mode=uncond_gen \
--config.eval.eval_dir=/root/data/text2shape/meshdiffusion/outputs \
--config.eval.ckpt_path=/root/data/text2shape/meshdiffusion/ckpt/chair_res128.pt
2. 
cd nvdiffrec
python eval.py --config configs/res128.json \
--out-dir /root/data/text2shape/meshdiffusion/outputs \
--sample-path /root/data/text2shape/meshdiffusion/outputs/0.npy \
--deform_scale 3.0 --angle-ind 20
"""


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "diffusion configs", lock_config=False)
flags.DEFINE_enum("mode", None, ["train", "uncond_gen", "cond_gen"], "Running mode")
flags.mark_flags_as_required(["config", "mode"])


def main(argv):
    if FLAGS.mode == 'train':
        trainer.train(FLAGS.config)
    elif FLAGS.mode == 'uncond_gen':
        evaler.uncond_gen(FLAGS.config)
    elif FLAGS.mode == 'cond_gen':
        evaler.cond_gen(FLAGS.config)

if __name__ == "__main__":
  app.run(main)
