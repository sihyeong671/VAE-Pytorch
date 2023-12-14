import os
import argparse

from module.trainer import Trainer
from module.utils import Config, interpolate

# https://www.tensorflow.org/tutorials/generative/cvae?hl=ko#%EC%9D%B8%EC%BD%94%EB%8D%94_%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC
if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=0, type=int) # 0으로 하지 않으면 데이터로더값이 nan으로 나오는 오류 발생
    parser.add_argument("--model_path", type=str, default="models/VAE_model.pth")
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    config = Config(args=args)
    print(config)
    trainer = Trainer(config=config)
    trainer.setup()
    
    if config.mode == "train":
        trainer.train()
    elif config.mode == "test":
        trainer.generate("test/test.png")
    elif config.mode == "interpolate":
        trainer.interpolate()
    
    
    