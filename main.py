import base64
import nsml
import numpy as np
from PIL import Image
from io import BytesIO
from parameter import get_parameters
from finetuner import Finetuner
from data_loader import get_loader
from utils import make_folder
from torch.backends import cudnn


def get_bindings(trainer):
    def save(filename, *args):
        trainer.save(filename)

    def load(filename, *args):
        trainer.load(filename)

    def infer(input):
        result = trainer.infer(input)
        # convert tensor to dataurl
        data_url_list = [''] * input
        for idx, sample in enumerate(result):
            numpy_array = np.uint8(sample.cpu().numpy()*255)
            image = Image.fromarray(np.transpose(numpy_array, axes=(1, 2, 0)), 'RGB')
            temp_out = BytesIO()
            image.save(temp_out, format='png')
            byte_data = temp_out.getvalue()
            data_url_list[idx] = u'data:image/{format};base64,{data}'.\
                format(format='png',
                       data=base64.b64encode(byte_data).decode('ascii'))
        return data_url_list

    return save, load, infer


def main(config):
    # For fast training
    cudnn.benchmark = True

    # Data loader
    loader = get_loader(config.dataset, config.batch_size, config.num_workers, config.img_size)
    print("Succesfully load dataset : {}".format(len(loader)))
    
    # Create directories if not exist
    make_folder(config.model_save_path)
    if config.mode == 'train':
        from trainer_jjy import Trainer
        trainer = Trainer(loader, config)
    elif config.mode == 'fixedPool':
        from trainer_fixedP_jjy import Trainer
        trainer = Trainer(loader, config)
    elif config.mode == 'finetune':
        trainer = Finetuner(loader, config)
    else:
        raise NotImplementedError()    
    
    save, load, infer = get_bindings(Trainer)
    nsml.bind(save=save, load=load, infer=infer)
    
    trainer.train()


if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)
