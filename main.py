import argparse
import os
import sys
import logging
import yaml
import shutil

from utils.train_utils import set_backend_random_seed, get_optimizer, get_scheduler
from utils.model_utils import get_model, freeze_model, load_check_point

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./experiments/apag_fpn_widerface2.yml")
    args, unknown = parser.parse_known_args()
    return args

def main( config_path):

    # configure
    with open(config_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(conf['MODEL']['PATH_SAVE'], exist_ok=True)
    shutil.copy2(config_path, os.path.join(conf['MODEL']['PATH_SAVE'], os.path.basename(config_path)))
    path_log = os.path.join(conf['MODEL']['PATH_SAVE'], 'log.txt')

    if os.path.exists(path_log):
        os.remove(path_log)

    log_targets = logging.StreamHandler(sys.stdout), logging.FileHandler(path_log)
    logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=log_targets)

    # model -----
    set_backend_random_seed(conf)
    model = get_model(conf, print_model_info=False)
    model = freeze_model(conf, model, printInfo=True)
    model = load_check_point(conf, model)

    # data loader -----
    dataset_name = conf['DATASET']['NAME']
    exec('from dataset.' + dataset_name + ' import get_train_loader, get_valid_loader')
    logging.info('load train dataset --------------------------------')
    train_dataloader = eval('get_train_loader(conf)')
    conf['NUM_TRAIN_STEPS'] = len(train_dataloader)
    logging.info('load valid dataset --------------------------------')
    valid_dataloader = eval('get_valid_loader(conf)')
    conf['NUM_VALID_STEPS'] = len(valid_dataloader)

    # optimizer, scheduler -----
    optimizer_conf = conf["OPTIMIZER"]
    opt = get_optimizer(model, **optimizer_conf)
    scheduler = get_scheduler(opt, conf)

    # trainer, validator -----
    trainer_name = conf['MODEL']['OUTPUT_FORMAT']
    exec('from train_val.' + trainer_name + ' import Trainer, Validationer')
    trainer = eval('Trainer(conf, train_dataloader, opt, scheduler)')
    validationer = eval('Validationer(conf)')

    os.makedirs(conf['MODEL']['PATH_SAVE'], exist_ok=True)

    logging.info('start training --------------------------------------')
    end_epoch = conf["TRAIN"]["END_EPOCH"]
    for epoch in range(1, end_epoch + 1):
        # train
        trainer(model, epoch)

        # validation
        validationer(model, epoch, valid_dataloader)

if __name__ == "__main__":
    args = parse_args()
    main(args.config_path)