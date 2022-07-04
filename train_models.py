import os
import sys
import argparse
import logging
import shutil
import datetime
import torch
import torch.utils.data
import torch.optim as optim

from torchsummary import summary

import tensorboardX

from pathlib import Path

from utils.data import get_dataset
from models import get_network
from train_ggcnn import train, validate
from tqdm import tqdm


# GMDATA_PATH = Path.home().joinpath('Project/gmdata')
# DATASET_PATH = GMDATA_PATH.joinpath('datasets/train_datasets')
# INPUT_DATA_PATH = DATASET_PATH.joinpath('gg_data/small_data')#gg_data/shahao_data
# OUT_PATH = GMDATA_PATH.joinpath('datasets/models/gg2/shahao_model')



GMDATA_PATH = Path("/media/shahao/F07EE98F7EE94F42/win_stevehao/Research/gmdata")
DATASET_PATH= GMDATA_PATH.joinpath('datasets')
INPUT_DATA_PATH = DATASET_PATH.joinpath('train_datasets/gg_data/small_data/cor')



def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')

    # Network
    parser.add_argument('--network', type=str, default='ggcnn2', help='Network Name in .models')
    parser.add_argument('--resume', type=str, default=False,
                        help='to resume the interrupted model training')
    parser.add_argument('--start-epoch', type=int, default=44,
                        help='the next epoch to start')
    parser.add_argument('--resume-path', type=str, default="output/models/cor/epoch_43_iou_0.92",
                        help='to resume the interrupted model training')    
    # Dataset & Data & Training
    parser.add_argument('--input-size', type=int, default=1024,
                        help='Input image size for the network')
    parser.add_argument('--output-size', type=int, default=300,
                        help='output image size for the network')
    parser.add_argument('--dataset', type=str, default='cornell',
                        help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str, default=INPUT_DATA_PATH, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    #
    parser.add_argument('--lr', type=float, default=0.00003, help='learning rate')#0.001#gmd0.0001
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')#8
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=99, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=50, help='Validation Batches')

    # Logging etc.
    parser.add_argument('--description', type=str, default='gg_cor', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')
    parser.add_argument('--vis', action='store_true', help='Visualise the training process')

    args = parser.parse_args()
    return args


def run(args, save_folder, log_folder):
    tb = tensorboardX.SummaryWriter(log_folder)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    train_dataset = Dataset(args.dataset_path, start=0.0, end=args.split, ds_rotate=args.ds_rotate,
                            input_size=args.input_size, 
                            output_size=args.output_size,
                            random_rotate=True, random_zoom=True,
                            include_depth=args.use_depth, include_rgb=args.use_rgb)
    print("沙昊1",len(train_dataset.grasp_files))
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,        
        num_workers=args.num_workers
    )
    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=True,
                          include_depth=args.use_depth, include_rgb=args.use_rgb)
    print("沙昊",len(val_dataset.grasp_files))
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1*args.use_depth + 3*args.use_rgb

    if args.resume :
        net = torch.load(args.resume_path)
        start_epoch = args.start_epoch
        
    else :
        ggcnn = get_network(args.network)
        net = ggcnn(input_channels=input_channels)
        # target_net = ggcnn(input_channels=input_channels)
        start_epoch = 0

    device = torch.device("cuda:0")
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(),lr=args.lr)
    logging.info('Done')

    # Print model architecture.
    summary(net, (input_channels, 300, 300))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, 300, 300))
    sys.stdout = sys.__stdout__
    f.close()

    best_iou = 0.0
    for epoch in tqdm(range(start_epoch,args.epochs),desc="training"):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer,
                              args.batches_per_epoch, vis=args.vis)

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        # import pdb; pdb.set_trace()
        test_results = validate(net, device, val_data, args.val_batches)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))

        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] /
                      (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            torch.save(net.state_dict(), os.path.join(
                save_folder, 'epoch_%02d_iou_%0.2f_statedict.pt' % (epoch, iou)))
            best_iou = iou




def main():
    args = parse_args()
    # for t in 'gmd jaq cor'.split():
    #     if t == 'cor':
    #         args.dataset = 'cornell'
    #         args.batches_per_epoch = 562 
    #     elif t == 'jaq':
    #         continue
    #         args.dataset = 'jacquard'
    #         args.batches_per_epoch = 1162
    #     elif t == 'gmd':
    #         continue
    #         args.dataset = 'gmd'
    #         args.batches_per_epoch = 1440
    # args.dataset= 'gmd'
    # args.description = 'train %s' % (t)
    # args.dataset_path = INPUT_DATA_PATH.joinpath().as_posix()
    if args.resume : 
        net_desc= args.resume_path.split("/")[-2]
    else:
        dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
        net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))
    
    save_folder = os.path.join(args.outdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    log_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    run(args, save_folder, log_folder)


if __name__ == '__main__':
    main()
