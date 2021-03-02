from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pl_bolts.callbacks import PrintTableMetricsCallback

from model import MInterface
from data import DInterface


def load_callbacks():
    callbacks = []
    callbacks.append(ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True
    ))

    callbacks.append(EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.01
    ))

    callbacks.append(PrintTableMetricsCallback())
    return callbacks


def main(args):
    model = MInterface(**vars(args))
    data_module = DInterface(**vars(args))
    callbacks = load_callbacks()
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--num_workers', default=8, type=int)

    # Training Info
    parser.add_argument('--dataset', default='satup_data', type=str)
    parser.add_argument('--data_dir', default='dataset', type=str)
    parser.add_argument('--model_name', default='rdnfuse', type=str)
    parser.add_argument('--loss', default='l1', type=str)
    parser.add_argument('--augment', action='store_true')

    # Model Hyperparameters
    parser.add_argument('--scale', default=2, type=int)
    parser.add_argument('--in_bands_num', default=12, type=int)
    parser.add_argument('--hid', default=64, type=int)
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--rdn_size', default=3, type=int)
    parser.add_argument('--rdb_growrate', default=64, type=int)
    parser.add_argument('--rdb_conv_num', default=8, type=int)

    # Other
    parser.add_argument('--color_range', default=255, type=int)

    parser = Trainer.add_argparse_args(
        parser.add_argument_group(title="pl.Trainer args"))
    args = parser.parse_args()

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=250)

    # List Arguments
    args.mean_sen = [1.315, 1.211, 1.948, 1.892, 3.311,
                     6.535, 7.634, 8.197, 8.395, 8.341, 5.89, 3.616]
    args.std_sen = [5.958, 2.273, 2.299, 2.668, 2.895,
                    4.276, 4.978, 5.237, 5.304, 5.103, 4.298, 3.3]

    main(args)
