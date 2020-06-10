import sys
import os
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from cbow_skip_gram.model import CBOW
from utilities.data_preprocessing import data_prep_from_file

import logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

log = logging.getLogger()

def setup_arg_parser():
    arg_parser = argparse.ArgumentParser(description='Parameter for training')
    arg_parser.add_argument('--data-file', dest="data_file_location",
                            help='Data file location')
    arg_parser.add_argument('--embedding_dims', dest = "embedding_dims", type = int,
                    help='embedding dimensions')
    arg_parser.add_argument('--batch_size', dest="batch_size", type = int,
                            help='batch size for training')
    arg_parser.add_argument('--lr', dest="lr", type = float,
                            help='Learning rate for training')
    arg_parser.add_argument('--epochs', dest="num_epochs", type = int,
                            help='Number of epochs')


    return arg_parser


def train(args):
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args(args)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Create data for train, valid, test
    log.info("Preparing data for training.....")
    train_dl, valid_dl, test_dl, vocab = data_prep_from_file(args.data_file_location)
    log.info("Data preparation completed. ")

    model = CBOW(vocab.vocab_size, args.embedding_dims)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    train_iter = iter(train_dl)
    valid_iter = iter(valid_dl)


    for epoch in tqdm(range(args.num_epochs)):
        total_loss = 0
        log.info("Processing epoch {}...".format(epoch))
        for batch in tqdm(train_iter):
            output_logits = model(batch[:, :-1].type(torch.long))
            loss = loss_function(output_logits, batch[:, -1].type(torch.long) )
            total_loss += loss
            loss.backward()
            optimizer.step()
        log.info("Loss : {}".format(total_loss))







if __name__ == "__main__":
    train(sys.argv[1:])

