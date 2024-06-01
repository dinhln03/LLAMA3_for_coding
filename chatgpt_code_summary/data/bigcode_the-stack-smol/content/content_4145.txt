import os
import csv
import shutil
from datetime import datetime
from numpy import logspace

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset.e_piano import create_epiano_datasets, create_pop909_datasets

from model.music_transformer import MusicTransformer

from model.discriminator import MusicDiscriminator
from model.classifier import CNNDiscriminator

from model.loss import SmoothCrossEntropyLoss

from utilities.constants import *
from utilities.WGAN_GP import WassersteinLoss
from utilities.device import get_device, use_cuda
from utilities.lr_scheduling import LrStepTracker, get_lr
from utilities.argument_funcs import parse_train_args, print_train_args, write_model_params
from utilities.run_model import train_epoch, eval_model

CSV_HEADER = ["Epoch", "Learn rate", "Avg Train loss", "Train Accuracy", "Avg Eval loss", "Eval accuracy"]

dis_filter_sizes = [2, 3, 4, 5]
dis_num_filters = [300, 300, 300, 300]

# Baseline is an untrained epoch that we evaluate as a baseline loss and accuracy
BASELINE_EPOCH = -1

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Trains a model specified by command line arguments
    ----------
    """

    args = parse_train_args()
    print_train_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    eventid = f"{datetime.now().strftime('MusicTransformer-%Y.%m.%d')}_gan_{args.gan}_creative_{args.creative}_ce_{args.ce_smoothing}"

    args.output_dir = args.output_dir  + "/" +  eventid

    os.makedirs(args.output_dir, exist_ok=True)

    ##### Output prep #####
    params_file = os.path.join(args.output_dir, "model_params.txt")
    write_model_params(args, params_file)

    weights_folder = os.path.join(args.output_dir, "weights")
    os.makedirs(weights_folder, exist_ok=True)

    results_folder = os.path.join(args.output_dir, "results")
    os.makedirs(results_folder, exist_ok=True)

    results_file = os.path.join(results_folder, "results.csv")
    best_loss_file = os.path.join(results_folder, "best_loss_weights.pickle")
    best_acc_file = os.path.join(results_folder, "best_acc_weights.pickle")
    best_loss_critic_file = os.path.join(results_folder, "best_loss_critic_weights.pickle")
    best_acc_critic_file = os.path.join(results_folder, "best_acc_critic_weights.pickle")

    best_loss_classifier_file = os.path.join(
        results_folder, "best_loss_classifier_weights.pickle")
    best_acc_classifier_file = os.path.join(
        results_folder, "best_acc_classifier_weights.pickle")

    best_text = os.path.join(results_folder, "best_epochs.txt")

    ##### Tensorboard #####
    if(args.no_tensorboard):
        tensorboard_summary = None
    else:
        from torch.utils.tensorboard import SummaryWriter        

        tensorboad_dir = os.path.join(args.output_dir, "tensorboard/" + eventid)
        tensorboard_summary = SummaryWriter(log_dir=tensorboad_dir)

    ##### Datasets #####
    # 데이터셋이 바뀌기 때문에 아래와같이 해주어야함
    if args.interval and args.octave:
        print("octave interval dataset!!")
        classic_train, classic_val, classic_test = create_epiano_datasets('./dataset/octave_interval_e_piano', args.max_sequence,
                                                                          condition_token=args.condition_token, interval = args.interval, octave = args.octave)
        pop909_dataset = create_pop909_datasets('./dataset/logscale_pop909', args.max_sequence, condition_token=args.condition_token, interval = args.interval, octave = args.octave)
        pop_train, pop_valid, pop_test = torch.utils.data.random_split(pop909_dataset,
                                                                       [int(len(pop909_dataset) * 0.8), int(len(pop909_dataset) * 0.1),
                                                                        len(pop909_dataset) - int(len(pop909_dataset) * 0.8) - int(len(pop909_dataset) * 0.1)],
                                                                       generator=torch.Generator().manual_seed(42))
    elif args.octave and args.fusion_encoding and args.absolute:
        print("absolute dataset!!")
        classic_train, classic_val, classic_test = create_epiano_datasets('./dataset/octave_fusion_absolute_e_piano', args.max_sequence,
                                                                          condition_token=args.condition_token, interval = args.interval, octave = args.octave, fusion = args.fusion_encoding, absolute = args.absolute)
        pop909_dataset = create_pop909_datasets('./dataset/pop909_absolute', args.max_sequence, condition_token=args.condition_token, interval = args.interval, octave = args.octave, fusion = args.fusion_encoding, absolute = args.absolute)
        pop_train, pop_valid, pop_test = torch.utils.data.random_split(pop909_dataset,
                                                                       [int(len(pop909_dataset) * 0.8), int(len(pop909_dataset) * 0.1),
                                                                        len(pop909_dataset) - int(len(pop909_dataset) * 0.8) - int(len(pop909_dataset) * 0.1)],
                                                                       generator=torch.Generator().manual_seed(42))
    elif args.interval and not args.octave:
        print("interval dataset!!")
        classic_train, classic_val, classic_test = create_epiano_datasets('./dataset/logscale_e_piano', args.max_sequence,
                                                                          condition_token=args.condition_token, interval = args.interval, octave = args.octave)
        pop909_dataset = create_pop909_datasets('./dataset/logscale_pop909', args.max_sequence, condition_token=args.condition_token, interval = args.interval, octave = args.octave)
        pop_train, pop_valid, pop_test = torch.utils.data.random_split(pop909_dataset,
                                                                       [int(len(pop909_dataset) * 0.8), int(len(pop909_dataset) * 0.1),
                                                                        len(pop909_dataset) - int(len(pop909_dataset) * 0.8) - int(len(pop909_dataset) * 0.1)],
                                                                       generator=torch.Generator().manual_seed(42))
    elif args.octave and args.fusion_encoding:
        print("Octave_fusion dataset!!")
        classic_train, classic_val, classic_test = create_epiano_datasets('./dataset/octave_fusion_e_piano', args.max_sequence,
                                                                          condition_token=args.condition_token, interval = args.interval, octave = args.octave, fusion = args.fusion_encoding)
        pop909_dataset = create_pop909_datasets('./dataset/logscale_pop909', args.max_sequence, condition_token=args.condition_token, interval = args.interval, octave = args.octave, fusion = args.fusion_encoding)
        pop_train, pop_valid, pop_test = torch.utils.data.random_split(pop909_dataset,
                                                                       [int(len(pop909_dataset) * 0.8), int(len(pop909_dataset) * 0.1),
                                                                        len(pop909_dataset) - int(len(pop909_dataset) * 0.8) - int(len(pop909_dataset) * 0.1)],
                                                                       generator=torch.Generator().manual_seed(42))
    elif not args.interval and args.octave and not args.fusion_encoding:
        print("Octave dataset!!")
        classic_train, classic_val, classic_test = create_epiano_datasets('./dataset/octave_e_piano', args.max_sequence,
                                                                          condition_token=args.condition_token, interval = args.interval, octave = args.octave)
        pop909_dataset = create_pop909_datasets('./dataset/pop909_octave', args.max_sequence, condition_token=args.condition_token, interval = args.interval, octave = args.octave)
        pop_train, pop_valid, pop_test = torch.utils.data.random_split(pop909_dataset,
                                                                       [int(len(pop909_dataset) * 0.8), int(len(pop909_dataset) * 0.1),
                                                                        len(pop909_dataset) - int(len(pop909_dataset) * 0.8) - int(len(pop909_dataset) * 0.1)],
                                                                       generator=torch.Generator().manual_seed(42))
    elif args.logscale:
        print("logscvale dataset")
        classic_train, classic_val, classic_test = create_epiano_datasets('./dataset/logscale_epiano0420', args.max_sequence, random_seq=True,
                                                                            condition_token=args.condition_token, interval = args.interval, octave = args.octave, logscale=args.logscale, absolute = args.absolute)
        pop909_dataset = create_pop909_datasets('./dataset/logscale_pop0420', args.max_sequence, random_seq=True, condition_token=args.condition_token, interval = args.interval, octave = args.octave, logscale=args.logscale, absolute = args.absolute)
        pop_train, pop_valid, pop_test = torch.utils.data.random_split(pop909_dataset,
                                                                        [int(len(pop909_dataset) * 0.8), int(len(pop909_dataset) * 0.1),
                                                                        len(pop909_dataset) - int(len(pop909_dataset) * 0.8) - int(len(pop909_dataset) * 0.1)],
                                                                        generator=torch.Generator().manual_seed(42))
    else:
        classic_train, classic_val, classic_test = create_epiano_datasets(args.classic_input_dir, args.max_sequence,
                                                                          condition_token = args.condition_token, octave = args.octave)
        pop909_dataset = create_pop909_datasets('dataset/pop_pickle/', args.max_sequence, condition_token = args.condition_token, octave = args.octave)
        pop_train, pop_valid, pop_test = torch.utils.data.random_split(pop909_dataset,
                                                                       [int(len(pop909_dataset) * 0.8), int(len(pop909_dataset) * 0.1), len(pop909_dataset) - int(len(pop909_dataset) * 0.8) - int(len(pop909_dataset) * 0.1)],
                                                                       generator=torch.Generator().manual_seed(42))

    if args.data == 'both':
        print("Dataset: both")
        train_dataset = torch.utils.data.ConcatDataset([ classic_train, pop_train])
        val_dataset = torch.utils.data.ConcatDataset([ classic_val, pop_valid])
    elif args.data == 'classic':
        print("Dataset: classic")
        train_dataset = torch.utils.data.ConcatDataset([classic_train])
        val_dataset = torch.utils.data.ConcatDataset([classic_val])
    else:
        print("Dataset: pop")
        train_dataset = torch.utils.data.ConcatDataset([pop_train])
        val_dataset = torch.utils.data.ConcatDataset([pop_valid])

    test_dataset = torch.utils.data.ConcatDataset([classic_test, pop_test])


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                                    max_sequence=args.max_sequence, rpr=args.rpr, 
                                    condition_token = args.condition_token, interval = args.interval, octave = args.octave, 
                                    fusion = args.fusion_encoding, absolute = args.absolute, logscale=args.logscale).to(get_device())

    # EY critic
    # num_prime = args.num_prime
    critic = MusicDiscriminator(n_layers=args.n_layers // 2, num_heads=args.num_heads // 2,
                d_model=args.d_model // 2, dim_feedforward=args.dim_feedforward // 2, dropout=args.dropout,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    classifier = MusicDiscriminator(n_layers=args.n_layers // 2, num_heads=args.num_heads // 2,
                                 d_model=args.d_model // 2, dim_feedforward=args.dim_feedforward // 2, dropout=args.dropout,
                                 max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())


    if args.creative:
        classifier.load_state_dict(torch.load('best_classifier_acc_0.9883.pickle'))

    ##### Continuing from previous training session #####
    start_epoch = BASELINE_EPOCH
    if(args.continue_weights is not None):
        if(args.continue_epoch is None):
            print("ERROR: Need epoch number to continue from (-continue_epoch) when using continue_weights")
            return
        else:
            model.load_state_dict(torch.load(args.continue_weights))
            start_epoch = args.continue_epoch
    elif(args.continue_epoch is not None):
        print("ERROR: Need continue weights (-continue_weights) when using continue_epoch")
        return

    ##### Lr Scheduler vs static lr #####
    if(args.lr is None):
        if(args.continue_epoch is None):
            init_step = 0
        else:
            init_step = args.continue_epoch * len(train_loader)

        lr = LR_DEFAULT_START
        lr_stepper = LrStepTracker(args.d_model, SCHEDULER_WARMUP_STEPS, init_step)
    else:
        lr = args.lr

    ##### Not smoothing evaluation loss #####
    if args.interval and args.octave:
        eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD_OCTAVE_INTERVAL)
    elif args.interval and not args.octave:
        eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD_INTERVAL)
    elif args.octave and args.fusion_encoding and args.absolute:
        eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD_OCTAVE_FUSION_ABSOLUTE)
    elif args.octave and args.fusion_encoding:
        eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD_OCTAVE_FUSION)
    elif not args.interval and args.octave and not args.fusion_encoding:
        eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD_OCTAVE)
    elif args.logscale:
        eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD_RELATIVE)
    else:
        eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)


    ##### SmoothCrossEntropyLoss or CrossEntropyLoss for training #####
    if(args.ce_smoothing is None):
        train_loss_func = eval_loss_func
    else:
        if args.interval and args.octave:
            train_loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE_OCTAVE_INTERVAL, ignore_index=TOKEN_PAD_INTERVAL)
        elif args.interval and not args.octave:
            train_loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE_INTERVAL, ignore_index=TOKEN_PAD_INTERVAL)
        elif not args.interval and args.octave and args.fusion_encoding and args.absolute:
            train_loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE_OCTAVE_FUSION_ABSOLUTE, ignore_index=TOKEN_PAD_OCTAVE_FUSION_ABSOLUTE)
        elif not args.interval and args.octave and args.fusion_encoding:
            train_loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE_OCTAVE_FUSION, ignore_index=TOKEN_PAD_OCTAVE_FUSION)
        elif not args.interval and args.octave and not args.fusion_encoding:
            train_loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE_OCTAVE, ignore_index=TOKEN_PAD_OCTAVE)
        elif args.logscale:
            train_loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE_RELATIVE, ignore_index=TOKEN_PAD_RELATIVE)
        else:
            train_loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE, ignore_index=TOKEN_PAD)

    ##### EY - WGAN Loss #####
    classifier_loss_func = nn.MSELoss()

    ##### Optimizer #####
    opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
    critic_opt = Adam(critic.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
    classifier_opt = Adam(classifier.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)

    if(args.lr is None):
        lr_scheduler = LambdaLR(opt, lr_stepper.step)
        critic_lr_scheduler = LambdaLR(critic_opt, lr_stepper.step)
        classifier_lr_scheduler = LambdaLR(classifier_opt, lr_stepper.step)
    else:
        lr_scheduler = None

    ##### Tracking best evaluation accuracy #####
    best_eval_acc        = 0.0
    best_eval_acc_epoch  = -1
    best_eval_loss       = float("inf")
    best_eval_loss_epoch = -1

    ##### Results reporting #####
    if(not os.path.isfile(results_file)):
        with open(results_file, "w", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow(CSV_HEADER)


    ##### TRAIN LOOP #####
    for epoch in range(start_epoch, args.epochs):
        # Baseline has no training and acts as a base loss and accuracy (epoch 0 in a sense)
        if(epoch >= BASELINE_EPOCH):
            print(SEPERATOR)
            print("NEW EPOCH:", epoch+1)
            print(SEPERATOR)
            print("")

            # Train
            # EY 고쳐야 할 부분의 시작
            train_loss, train_acc, dis_loss, gen_loss, cre_loss, gan_accuracy, class_accuracy, creativity = train_epoch(epoch+1, model, critic, classifier, train_loader, train_loss_func, classifier_loss_func, opt, critic_opt, classifier_opt, lr_scheduler, critic_lr_scheduler, classifier_lr_scheduler, args)

            print(SEPERATOR)
            print("Evaluating:")
        else:
            print(SEPERATOR)
            print("Baseline model evaluation (Epoch 0):")

        # Eval
        # train_loss, train_acc = eval_model(model, train_loader, train_loss_func)
        eval_loss, eval_acc = eval_model(model, val_loader, eval_loss_func, args)

        # Learn rate
        lr = get_lr(opt)

        print("Epoch:", epoch+1)
        print("Avg train loss:", train_loss)
        print("Avg train acc:", train_acc)
        print("Avg eval loss:", eval_loss)
        print("Avg eval acc:", eval_acc)
        print(SEPERATOR)
        print("")

        new_best = False

        if(eval_acc > best_eval_acc):
            best_eval_acc = eval_acc
            best_eval_acc_epoch  = epoch+1
            torch.save(model.state_dict(), best_acc_file)
            torch.save(critic.state_dict(), best_acc_critic_file)
            torch.save(classifier.state_dict(), best_acc_classifier_file)
            new_best = True

        if(eval_loss < best_eval_loss):
            best_eval_loss       = eval_loss
            best_eval_loss_epoch = epoch+1
            torch.save(model.state_dict(), best_loss_file)
            torch.save(critic.state_dict(), best_loss_critic_file)
            torch.save(classifier.state_dict(), best_loss_classifier_file)
            new_best = True

        # Writing out new bests
        if(new_best):
            with open(best_text, "w") as o_stream:
                print("Best eval acc epoch:", best_eval_acc_epoch, file=o_stream)
                print("Best eval acc:", best_eval_acc, file=o_stream)
                print("")
                print("Best eval loss epoch:", best_eval_loss_epoch, file=o_stream)
                print("Best eval loss:", best_eval_loss, file=o_stream)


        if(not args.no_tensorboard):
            tensorboard_summary.add_scalar("Avg_CE_loss/train", train_loss, global_step=epoch+1)
            tensorboard_summary.add_scalar("Avg_CE_loss/eval", eval_loss, global_step=epoch+1)
            tensorboard_summary.add_scalar("Accuracy/train", train_acc, global_step=epoch+1)
            tensorboard_summary.add_scalar("Accuracy/eval", eval_acc, global_step=epoch+1)
            tensorboard_summary.add_scalar("Learn_rate/train", lr, global_step=epoch+1)

            tensorboard_summary.add_scalar("Critic_loss/train", dis_loss, global_step=epoch+1)
            tensorboard_summary.add_scalar("Gen_loss/train", gen_loss, global_step=epoch+1)
            tensorboard_summary.add_scalar("Creativity_loss/train", cre_loss, global_step=epoch+1)
            tensorboard_summary.add_scalar("GAN_accuracy/train", gan_accuracy, global_step=epoch+1)
            tensorboard_summary.add_scalar("Class_accuracy/train", class_accuracy, global_step=epoch+1)
            tensorboard_summary.add_scalar("Creativity/train", creativity, global_step=epoch+1)

            tensorboard_summary.flush()

        if((epoch+1) % args.weight_modulus == 0):
            epoch_str = str(epoch+1).zfill(PREPEND_ZEROS_WIDTH)
            path = os.path.join(weights_folder, "epoch_" + epoch_str + ".pickle")
            torch.save(model.state_dict(), path)

        with open(results_file, "a", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow([epoch+1, lr, train_loss, train_acc, eval_loss, eval_acc])

    # Sanity check just to make sure everything is gone
    if(not args.no_tensorboard):
        tensorboard_summary.flush()

    return


if __name__ == "__main__":
    main()
