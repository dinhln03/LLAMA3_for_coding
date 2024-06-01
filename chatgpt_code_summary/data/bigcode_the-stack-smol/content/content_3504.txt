import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from test import run_test

if __name__ == '__main__':
    opt = TrainOptions().parse()
    # opt.serial_batches = True  # no shuffle
    print('Creating DataLoader...')
    dataset = DataLoader(opt)
    print('DataLoader created!')

    print('#training meshes = %d' % dataset_size)

    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, 
                       opt.niter + opt.niter_decay + 1):

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        o_ncorrect = 0
        o_nexamples = 0
        o_pr = 0
        o_re = 0
        model.save_network(0)
        for i, data in enumerate(dataset):
            print(i)
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            ncorrect, nexamples, pr, re = model.optimize_parameters()
            
            o_ncorrect += ncorrect
            o_nexamples += nexamples
            o_pr += pr
            o_re += re
            
            if total_steps % opt.print_freq == 0:
                loss = model.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest')
            model.save_network(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            acc, pr, re = run_test(epoch)
            writer.plot_acc(acc, epoch)
            writer.plot_pr(pr, epoch)
            writer.plot_re(re, epoch)
            writer.plot_train_acc(float(o_ncorrect)/o_nexamples, epoch)
            writer.plot_train_pr(float(o_pr)/o_nexamples, epoch)
            writer.plot_train_re(float(o_re)/o_nexamples, epoch)


    writer.close()
