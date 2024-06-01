#!/usr/bin/python3

# System imports
import argparse
import sys
import serial

# Data processing imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns

def checkparams(pwm_freq, pwm_duty, num_samples):
    check_ok = True
    if pwm_freq < 20 or pwm_freq > 100:
        print("Allowed PWM freq is between in [20, 100] kHz interval.")
        check_ok = False
    if pwm_duty < 5 or pwm_duty > 80:
        print("Allowed PWM duty is between in [5, 80] percent interval.")
        check_ok = False
    if num_samples < 1 or num_samples > 20000:
        print("Allowed samples num is between in [1, 8192] interval.")
        check_ok = False
    if check_ok == False:
        sys.exit(1);

def main(baudrate, pwm_freq, pwm_duty, num_samples, delays_file):

    ser = serial.Serial(
        port='/dev/ttyUSB0',
        baudrate=baudrate,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        rtscts=0
    )

    if not ser.is_open:
        print("Error opening serial port device.")
        sys.exit(1)

    checkparams(pwm_freq, pwm_duty, num_samples)

    print("Params OK!")

    delays = np.empty(num_samples)

    ser.write(str.encode('{},{},{}\r\n'.format(
        pwm_freq, 
        pwm_duty, 
        num_samples)))

    timer_frequency = int(ser.readline().strip()) # MHz
    ser.write(str.encode('\n')); # start measurement

    for i in range(num_samples):
        delays[i] = int(ser.readline().strip())

    ser.close()

    delays *= (1e-6 / timer_frequency);

    delays = np.delete(delays, 0);
    delays = np.delete(delays, 0);

    print("min: {}, avg: {}, max = {}".format(
        np.min(delays),
        np.mean(delays),
        np.max(delays)));

    print("std: ", np.std(delays))


    LOG_FILE = open(delays_file, 'w')

    np.save(delays_file, delays);

    # mean = np.mean(delays);
    # maxi = np.max(delays);
    # mini = np.min(delays);

    # # sns.distplot(delays, norm_hist=True);

    # # plt.show();
    # # 
    # delays *= 1e6;

    # plt.plot(delays)
    # plt.ylabel('Vrijeme kašnjenja (${\mu}s$)')
    # plt.xlabel('Uzorci (padajući brid odziva)')
    # plt.show()

    # plt.figure(0)
    # n, bins, patches = plt.hist(delays, 50, normed=True, 
    #     histtype='step');

    # y = mlab.normpdf(bins, 
    #     np.mean(delays), 
    #     np.std(delays))

    # plt.show()
    # plt.figure(1)
    # plt.plot(bins, y)
    # plt.xlabel('Vrijeme kašnjenja (${\mu}s$)')
    # plt.ylabel('Funkcija gustoće vjerojatnosti')
    # plt.show();
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--pwm_freq', type=int, default=20)
    parser.add_argument('--pwm_duty', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=20000)
    parser.add_argument('--delays_file', type=str, default='novo.npy')

    ARGS, other = parser.parse_known_args()

    main(ARGS.baudrate, ARGS.pwm_freq, ARGS.pwm_duty, ARGS.num_samples, 
        ARGS.delays_file);
