import os
import argparse
from ops.os_operation import mkdir
import time

def write_slurm_sh_multi_H2(id,command_line, queue_name="learnfair",nodes=1,
                   gpu_per_node=8,wall_time=3*24*60,username="wang3702",CPU_PER_GPU=8):
    import time
    import datetime
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    now = time.strftime("%H:%M:%S")
    dependency_handler_path = os.path.join(os.getcwd(), "ops")
    dependency_handler_path = os.path.join(dependency_handler_path, "handler.txt")
    run_path = os.path.join(os.getcwd(), "log")
    mkdir(run_path)
    run_path = os.path.abspath(run_path)
    prefix = "node%d_gpu%d"%(nodes,gpu_per_node)
    batch_file = os.path.join(run_path, prefix+"slurm_job_" + str(id) + ".sh")
    output_path = os.path.join(run_path, prefix+"output_" + str(id) + "_" + str(formatted_today + now) + ".log")
    error_path = os.path.join(run_path, prefix+"error_" + str(id) + "_" + str(formatted_today + now) + ".log")
    with open(batch_file, "w") as file:
        file.write("#! /bin/bash\n")#!/bin/bash
        file.write("#SBATCH --job-name=%s\n" % id)
        file.write("#SBATCH --output=%s\n" % output_path)
        file.write("#SBATCH --error=%s\n" % error_path)
        file.write("#SBATCH --partition=%s\n"%queue_name)
        file.write("#SBATCH --signal=USR1@600\n")
        file.write("#SBATCH --nodes=%d\n" % nodes)
        file.write("#SBATCH --ntasks-per-node=%d\n" % 1)
        file.write("#SBATCH --mem=%dG\n"%(350/8*gpu_per_node))
        file.write("#SBATCH --gpus=%d\n" % (nodes * gpu_per_node))
        file.write("#SBATCH --gpus-per-node=%d\n" % (gpu_per_node))
        file.write("#SBATCH --cpus-per-task=%d\n"%(CPU_PER_GPU*gpu_per_node))
        file.write("#SBATCH --time=%d\n"%wall_time)
        file.write("#SBATCH --mail-user=%s@fb.com\n"%username)
        file.write("#SBATCH --mail-type=FAIL\n")
        file.write("#SBATCH --mail-type=end \n")
        file.write('#SBATCH --constraint="volta"\n')
        report_info = "%s job failed; \t" % id
        report_info += "log path: %s; \t" % output_path
        report_info += "error record path: %s\t" % error_path
        report_info += "command line path: %s\t" % batch_file
        file.write('#SBATCH --comment="%s"\n' % (report_info))
        with open(dependency_handler_path, 'r') as rfile:
            line = rfile.readline()
            while line:
                file.write(line)
                line = rfile.readline()

        file.write("export GLOO_SOCKET_IFNAME=\nexport NCCL_SOCKET_IFNAME=\n")
        file.write("module load cuda/10.2 cudnn/v7.6.5.32-cuda.10.2 gcc/7.3.0\n")
        #file.write("bash /private/home/wang3702/.bashrc\n")
        #file.write("module load anaconda3\n")
        file.write("/private/home/wang3702/anaconda3/bin/conda init\n")
        file.write("CONDA_BASE=$(conda info --base) ; source $CONDA_BASE/etc/profile.d/conda.sh\n")
        file.write("conda activate pytorch2\n")
        file.write("master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}\n")
        file.write('dist_url="tcp://"\n')
        file.write("dist_url+=$master_node\n")
        file.write("dist_url+=:40000\n")
        file.write("export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}\n")
        file.write("export MASTER_PORT=29500\n")
        file.write("srun --label "+command_line + " --slurm=1 --dist_url=$dist_url &\n")
        file.write("wait $!\n")
        file.write("set +x \n")
        file.write("echo ..::Job Finished, but No, AGI is to BE Solved::.. \n")
        # signal that job is finished
    os.system('sbatch ' + batch_file)
def find_checkpoint(current_dir,checkpoint_name):
    if not os.path.isdir(current_dir):
        return None
    listfiles = os.listdir(current_dir)
    for item in listfiles:
        sub_dir = os.path.join(current_dir,item)
        if item==checkpoint_name:
            return sub_dir
        elif os.path.isdir(sub_dir):
            search_result = find_checkpoint(sub_dir,checkpoint_name)
            if search_result is not None:
                return search_result
    return None

def write_slurm_sh_multi(id,command_line, queue_name="learnfair",nodes=1,
                   gpu_per_node=8,wall_time=3*24*60,username="wang3702",
                         CPU_PER_GPU=8,gpu_memory=False,environment=0):
    import time
    import datetime
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    now = time.strftime("%H:%M:%S")
    dependency_handler_path = os.path.join(os.getcwd(), "ops")
    dependency_handler_path = os.path.join(dependency_handler_path, "handler.txt")
    run_path = os.path.join(os.getcwd(), "log")
    mkdir(run_path)
    run_path = os.path.abspath(run_path)
    prefix = "node%d_gpu%d"%(nodes,gpu_per_node)
    batch_file = os.path.join(run_path, prefix+"slurm_job_" + str(id) + ".sh")
    output_path = os.path.join(run_path, prefix+"output_" + str(id) + "_" + str(formatted_today + now) + ".log")
    error_path = os.path.join(run_path, prefix+"error_" + str(id) + "_" + str(formatted_today + now) + ".log")
    with open(batch_file, "w") as file:
        file.write("#! /bin/bash\n")#!/bin/bash
        file.write("#SBATCH --job-name=%s\n" % id)
        file.write("#SBATCH --output=%s\n" % output_path)
        file.write("#SBATCH --error=%s\n" % error_path)
        file.write("#SBATCH --partition=%s\n"%queue_name)
        file.write("#SBATCH --signal=USR1@600\n")
        file.write("#SBATCH --nodes=%d\n" % nodes)
        file.write("#SBATCH --ntasks-per-node=%d\n" % 1)
        file.write("#SBATCH --mem=%dG\n"%(350/8*gpu_per_node))#--mem : Specify the real memory required per node.
        file.write("#SBATCH --gpus=%d\n" % (nodes * gpu_per_node))
        file.write("#SBATCH --gpus-per-node=%d\n" % (gpu_per_node))
        file.write("#SBATCH --cpus-per-task=%d\n"%(CPU_PER_GPU*gpu_per_node))
        file.write("#SBATCH --time=%d\n"%wall_time)
        file.write("#SBATCH --mail-user=%s@fb.com\n"%username)
        file.write("#SBATCH --mail-type=FAIL\n")
        file.write("#SBATCH --mail-type=end \n")
        if gpu_memory is False:
            file.write('#SBATCH --constraint="volta"\n')
        else:
            file.write('#SBATCH --constraint="volta32gb"\n')
        #file.write('#SBATCH --constraint="volta"\n')
        report_info = "%s job failed; \t" % id
        report_info += "log path: %s; \t" % output_path
        report_info += "error record path: %s\t" % error_path
        report_info += "command line path: %s\t" % batch_file
        file.write('#SBATCH --comment="%s"\n' % (report_info))
        with open(dependency_handler_path, 'r') as rfile:
            line = rfile.readline()
            while line:
                file.write(line)
                line = rfile.readline()

        file.write("export GLOO_SOCKET_IFNAME=\nexport NCCL_SOCKET_IFNAME=\n")
        file.write("module load cuda/10.2 cudnn/v7.6.5.32-cuda.10.2 gcc/7.3.0\n")
        #file.write("bash /private/home/wang3702/.bashrc\n")
        file.write("/private/home/wang3702/anaconda3/bin/conda init\n")
        #file.write("module load anaconda3\n")
        file.write("CONDA_BASE=$(conda info --base) ; source $CONDA_BASE/etc/profile.d/conda.sh\n")
        if environment==0:
            file.write("conda activate pytorch2\n")
        else:
            file.write("conda activate pytorch\n")
        file.write("master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}\n")
        file.write('dist_url="tcp://"\n')
        file.write("dist_url+=$master_node\n")
        file.write("dist_url+=:40000\n")
        file.write("export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}\n")
        file.write("export MASTER_PORT=29500\n")
        file.write("srun --label "+command_line + " --slurm=1 --dist_url=$dist_url &\n")
        file.write("wait $!\n")
        file.write("set +x \n")
        file.write("echo ..::Job Finished, but No, AGI is to BE Solved::.. \n")
        # signal that job is finished
    os.system('sbatch ' + batch_file)

def write_slurm_sh_multi2(id,command_line, queue_name="learnfair",nodes=1,
                   gpu_per_node=8,wall_time=3*24*60,username="wang3702",CPU_PER_GPU=8,
                          gpu_memory=False,environment=0):
    import time
    import datetime
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    now = time.strftime("%H:%M:%S")
    dependency_handler_path = os.path.join(os.getcwd(), "ops")
    dependency_handler_path = os.path.join(dependency_handler_path, "handler.txt")
    run_path = os.path.join(os.getcwd(), "log")
    mkdir(run_path)
    run_path = os.path.abspath(run_path)
    prefix = "node%d_gpu%d"%(nodes,gpu_per_node)
    batch_file = os.path.join(run_path, prefix+"slurm_job_" + str(id) + ".sh")
    output_path = os.path.join(run_path, prefix+"output_" + str(id) + "_" + str(formatted_today + now) + ".log")
    error_path = os.path.join(run_path, prefix+"error_" + str(id) + "_" + str(formatted_today + now) + ".log")
    with open(batch_file, "w") as file:
        file.write("#! /bin/bash\n")#!/bin/bash
        file.write("#SBATCH --job-name=%s\n" % id)
        file.write("#SBATCH --output=%s\n" % output_path)
        file.write("#SBATCH --error=%s\n" % error_path)
        file.write("#SBATCH --partition=%s\n"%queue_name)
        file.write("#SBATCH --signal=USR1@600\n")
        file.write("#SBATCH --nodes=%d\n" % nodes)
        file.write("#SBATCH --ntasks-per-node=%d\n" % 1)
        file.write("#SBATCH --mem=%dG\n"%(350/8*gpu_per_node))
        file.write("#SBATCH --gpus=%d\n" % (nodes * gpu_per_node))
        file.write("#SBATCH --gpus-per-node=%d\n" % (gpu_per_node))
        file.write("#SBATCH --cpus-per-task=%d\n"%(CPU_PER_GPU*gpu_per_node))
        file.write("#SBATCH --time=%d\n"%wall_time)
        file.write("#SBATCH --mail-user=%s@fb.com\n"%username)
        file.write("#SBATCH --mail-type=FAIL\n")
        file.write("#SBATCH --mail-type=end \n")
        if gpu_memory is False:
            file.write('#SBATCH --constraint="volta"\n')
        else:
            file.write('#SBATCH --constraint="volta32gb"\n')
        report_info = "%s job failed; \t" % id
        report_info += "log path: %s; \t" % output_path
        report_info += "error record path: %s\t" % error_path
        report_info += "command line path: %s\t" % batch_file
        file.write('#SBATCH --comment="%s"\n' % (report_info))
        with open(dependency_handler_path, 'r') as rfile:
            line = rfile.readline()
            while line:
                file.write(line)
                line = rfile.readline()

        file.write("export GLOO_SOCKET_IFNAME=\nexport NCCL_SOCKET_IFNAME=\n")
        file.write("module load cuda/10.2 cudnn/v7.6.5.32-cuda.10.2 gcc/7.3.0\n")
        #file.write("bash /private/home/wang3702/.bashrc\n")
        # file.write("/private/home/wang3702/anaconda3/bin/conda init\n")
        file.write("/private/home/wang3702/anaconda3/bin/conda init\n")
        #file.write("module load anaconda3\n")
        file.write("CONDA_BASE=$(conda info --base) ; source $CONDA_BASE/etc/profile.d/conda.sh\n")
        if environment==0:
            file.write("conda activate pytorch2\n")
        else:
            file.write("conda activate pytorch\n")
        #file.write("source activate\n")
        file.write("master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:3}\n")
        file.write('dist_url="tcp://"\n')
        file.write("dist_url+=$master_node\n")
        file.write("dist_url+=:40000\n")
        file.write("export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:3}\n")
        file.write("export MASTER_PORT=29500\n")
        file.write("srun --label "+command_line + " &\n")
        file.write("wait $!\n")
        file.write("set +x \n")
        file.write("echo ..::Job Finished, but No, AGI is to BE Solved::.. \n")
        # signal that job is finished
    os.system('sbatch ' + batch_file)


def write_slurm_sh_faster(id,command_line, queue_name="learnfair",nodes=1,
                   gpu_per_node=8,wall_time=3*24*60,username="wang3702",CPU_PER_GPU=8,
                          gpu_memory=False,environment=0):
    import time
    import datetime
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    now = time.strftime("%H:%M:%S")
    dependency_handler_path = os.path.join(os.getcwd(), "ops")
    dependency_handler_path = os.path.join(dependency_handler_path, "handler.txt")
    run_path = os.path.join(os.getcwd(), "log")
    mkdir(run_path)
    run_path = os.path.abspath(run_path)
    batch_file = os.path.join(run_path, "slurm_job_" + str(id) + ".sh")
    output_path = os.path.join(run_path, "output_" + str(id) + "_" + str(formatted_today + now) + ".log")
    error_path = os.path.join(run_path, "error_" + str(id) + "_" + str(formatted_today + now) + ".log")
    with open(batch_file, "w") as file:
        file.write("#!/bin/bash\n")#!/bin/bash
        file.write("#SBATCH --job-name=%s\n" % id)
        file.write("#SBATCH --output=%s\n" % output_path)
        file.write("#SBATCH --error=%s\n" % error_path)
        file.write("#SBATCH --partition=%s\n"%queue_name)
        file.write("#SBATCH --signal=USR1@600\n")
        file.write("#SBATCH --nodes=%d\n" % nodes)
        file.write("#SBATCH --ntasks-per-node=%d\n" % gpu_per_node)
        file.write("#SBATCH --mem=%dG\n"%(int(350/8*gpu_per_node)))
        file.write("#SBATCH --gpus=%d\n" % (nodes * gpu_per_node))
        file.write("#SBATCH --cpus-per-task=%d\n"%(CPU_PER_GPU))
        file.write("#SBATCH --time=%d\n"%wall_time)
        file.write("#SBATCH --mail-user=%s@fb.com\n"%username)
        file.write("#SBATCH --mail-type=FAIL\n")
        file.write("#SBATCH --mail-type=end \n")
        if gpu_memory:
            file.write('#SBATCH --constraint="volta32gb"\n')
        else:
            file.write('#SBATCH --constraint="volta"\n')
        report_info = "%s job failed; \t" % id
        report_info += "log path: %s; \t" % output_path
        report_info += "error record path: %s\t" % error_path
        report_info += "command line path: %s\t" % batch_file
        file.write('#SBATCH --comment="%s"\n' % (report_info))
        with open(dependency_handler_path, 'r') as rfile:
            line = rfile.readline()
            while line:
                file.write(line)
                line = rfile.readline()
        file.write("module load cuda/10.2 cudnn/v7.6.5.32-cuda.10.2 gcc/7.3.0\n")
        #file.write("bash /private/home/wang3702/.bashrc\n")
        file.write("/private/home/wang3702/anaconda3/bin/conda init\n")
        #file.write("module load anaconda3\n")
        file.write("CONDA_BASE=$(conda info --base) ; source $CONDA_BASE/etc/profile.d/conda.sh\n")
        if environment==0:
            file.write("conda activate pytorch2\n")
        else:
            file.write("conda activate pytorch\n")
        #file.write("source activate\n")
        file.write(command_line + " &\n")
        file.write("wait $!\n")
        file.write("set +x \n")
        file.write("echo ..::Job Finished, but No, AGI is to BE Solved::.. \n")
        # signal that job is finished
    os.system('sbatch ' + batch_file)
def write_slurm_sh(id,command_line, queue_name="learnfair",nodes=1,
                   gpu_per_node=8,wall_time=3*24*60,username="wang3702",CPU_PER_GPU=10):
    """
    Args:
        id: running id
        command_line: command line
        outlog_path: saving path
    Returns:

    """
    import time
    import datetime
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    now = time.strftime("%H:%M:%S")
    dependency_handler_path = os.path.join(os.getcwd(),"ops")
    dependency_handler_path = os.path.join(dependency_handler_path,"handler.txt")
    run_path = os.path.join(os.getcwd(),"log")
    mkdir(run_path)
    run_path = os.path.abspath(run_path)
    batch_file = os.path.join(run_path,"slurm_job_"+str(id)+".sh")
    output_path = os.path.join(run_path,"output_"+str(id)+"_"+str(formatted_today+now)+".log")
    error_path = os.path.join(run_path,"error_"+str(id)+"_"+str(formatted_today+now)+".log")
    with open(batch_file,"w") as file:
        file.write("#!/bin/sh\n")
        file.write("#SBATCH --job-name=%s\n"%id)
        file.write("#SBATCH --output=%s\n"%output_path)
        file.write("#SBATCH --error=%s\n"%error_path)
        file.write("#SBATCH --partition=%s\n"%queue_name)
        file.write("#SBATCH --signal=USR1@600\n")
        file.write("#SBATCH --nodes=%d\n"%nodes )
        file.write("#SBATCH --ntasks-per-node=1\n")
        file.write("#SBATCH --mem=350G\n")
        file.write("#SBATCH --gpus=%d\n"%(nodes*gpu_per_node))
        file.write("#SBATCH --gpus-per-node=%d\n" % (gpu_per_node))
        file.write("#SBATCH --cpus-per-task=%d\n"%(CPU_PER_GPU*gpu_per_node))
        file.write("#SBATCH --time=%d\n"%wall_time)
        file.write("#SBATCH --mail-user=%s@fb.com\n"%username)
        file.write("#SBATCH --mail-type=FAIL\n")
        file.write("#SBATCH --mail-type=end \n")
        file.write('#SBATCH --constraint="volta"\n')
        report_info ="%s job failed; \t"%id
        report_info += "log path: %s; \t"%output_path
        report_info += "error record path: %s\t"%error_path
        report_info += "command line path: %s\t"%batch_file
        file.write('#SBATCH --comment="%s"\n'%(report_info))
        with open(dependency_handler_path,'r') as rfile:
            line = rfile.readline()
            while line:
                file.write(line)
                line = rfile.readline()

        #file.write("bash /private/home/wang3702/.bashrc\n")
       # file.write("/private/home/wang3702/anaconda3/bin/conda init\n")
        #file.write("CONDA_BASE=$(conda info --base) ; source $CONDA_BASE/etc/profile.d/conda.sh\n")

        #file.write("module load anaconda3\n")
        #file.write("conda activate pytorch2\n")
        file.write("module load cuda/10.2 cudnn/v7.6.5.32-cuda.10.2 gcc/7.3.0\n")
        file.write("/private/home/wang3702/anaconda3/bin/conda init\n")
        file.write("CONDA_BASE=$(conda info --base) ; source $CONDA_BASE/etc/profile.d/conda.sh\n")
        file.write("conda activate pytorch2\n")
        file.write(command_line+" &\n")
        file.write("wait $!\n")
        file.write("set +x \n")
        file.write("echo ..::Job Finished, but No, AGI is to BE Solved::.. \n")
        # signal that job is finished
    os.system('sbatch ' + batch_file)
parser = argparse.ArgumentParser(description='slurm job submission')
parser.add_argument('--data', default="imagenet", type=str, metavar='DIR',
                        help='path to dataset')
parser.add_argument("--mode",type=int,default=0,help="control mode for training")
parser.add_argument("--type",type=int,default=0,help="running type control")
parser.add_argument("--roi",type=int,default = 20, help="number of rois sampled here")
parser.add_argument("--queue",type=int,default=0, help="queue specified list")
parser.add_argument("-F",type=str, default=None, help="resume path for running again")
parser.add_argument("--comment", type=str,default=None,help="adding comment for script names")
parser.add_argument("--node",type=int,default=1,help="nodes needed for training")
parser.add_argument("--gpu",type=int,default=8,help="number of gpus per node")
args = parser.parse_args()
if args.queue ==0:
    queue_name = "learnfair"
elif args.queue ==1:
    queue_name = "dev"
elif args.queue ==2:
    queue_name = "scavenge"
elif args.queue ==3:
    queue_name = 'priority'
elif args.queue ==4:
    queue_name = 'learnlab'
elif args.queue==5:
    queue_name = 'devlab'
elif args.queue==6:
    queue_name = 'prioritylab'
dump_path= os.path.join(os.getcwd(),"swav_dump_100")
from ops.os_operation import mkdir
mkdir(dump_path)
import time
import datetime
today = datetime.date.today()
formatted_today = today.strftime('%y%m%d')
now = time.strftime("%H:%M:%S")
dump_path = os.path.join(dump_path, formatted_today + now)
if args.mode==1:
    if args.type==0:
        # command_line = "python3 main_adco.py --mode=1 --lr=0.06 --data=%s " \
        #                "--dist_url=tcp://localhost:10031 --epochs=100 " \
        #                "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0006 " \
        #                "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
        #                "--rank=0 --multiprocessing_distributed=1 --moco_dim=128 " \
        #                "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
        #                "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=57" % args.data
        # write_slurm_sh("baseline_sym_moco_lr0.06_proj", command_line, queue_name)
        command_line = "python3 main_adco.py --mode=1 --lr=0.06 --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=100 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0006 " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=16 --mask_size=32 " \
                       "--num_roi=1 " % args.data
        write_slurm_sh("baseline_sym_moco_lr0.06", command_line, queue_name)
        # command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
        #                "--dist_url=tcp://localhost:10031 --epochs=100 " \
        #                "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
        #                "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
        #                "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
        #                "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
        #                "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=16 --mask_size=32 " \
        #                "--num_roi=1 --img_size=96 " % args.data
        # write_slurm_sh("baseline_sym_moco_input96", command_line, queue_name)
        #running all the baseline with 100 epochs
        #base line moco
        # command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
        #                "--dist_url=tcp://localhost:10031 --epochs=100 " \
        #                "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
        #                "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
        #                "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
        #                "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
        #                "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=35 --mask_size=32 " \
        #                " --num_roi=1 " % args.data
        # write_slurm_sh("baseline_sym_mocobn_100", command_line, queue_name)
        # #moco multi baseline
        # command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
        #                "--dist_url=tcp://localhost:10031 --epochs=100 " \
        #                "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
        #                "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
        #                "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
        #                "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
        #                "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=18 --nmb_crops 2 6 " \
        #                "--size_crops 224 96 --min_scale_crops 0.14 0.05 --max_scale_crops 1.0 0.14 " % (args.data)
        # write_slurm_sh("multi_moco_baseline_100_new", command_line, queue_name)
        # # #moco multi sym baseline
        # command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
        #                "--dist_url=tcp://localhost:10031 --epochs=100 " \
        #                "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
        #                "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
        #                "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
        #                "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
        #                "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=20 --nmb_crops 4 " \
        #                "--size_crops 224  --min_scale_crops 0.14  --max_scale_crops 1.0  " % (args.data)
        # write_slurm_sh("2key_multi_moco_baseline_4_224", command_line, queue_name)
        # #swav multi baseline
        # command_line = "python3 main_adco.py --mode=5 --type=0 --data=%s --epochs 100 --lr=0.6 " \
        #                "--lr_final 0.0006 --batch_size=256 --warmup_epochs 0 --freeze_prototypes_niters 5005 " \
        #                "--queue_length 3840 --epoch_queue_starts 15 --dist_url=tcp://localhost:10031 " \
        #                "--knn_batch_size=256 --cos=1 --momentum=0.9 --weight_decay=1e-6 --world_size=1 " \
        #                "--rank=0 --multiprocessing_distributed=1 --moco_dim=128 --moco_k=3000 --moco_t=0.1 " \
        #                "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --nmb_crops 2 " \
        #                "--size_crops 224 --min_scale_crops 0.14  --max_scale_crops 1.0  --dump_path %s " % (args.data,dump_path)
        # write_slurm_sh("swav_baseline_100_only224", command_line, queue_name)

        # command_line = "python3 main_adco.py --mode=5 --type=0 --data=%s --epochs 100 --lr=0.6 " \
        #                "--lr_final 0.0006 --batch_size=256 --warmup_epochs 0 --freeze_prototypes_niters 5005 " \
        #                "--queue_length 3840 --epoch_queue_starts 15 --dist_url=tcp://localhost:10031 " \
        #                "--knn_batch_size=256 --cos=1 --momentum=0.9 --weight_decay=1e-6 --world_size=1 " \
        #                "--rank=0 --multiprocessing_distributed=1 --moco_dim=128 --moco_k=3000 --moco_t=0.1 " \
        #                "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --nmb_crops 2 6 " \
        #                "--size_crops 224 96 --min_scale_crops 0.14 0.05 --max_scale_crops 1.0 0.14 --dump_path %s " % (
        #                args.data, dump_path)
        # write_slurm_sh("swav_baseline_100", command_line, queue_name)




    elif args.type==10:
        #half dropout results
        command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=100 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=10 " % args.data
        if args.F is not None:
            resume_name = os.path.split(os.path.abspath(args.F))[1]
            command_line += "--resume=%s"%args.F
            write_slurm_sh("halfdropoutnew_resume%s"%resume_name, command_line, queue_name)
        else:
            write_slurm_sh("halfdropoutnew", command_line, queue_name)


    elif args.type==11:
        # to make sure overlap region can really not work
        for mask_size in [96, 160]:
            command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=200 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=11 --shift_ratio=0 " \
                           " --mask_size=%d " % (args.data,mask_size)
            write_slurm_sh("type11_roimatch_%s"%mask_size, command_line, queue_name)
    elif args.type==13:
        for mask_size in [96,160]:
            command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                   "--dist_url=tcp://localhost:10031 --epochs=200 " \
                   "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                   "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=13 " \
                           "--mask_size=%d "%(args.data,mask_size)
            write_slurm_sh("type13_singleroi_vs_global_%d"%mask_size,command_line,queue_name)
            time.sleep(1)

    elif args.type==14:
        #roi vs global
        for mask_size in [96,160]:
            command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                   "--dist_url=tcp://localhost:10031 --epochs=200 " \
                   "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                   "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=14 " \
                           "--mask_size=%d "%(args.data,mask_size)
            write_slurm_sh("type14_singleroi_vs_global_%d"%mask_size,command_line,queue_name)
    elif args.type==16:
        for mask_size in [96,128,160]:
            command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                   "--dist_url=tcp://localhost:10031 --epochs=200 " \
                   "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                   "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=16 " \
                           "--mask_size=%d --num_roi=10 "%(args.data,mask_size)
            write_slurm_sh("type16_roi+global_vs_global_%d"%mask_size,command_line,queue_name)
    elif args.type==-16:
        command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=200 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=16 --mask_size=32 --num_roi=1 " % args.data
        if args.F is not None:
            resume_name = os.path.split(os.path.abspath(args.F))[1]
            command_line += " --resume=%s"%args.F
            write_slurm_sh("baseline_sym_moco_resume%s"%resume_name, command_line, queue_name)
        else:
            write_slurm_sh("baseline_sym_moco", command_line,queue_name)
    elif args.type==17:
        command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=200 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=17 --mask_size=32" \
                       " --num_roi=%d" % (args.data,args.roi)

        write_slurm_sh("type17_randroi_%d"%args.roi, command_line,queue_name)
    elif args.type==-17:
        #roi vs roi,with global as negative
        for roi in [10,20,50,100]:
            for mask_size in [32, 96, 160, 196]:
                command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                               "--dist_url=tcp://localhost:10031 --epochs=200 " \
                               "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                               "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=17 --mask_size=%d" \
                               " --num_roi=%d" % (args.data,mask_size, roi)
                write_slurm_sh("type17_randroi_%d_masksize_%d" % (roi,mask_size), command_line,queue_name)
    elif args.type==18:
        command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=200 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=18 --nmb_crops 2 6 " \
                       "--size_crops 224 96 --min_scale_crops 0.14 0.05 --max_scale_crops 1.0 0.14 "% (args.data)
        if args.F is not None:
            resume_name = os.path.split(os.path.abspath(args.F))[1]
            command_line += "--resume=%s"%args.F
            write_slurm_sh("multi_moco_baseline_resume%s"%resume_name, command_line, queue_name)
        else:
            write_slurm_sh("multi_moco_baseline" , command_line, queue_name)
    elif args.type==19:
        for roi in [20]:
            for mask_size in [32,160]:
                command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                               "--dist_url=tcp://localhost:10031 --epochs=200 " \
                               "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                               "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=19 --mask_size=%d" \
                               " --num_roi=%d" % (args.data,mask_size, roi)
                write_slurm_sh("type19_randroi_%d_masksize_%d" % (roi,mask_size), command_line,queue_name)


    elif args.type==20:
        command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=200 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=20 --nmb_crops 2 6 " \
                       "--size_crops 224 96 --min_scale_crops 0.14 0.05 --max_scale_crops 1.0 0.14 "% (args.data)
        if args.F is not None:
            resume_name = os.path.split(os.path.abspath(args.F))[1]
            command_line += " --resume=%s"%args.F
            write_slurm_sh("2key_multi_moco_baseline_correct_resume%s"%resume_name, command_line, queue_name)
        else:
            write_slurm_sh("2key_multi_moco_baseline_correct", command_line, queue_name)

    elif args.type==21:
        for roi in [20]:
            for mask_size in [96]:
                command_line = "python3 main_adco.py --mode=1 --lr=0.09 --data=%s " \
                               "--dist_url=tcp://localhost:10031 --epochs=200 " \
                               "--batch_size=768 --knn_batch_size=256 --cos=1 --lr_final=0.0009 " \
                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                               "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=21 --mask_size=%d" \
                               " --num_roi=%d" % (args.data,mask_size, roi)
                write_slurm_sh("type21_randroi_%d_masksize_%d" % (roi,mask_size), command_line,queue_name)
    elif args.type==22:
        for roi in [50]:
            for mask_size in [96]:
                command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                               "--dist_url=tcp://localhost:10031 --epochs=200 " \
                               "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                               "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=22 --mask_size=%d" \
                               " --num_roi=%d" % (args.data, mask_size, roi)
                write_slurm_sh("type22_randroi_%d_masksize_%d" % (roi,mask_size), command_line,queue_name)
    elif args.type==23:
        command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=200 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=23 --nmb_crops 2 2 2 2 2 2 2 2"  \
                       " --size_crops 96 112 128 144 160 176 192 208 " % args.data
        write_slurm_sh("type23_specifyroi", command_line, queue_name)
    elif args.type==-23:
        # command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
        #                "--dist_url=tcp://localhost:10031 --epochs=200 " \
        #                "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
        #                "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
        #                "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
        #                "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
        #                "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=23 --nmb_crops 6"  \
        #                " --size_crops 96 " % args.data
        # write_slurm_sh("type23_specifyroi_6_96", command_line, queue_name)
        min_scale = 64
        max_scale = 224
        divide_list = [2,4,8,16,32]
        pick_times = [1,2,3]
        for pick_time in pick_times:
            for divide in divide_list:
                check_list = ""
                num_list = ""
                current_scale = min_scale
                while current_scale<max_scale:
                    check_list+=str(current_scale)+" "
                    num_list+=str(pick_time)+" "
                    current_scale+=divide
                print(check_list)
                command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                               "--dist_url=tcp://localhost:10031 --epochs=200 " \
                               "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                               "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=23 --nmb_crops %s " \
                               " --size_crops %s " % (args.data,num_list,check_list)
                write_slurm_sh("type23_specifyroi_%d_%d"%(pick_time,divide), command_line, queue_name)
    elif args.type==24:
        for alpha in [0.5, 1.0, 2.0]:
            for local_t in [0.1,0.2,0.3]:
                command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                           "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=24 --nmb_crops 1 6" \
                           " --size_crops 224 96 --local_t=%.4f --alpha=1.0 " % (args.data,local_t)
                write_slurm_sh("type24_lg_t_%.3f_alpha_%.2f"%(local_t,alpha), command_line, queue_name)
    elif args.type==25:
        for alpha in [0.5]:
            for local_t in [0.2]:
                command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                               "--dist_url=tcp://localhost:10031 --epochs=100 " \
                               "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                               "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=24 --nmb_crops 1 6" \
                               " --size_crops 224 96 --local_t=%.4f --alpha=%.2f " % (args.data, local_t,alpha)
                write_slurm_sh("type25_lgq_t_%.3f_alpha_%.2f" %(local_t,alpha), command_line, queue_name)
    elif args.type==26:
        for alpha in [0.5,1.0]:
            for local_t in [0.2]:
                command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                               "--dist_url=tcp://localhost:10031 --epochs=100 " \
                               "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                               "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=26 --nmb_crops 1 6" \
                               " --size_crops 224 96 --local_t=%.4f --alpha=%.2f " % (args.data, local_t,alpha)
                write_slurm_sh("type26_lgq_t_%.3f_alpha_%.2f" %(local_t,alpha), command_line, queue_name)
    elif args.type == 27:
        min_scale = 96
        max_scale = 224
        divide_list = [16]
        pick_times = [1]
        for learning_rate in [0.05]:#[0.02,0.03,0.04,0.05,0.06,0.1,0.15]:
            for pick_time in pick_times:
                for divide in divide_list:
                    check_list = ""
                    num_list = ""
                    current_scale = min_scale
                    while current_scale < max_scale:
                        check_list += str(current_scale) + " "
                        num_list += str(pick_time) + " "
                        current_scale += divide
                    print(check_list)
                    print(num_list)
                    for alpha in [0.1,0.15,0.2,0.3]:#[0.3, 0.5, 1.0]:
                        for local_t in [0.12,0.15,0.18]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=27 --nmb_crops 1 %s" \
                                           " --size_crops 224 %s --local_t=%.4f --alpha=%.2f " % \
                                           (learning_rate,args.data, local_t,num_list, check_list, local_t, alpha)
                            write_slurm_sh("type27_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (local_t, alpha, pick_time, divide,learning_rate),
                                           command_line, queue_name)
                            time.sleep(1)
    elif args.type == -270:
        for num_roi in [6,10,20,30]:
            for crop_size in [64, 96, 128, 160, 192]:
                for learning_rate in [0.05]:
                    for alpha in [0.1]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.18]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=27 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f " % \
                                           (learning_rate, args.data,local_t, num_roi,crop_size, local_t, alpha)
                            write_slurm_sh(
                                "type27crop_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (local_t, alpha, num_roi,crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==-271:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.05]:
                    for alpha in [0.1]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.18,0.2]:
                            for moco_dim in [256,512]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                               "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=%d  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=27 --nmb_crops 1 %d" \
                                               " --size_crops 224 %d --local_t=%.4f --alpha=%.2f " % \
                                               (learning_rate, args.data,moco_dim, local_t, num_roi, crop_size, local_t, alpha)
                                write_slurm_sh(
                                    "type27dim_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f_dim%d" % (
                                    local_t, alpha, num_roi, crop_size, learning_rate,moco_dim),
                                    command_line, queue_name)
                                time.sleep(1)
    elif args.type == -27:
        #calculate baseline 6*96 for type 27 as a direct cmp with SWAV
        for learning_rate in [0.05]:
            for alpha in [0.1]:  # [0.3, 0.5, 1.0]:
                for local_t in [0.18]:
                    for moco_dim in [128,256,512]:
                        command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                   "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=27 --nmb_crops 1 6" \
                                   " --size_crops 224 96 --local_t=%.4f --alpha=%.2f " % \
                                   (learning_rate, args.data, local_t,local_t, alpha)
                        write_slurm_sh("type27baseline_lgq_t_%.3f_alpha_%.2f_6_96_lr%.4f" % (local_t, alpha,learning_rate),
                                   command_line, queue_name)
                        time.sleep(1)
    elif args.type ==  28:
        command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=100 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=28 --nmb_crops 2 6 " \
                       "--size_crops 224 96 --min_scale_crops 0.14 0.05 --max_scale_crops 1.0 0.14 " % (args.data)
        write_slurm_sh("type28_small_inside", command_line, queue_name)
    elif args.type==29:
        for learning_rate in [0.03]:
            for alpha in [0.5,1.0]:
                for local_t in [0.2]:
                    command_line = "python3 main_adco.py --mode=1 --lr=%.2f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                   "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=%.5f " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=29 --nmb_crops 1 6" \
                                   " --size_crops 224 96 --local_t=%.4f --alpha=%.2f " \
                                   "" % (learning_rate,args.data, learning_rate/100,local_t, alpha)
                    write_slurm_sh("type29_lgq_t_%.3f_alpha_%.2f_lr_%.4f" % (local_t, alpha,learning_rate), command_line, queue_name)
    elif args.type==30:
        for learning_rate in [0.03]:
            for alpha in [0.5,1.0]:
                for local_t in [0.2]:
                    command_line = "python3 main_adco.py --mode=1 --lr=%.2f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                   "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=%.5f " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=30 --nmb_crops 6 " \
                                   " --size_crops 96 --local_t=%.4f --alpha=%.2f " \
                                   "" % (learning_rate,args.data, learning_rate/100,local_t, alpha)
                    write_slurm_sh("type30_lgq_t_%.3f_alpha_%.2f_lr_%.4f" % (local_t, alpha,learning_rate), command_line, queue_name)
    elif args.type==31:
        for learning_rate in [0.03]:
            for alpha in [0.5]:
                for local_t in [0.2]:
                    for num_roi in [5, 10, 20]:
                        for mask_size in [96]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.2f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=%.5f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=0.2  " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=31 " \
                                           "--local_t=%.4f --alpha=%.2f --num_roi=%d --mask_size=%d " \
                                           "" % (learning_rate, args.data, learning_rate / 100,
                                                 local_t, alpha,num_roi,mask_size)
                            write_slurm_sh("type31_lgq_t_%.3f_alpha_%.2f_lr_%.4f_roi%d_mask%d" %
                                           (local_t, alpha, learning_rate,num_roi,mask_size),
                                           command_line, queue_name)
    elif args.type==32:
        for learning_rate in [0.03]:
            for alpha in [0.5]:
                for local_t in [0.2]:
                    for num_roi in [5, 10, 20]:
                        for mask_size in [96]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.2f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=%.5f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=0.2  " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=32 " \
                                           "--local_t=%.4f --alpha=%.2f --num_roi=%d --mask_size=%d " \
                                           "" % (learning_rate, args.data, learning_rate / 100,
                                                 local_t, alpha,num_roi,mask_size)
                            write_slurm_sh("type32_lgq_t_%.3f_alpha_%.2f_lr_%.4f_roi%d_mask%d" %
                                           (local_t, alpha, learning_rate,num_roi,mask_size),
                                           command_line, queue_name)
    elif args.type==33:
        for learning_rate in [0.03,0.04,0.05,0.06,0.09,0.12]:
            for alpha in [0.5,1.0,2.0,5.0]:
                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=100 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=%.4f " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=33 --nmb_crops 2 6 " \
                       "--size_crops 224 96 --min_scale_crops 0.14 0.05 --max_scale_crops 1.0 0.14 " \
                        "--alpha=%.4f " \
                               " " % (learning_rate,args.data,learning_rate/100,alpha)
                write_slurm_sh("multimoco_alpha_%.2f_lr_%.4f"%(alpha,learning_rate), command_line, queue_name)
    elif args.type==-28:
        for learning_rate in [0.06]:
            for alpha in [1.0]:
                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=100 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=%.4f " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=28 --nmb_crops 2 6 " \
                       "--size_crops 224 96 --min_scale_crops 0.14 0.05 --max_scale_crops 1.0 0.14 " \
                        "--alpha=%.4f " \
                               " " % (learning_rate,args.data,learning_rate/100,alpha)
                write_slurm_sh("multimocoinside_alpha_%.2f_lr_%.4f"%(alpha,learning_rate), command_line, queue_name)
    elif args.type==34:
        min_scale = 96
        max_scale = 224
        divide_list = [16]
        pick_times = [1]
        for learning_rate in [0.04, 0.05]:
            for pick_time in pick_times:
                for divide in divide_list:
                    check_list = ""
                    num_list = ""
                    current_scale = min_scale
                    while current_scale < max_scale:
                        check_list += str(current_scale) + " "
                        num_list += str(pick_time) + " "
                        current_scale += divide
                    print(check_list)
                    print(num_list)
                    for alpha in [0.1, 0.3, 0.5,1.0]:
                        for local_t in [0.2]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=34 --nmb_crops 1 %s" \
                                           " --size_crops 224 %s --local_t=%.4f --alpha=%.2f " % \
                                           (learning_rate, args.data, num_list, check_list, local_t, alpha)
                            write_slurm_sh("type34_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (
                            local_t, alpha, pick_time, divide, learning_rate),
                                           command_line, queue_name)
                            time.sleep(1)
    elif args.type == 36:
        min_scale = 96
        max_scale = 224
        divide_list = [16]
        pick_times = [1]
        for learning_rate in [0.04,0.05]:#[0.02,0.03,0.04,0.05,0.06,0.1,0.15]:
            for pick_time in pick_times:
                for divide in divide_list:
                    check_list = ""
                    num_list = ""
                    current_scale = min_scale
                    while current_scale < max_scale:
                        check_list += str(current_scale) + " "
                        num_list += str(pick_time) + " "
                        current_scale += divide
                    print(check_list)
                    print(num_list)
                    for alpha in [0.1]:#[0.3, 0.5, 1.0]:
                        for local_t in [0.2]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=36 --nmb_crops 1 %s" \
                                           " --size_crops 224 %s --local_t=%.4f --alpha=%.2f " % \
                                           (learning_rate,args.data, local_t,num_list, check_list, local_t, alpha)
                            write_slurm_sh("type36_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (local_t, alpha, pick_time, divide,learning_rate),
                                           command_line, queue_name)
                            time.sleep(1)
    elif args.type==37:
        for learning_rate in [0.03,0.04,0.05,0.06]:
            for alpha in [0.1,0.3,0.5,1.0]:  # [0.3, 0.5, 1.0]:
                for local_t in [0.2]:
                    command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                   "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=37 --nmb_crops 1 6" \
                                   " --size_crops 224 96 --local_t=%.4f --alpha=%.2f " % \
                                   (learning_rate, args.data, local_t,local_t, alpha)
                    write_slurm_sh("type37baseline_lgq_t_%.3f_alpha_%.2f_6_96_lr%.4f" % (local_t, alpha,learning_rate),
                                   command_line, queue_name)
                    time.sleep(1)
    elif args.type==38:
        min_scale = 96
        max_scale = 224
        divide_list = [16]
        pick_times = [1]
        for learning_rate in [0.05]:  # [0.02,0.03,0.04,0.05,0.06,0.1,0.15]:
            for pick_time in pick_times:
                for divide in divide_list:
                    check_list = ""
                    num_list = ""
                    current_scale = min_scale
                    while current_scale < max_scale:
                        check_list += str(current_scale) + " "
                        num_list += str(pick_time) + " "
                        current_scale += divide
                    print(check_list)
                    print(num_list)
                    for alpha in [0]: #[0.1, 0.3, 0.5, 1.0]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.2]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=38 --nmb_crops 1 %s" \
                                           " --size_crops 224 %s --local_t=%.4f --alpha=%.2f " % \
                                           (learning_rate, args.data, local_t,"", "", local_t, alpha)
                            write_slurm_sh("type38_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (
                            local_t, alpha, pick_time, divide, learning_rate),
                                           command_line, queue_name)
                            time.sleep(1)
    elif args.type==-38:
        for learning_rate in [0.05]:
            for alpha in [0.1,0.3,0.5,1.0]:  # [0.3, 0.5, 1.0]:
                for local_t in [0.2]:
                    command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                   "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=38 --nmb_crops 1 6" \
                                   " --size_crops 224 96 --local_t=%.4f --alpha=%.2f " % \
                                   (learning_rate, args.data, local_t,local_t, alpha)
                    write_slurm_sh("type38baseline_lgq_t_%.3f_alpha_%.2f_6_96_lr%.4f" % (local_t, alpha,learning_rate),
                                   command_line, queue_name)
                    time.sleep(1)
    elif args.type==39:
        for learning_rate in [0.05]:
            for alpha in [0.1]:  # [0.3, 0.5, 1.0]:
                for local_t in [0.2]:
                    command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                   "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=39 --nmb_crops 1 6" \
                                   " --size_crops 224 96 --local_t=%.4f --alpha=%.2f " % \
                                   (learning_rate, args.data, local_t,local_t, alpha)
                    write_slurm_sh("type39baseline_lgq_t_%.3f_alpha_%.2f_6_96_lr%.4f" % (local_t, alpha,learning_rate),
                                   command_line, queue_name)
                    time.sleep(1)

    elif args.type==40:
        for learning_rate in [0.05]:
            for alpha in [0.5]:  # [0.3, 0.5, 1.0]:
                for local_t in [0.2]:
                    command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                   "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=40 --nmb_crops 1 6" \
                                   " --size_crops 224 96 --local_t=%.4f --alpha=%.2f " % \
                                   (learning_rate, args.data, local_t,local_t, alpha)
                    write_slurm_sh("type40baseline_lgq_t_%.3f_alpha_%.2f_6_96_lr%.4f" % (local_t, alpha,learning_rate),
                                   command_line, queue_name)
                    time.sleep(1)
    elif args.type==41:
        for mask_size in [96]:
            command_line = "python3 main_adco.py --mode=1 --lr=0.03 --data=%s " \
                   "--dist_url=tcp://localhost:10031 --epochs=100 " \
                   "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                   "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=41 " \
                           "--mask_size=%d "%(args.data,mask_size)
            write_slurm_sh("type41_singleroi_vs_global_%d"%mask_size,command_line,queue_name)
    elif args.type==42:
        for learning_rate in [0.05]:
            for alpha in [0.1,0.5]:  # [0.3, 0.5, 1.0]:
                for local_t in [0.15,0.2]:
                    command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                   "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=42 --nmb_crops 1 6" \
                                   " --size_crops 224 96 --local_t=%.4f --alpha=%.2f " % \
                                   (learning_rate, args.data, local_t,local_t, alpha)
                    write_slurm_sh("type42baseline_lgq_t_%.3f_alpha_%.2f_6_96_lr%.4f" % (local_t, alpha,learning_rate),
                                   command_line, queue_name)
                    time.sleep(1)
    elif args.type==43:
        for learning_rate in [0.05]:
            for alpha in [0.1,0.5]:  # [0.3, 0.5, 1.0]:
                for local_t in [0.15,0.2]:
                    command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                   "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=43 --nmb_crops 1 6" \
                                   " --size_crops 224 96 --local_t=%.4f --alpha=%.2f " % \
                                   (learning_rate, args.data, local_t,local_t, alpha)
                    write_slurm_sh("type43baseline_lgq_t_%.3f_alpha_%.2f_6_96_lr%.4f" % (local_t, alpha,learning_rate),
                                   command_line, queue_name)
                    time.sleep(1)
    elif args.type == 44:
        # for num_roi in [6]:
        #     for crop_size in [96]:
        #         for learning_rate in [0.05]:
        #             for alpha in [0.1]:  # [0.3, 0.5, 1.0]:
        #                 for local_t in [0.15, 0.18, 0.2]:
        #                     for sample_ratio in [2,4]:
        #                         command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
        #                                    "--dist_url=tcp://localhost:10031 --epochs=100 " \
        #                                    "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
        #                                    "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
        #                                    "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
        #                                    "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
        #                                    "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=44 --nmb_crops 1 %d" \
        #                                    " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --sample_ratio=%d " % \
        #                                    (learning_rate, args.data,local_t, num_roi,crop_size, local_t, alpha,sample_ratio)
        #                         write_slurm_sh(
        #                         "type44crop_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f_ratio%d" % (local_t, alpha, num_roi,crop_size, learning_rate,sample_ratio),
        #                         command_line, queue_name)
        #                         time.sleep(1)
        for num_roi in [6]:
            for crop_size in [96,192]:
                for learning_rate in [0.03,0.05,0.06]:
                    for alpha in [0.1,0.3,0.5,1.0]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.20]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=44 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f " % \
                                           (learning_rate, args.data,local_t, num_roi,crop_size, local_t, alpha)
                            write_slurm_sh(
                                "type44_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (local_t, alpha, num_roi,crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==-44:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.05]:
                    for alpha in [0.1,0.5]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.20]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=44 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data,local_t, num_roi,crop_size, local_t, alpha)
                            write_slurm_sh(
                                "type44align_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (local_t, alpha, num_roi,crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==45 or args.type==46:
        for crop_size in [96]:
            for learning_rate in [0.03,0.04,0.05]:
                for alpha in [0.1,0.3,0.5,1,2]:
                    for local_t in [0.2]:
                        command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                       "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                       "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d --mask_size %d" \
                                       " --local_t=%.4f --alpha=%.2f  " % \
                                       (learning_rate, args.data, local_t, args.type, crop_size,local_t, alpha)
                        write_slurm_sh(
                            "type%d_crop_lgq_t_%.3f_alpha_%.2f_%d_lr%.4f" % (args.type, local_t,alpha,
                                                                             crop_size, learning_rate),
                            command_line, queue_name)
                        time.sleep(1)
    elif args.type ==47:
        min_scale = 96
        max_scale = 224
        divide_list = [16]
        pick_times = [1]
        for learning_rate in [0.03,0.05]:  # [0.02,0.03,0.04,0.05,0.06,0.1,0.15]:
            for pick_time in pick_times:
                for divide in divide_list:
                    check_list = ""
                    num_list = ""
                    current_scale = min_scale
                    while current_scale < max_scale:
                        check_list += str(current_scale) + " "
                        num_list += str(pick_time) + " "
                        current_scale += divide
                    print(check_list)
                    print(num_list)
                    for alpha in [0.1,0.5,1.0]:  # [0.1, 0.3, 0.5, 1.0]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.2]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=47 " \
                                           " --size_crops 224 %s --local_t=%.4f --alpha=%.2f " % \
                                           (learning_rate, args.data, local_t,  check_list, local_t, alpha)
                            write_slurm_sh("type47_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (
                                local_t, alpha, pick_time, divide, learning_rate),
                                           command_line, queue_name)
                            time.sleep(1)
    elif args.type ==49:
        min_scale = 96
        max_scale = 224
        divide_list = [2,4,8,16,32]
        pick_times = [1]
        for learning_rate in [0.06]:  # [0.02,0.03,0.04,0.05,0.06,0.1,0.15]:
            for pick_time in pick_times:
                for divide in divide_list:
                    check_list = ""
                    num_list = ""
                    current_scale = min_scale
                    while current_scale < max_scale:
                        check_list += str(current_scale) + " "
                        num_list += str(pick_time) + " "
                        current_scale += divide
                    print(check_list)
                    print(num_list)
                    for alpha in [0]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.2]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=49 --nmb_crops 1 %s" \
                                           " --size_crops 224 %s --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data, local_t, num_list,check_list, local_t, alpha)
                            write_slurm_sh_faster(
                                "type49crop_lgq_t_%.3f_alpha_%.2f_divide%d_lr%.4f" % (
                                local_t, alpha, divide, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)

    elif args.type==-49:
        #only run on pytorch environment, not base environment
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for alpha in [-0.1,-0.3,-0.5,-1]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.18]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=49 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data,local_t, num_roi,crop_size, local_t, alpha)
                            write_slurm_sh_faster(
                                "type49align_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (local_t, alpha, num_roi,crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==50:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.05]:
                    for alpha in [0, 0.1,0.5,1.0,2.0]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.20]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=50 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data,local_t, num_roi,crop_size, local_t, alpha)
                            write_slurm_sh_faster(
                                "type50align_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (local_t, alpha, num_roi,crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==51:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.05]:
                    for alpha in [0, 0.1,0.5,1.0]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.20]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=51 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data,local_t, num_roi,crop_size, local_t, alpha)
                            write_slurm_sh_faster(
                                "type51align_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (local_t, alpha, num_roi,crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==52:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for alpha in [0, 0.1,0.2,0.5,1.0]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.20]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=52 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data,local_t, num_roi,crop_size, local_t, alpha)
                            write_slurm_sh_faster(
                                "type52_1v1_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (local_t, alpha, num_roi,crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==53:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.05]:
                    for alpha in [0, 0.1,0.5,1.0]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.20]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=53 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data,local_t, num_roi,crop_size, local_t, alpha)
                            write_slurm_sh_faster(
                                "type53align_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (local_t, alpha, num_roi,crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==54:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.05]:
                    for alpha in [0, 0.1,0.5,1.0]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.15,0.18,0.20]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=54 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data,local_t, num_roi,crop_size, local_t, alpha)
                            write_slurm_sh_faster(
                                "type54align_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (local_t, alpha, num_roi,crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==55:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for alpha in [0.1]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.20]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=55 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data,local_t, num_roi,crop_size, local_t, alpha)
                            write_slurm_sh_faster(
                                "type55align_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (local_t, alpha, num_roi,crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==551:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for alpha in [0.1]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.20]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=55 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data,local_t, num_roi,crop_size, local_t, alpha)
                            write_slurm_sh_faster(
                                "type55align_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (local_t, alpha, num_roi,crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==550:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for alpha in [0.1]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.20]:
                            for pred_dim in [256,1024,2048]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=55 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 --pred_dim=%d " % \
                                           (learning_rate, args.data,local_t, num_roi,crop_size, local_t, alpha,pred_dim)
                                write_slurm_sh_faster(
                                "type55dim%d_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (pred_dim,local_t, alpha, num_roi,crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==56:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.05,0.06]:
                    for alpha in [0, 0.05,0.1,0.2]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.18, 0.20]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=56 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data,local_t, num_roi,crop_size, local_t, alpha)
                            write_slurm_sh_faster(
                                "type56align_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (local_t, alpha, num_roi,crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==58:
        for learning_rate in [0.06]:
            for alpha in [1.0]:
                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=100 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=%.4f " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=58 --nmb_crops 2 6 " \
                       "--size_crops 224 96 --min_scale_crops 0.14 0.05 --max_scale_crops 1.0 0.14 " \
                        "--alpha=%.4f " \
                               " " % (learning_rate,args.data,learning_rate/100,alpha)
                write_slurm_sh("multimoco_proj_alpha_%.2f_lr_%.4f"%(alpha,learning_rate), command_line, queue_name)
    elif args.type==59:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=59 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, num_roi, crop_size, local_t)
                            write_slurm_sh_faster(
                                "type59_lgq_t_%.3f_%d_%d_lr%.4f" % (local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==60:
        for num_roi in [3,6,10,15,20,25,30]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=60 --num_roi=%d " \
                                           " --mask_size=%d --local_t=%.4f --align=1 " % \
                                           (learning_rate, args.data, epoch, 256,
                                            256,learning_rate * args.node / 100,
                                            local_t, num_roi, crop_size, local_t)
                            write_slurm_sh_faster(
                                "type60_lgq_t_%.3f_%d_%d_lr%.4f" % (local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==61:
        #for num_roi in ['','6']:
        #    for crop_size in ['','96']:
        indicate_list=[['',''],['6','96']]
        for indication in indicate_list:
            num_roi = indication[0]
            crop_size= indication[1]
            for learning_rate in [0.06]:
                for local_t in [0.2]:
                    for epoch in [100]:
                        command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                       "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                       "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                       "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=61 --nmb_crops 1 %s" \
                                       " --size_crops 224 %s --local_t=%.4f --align=1 " % \
                                       (learning_rate, args.data, epoch, 256, 256,
                                        learning_rate / 100,
                                        local_t, num_roi, crop_size, local_t)
                        write_slurm_sh_faster(
                            "type61_lgq_t_%.3f_%s_%s_lr%.4f" % (local_t, num_roi, crop_size, learning_rate),
                            command_line, queue_name)
                        time.sleep(1)
    elif args.type==62:
        for learning_rate in [0.06]:
            for alpha in [0,1.0]:#0 denotes only shuffling to influence
                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=100 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=%.4f " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=62 --nmb_crops 2 6 " \
                       "--size_crops 224 96 --min_scale_crops 0.14 0.05 --max_scale_crops 1.0 0.14 " \
                        "--alpha=%.4f " \
                               " " % (learning_rate,args.data,learning_rate/100,alpha)
                write_slurm_sh("pixelembedshufflemoco_alpha_%.2f_lr_%.4f"%(alpha,learning_rate), command_line, queue_name)
    elif args.type==63:
        for learning_rate in [0.06]:
            for alpha in [0,1.0]:#0 denotes only shuffling to influence
                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=100 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=%.4f " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 --choose=0,1,2,3,4,5,6,7 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=63 --nmb_crops 2 6 " \
                       "--size_crops 224 96 --min_scale_crops 0.14 0.05 --max_scale_crops 1.0 0.14 " \
                        "--alpha=%.4f " \
                               " " % (learning_rate,args.data,learning_rate/100,alpha)
                write_slurm_sh("pixelGLsync_alpha_%.2f_lr_%.4f"%(alpha,learning_rate), command_line, queue_name)
    elif args.type == 64:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for alpha in [0,0.1,0.2,0.5, 1.0]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.20]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=64 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data, local_t, num_roi, crop_size, local_t, alpha)
                            write_slurm_sh_faster(
                                "type64align_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (
                                local_t, alpha, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type == 65:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for alpha in [0,0.1,0.2,0.5, 1.0]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.20]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=65 --nmb_crops 1 %d " \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data, local_t, num_roi, crop_size, local_t, alpha)
                            write_slurm_sh_faster(
                                "type65align_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (
                                local_t, alpha, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type == 66:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for alpha in [0, 0.1, 0.2, 0.5, 1.0]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.20]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=66 --nmb_crops 1 %d " \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data, local_t, num_roi, crop_size, local_t, alpha)
                            write_slurm_sh_faster(
                                "type66align_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (
                                    local_t, alpha, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type == 67:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06,0.08,0.09]:
                    for alpha in [0, 0.1, 0.2, 0.5]:  # [0.3, 0.5, 1.0]:
                        for local_t in [0.20]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=100 " \
                                           "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=0.0003 " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f --choose=0,1,2,3,4,5,6,7 " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=67 --nmb_crops 1 %d " \
                                           " --size_crops 224 %d --local_t=%.4f --alpha=%.2f --align=1 " % \
                                           (learning_rate, args.data, local_t, num_roi, crop_size, local_t, alpha)
                            write_slurm_sh_faster(
                                "type67align_lgq_t_%.3f_alpha_%.2f_%d_%d_lr%.4f" % (
                                    local_t, alpha, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==68:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=68 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, num_roi, crop_size, local_t)
                            write_slurm_sh_faster(
                                "type68_lgq_t_%.3f_%d_%d_lr%.4f" % (local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==69:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=69 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, num_roi, crop_size, local_t)
                            write_slurm_sh_faster(
                                "type69_lgq_t_%.3f_%d_%d_lr%.4f" % (local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==70:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=70 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, num_roi, crop_size, local_t)
                            write_slurm_sh_faster(
                                "type70_lgq_t_%.3f_%d_%d_lr%.4f" % (local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==71:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for alpha in [0,0.05,0.1,0.2]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=71 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 --alpha=%.4f " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, num_roi, crop_size, local_t,alpha)
                                write_slurm_sh_faster(
                                "type71_lgq_t_%.3f_%d_%d_lr%.4f_alpha%.4f" % (local_t, num_roi, crop_size, learning_rate,alpha),
                                command_line, queue_name)
                                time.sleep(1)
    elif args.type==72:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=72 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, num_roi, crop_size, local_t)
                            write_slurm_sh_faster(
                                "type72_lgq_t_%.3f_%d_%d_lr%.4f" % (local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==73:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=73 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, num_roi, crop_size, local_t)
                            write_slurm_sh_faster(
                                "type73_lgq_t_%.3f_%d_%d_lr%.4f" % (local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==74:
        for crop_size in [64,96,128,160,192]:
            for learning_rate in [0.06]:
                for local_t in [0.2]:
                    for epoch in [100]:
                        command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=74 --mask_size %d " \
                                           " --local_t=%.4f --align=1 " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t,  crop_size, local_t)
                        write_slurm_sh_faster(
                                "type74_lgq_t_%.3f_mask%d_lr%.4f" % (local_t, crop_size, learning_rate),
                                command_line, queue_name)
                        time.sleep(1)
    elif args.type==75:
        for num_roi in [3,6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=75 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, num_roi, crop_size, local_t)
                            write_slurm_sh_faster(
                                "type75_lgq_t_%.3f_%d_%d_lr%.4f" % (local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==76 or args.type==98:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for shuffle_mode in range(9):
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 --shuffle_mode=%d " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, args.type,num_roi, crop_size, local_t,shuffle_mode)
                                write_slurm_sh_faster(
                                "type%d_%d_lgq_t_%.3f_%d_%d_lr%.4f" % (args.type,shuffle_mode,local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==-76:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for shuffle_mode in [0,1,7]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=76 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 --shuffle_mode=%d --mlp_bn_stat=0 " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, num_roi, crop_size, local_t,shuffle_mode)
                                write_slurm_sh_faster(
                                "type76_%d_lgq_t_%.3f_%d_%d_lr%.4f" % (shuffle_mode,local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==77:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for shuffle_mode in [0,1,2,3,5,6]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=77 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 --shuffle_mode=%d " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, num_roi, crop_size, local_t,shuffle_mode)
                                write_slurm_sh_faster(
                                "type77_%d_lgq_t_%.3f_%d_%d_lr%.4f" % (shuffle_mode,local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)

    elif args.type==78:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for shuffle_mode in [0,1,3,4,5,7]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=78 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 --shuffle_mode=%d " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, num_roi, crop_size, local_t,shuffle_mode)
                                write_slurm_sh_faster(
                                "type78_%d_lgq_t_%.3f_%d_%d_lr%.4f" % (shuffle_mode,local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==79:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for shuffle_mode in range(2,11):
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=79 --nmb_crops 1 %d" \
                                               " --size_crops 224 %d --local_t=%.4f --align=1 --shuffle_mode=%d " % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                args.node * 256,
                                                learning_rate * args.node / 100,
                                                local_t, num_roi, crop_size, local_t, shuffle_mode)
                                write_slurm_sh_faster(
                                    "type79_%d_lgq_t_%.3f_%d_%d_lr%.4f" % (
                                    shuffle_mode, local_t, num_roi, crop_size, learning_rate),
                                    command_line, queue_name)
                            time.sleep(1)
    elif args.type==80:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for shuffle_mode in [1,5,7]:
                                for mlp_bn_stat in [0,1]:
                                    command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=80 --nmb_crops 1 %d" \
                                               " --size_crops 224 %d --local_t=%.4f --align=1 --shuffle_mode=%d" \
                                                   " --mlp_bn_stat=%d " % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                args.node * 256,
                                                learning_rate * args.node / 100,
                                                local_t, num_roi, crop_size, local_t, shuffle_mode,mlp_bn_stat)
                                    write_slurm_sh_faster(
                                    "type80_%d_lgq_t_%.3f_%d_%d_lr%.4f_bnmode%d" % (
                                    shuffle_mode, local_t, num_roi, crop_size, learning_rate,mlp_bn_stat),
                                    command_line, queue_name)
                                    time.sleep(1)
    elif args.type==81:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for shuffle_mode in [1,5,7]:
                                for mlp_bn_stat in [1]:
                                    command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=81 --nmb_crops 1 %d" \
                                               " --size_crops 224 %d --local_t=%.4f --align=1 --shuffle_mode=%d" \
                                                   " --mlp_bn_stat=%d " % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                args.node * 256,
                                                learning_rate * args.node / 100,
                                                local_t, num_roi, crop_size, local_t, shuffle_mode,mlp_bn_stat)
                                    write_slurm_sh_faster(
                                    "type81_%d_lgq_t_%.3f_%d_%d_lr%.4f_bnmode%d" % (
                                    shuffle_mode, local_t, num_roi, crop_size, learning_rate,mlp_bn_stat),
                                    command_line, queue_name)
                                    time.sleep(1)
    elif args.type==82:
        for num_roi in [6,16,32,64]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for shuffle_mode in [1,5]:
                                for mlp_bn_stat in [1]:
                                    command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=82 --nmb_crops 1 %d" \
                                               " --size_crops 224 %d --local_t=%.4f --align=1 --shuffle_mode=%d" \
                                                   " --mlp_bn_stat=%d " % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                args.node * 256,
                                                learning_rate * args.node / 100,
                                                local_t, num_roi, crop_size, local_t, shuffle_mode,mlp_bn_stat)
                                    write_slurm_sh_faster(
                                    "type82_%d_lgq_t_%.3f_%d_%d_lr%.4f_bnmode%d" % (
                                    shuffle_mode, local_t, num_roi, crop_size, learning_rate,mlp_bn_stat),
                                    command_line, queue_name)
                                    time.sleep(1)
    elif args.type == 83 or args.type==84:
        for num_roi in [1,3,5,10]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for alpha in [0.1,0.2,0.5,1.0,2.0]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d --num_roi %d" \
                                               " --mask_size %d --local_t=%.4f --align=1 --alpha=%f " \
                                               " " % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                args.node * 256,
                                                learning_rate * args.node / 100,
                                                local_t,args.type, num_roi, crop_size, local_t, alpha)
                                write_slurm_sh_faster(
                                    "type%d_lgq_t_%.3f_%d_%d_lr%.4f_alpha%f" % (args.type,
                                        local_t, num_roi, crop_size, learning_rate,alpha),
                                    command_line, queue_name)
                                time.sleep(1)
    elif args.type==85:
        for num_roi in [6,16,32,64]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for shuffle_mode in [1,5]:
                                for mlp_bn_stat in [1]:
                                    command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=85 --nmb_crops 1 %d" \
                                               " --size_crops 224 %d --local_t=%.4f --align=1 --shuffle_mode=%d" \
                                                   " --mlp_bn_stat=%d " % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                args.node * 256,
                                                learning_rate * args.node / 100,
                                                local_t, num_roi, crop_size, local_t, shuffle_mode,mlp_bn_stat)
                                    write_slurm_sh_faster(
                                    "type85_%d_lgq_t_%.3f_%d_%d_lr%.4f_bnmode%d" % (
                                    shuffle_mode, local_t, num_roi, crop_size, learning_rate,mlp_bn_stat),
                                    command_line, queue_name)
                                    time.sleep(1)
    elif args.type==86:
        for num_roi in [6,16,32]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for shuffle_mode in [1,5,7]:
                                for mlp_bn_stat in [1]:
                                    command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=86 --nmb_crops 1 %d" \
                                               " --size_crops 224 %d --local_t=%.4f --align=1 --shuffle_mode=%d" % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                args.node * 256,
                                                learning_rate * args.node / 100,
                                                local_t, num_roi, crop_size, local_t, shuffle_mode)
                                    write_slurm_sh_faster(
                                    "type86_%d_lgq_t_%.3f_%d_%d_lr%.4f" % (
                                    shuffle_mode, local_t, num_roi, crop_size, learning_rate),
                                    command_line, queue_name)
                                    time.sleep(1)
    elif args.type==87 or args.type==88 or args.type==93 or args.type==94 or args.type==95 or args.type==96:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t,args.type, num_roi, crop_size, local_t)
                            write_slurm_sh_faster(
                                "type%d_lgq_t_%.3f_%d_%d_lr%.4f" % (args.type,
                                     local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
    elif args.type==89 or args.type==90:
        for num_roi in [1,5,10]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for alpha in [0.1,0.2,0.5,1.0,2.0]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d --num_roi %d" \
                                               " --mask_size %d --local_t=%.4f --align=1 --alpha=%f " \
                                               " " % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                args.node * 256,
                                                learning_rate * args.node / 100,
                                                local_t,args.type, num_roi, crop_size, local_t, alpha)
                                write_slurm_sh_faster(
                                    "type%d_lgq_t_%.3f_%d_%d_lr%.4f_alpha%f" % (args.type,
                                        local_t, num_roi, crop_size, learning_rate,alpha),
                                    command_line, queue_name)
                                time.sleep(1)
    elif args.type==91:
        for learning_rate in [0.06]:
            for local_t in [0.2]:
                for epoch in [100]:
                    command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                   "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d " \
                                   " --local_t=%.4f --align=1 " % \
                                   (learning_rate * args.node, args.data, epoch, args.node * 256,
                                    args.node * 256,
                                    learning_rate * args.node / 100,
                                    local_t, args.type, local_t)
                    write_slurm_sh_faster(
                        "type%d_lgq_t_%.3f_lr%.4f" % (args.type, local_t, learning_rate),
                        command_line, queue_name)
                    time.sleep(1)
    elif args.type==92:
        for learning_rate in [0.06]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for shuffle_mode in range(4):
                        command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                   "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d " \
                                   " --local_t=%.4f --align=1 --shuffle_mode=%d " % \
                                   (learning_rate * args.node, args.data, epoch, args.node * 256,
                                    args.node * 256,
                                    learning_rate * args.node / 100,
                                    local_t, args.type, local_t,shuffle_mode)
                        write_slurm_sh_faster(
                        "type%d_%d_lgq_t_%.3f_lr%.4f" % (args.type,shuffle_mode, local_t, learning_rate),
                        command_line, queue_name)
                        time.sleep(1)
    elif args.type==97:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for shuffle_mode in range(4):

                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=97 --nmb_crops 1 %d" \
                                               " --size_crops 224 %d --local_t=%.4f --align=1 --shuffle_mode=%d" % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                args.node * 256,
                                                learning_rate * args.node / 100,
                                                local_t, num_roi, crop_size, local_t, shuffle_mode)
                                write_slurm_sh_faster(
                                    "type97_%d_lgq_t_%.3f_%d_%d_lr%.4f" % (
                                    shuffle_mode, local_t, num_roi, crop_size, learning_rate),
                                    command_line, queue_name)
                                time.sleep(1)
    elif args.type==99 or args.type==103 or args.type==104 or args.type==105 \
            or args.type==106 or args.type==107 or args.type==108 or args.type==109 \
            or args.type==110 or args.type==111 or args.type==112 or args.type==113:
        for learning_rate in [0.06]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for shuffle_mode in [1]:
                        command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                   "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d " \
                                   " --local_t=%.4f --align=1 --shuffle_mode=%d " % \
                                   (learning_rate * args.node, args.data, epoch, args.node * 256,
                                    args.node * 256,
                                    learning_rate * args.node / 100,
                                    local_t, args.type, local_t,shuffle_mode)
                        write_slurm_sh_faster(
                        "type%d_%d_lgq_t_%.3f_lr%.4f" % (args.type,shuffle_mode, local_t, learning_rate),
                        command_line, queue_name)
                        time.sleep(1)
    elif args.type==126 or args.type==127 or args.type==129 or args.type==131:
        for learning_rate in [0.03]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for shuffle_mode in range(8):
                        command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                   "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d " \
                                   " --local_t=%.4f --align=1 --shuffle_mode=%d --use_fp16=1 " % \
                                   (learning_rate * args.node, args.data, epoch, args.node * 256,
                                    args.node * 256,
                                    learning_rate * args.node / 100,
                                    local_t, args.type, local_t,shuffle_mode)
                        write_slurm_sh_faster(
                        "type%dablation_%d_lgq_t_%.3f_lr%.4f" % (args.type,shuffle_mode, local_t, learning_rate),
                        command_line, queue_name,environment=1)
                        time.sleep(1)
    elif args.type==133 or args.type==134:
        for learning_rate in [0.03]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for shuffle_mode in range(3):
                        for momentum_weight_decay in [0.9,0.99,0.999]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d " \
                                           " --local_t=%.4f --align=1 --shuffle_mode=%d --use_fp16=1 --momentum_stat=%f" % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, args.type, local_t, shuffle_mode,momentum_weight_decay)
                            write_slurm_sh_faster(
                                "type%dablation_%d_%f_lgq_t_%.3f_lr%.4f" % (
                                args.type, shuffle_mode,momentum_weight_decay, local_t, learning_rate),
                                command_line, queue_name, environment=1)
                            time.sleep(1)
    elif args.type==128 or args.type==130 or args.type==132 or args.type==135 or args.type==136:
        for learning_rate in [0.03]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for group_norm_size in [1,2,4,8,16,32,64,128]:
                        command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                   "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d " \
                                   " --local_t=%.4f --align=1 --group_norm_size=%d --use_fp16=1 " % \
                                   (learning_rate * args.node, args.data, epoch, args.node * 256,
                                    args.node * 256,
                                    learning_rate * args.node / 100,
                                    local_t, args.type, local_t,group_norm_size)
                        write_slurm_sh_faster(
                        "type%dgroupablation_%d_lgq_t_%.3f_lr%.4f" % (args.type,group_norm_size, local_t, learning_rate),
                        command_line, queue_name,environment=1)
                        time.sleep(1)
    elif args.type==152:
        for learning_rate in [0.03]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for group_norm_size in [1,2,4,8,16,32,64,128]:
                        command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                   "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d " \
                                   " --local_t=%.4f --align=1 --group_norm_size=%d --use_fp16=1 " % \
                                   (learning_rate * args.node, args.data, epoch, args.node * 256,
                                    args.node * 256,
                                    learning_rate * args.node / 100,
                                    local_t, args.type, local_t,group_norm_size)
                        write_slurm_sh_faster(
                        "type%dgroup_%d_lgq_t_%.3f_lr%.4f" % (args.type,group_norm_size, local_t, learning_rate),
                        command_line, queue_name,environment=0)
                        time.sleep(1)
    elif args.type==137 or args.type==138:
        for learning_rate in [0.03]:
            for local_t in [0.2]:
                for epoch in [100]:
                    command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                   "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d " \
                                   " --local_t=%.4f --align=1 --use_fp16=1 " % \
                                   (learning_rate * args.node, args.data, epoch, args.node * 256,
                                    args.node * 256,
                                    learning_rate * args.node / 100,
                                    local_t, args.type, local_t)
                    write_slurm_sh_faster(
                        "type%d2bnablation_lgq_t_%.3f_lr%.4f" % (args.type,local_t, learning_rate),
                        command_line, queue_name,environment=1)
                    time.sleep(1)
    elif args.type==118:
        for learning_rate in [0.06]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for shuffle_mode in [1]:
                        for conv_size in [1,2,3,4]:
                            for stride_size in [1,2,3]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d " \
                                               " --local_t=%.4f --align=1 --shuffle_mode=%d --loco_conv_size=%d " \
                                               "--loco_conv_stride=%d" % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                args.node * 256,
                                                learning_rate * args.node / 100,
                                                local_t, args.type, local_t, shuffle_mode,conv_size,stride_size)
                                write_slurm_sh_faster(
                                    "type%d_%d_conv%d_%d_lr%.4f" % (args.type, shuffle_mode, conv_size,
                                                                    stride_size,learning_rate),
                                    command_line, queue_name)
                                time.sleep(1)
    elif args.type==114:
        for learning_rate in [0.06]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for group_norm_size in [1,2,4,8]:
                        command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                   "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d " \
                                   " --local_t=%.4f --align=1 --group_norm_size=%d " % \
                                   (learning_rate * args.node, args.data, epoch, args.node * 256,
                                    args.node * 256,
                                    learning_rate * args.node / 100,
                                    local_t, args.type, local_t,group_norm_size)
                        write_slurm_sh_faster(
                        "type%d_%d_lgq_t_%.3f_lr%.4f" % (args.type,group_norm_size, local_t, learning_rate),
                        command_line, queue_name)
                        time.sleep(1)
    elif args.type==115 or args.type==116 or args.type==117 or args.type==120 \
            or args.type==121 or args.type==122 or args.type==123 or args.type==124:
        for learning_rate in [0.06]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for group_norm_size in [1,8]:
                        for alpha in [1.0,3.0]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                   "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --min_scale_crops 0.14 0.05" \
                                    " --size_crops 224 96 --nmb_crops 2 6  --max_scale_crops 1.0 0.14 --type=%d " \
                                   " --local_t=%.4f --align=1 --group_norm_size=%d --alpha=%f " % \
                                   (learning_rate * args.node, args.data, epoch, args.node * 256,
                                    args.node * 256,
                                    learning_rate * args.node / 100,
                                    local_t, args.type, local_t,group_norm_size,alpha)
                            write_slurm_sh_faster(
                        "type%d_%d_alpha%f_lgq_t_%.3f_lr%.4f" % (args.type,group_norm_size,alpha, local_t, learning_rate),
                        command_line, queue_name,gpu_memory=True)
                            time.sleep(1)
    elif args.type==-120:
        for learning_rate in [0.06]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for group_norm_size in [1]:
                        for num_crops in [4,8,16,32]:
                            same_alpha = int(num_crops / 2) - 1
                            iter_alpha =[same_alpha,1.0] if same_alpha!=1 else [1.0]
                            for alpha in iter_alpha:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --min_scale_crops 0.14 " \
                                               " --size_crops 96 --nmb_crops %d  --max_scale_crops 1.0 --type=%d " \
                                               " --local_t=%.4f --align=1 --group_norm_size=%d --alpha=%f --use_fp16=1" % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                args.node * 256,
                                                learning_rate * args.node / 100,
                                                local_t, num_crops,abs(args.type), local_t, group_norm_size, alpha)
                                write_slurm_sh_faster(
                                    "type%d_%d_%d_alpha%f_lgq_t_%.3f_lr%.4f" % (
                                    args.type,num_crops, group_norm_size, alpha, local_t, learning_rate),
                                    command_line, queue_name, gpu_memory=True,environment=1)
                                time.sleep(1)

    elif args.type==139 or args.type==140 or args.type==141 or args.type==142 \
            or args.type==143 or args.type==144 or args.type==145 or args.type==146 or args.type==147:
        for learning_rate in [0.06]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for group_norm_size in [1]:
                        for num_crops in [4,8,16]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --min_scale_crops 0.14 " \
                                           " --size_crops 96 --nmb_crops %d --max_scale_crops 1.0 --type=%d " \
                                           " --local_t=%.4f --align=1 --group_norm_size=%d --use_fp16=1 " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, num_crops,args.type, local_t, group_norm_size)
                            write_slurm_sh_faster(
                                "type%dviewnorm_%d_%d_lgq_t_%.3f_lr%.4f" % (
                                args.type, num_crops,group_norm_size, local_t, learning_rate),
                                command_line, queue_name, gpu_memory=True,environment=1)
                            time.sleep(1)
    elif args.type==148 or args.type==149 or args.type==150:
        for learning_rate in [0.06]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for group_norm_size in [1]:
                        for num_crops in [4,8,16,32]:
                            for crop_size in [224,96]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --min_scale_crops 0.2 " \
                                               " --size_crops %d --nmb_crops %d --max_scale_crops 1.0 --type=%d " \
                                               " --local_t=%.4f --align=1 --group_norm_size=%d --use_fp16=1 " % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                args.node * 256,
                                                learning_rate * args.node / 100,
                                                local_t, crop_size,num_crops, args.type, local_t, group_norm_size)
                                write_slurm_sh_faster(
                                    "type%dviewnorm_%d_%d_group%d_lgq_t_%.3f_lr%.4f" % (
                                        args.type, num_crops,crop_size, group_norm_size, local_t, learning_rate),
                                    command_line, queue_name, gpu_memory=True, environment=1)
                                time.sleep(1)
    elif args.type==151:
        for learning_rate in [0.06]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for group_norm_size in [1]:
                        for alpha in [1.0]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1  " \
                                           " --type=%d --min_scale_crops 0.14 0.05 " \
                                            " --size_crops 224 96 --nmb_crops 4 6  --max_scale_crops 1.0 0.14" \
                                           " --local_t=%.4f --align=1 --group_norm_size=%d --use_fp16=1 --alpha 1.0" % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, args.type, local_t, group_norm_size)
                            write_slurm_sh_faster(
                                "type%dmultiquery_viewkey_group%d_lgq_t_%.3f_lr%.4f" % (
                                    args.type, group_norm_size, local_t, learning_rate),
                                command_line, queue_name, gpu_memory=True, environment=1)
                            time.sleep(1)


    elif args.type==125:
        for learning_rate in [0.06]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for momentum_stat in [0.9,0.99,0.999]:
                        command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                       "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                       "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                       "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --min_scale_crops 0.14 0.05" \
                                       " --size_crops 224 96 --nmb_crops 2 6  --max_scale_crops 1.0 0.14 --type=%d " \
                                       " --local_t=%.4f --align=1 --momentum_stat=%f " % \
                                       (learning_rate * args.node, args.data, epoch, args.node * 256,256,
                                        learning_rate * args.node / 100,
                                        local_t, args.type, local_t, momentum_stat)
                        write_slurm_sh_faster(
                            "type%d_momentum%f_lgq_t_%.3f_lr%.4f" % (
                            args.type, momentum_stat, local_t, learning_rate),
                            command_line, queue_name, gpu_memory=True)
                        time.sleep(1)
    elif args.type==-108:
        for learning_rate in [0.06]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for batch_size in [1024]:
                        for shuffle_mode in [1]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                   "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                   "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d " \
                                   " --local_t=%.4f --align=1 --shuffle_mode=%d " % \
                                   (learning_rate * batch_size/256, args.data, epoch, batch_size,
                                    256,
                                    learning_rate * batch_size/256/ 100,
                                    local_t, abs(args.type), local_t,shuffle_mode)
                            write_slurm_sh_faster(
                            "type%d_%d_lgq_t_%.3f_lr%.4f" % (args.type,shuffle_mode, local_t, learning_rate*batch_size/256),
                            command_line, queue_name,gpu_memory=True)
                            time.sleep(1)
    elif args.type==100:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for group_norm_size in [1,2,4,8]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 --group_norm_size=%d " % \
                                           (learning_rate/2, args.data, epoch, 128,
                                            128,
                                            learning_rate/ 200,
                                            local_t,args.type, num_roi, crop_size, local_t,group_norm_size)
                                write_slurm_sh_faster(
                                "type%d_group%d_lgq_t_%.3f_%d_%d_lr%.4f" % (args.type,group_norm_size,
                                     local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name,gpu_per_node=args.gpu)
                                time.sleep(1)
    elif args.type==101:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for group_num in [1,2,4,8]:

                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=101 --nmb_crops 1 %d" \
                                               " --size_crops 224 %d --local_t=%.4f --align=1 --group_norm_size=%d " % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                args.node * 256,
                                                learning_rate * args.node / 100,
                                                local_t, num_roi, crop_size, local_t, group_num)
                                write_slurm_sh_faster(
                                    "type101_%d_lgq_t_%.3f_%d_%d_lr%.4f" % (
                                    group_num, local_t, num_roi, crop_size, learning_rate),
                                    command_line, queue_name)
                                time.sleep(1)
    elif args.type==102:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for shuffle_mode in [0,1,7]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 --shuffle_mode=%d " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, args.type,num_roi, crop_size, local_t,shuffle_mode)
                                write_slurm_sh_faster(
                                "type%d_%d_lgq_t_%.3f_%d_%d_lr%.4f" % (args.type,shuffle_mode,local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name)
                            time.sleep(1)
elif args.mode==2:
    if args.type==58:
        for learning_rate in [0.06]:
            for alpha in [1.0]:
                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                       "--dist_url=tcp://localhost:10031 --epochs=100 " \
                       "--batch_size=256 --knn_batch_size=256 --cos=1 --lr_final=%.4f " \
                       "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                       "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                       "--moco_m=0.999 --moco_k=65536 --moco_t=0.2 " \
                       "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=58 --nmb_crops 2 6 " \
                       "--size_crops 224 96 --min_scale_crops 0.14 0.05 --max_scale_crops 1.0 0.14 " \
                        "--alpha=%.4f " \
                               " " % (learning_rate,args.data,learning_rate/100,alpha)
                write_slurm_sh_multi("multimoco_proj_alpha_%.2f_lr_%.4f"%(alpha,learning_rate), command_line, queue_name,
                                     nodes=args.node,gpu_per_node=args.gpu)
    elif args.type==59:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [800]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=59 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 " % \
                                           (learning_rate*args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate*args.node / 100,
                                            local_t, num_roi, crop_size, local_t)
                            write_slurm_sh_multi(
                                "type59_lgq_t_%.3f_%d_%d_lr%.4f" % (local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)

    elif args.type==61:

        for num_roi in ['','6']:
            for crop_size in ['','96']:
                for learning_rate in [0.04,0.06,0.08]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=61 --nmb_crops 1 %s" \
                                           " --size_crops 224 %s --local_t=%.4f --align=1 --ngpu=%d " % \
                                           (learning_rate, args.data, epoch, 256,256,
                                            learning_rate / 100,
                                            local_t, num_roi, crop_size, local_t,args.gpu)
                            write_slurm_sh_multi(
                                "type61_lgq_t_%.3f_%s_%s_lr%.4f" % (local_t, num_roi, crop_size, learning_rate),
                                command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                            time.sleep(1)
    elif args.type==77:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for shuffle_mode in [5]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=77 --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 --shuffle_mode=%d " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t, num_roi, crop_size, local_t,shuffle_mode)
                                write_slurm_sh_multi(
                                "type77_%d_lgq_t_%.3f_%d_%d_lr%.4f" % (shuffle_mode,local_t, num_roi, crop_size, learning_rate*args.node),
                                command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                            time.sleep(1)
    elif args.type==87 or args.type==88 or args.type==94:
        if args.type==87:
            roi_num_list=[32]
        elif args.type==88:
            roi_num_list = [6,32]
        else:
            roi_num_list = [0]
        for num_roi in roi_num_list:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [800]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 128,
                                            learning_rate * args.node / 100,
                                            local_t,args.type, num_roi, crop_size, local_t)
                            if args.queue<=1:
                                write_slurm_sh_multi2(
                                    "type%d_lgq_t_%.3f_%d_%d_lr%.4f_epoch%d" % (args.type,
                                                                                local_t, num_roi, crop_size,
                                                                                learning_rate, epoch),
                                    command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                            else:
                                write_slurm_sh_multi(
                                "type%d_lgq_t_%.3f_%d_%d_lr%.4f_epoch%d" % (args.type,
                                     local_t, num_roi, crop_size, learning_rate,epoch),
                                command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                            time.sleep(1)
    elif args.type == 100:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for group_norm_size in [1,2,4,8,16]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d --nmb_crops 1 %d" \
                                           " --size_crops 224 %d --local_t=%.4f --align=1 --group_norm_size=%d " % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            args.node * 256,
                                            learning_rate * args.node / 100,
                                            local_t,args.type, num_roi, crop_size, local_t,group_norm_size)
                                if args.node>=4:
                                    command_line += " --warmup_epochs=10 "
                                if args.queue <= 1:
                                    write_slurm_sh_multi2(
                                    "type%d_group%d_lgq_t_%.3f_%d_%d_lr%.4f" % (args.type,group_norm_size,
                                     local_t, num_roi, crop_size, learning_rate),
                                    command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                                else:
                                    write_slurm_sh_multi(
                                        "type%d_group%d_lgq_t_%.3f_%d_%d_lr%.4f" % (args.type, group_norm_size,
                                                                                    local_t, num_roi, crop_size,
                                                                                    learning_rate),
                                        command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                                time.sleep(1)
    elif args.type==101:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [100]:
                            for group_num in [1,2,4,8,16]:

                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=101 --nmb_crops 1 %d" \
                                               " --size_crops 224 %d --local_t=%.4f --align=1 --group_norm_size=%d " % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                args.node * 256,
                                                learning_rate * args.node / 100,
                                                local_t, num_roi, crop_size, local_t, group_num)
                                if args.node >= 4:
                                    command_line += " --warmup_epochs=10 "
                                if args.queue <= 1:
                                   write_slurm_sh_multi2(
                                    "type101_%d_lgq_t_%.3f_%d_%d_lr%.4f" % (
                                    group_num, local_t, num_roi, crop_size, learning_rate),
                                    command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                                else:
                                    write_slurm_sh_multi(
                                        "type101_%d_lgq_t_%.3f_%d_%d_lr%.4f" % (
                                            group_num, local_t, num_roi, crop_size, learning_rate),
                                        command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                                time.sleep(1)
    elif args.type==119:
        for batch_size in [4096]:
            #for crop_size in [96]:
            if True:
                for learning_rate in [0.06]:
                    for local_t in [0.2]:
                        for epoch in [800]:
                            for group_num in [1,8,16,32]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --type=%d " \
                                               " --local_t=%.4f --align=1  --group_norm_size=%d --use_fp16=1 " % \
                                               (learning_rate * batch_size / 256, args.data, epoch, batch_size,
                                                256,
                                                learning_rate * batch_size / 256 / 100,
                                                local_t, abs(args.type), local_t,group_num)
                                command_line += " --warmup_epochs=10 "
                                write_slurm_sh_multi(
                                    "mocov2bigbatch_type%d_%d_%f_lgq_t_%.3f_epoch%d" % (
                                        args.type, group_num, learning_rate, local_t, epoch),
                                    command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu, gpu_memory=True,
                                    environment=1)
    elif args.type==115 or args.type==120:
        for batch_size in [2048]:
            for learning_rate in [0.045]:
                for local_t in [0.2]:
                    for epoch in [800]:
                        for group_norm_size in [64]:
                            for alpha in [1.0]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                   "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                   "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                   "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                   "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                   "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                   "--knn_neighbor=20 --knn_freq=10 --tensorboard=1 --min_scale_crops 0.14 0.05" \
                                    " --size_crops 224 96 --nmb_crops 2 6  --max_scale_crops 1.0 0.14 --type=%d " \
                                   " --local_t=%.4f --align=1 --group_norm_size=%d --alpha=%f --use_fp16=1 " % \
                                   (learning_rate * batch_size/256, args.data, epoch, batch_size,
                                    256,
                                    learning_rate * batch_size/256/ 100,
                                    local_t, args.type, local_t,group_norm_size,alpha)
                                write_slurm_sh_multi(
                                "multimoco_type%d_%d_alpha%f_lgq_t_%.3f_lr%.4f" % (args.type,group_norm_size,alpha, local_t, learning_rate),
                                command_line, queue_name,nodes=args.node, gpu_per_node=args.gpu,gpu_memory=True,environment=1)
                                time.sleep(1)
    elif args.type==149:
        for learning_rate in [0.06]:
            for local_t in [0.2]:
                for epoch in [1000]:
                    for group_norm_size in [1]:
                        for num_crops in [4]:
                            for crop_size in [224]:
                                command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                               "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                               "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                               "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                               "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                               "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                               "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --min_scale_crops 0.2 " \
                                               " --size_crops %d --nmb_crops %d --max_scale_crops 1.0 --type=%d " \
                                               " --local_t=%.4f --align=1 --group_norm_size=%d --use_fp16=1 " % \
                                               (learning_rate * args.node, args.data, epoch, args.node * 256,
                                                512,
                                                learning_rate * args.node / 100,
                                                local_t, crop_size,num_crops, args.type, local_t, group_norm_size)
                                write_slurm_sh_multi2(
                                    "mocov2_%dview_type%d_%d_%f_lgq_t_%.3f_epoch%d" % (
                                        args.type, num_crops,group_norm_size, learning_rate, local_t, epoch),
                                    command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu, gpu_memory=False,
                                    environment=0)
                                time.sleep(1)
    elif args.type==151:
        for learning_rate in [0.06]:
            for local_t in [0.2]:
                for epoch in [1000]:
                    for group_norm_size in [1]:
                        for alpha in [1.0]:
                            command_line = "python3 main_adco.py --mode=1 --lr=%.4f --data=%s " \
                                           "--dist_url=tcp://localhost:10031 --epochs=%d " \
                                           "--batch_size=%d --knn_batch_size=%d --cos=1 --lr_final=%.8f " \
                                           "--momentum=0.9 --weight_decay=1e-4 --world_size=1 " \
                                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128  " \
                                           "--moco_m=0.999 --moco_k=65536 --moco_t=%.4f " \
                                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1  " \
                                           " --type=%d --min_scale_crops 0.14 0.05 " \
                                            " --size_crops 224 96 --nmb_crops 4 6  --max_scale_crops 1.0 0.14" \
                                           " --local_t=%.4f --align=1 --group_norm_size=%d --use_fp16=1 --alpha=1.0" % \
                                           (learning_rate * args.node, args.data, epoch, args.node * 256,
                                            512,
                                            learning_rate * args.node / 100,
                                            local_t, args.type, local_t, group_norm_size)
                            write_slurm_sh_multi(
                                "type%dmultiquery_viewkey_group%d_lgq_t_%.3f_lr%.4f" % (
                                    args.type, group_norm_size, local_t, learning_rate),
                                command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                gpu_memory=True, environment=1)
                            time.sleep(1)
elif args.mode==6:
    if args.type==0 or args.type==1 or args.type==2 or args.type==3:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.9]:
                    for local_t in [1.0]:
                        for epoch in [100]:
                            for batch_size in [512]:
                                command_line = "python3 main_adco.py --mode=6 --type=%d --data=%s " \
                                               "--epochs=%d --start_epoch=0 --batch_size=%d --lr=0.9 " \
                                               "--weight_decay=1e-6 --dist_url=tcp://localhost:10031 --rank=0 " \
                                               "--multiprocessing_distributed=1 --world_size=1  --moco_dim=256 " \
                                               "--mlp_dim=4096 --moco_m=0.99 --moco_t=%f --warmup_epochs=10 --nmb_crops 1 %d " \
                                               " --size_crops 224 %d --align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 " \
                                               "--knn_batch_size=%d  " \
                                               % (
                                               args.type, args.data, epoch, batch_size,local_t, num_roi, crop_size, args.node * 64)
                                if args.node == 1:
                                    write_slurm_sh_faster("mocov3type%d_lgq_t_%.3f_%d_%d_epoch%d" % (args.type,
                                                                                                     local_t, num_roi,
                                                                                                     crop_size,
                                                                                                     epoch),
                                                          command_line, queue_name)
                                else:
                                    if args.queue <= 1:
                                        write_slurm_sh_multi2(
                                            "mocov3type%d_lgq_t_%.3f_%d_%d_epoch%d" % (args.type,
                                                                                       local_t, num_roi, crop_size,
                                                                                       epoch),
                                            command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                                    else:
                                        write_slurm_sh_multi(
                                            "mocov3type%d_lgq_t_%.3f_%d_%d_epoch%d" % (
                                                args.type, local_t, num_roi, crop_size, epoch),
                                            command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                                time.sleep(1)
    elif args.type==4 or args.type==5 or args.type==6:
        for num_roi in [1]:
            for crop_size in [96]:
                for learning_rate in [0.9]:
                    for local_t in [1.0]:
                        for epoch in [100]:
                            for batch_size in [1024]:
                                for group_norm_size in [1,2,4,8]:
                                    command_line = "python3 main_adco.py --mode=6 --type=%d --data=%s " \
                                                   "--epochs=%d --start_epoch=0 --batch_size=%d --lr=%f " \
                                                   "--weight_decay=1e-6 --dist_url=tcp://localhost:10031 --rank=0 " \
                                                   "--multiprocessing_distributed=1 --world_size=1  --moco_dim=256 " \
                                                   "--mlp_dim=4096 --moco_m=0.99 --moco_t=%f --warmup_epochs=10 --nmb_crops 1 %d " \
                                                   " --size_crops 224 %d --align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 " \
                                                   "--knn_batch_size=%d  --group_norm_size=%d " \
                                                   % (args.type, args.data, epoch, batch_size, learning_rate,local_t, num_roi,
                                                       crop_size, args.node * 64,group_norm_size)
                                    if args.node == 1:
                                        write_slurm_sh_faster("mocov3type%d_%d_%flgq_t_%.3f_%d_%d_epoch%d" % (args.type,
                                                                                                           group_norm_size,
                                                                                                              learning_rate,
                                                                                                         local_t,
                                                                                                         num_roi,
                                                                                                         crop_size,
                                                                                                         epoch),
                                                              command_line, queue_name,gpu_memory=True)
                                    else:
                                        if args.queue <= 1:
                                            write_slurm_sh_multi2(
                                                "mocov3type%d_%d_%f_lgq_t_%.3f_%d_%d_epoch%d" % (args.type,group_norm_size,learning_rate,
                                                                                           local_t, num_roi, crop_size,
                                                                                           epoch),
                                                command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                                        else:
                                            write_slurm_sh_multi(
                                                "mocov3type%d_%d_%f_lgq_t_%.3f_%d_%d_epoch%d" % (
                                                    args.type, group_norm_size,learning_rate,local_t, num_roi, crop_size, epoch),
                                                command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                                    time.sleep(1)
    elif args.type==7 or args.type==8:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.9]:
                    for local_t in [1.0]:
                        for epoch in [100]:
                            for batch_size in [1024]:
                                for group_norm_size in [1,2,4,8]:
                                    command_line = "python3 main_adco.py --mode=6 --type=%d --data=%s " \
                                                   "--epochs=%d --start_epoch=0 --batch_size=%d --lr=%f " \
                                                   "--weight_decay=1e-6 --dist_url=tcp://localhost:10031 --rank=0 " \
                                                   "--multiprocessing_distributed=1 --world_size=1  --moco_dim=256 " \
                                                   "--mlp_dim=4096 --moco_m=0.99 --moco_t=%f --warmup_epochs=10 --nmb_crops 1 %d " \
                                                   " --size_crops 224 %d --align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 " \
                                                   "--knn_batch_size=%d  --group_norm_size=%d --use_fp16=1 " \
                                                   % (args.type, args.data, epoch, batch_size, learning_rate,local_t, num_roi,
                                                       crop_size, args.node * 64,group_norm_size)
                                    if args.node == 1:
                                        write_slurm_sh_faster("mocov3type%d_%d_%flgq_t_%.3f_%d_%d_epoch%d" % (args.type,
                                                                                                           group_norm_size,
                                                                                                              learning_rate,
                                                                                                         local_t,
                                                                                                         num_roi,
                                                                                                         crop_size,
                                                                                                         epoch),
                                                              command_line, queue_name,gpu_memory=True,environment=1)
                                    else:
                                        if args.queue <= 1:
                                            write_slurm_sh_multi2(
                                                "mocov3type%d_%d_%f_lgq_t_%.3f_%d_%d_epoch%d" % (args.type,group_norm_size,learning_rate,
                                                                                           local_t, num_roi, crop_size,
                                                                                           epoch),
                                                command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,gpu_memory=True,environment=1)
                                        else:
                                            write_slurm_sh_multi(
                                                "mocov3type%d_%d_%f_lgq_t_%.3f_%d_%d_epoch%d" % (
                                                    args.type, group_norm_size,learning_rate,local_t, num_roi, crop_size, epoch),
                                                command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,gpu_memory=True,environment=1)
                                    time.sleep(1)
    elif args.type==-7:
        combine_choice=[1024,16]#[[1024,16],[2048,32],[4096,64]]
        for num_roi in [10]:
            for crop_size in [96]:
                for learning_rate in [0.3]:
                    for local_t in [1.0]:
                        for epoch in [1000]:
                            for batch_size,group_norm_size in combine_choice:
                                    command_line = "python3 main_adco.py --mode=6 --type=7 --data=%s " \
                                                   "--epochs=%d --start_epoch=0 --batch_size=%d --lr=%f " \
                                                   "--weight_decay=1.5e-6 --dist_url=tcp://localhost:10031 --rank=0 " \
                                                   "--multiprocessing_distributed=1 --world_size=1  --moco_dim=256 " \
                                                   "--mlp_dim=4096 --moco_m=0.996 --moco_t=%f --warmup_epochs=10 --nmb_crops 1 %d " \
                                                   " --size_crops 224 %d --align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 " \
                                                   "--knn_batch_size=%d  --group_norm_size=%d --use_fp16=1 " \
                                                   % ( args.data, epoch, batch_size, learning_rate,local_t, num_roi,
                                                       crop_size, args.node * 64,group_norm_size)
                                    if args.queue <= 1:
                                        write_slurm_sh_multi2(
                                            "mocov3type%d_%d_%f_lgq_t_%.3f_%d_%d_epoch%d" % (
                                            args.type, group_norm_size, learning_rate,
                                            local_t, num_roi, crop_size,
                                            epoch),
                                            command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                            gpu_memory=True, environment=1)
                                    else:
                                        write_slurm_sh_multi(
                                            "mocov3type%d_%d_%f_lgq_t_%.3f_%d_%d_epoch%d" % (
                                                args.type, group_norm_size, learning_rate, local_t, num_roi, crop_size,
                                                epoch),
                                            command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                            gpu_memory=True, environment=1)
                                    time.sleep(1)
    elif args.type==-13:
        combine_choice=[[4096,1],[4096,64]]#[[1024,16],[2048,32],[4096,64]]
        for num_roi in [20]:
            for crop_size in [96]:
                for learning_rate in [0.3]:
                    for local_t in [1.0]:
                        for epoch in [1000]:
                            for batch_size,group_norm_size in combine_choice:
                                    command_line = "python3 main_adco.py --mode=6 --type=13 --data=%s " \
                                                   "--epochs=%d --start_epoch=0 --batch_size=%d --lr=%f " \
                                                   "--weight_decay=1.5e-6 --dist_url=tcp://localhost:10031 --rank=0 " \
                                                   "--multiprocessing_distributed=1 --world_size=1  --moco_dim=256 " \
                                                   "--mlp_dim=4096 --moco_m=0.996 --moco_t=%f --warmup_epochs=10 --nmb_crops 1 %d " \
                                                   " --size_crops 224 %d --align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 " \
                                                   "--knn_batch_size=%d  --group_norm_size=%d --use_fp16=1 " \
                                                   % ( args.data, epoch, batch_size, learning_rate,local_t, num_roi,
                                                       crop_size, args.node * 64,group_norm_size)
                                    if args.queue <= 1:
                                        write_slurm_sh_multi2(
                                            "mocov3type%d_%d_%f_lgq_t_%.3f_%d_%d_epoch%d" % (
                                            args.type, group_norm_size, learning_rate,
                                            local_t, num_roi, crop_size,
                                            epoch),
                                            command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                            gpu_memory=True, environment=1)
                                    else:
                                        write_slurm_sh_multi(
                                            "mocov3type%d_%d_%f_lgq_t_%.3f_%d_%d_epoch%d" % (
                                                args.type, group_norm_size, learning_rate, local_t, num_roi, crop_size,
                                                epoch),
                                            command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                            gpu_memory=True, environment=1)
                                    time.sleep(1)
    elif args.type==9 or args.type==10:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.9]:
                    for local_t in [1.0]:
                        for epoch in [100]:
                            for batch_size in [1024]:
                                for ema_param in [0.001,0.01,0.1]:
                                    command_line = "python3 main_adco.py --mode=6 --type=%d --data=%s " \
                                                   "--epochs=%d --start_epoch=0 --batch_size=%d --lr=%f " \
                                                   "--weight_decay=1e-6 --dist_url=tcp://localhost:10031 --rank=0 " \
                                                   "--multiprocessing_distributed=1 --world_size=1  --moco_dim=256 " \
                                                   "--mlp_dim=4096 --moco_m=0.99 --moco_t=%f --warmup_epochs=10 --nmb_crops 1 %d " \
                                                   " --size_crops 224 %d --align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 " \
                                                   "--knn_batch_size=%d  --momentum_stat=%f --use_fp16=1 " \
                                                   % (args.type, args.data, epoch, batch_size, learning_rate,local_t, num_roi,
                                                       crop_size, args.node * 64,ema_param)
                                    if args.node == 1:
                                        write_slurm_sh_faster("mocov3type%d_%f_%flgq_t_%.3f_%d_%d_epoch%d" % (args.type,
                                                                                                           ema_param,
                                                                                                              learning_rate,
                                                                                                         local_t,
                                                                                                         num_roi,
                                                                                                         crop_size,
                                                                                                         epoch),
                                                              command_line, queue_name,gpu_memory=True,environment=1)
                                    else:
                                        if args.queue <= 1:
                                            write_slurm_sh_multi2(
                                                "mocov3type%d_%d_%f_lgq_t_%.3f_%d_%d_epoch%d" % (args.type,group_norm_size,learning_rate,
                                                                                           local_t, num_roi, crop_size,
                                                                                           epoch),
                                                command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,gpu_memory=True,environment=1)
                                        else:
                                            write_slurm_sh_multi(
                                                "mocov3type%d_%d_%f_lgq_t_%.3f_%d_%d_epoch%d" % (
                                                    args.type, group_norm_size,learning_rate,local_t, num_roi, crop_size, epoch),
                                                command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,gpu_memory=True,environment=1)
                                    time.sleep(1)
    elif args.type==11:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.9]:
                    for local_t in [1.0]:
                        for epoch in [100]:
                            for batch_size in [1024]:
                                for ema_param in [0.999]:
                                    for group_norm_size in [1,4,8,16]:
                                        command_line = "python3 main_adco.py --mode=6 --type=%d --data=%s " \
                                                   "--epochs=%d --start_epoch=0 --batch_size=%d --lr=%f " \
                                                   "--weight_decay=1e-6 --dist_url=tcp://localhost:10031 --rank=0 " \
                                                   "--multiprocessing_distributed=1 --world_size=1  --moco_dim=256 " \
                                                   "--mlp_dim=4096 --moco_m=0.99 --moco_t=%f --warmup_epochs=10 --nmb_crops 1 %d " \
                                                   " --size_crops 224 %d --align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 " \
                                                   "--knn_batch_size=%d  --momentum_stat=%f --use_fp16=1 --group_norm_size=%d " \
                                                   % (args.type, args.data, epoch, batch_size, learning_rate,local_t, num_roi,
                                                       crop_size, args.node * 64,ema_param,group_norm_size)
                                        if args.node == 1:
                                            write_slurm_sh_faster(
                                                "mocov3type%d_%f_%d_%flgq_t_%.3f_%d_%d_epoch%d" % (args.type,
                                                                                                   group_norm_size,
                                                                                                ema_param,
                                                                                                learning_rate,
                                                                                                local_t,
                                                                                                num_roi,
                                                                                                crop_size,
                                                                                                epoch),
                                                command_line, queue_name, gpu_memory=True, environment=1)
                                        else:
                                            if args.queue <= 1:
                                                write_slurm_sh_multi2(
                                                    "mocov3type%d_%d_%f_%f_lgq_t_%.3f_%d_%d_epoch%d" % (
                                                    args.type, group_norm_size,ema_param, learning_rate,
                                                    local_t, num_roi, crop_size,
                                                    epoch),
                                                    command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                                    gpu_memory=True, environment=1)
                                            else:
                                                write_slurm_sh_multi(
                                                    "mocov3type%d_%d_%f_%f_lgq_t_%.3f_%d_%d_epoch%d" % (
                                                        args.type, group_norm_size,ema_param, learning_rate, local_t, num_roi,
                                                        crop_size, epoch),
                                                    command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                                    gpu_memory=True, environment=1)
                                        time.sleep(1)
    elif args.type==12:
        for num_roi in [6]:
            for crop_size in [96]:
                for learning_rate in [0.9]:
                    for local_t in [1.0]:
                        for epoch in [100]:
                            for batch_size in [1024]:
                                for group_norm_size in [8]:
                                    command_line = "python3 main_adco.py --mode=6 --type=%d --data=%s " \
                                                   "--epochs=%d --start_epoch=0 --batch_size=%d --lr=%f " \
                                                   "--weight_decay=1e-6 --dist_url=tcp://localhost:10031 --rank=0 " \
                                                   "--multiprocessing_distributed=1 --world_size=1  --moco_dim=256 " \
                                                   "--mlp_dim=4096 --moco_m=0.99 --moco_t=%f --warmup_epochs=10 --nmb_crops 1 %d " \
                                                   " --size_crops 224 %d --align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 " \
                                                   "--knn_batch_size=%d  --group_norm_size=%d --use_fp16=1 " \
                                                   % (args.type, args.data, epoch, batch_size, learning_rate,local_t, num_roi,
                                                       crop_size, args.node * 64,group_norm_size)
                                    if args.node == 1:
                                        write_slurm_sh_faster("mocov3type%d_%d_%flgq_t_%.3f_%d_%d_epoch%d" % (args.type,
                                                                                                           group_norm_size,
                                                                                                              learning_rate,
                                                                                                         local_t,
                                                                                                         num_roi,
                                                                                                         crop_size,
                                                                                                         epoch),
                                                              command_line, queue_name,gpu_memory=True,environment=1)
                                    else:
                                        if args.queue <= 1:
                                            write_slurm_sh_multi2(
                                                "mocov3type%d_%d_%f_lgq_t_%.3f_%d_%d_epoch%d" % (args.type,group_norm_size,learning_rate,
                                                                                           local_t, num_roi, crop_size,
                                                                                           epoch),
                                                command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,gpu_memory=False,environment=0)
                                        else:
                                            write_slurm_sh_multi(
                                                "mocov3type%d_%d_%f_lgq_t_%.3f_%d_%d_epoch%d" % (
                                                    args.type, group_norm_size,learning_rate,local_t, num_roi, crop_size, epoch),
                                                command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,gpu_memory=True,environment=1)
                                    time.sleep(1)
    elif args.type==13 or args.type==14 or args.type==15:
        for learning_rate in [0.9]:
            for local_t in [1.0]:
                for epoch in [100]:
                    for batch_size in [1024]:
                        for group_norm_size in [1,4,8,16]:
                            command_line = "python3 main_adco.py --mode=6 --type=%d --data=%s " \
                                           "--epochs=%d --start_epoch=0 --batch_size=%d --lr=%f " \
                                           "--weight_decay=1e-6 --dist_url=tcp://localhost:10031 --rank=0 " \
                                           "--multiprocessing_distributed=1 --world_size=1  --moco_dim=256 " \
                                           "--mlp_dim=4096 --moco_m=0.99 --moco_t=%f --warmup_epochs=10  " \
                                           " --align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 " \
                                           "--knn_batch_size=%d  --group_norm_size=%d --use_fp16=1 " \
                                           % (args.type, args.data, epoch, batch_size, learning_rate,
                                              local_t,  args.node * 64, group_norm_size)
                            if args.node == 1:
                                write_slurm_sh_faster("mocov3type%d_%d_%flgq_t_%.3f_epoch%d" % (args.type,
                                                                                                group_norm_size,
                                                                                                learning_rate,
                                                                                              local_t,
                                                                                            epoch),
                                                      command_line, queue_name, gpu_memory=True, environment=1)
                            else:
                                if args.queue <= 1:
                                    write_slurm_sh_multi2(
                                        "mocov3type%d_%d_%f_lgq_t_%.3f_epoch%d" % (
                                        args.type, group_norm_size, learning_rate,
                                        local_t, epoch),
                                        command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                        gpu_memory=False, environment=0)
                                else:
                                    write_slurm_sh_multi(
                                        "mocov3type%d_%d_%f_lgq_t_%.3f_epoch%d" % (
                                            args.type, group_norm_size, learning_rate, local_t, epoch),
                                        command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                        gpu_memory=True, environment=1)
                            time.sleep(1)
    elif args.type==19:
        for learning_rate in [0.9]:
            for local_t in [1.0]:
                for epoch in [100]:
                    for batch_size in [1024]:
                        for group_norm_size in [1,4,8,16,32]:
                            for key_group_norm_size in [1,4,8,16,32]:
                                command_line = "python3 main_adco.py --mode=6 --type=%d --data=%s " \
                                               "--epochs=%d --start_epoch=0 --batch_size=%d --lr=%f " \
                                               "--weight_decay=1e-6 --dist_url=tcp://localhost:10031 --rank=0 " \
                                               "--multiprocessing_distributed=1 --world_size=1  --moco_dim=256 " \
                                               "--mlp_dim=4096 --moco_m=0.99 --moco_t=%f --warmup_epochs=10  " \
                                               " --align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 " \
                                               "--knn_batch_size=%d  --group_norm_size=%d --key_group=%d " \
                                               "--use_fp16=1 " \
                                               % (args.type, args.data, epoch, batch_size, learning_rate,
                                                  local_t, args.node * 64, group_norm_size,key_group_norm_size)
                                if args.node == 1:
                                    write_slurm_sh_faster("mocov3type%d_%d_%d_%flgq_t_%.3f_epoch%d" % (args.type,
                                                                                                    group_norm_size,
                                                                                                    key_group_norm_size,
                                                                                                    learning_rate,
                                                                                                    local_t,
                                                                                                    epoch),
                                                          command_line, queue_name, gpu_memory=True, environment=1)
                                else:
                                    if args.queue <= 3:
                                        write_slurm_sh_multi2(
                                            "mocov3type%d_%d_%d_%f_lgq_t_%.3f_epoch%d" % (
                                                args.type, group_norm_size, key_group_norm_size,learning_rate,
                                                local_t, epoch),
                                            command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                            gpu_memory=False, environment=0)
                                    else:
                                        write_slurm_sh_multi(
                                            "mocov3type%d_%d_%d_%f_lgq_t_%.3f_epoch%d" % (
                                                args.type, group_norm_size, key_group_norm_size,learning_rate, local_t, epoch),
                                            command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                            gpu_memory=True, environment=1)
                                time.sleep(1)
    elif args.type==16:
        for learning_rate in [0.9]:
            for local_t in [1.0]:
                for epoch in [100]:
                    for batch_size in [1024]:
                        for crop_size in [4,8,16]:
                            command_line = "python3 main_adco.py --mode=6 --type=%d --data=%s " \
                                           "--epochs=%d --start_epoch=0 --batch_size=%d --lr=%f " \
                                           "--weight_decay=1e-6 --dist_url=tcp://localhost:10031 --rank=0 " \
                                           "--multiprocessing_distributed=1 --world_size=1  --moco_dim=256 " \
                                           "--mlp_dim=4096 --moco_m=0.99 --moco_t=%f --warmup_epochs=10  " \
                                           " --align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 " \
                                           "--knn_batch_size=%d  --group_norm_size=1  --use_fp16=1 " \
                                           "--nmb_crops %d" \
                                           % (args.type, args.data, epoch, batch_size, learning_rate,
                                              local_t, args.node * 64,crop_size )
                            if args.node == 1:
                                write_slurm_sh_faster("mocov3type%d_%d_%flgq_t_%.3f_epoch%d" % (args.type,
                                                                                                crop_size,
                                                                                                learning_rate,
                                                                                                local_t,
                                                                                                epoch),
                                                      command_line, queue_name, gpu_memory=True, environment=1)
                            else:
                                if args.queue <= 1:
                                    write_slurm_sh_multi2(
                                        "mocov3type%d_%d_%f_lgq_t_%.3f_epoch%d" % (
                                            args.type, crop_size, learning_rate,
                                            local_t, epoch),
                                        command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                        gpu_memory=False, environment=0)
                                else:
                                    write_slurm_sh_multi(
                                        "mocov3type%d_%d_%f_lgq_t_%.3f_epoch%d" % (
                                            args.type, crop_size, learning_rate, local_t, epoch),
                                        command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                        gpu_memory=True, environment=1)
                            time.sleep(1)
    elif args.type==17 or args.type==18:
        warmup_epoch=10
        for learning_rate in [1.5e-4]:
            for local_t in [0.2]:
                for epoch in [100]:
                    for batch_size in [1024]:
                        if args.type==18:
                            group_list = [1,2,4,8,16,32,64,128]
                        else:
                            group_list = [1]
                        for group_norm_size in group_list:
                            command_line = "python3 main_adco.py --mode=6 --type=%d --data=%s " \
                                           "--epochs=%d --start_epoch=0 --batch_size=%d --lr=%f " \
                                           "--weight_decay=0.1 --dist_url=tcp://localhost:10031 --rank=0 " \
                                           "--multiprocessing_distributed=1 --world_size=1  --moco_dim=256 " \
                                           "--mlp_dim=4096 --moco_m=0.99 --moco_t=%f   " \
                                           " --align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 " \
                                           "--knn_batch_size=%d  --group_norm_size=%d  --use_fp16=1 " \
                                           "--warmup_epochs %d -a vit_small --crop_min 0.08 " \
                                           % (args.type, args.data, epoch, batch_size, learning_rate,
                                              local_t, 256 , group_norm_size,warmup_epoch)
                            if args.node == 1:
                                write_slurm_sh_faster("mocov3type%d_%d_%flgq_t_%.3f_epoch%d" % (args.type,
                                                                                                group_norm_size,
                                                                                                learning_rate,
                                                                                                local_t,
                                                                                                epoch),
                                                      command_line, queue_name, gpu_memory=True, environment=1)
                            else:
                                if args.queue <= 1:
                                    write_slurm_sh_multi2(
                                        "mocov3type%d_%d_%f_lgq_t_%.3f_epoch%d" % (
                                            args.type, group_norm_size, learning_rate,
                                            local_t, epoch),
                                        command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                        gpu_memory=False, environment=0)
                                else:
                                    write_slurm_sh_multi(
                                        "mocov3type%d_%d_%f_lgq_t_%.3f_epoch%d" % (
                                            args.type, group_norm_size, learning_rate, local_t, epoch),
                                        command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                        gpu_memory=True, environment=1)
                            time.sleep(1)
elif args.mode==7:
    if args.type==0 or args.type==1 or args.type==2 or args.type==3 or args.type==4:
        for num_roi in [16]:
            for crop_size in [96]:
                for learning_rate in [0.05]:
                    for barch_size in [512]:
                        for epoch in [100]:
                            command_line = "python3 main_adco.py --mode=7 --type=%d " \
                                       " --data=%s --epochs=%d --start_epoch=0 --batch_size=%d " \
                                       "--lr=%f --weight_decay=1e-4 --dist_url=tcp://localhost:10031 " \
                                       "--rank=0 --multiprocessing_distributed=1 --world_size=1  " \
                                       "--moco_dim=2048 --mlp_dim=512 --nmb_crops 1 %d --size_crops 224 %d " \
                                           "--align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 --knn_batch_size=%d "\
                                       %(args.type,args.data,epoch,barch_size,learning_rate,num_roi,crop_size,max(64*args.node,256))
                            if args.node==1:
                                write_slurm_sh_faster("simsiamtype%d_%d_%d_epoch%d" % (args.type, num_roi, crop_size,
                                                                         epoch),command_line, queue_name,)
                            else:
                                if args.queue <= 1:
                                    write_slurm_sh_multi2(
                                        "simsiamtype%d_%d_%d_epoch%d" % (args.type, num_roi, crop_size,
                                                                         epoch),
                                        command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                                else:
                                    write_slurm_sh_multi(
                                        "simsiamtype%d_%d_%d_epoch%d" % (args.type, num_roi, crop_size, epoch),
                                        command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                            time.sleep(1)
    elif args.type==5 or args.type==6 or args.type==7 or args.type==8 or args.type==9:
        for learning_rate in [0.05]:
            for barch_size in [512]:
                for epoch in [100]:
                    for group_norm_size in [1, 2, 4, 8,16,32,64]:
                        command_line = "python3 main_adco.py --mode=7 --type=%d " \
                                       " --data=%s --epochs=%d --start_epoch=0 --batch_size=%d " \
                                       "--lr=%f --weight_decay=1e-4 --dist_url=tcp://localhost:10031 " \
                                       "--rank=0 --multiprocessing_distributed=1 --world_size=1  " \
                                       "--moco_dim=2048 --mlp_dim=512  --group_norm_size=%d " \
                                       "--align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 --knn_batch_size=%d " \
                                       "--use_fp16=1 " \
                                       % (args.type, args.data, epoch, barch_size, learning_rate,group_norm_size,
                                          max(64 * args.node, 256))
                        if args.node == 1:
                            write_slurm_sh_faster("simsiamtype%d_%d_epoch%d" % (args.type,group_norm_size,
                                                                               epoch), command_line, queue_name,
                                                  gpu_memory=True,environment=1)
                        else:
                            if args.queue <= 1:
                                write_slurm_sh_multi2(
                                    "simsiamtype%d_%d_epoch%d" % (args.type,group_norm_size,
                                                                  epoch),
                                    command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                    gpu_memory=True,environment=1)
                            else:
                                write_slurm_sh_multi(
                                    "simsiamtype%d_%d_epoch%d" % (args.type,group_norm_size, epoch),
                                    command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                    gpu_memory=True,environment=1)
                        time.sleep(1)
    elif args.type==-6:
        for learning_rate in [0.05]:
            for barch_size in [256,512]:
                for epoch in [800]:
                    for group_norm_size in [8]:
                        command_line = "python3 main_adco.py --mode=7 --type=%d " \
                                       " --data=%s --epochs=%d --start_epoch=0 --batch_size=%d " \
                                       "--lr=%f --weight_decay=1e-4 --dist_url=tcp://localhost:10031 " \
                                       "--rank=0 --multiprocessing_distributed=1 --world_size=1  " \
                                       "--moco_dim=2048 --mlp_dim=512  --group_norm_size=%d " \
                                       "--align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 --knn_batch_size=%d " \
                                       "--use_fp16=1 " \
                                       % (abs(args.type), args.data, epoch, barch_size, learning_rate,group_norm_size,
                                          max(64 * args.node, 256))
                        if args.node == 1:
                            write_slurm_sh_faster("simsiamtype%d_%d_epoch%d" % (args.type,group_norm_size,
                                                                               epoch), command_line, queue_name,
                                                  gpu_memory=True )
                        else:
                            if args.queue <= 1:
                                write_slurm_sh_multi2(
                                    "simsiamtype%d_%d_epoch%d" % (args.type,group_norm_size,
                                                                  epoch),
                                    command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                            else:
                                write_slurm_sh_multi(
                                    "simsiamtype%d_%d_epoch%d" % (args.type,group_norm_size, epoch),
                                    command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu)
                        time.sleep(1)
    elif args.type==10:
        for learning_rate in [0.05]:
            for barch_size in [512]:
                for epoch in [100]:
                    for crop_size in [4, 8,16]:
                        command_line = "python3 main_adco.py --mode=7 --type=%d " \
                                       " --data=%s --epochs=%d --start_epoch=0 --batch_size=%d " \
                                       "--lr=%f --weight_decay=1e-4 --dist_url=tcp://localhost:10031 " \
                                       "--rank=0 --multiprocessing_distributed=1 --world_size=1  " \
                                       "--moco_dim=2048 --mlp_dim=512  --nmb_crops %d " \
                                       "--align=1 --knn_neighbor=20 --knn_freq=1 --tensorboard=1 --knn_batch_size=%d " \
                                       "--use_fp16=1 " \
                                       % (args.type, args.data, epoch, barch_size, learning_rate,crop_size,
                                          max(64 * args.node, 256))
                        if args.node == 1:
                            write_slurm_sh_faster("simsiamtype%d_%d_epoch%d" % (args.type,crop_size,
                                                                               epoch), command_line, queue_name,
                                                  gpu_memory=True,environment=1)
                        else:
                            if args.queue <= 1:
                                write_slurm_sh_multi2(
                                    "simsiamtype%d_%d_epoch%d" % (args.type,crop_size,
                                                                  epoch),
                                    command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                    gpu_memory=True,environment=1)
                            else:
                                write_slurm_sh_multi(
                                    "simsiamtype%d_%d_epoch%d" % (args.type,crop_size, epoch),
                                    command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                    gpu_memory=True,environment=1)
                        time.sleep(1)


elif args.mode==5:
    #run swav baseline
    if args.type==0:

        if args.F is None:
            command_line = "python3 main_adco.py --mode=5 --type=0 --data=%s --epochs 200 --lr=0.6 "\
                       "--lr_final 0.0006 --batch_size=256 --warmup_epochs 0 --freeze_prototypes_niters 5005 "\
                        "--queue_length 3840 --epoch_queue_starts 15 --dist_url=tcp://localhost:10031 "\
                        "--knn_batch_size=256 --cos=1 --momentum=0.9 --weight_decay=1e-6 --world_size=1 "\
                        "--rank=0 --multiprocessing_distributed=1 --moco_dim=128 --moco_k=3000 --moco_t=0.1 "\
                    "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --nmb_crops 2 6 " \
                       "--size_crops 224 96 --min_scale_crops 0.14 0.05 --max_scale_crops 1.0 0.14 --dump_path %s"%(args.data,dump_path)
            write_slurm_sh("swav_baseline" , command_line, queue_name)
        else:
            args.F= os.path.abspath(args.F)
            command_line = "python3 main_adco.py --mode=5 --type=0 --data=%s --epochs 200 --lr=0.6 " \
                           "--lr_final 0.0006 --batch_size=256 --warmup_epochs 0 --freeze_prototypes_niters 5005 " \
                           "--queue_length 3840 --epoch_queue_starts 15 --dist_url=tcp://localhost:10031 " \
                           "--knn_batch_size=256 --cos=1 --momentum=0.9 --weight_decay=1e-6 --world_size=1 " \
                           "--rank=0 --multiprocessing_distributed=1 --moco_dim=128 --moco_k=3000 --moco_t=0.1 " \
                           "--knn_neighbor=20 --knn_freq=1 --tensorboard=1 --nmb_crops 2 6 " \
                           "--size_crops 224 96 --min_scale_crops 0.14 0.05 --max_scale_crops 1.0 0.14 " \
                           "--resume=%s --dump_path %s " % (args.data,args.F,dump_path)
            resume_name= os.path.split(os.path.abspath(args.F))[1]
            write_slurm_sh("swav_baseline_resume%s"%resume_name, command_line, queue_name)
elif args.mode==8:
    if args.type==0 or args.type==1:
        for epoch in [100]:
            for batch_size in [2048]:
                for lr_w in [0.2]:
                    for lr_bias in [0.0048]:
                        for alpha in [0.51]:
                            command_line="python3 main.py %s --epochs=%d " \
                                         "--batch-size=%d --learning-rate-weights=%f --learning-rate-biases=%f " \
                                        "--weight-decay=1e-6 --lambd=%f --type=%d --knn_neighbor=20 " \
                                         "--knn_freq=1 --knn_batch_size=%d --tensorboard=1 "%(args.data,epoch,
                                           batch_size,lr_w,lr_bias,alpha,args.type,256 )
                            if args.node==1:
                                write_slurm_sh_faster("BTtype%d_%d_epoch%d" % (args.type,batch_size,epoch), command_line, queue_name,
                                                  gpu_memory=False, environment=0)
                            else:
                                write_slurm_sh_multi2(
                                    "BTtype%d_%d_epoch%d" % (args.type, batch_size, epoch),
                                    command_line, queue_name, nodes=args.node, gpu_per_node=args.gpu,
                                    gpu_memory=False, environment=0)
    elif args.type==2:
        for epoch in [100]:
            for batch_size in [1024]:
                for lr_w in [0.2]:
                    for lr_bias in [0.0048]:
                        for alpha in [0.51]:
                            for group_size in [2,4,8,16,32]:
                                command_line = "python3 main.py %s --epochs=%d " \
                                               "--batch-size=%d --learning-rate-weights=%f --learning-rate-biases=%f " \
                                               "--weight-decay=1e-6 --lambd=%f --type=%d --knn_neighbor=20 " \
                                               "--knn_freq=1 --knn_batch_size=%d --tensorboard=1 --group_norm_size=%d " % (args.data, epoch,
                                                                                                      batch_size, lr_w,
                                                                                                      lr_bias, alpha,
                                                                                                      args.type, 256,group_size)
                                write_slurm_sh_faster("BTtype%d_%d_%d_epoch%d" % (args.type,group_size, batch_size,epoch), command_line, queue_name,
                                                      gpu_memory=False, environment=0)
elif args.mode==0:
    #used for finetuning, which will submit finetune jobs and a comment for which
    use_bn=args.type
    for lr in [20]:
        for weight_decay in [1e-6,1e-7,1e-8,1e-9]:
            command_line = "python3 lincls.py --data=%s --dist-url=tcp://localhost:10031 " \
                           "--pretrained='%s' --lr=%.4f --final_lr=%.8f --dataset=ImageNet --use_bn=%d --wd %.8f" % (
                               args.data, args.F, lr, lr / 100, use_bn,weight_decay)
            write_slurm_sh("linear_eval_%s_%.4f_bn%d_wd_%f" % (args.comment, lr, use_bn,weight_decay), command_line, queue_name)
            time.sleep(1)
elif args.mode==-2:
    use_bn = args.type
    #type 3:l2 norm linear
    for lr in [1.0]:
        for weight_decay in [1e-5,1e-6,1e-7,1e-8,1e-9]:
            command_line = "python3 lincls.py --data=%s --dist-url=tcp://localhost:10031 --batch-size=4096 " \
                           "--pretrained='%s' --lr=%.4f --final_lr=%.8f --dataset=ImageNet --use_bn=%d --wd %.8f" % (
                               args.data, args.F, lr, lr / 100, use_bn, weight_decay)
            write_slurm_sh("linearb4096_eval_%s_%.4f_bn%d_wd_%.8f" % (args.comment, lr, use_bn, weight_decay), command_line,
                           queue_name)
elif args.mode==-1:
    command_line = "python3 encode.py --data=%s --dist-url=tcp://localhost:10031 " \
                   "--pretrained='%s' --dataset=ImageNet " % (args.data, args.F)
    write_slurm_sh("encode_%s" % (args.comment), command_line, queue_name)

elif args.mode==-3:

    command_line = "python3 main_adco.py --sym=0 --lr=0.03 --memory_lr=3 --moco_t=0.12 " \
                    "--mem_t=0.02 --data=%s --dist_url=tcp://localhost:10001 --mode=0 " \
                    "--epochs=200 --moco_dim=128 --moco_m=0.999 --moco_k=65536 --cluster=65536 " \
                    "--knn_neighbor=20 --knn_freq=1 --data=imagenet --batch_size=256 --ad_init=1 "%(args.data)
    write_slurm_sh("type0",command_line,queue_name)

elif args.mode==-4:
    use_bn = args.type
    vit_model =True
    for lr in [0.05,0.1]:
        for weight_decay in [0]:
            for model_type in [0]:
                command_line ="python lincls_lars.py -a resnet50 --dist-url 'tcp://localhost:10001' " \
                          "--multiprocessing-distributed --world-size 1 --rank 0  --pretrained='%s' --lr %f --wd %f " \
                          "--lars --data %s --use_bn=%d --model_type=%d "%(args.F,lr,
                          weight_decay,args.data,use_bn,model_type)
                if vit_model:
                    command_line +=" --arch vit_small"
                write_slurm_sh("linear_larsb4096_eval_%s_bn%d_%.4f_wd_%.8f" % (args.comment, use_bn,lr,weight_decay),
                           command_line,
                           queue_name)
elif args.mode==-40:
    use_bn = args.type
    study_dir = os.path.abspath(args.F)
    checkpoint_name = "checkpoint_0099.pth.tar"
    for item in os.listdir(study_dir):
        if item== checkpoint_name:
            current_model_path = os.path.join(study_dir,item)
            current_dir = study_dir
            current_comment = os.path.split(current_dir)[1]
        else:
            current_dir = os.path.join(study_dir,item)
            current_comment = os.path.split(current_dir)[1]
            current_model_path = find_checkpoint(current_dir,checkpoint_name)
            if current_model_path is None:
                print("%s dir did not find checkpoint"%current_dir)
                continue
        if not os.path.exists(current_model_path):
            print("%s model path did not exist"%current_model_path)
            continue
        print("fintune %s model"%current_model_path)
        for lr in [0.05, 0.1]:
            for weight_decay in [0]:
                for model_type in [0]:
                    command_line = "python lincls_lars.py -a resnet50 --dist-url 'tcp://localhost:10001' " \
                                   "--multiprocessing-distributed --world-size 1 --rank 0  --pretrained='%s' --lr %f --wd %f " \
                                   "--lars --data %s --use_bn=%d --model_type=%d " % (current_model_path, lr,
                                                                                      weight_decay, args.data, use_bn,
                                                                                      model_type)
                    write_slurm_sh(
                        "linear_larsb4096_eval_%s_bn%d_%.4f_wd_%.8f" % (str(args.comment)+current_comment, use_bn, lr, weight_decay),
                        command_line,
                        queue_name)



elif args.mode==-5:
    config_dict={}
    config_path = os.path.join(os.getcwd(),"detection")
    config_path = os.path.join(config_path,"configs")
    config_dict['VOC']=os.path.join(config_path,"pascal_voc_R_50_C4_24k_loco.yaml")
    config_dict['VOC_freeze'] = os.path.join(config_path, "pascal_voc_R_50_C4_24k_loco_freeze.yaml")
    config_dict['COCO'] = os.path.join(config_path,"coco_R_50_C4_2x.yaml_loco.yaml")
    config_dict['COCO_freeze'] =os.path.join(config_path,"coco_R_50_C4_2x.yaml_loco_freeze.yaml")
    model_path = os.path.abspath(args.F)
    model_name = os.path.split(model_path)[1].replace(".pkl","")
    for kk in range(5):
        for config_now  in ['VOC','VOC_freeze']:
            command_line = "python detection/train_net.py --config-file %s --num-gpus 8" \
                           " MODEL.WEIGHTS %s"%(config_dict[config_now],args.F)
            write_slurm_sh_faster("detection_%s_run%d_%s" % (config_now, kk,model_name),
                                  command_line, queue_name, gpu_memory=True)
    for config_now in ['COCO',"COCO_freeze"]:
        command_line = "python detection/train_net.py --config-file %s --num-gpus 8" \
                       " MODEL.WEIGHTS %s" % (config_dict[config_now], args.F)
        write_slurm_sh_faster("detection_%s_%s" % (config_now, model_name),
                              command_line, queue_name, gpu_memory=True)
elif args.mode==-6:
    #finetune with mocov3 protocol
    for lr in [0.03,0.06,0.1,0.15,0.12]:
        for weight_decay in [0]:
            command_line ="python main_lincls.py -a resnet50 --dist-url 'tcp://localhost:10001' " \
                          "--multiprocessing-distributed --world-size 1 --rank 0  --pretrained='%s' --lr %f --wd %f " \
                          " %s "%(args.F,lr,weight_decay,args.data)
            write_slurm_sh("linear_main_lincls_%s_%.4f_wd_%.8f" % (args.comment, lr,weight_decay),
                           command_line,
                           queue_name)




