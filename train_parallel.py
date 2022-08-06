# Usage : python train_parallel.py --parallel True/False --f_name $f_name --ndim $ndim

from multiprocessing import Process
from functools import partial
import subprocess

from argparse import ArgumentParser

from src.utils import bool_flag

SCRIPT_PATH="./train.sh"

result = subprocess.run('chmod +x train.sh', shell=True, capture_output=True, text=True)
print(result)

def run_train(f_name, ndim, train_pct, weight_decay, lr, dropout, opt, random_seed):
    group_name=f"{f_name}:ndim={ndim}-tdf={train_pct}-wd={weight_decay}-lr={lr}-d={dropout}-opt={opt}"
    print("Start Group name %s"%group_name)
    print(f"Random seed : {random_seed}")

    command=f"{SCRIPT_PATH} {f_name} {ndim} {train_pct} {weight_decay} {lr} {dropout} {opt} {random_seed}"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    stdoutdata, _ = process.communicate()

    if process.returncode != 0 :
        print("Error %s"%group_name)
    else :
        print("Success %s"%group_name)

    print("Finish Group name %s"%group_name)

    output = stdoutdata.decode("utf-8")
    print("*"*10)
    print(output)
    print("*"*10,"\n")

    #return stdoutdata

if __name__ == '__main__':
    parser = ArgumentParser(description="Grokking for MLP")
    parser.add_argument("--parallel", type=bool_flag, default=False)
    parser.add_argument("--f_name", type=str)
    parser.add_argument("--ndim", type=int, default=2)
    params = parser.parse_args()
    parallel = params.parallel
    f_name = params.f_name
    ndim = params.ndim

    all_process = []    
    for train_pct in [80] :
        for weight_decay in [0.0] :
            for lr in [0.001] : 
                for dropout in [0.0] : 
                    for opt in ["adam"] :
                        for random_seed in [0, 100] :
                            if not parallel : 
                                run_train(f_name, ndim, train_pct, weight_decay, lr, dropout, opt, random_seed)
                            else :
                                task = partial(run_train, f_name, ndim, train_pct, weight_decay, lr, dropout, opt, random_seed)
                                p = Process(target=task)
                                p.start()
                                all_process.append(p)
            
    for p in all_process : p.join()