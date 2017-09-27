from __future__ import print_function

import pyslurm

def main():
    try:
        a = pyslurm.job()
        jobs = a.get()
        print(jobs)
    except ValueError as e:
        print("Job list error - {0}".format(e.args[0]))

if __name__=="__main__":
    main()
