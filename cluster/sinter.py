
#!/usr/bin/python
import os
import sys
import stat
import subprocess
from datetime import datetime


def get_walltime(params):
    for i, p in enumerate(params):
        if p == "-t" or p == "--time":
            return float(params[i + 1])
        elif p.startswith("-t"):
            return float(p[2:])
        elif p.startswith("--time="):
            return float(p[7:])
    return 60 * 24 * 7.

if __name__ == "__main__":
    params = sys.argv[1:]

    if "-h" in params or "--help" in params:
        print(subprocess.Popen(["srun", "--help"], stdout=subprocess.PIPE).communicate()[0])
        exit(0)

    walltime = get_walltime(params)

    curr_env = os.environ.copy()
    curr_env['LC_ALL'] = "en_CA.UTF-8"
    kticket = " ".join(subprocess.Popen(["klist"], env=curr_env, stdout=subprocess.PIPE).communicate()[0].strip().split('\n')[-1].strip().split(" ")[2:])
    kticket = (datetime.strptime(kticket, '%d/%m/%y %H:%M:%S') - datetime.now()).total_seconds() / 60.

    if walltime > kticket:
        keytab_path = os.path.join(os.environ["HOME"], ".kerb/client.keytab")
        if os.path.exists(keytab_path):
            if bool(os.stat(keytab_path).st_mode & (stat.S_IWGRP | stat.S_IWOTH | stat.S_IRGRP | stat.S_IROTH)):
                print("Unsecure keytab. Change the permission to 600.")
                exit(1)
            ret = subprocess.call(["kinit", "-kt", keytab_path, os.environ["USER"]])
        else:
            print("Your kerberos ticket will expire during your job.\nPlease enter your password now to renew it.\n")
            ret = subprocess.call(["kinit", os.environ["USER"]])

        if ret != 0:
            exit(ret)

    shell = os.readlink('/proc/%d/exe' % os.getppid())
    subprocess.call(["srun", "--qos=unkillable"] + params + ["--pty", shell])
