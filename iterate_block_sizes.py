import sys
import os
from os.path import abspath
import subprocess
from subprocess import Popen, PIPE
from time import sleep

if __name__ == "__main__":
    path = abspath(os.path.dirname(sys.argv[0]))
    N1_min = int(sys.argv[1])
    N1_max = int(sys.argv[2])
    N2_min = int(sys.argv[3])
    N2_max = int(sys.argv[4])
    
    kwargs = { "N": 512, "TRANSPOSE": 1, "TWO_LEVEL": 1}
    for kwarg in sys.argv[5:]:
        key, value = kwarg.split("=")
        if key in kwargs: kwargs[key] = value
    
    print "Path:", path
    
    N1 = [N1_min]
    while N1[-1] < N1_max: 
        N1.append( N1[-1]*2 )
    
    N2 = [N2_min]
    while N2[-1] < N2_max: N2.append( N2[-1]*2 )
    
    for n1 in N1:
        for n2 in N2:
            if n2 >= n1: continue
            with open(path + "/dgemm-blocked.c", 'r') as f:
                lines = f.read().splitlines()
                for i, line in enumerate(lines):
                    if line[:19] == "#define BLOCK_SIZE ":
                        lines[i] = line[:19] + str(n1)
                    if line[:20] == "#define BLOCK_SIZE2 ":
                        lines[i] = line[:20] + str(n2)
                    if line[:18] == "#define TRANSPOSE ":
                        lines[i] = line[:18] + str(kwargs["TRANSPOSE"])
                    if line[:18] == "#define TWO_LEVEL ":
                        lines[i] = line[:18] + str(kwargs["TWO_LEVEL"])
                        
            with open(path + "/dgemm-blocked.c", 'w') as f:
                for i, line in enumerate(lines):
                    if i > 0:  f.write("\n")
                    f.write(line)
            
            p = Popen( ['git clean -f'], cwd=path+"/", shell=True, stdout = PIPE, stderr = PIPE )
            p.wait()
            p = Popen( ['make'], cwd=path+"/", shell=True, stdout = PIPE, stderr = PIPE)
            p.wait()
            p = Popen( ['./benchmark-blocked -n ' + str(kwargs["N"])], cwd=path+"/", shell=True, stdout = PIPE)
            out = p.communicate()
            for line in out[0].splitlines()[2:]:
                print n1, "\t", n2, "\t", line
                sys.stdout.flush()
            
            
