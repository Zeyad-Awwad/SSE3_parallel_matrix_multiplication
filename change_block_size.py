import sys
import os
from os.path import abspath

if __name__ == "__main__":
    path = abspath(os.path.dirname(sys.argv[0]))
    N1 = sys.argv[1]
    N2 = sys.argv[2]
    
    with open(path + "/dgemm-blocked.c", 'r') as f:
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
            if line[:19] == "#define BLOCK_SIZE ":
                lines[i] = line[:19] + N1
            if line[:20] == "#define BLOCK_SIZE2 ":
                lines[i] = line[:20] + N2
    with open(path + "/dgemm-blocked.c", 'w+') as f:
        for i, line in enumerate(lines):
            if i > 0:  f.write("\n")
            f.write(line)
            
            