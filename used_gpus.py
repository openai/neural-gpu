"""List GPUs I'm using on this machine"""
import subprocess

def used_by_pid(pid):
    fname = '/proc/%s/environ' % pid
    for line in open(fname).read().split('\x00'):
        if line.split('=')[0] == 'CUDA_VISIBLE_DEVICES':
            return line.split('=')[1].split(',')
    return []

def possible_pids():
    return subprocess.check_output(['pgrep', '-u', 'ecprice', 'python']).split()


def main():
    used = set()
    for pid in possible_pids():
        try:
            used = used.union(used_by_pid(pid))
        except IOError as e:
            continue
    print ' '.join(sorted(used))

if __name__ == '__main__':
    main()
