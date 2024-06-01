with open('/home/pi/kown_hosts') as kown_f,open('/home/pi/cache_hosts') as cache_f:
    kown_hosts = kown_f.readlines()
    cache_hosts = set(cache_f.readlines())

kown_hosts = [host.split() for host in kown_hosts]

with open('/etc/ansible/hosts','w') as wf:
    wf.writelines([x.split()[1]+"\n" for x in cache_hosts])
