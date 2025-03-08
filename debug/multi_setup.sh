pkill -9 redis
pkill -9 python3 && ray stop --force
pkill -9 pt_main_thread
# just like this
ray start --head --port 6788
ray start --address=xxxx:6788