# test-whisper-
### whisperx install guide
(Read here for details)[https://github.com/m-bain/whisperX/blob/main/README.md] 
1. pip install git+https://github.com/m-bain/whisperx.git
2. sudo apt update && sudo apt install ffmpeg


### Jupyter notebook setup
1. Installation 
sudo apt update && sudo apt install python3-pip 
2. Password Configuration
python3
```
from notebook.auth import passwd
passwd()
from jupyter_server.auth import passwd
passwd()
```
copy the output 
3. Environment Configuration 
run hostname -I
run jupyter notebook --generate-config 
open generated config file. 
```
c = get_config() 
c.ServerApp.password = u'argon2:$argon2id$v=19$m=10240,t=10,p=생략'
c.ServerApp.ip = 'hotname -I 에서 얻은 값'
c.ServerApp.notebook_dir = '/'
c.ServerApp.port = 8888
```

4. Setting Firewall
run sudo iptables -I INPUT -p tcp -s 0.0.0.0/0 --dport 8888 -j ACCEPT 
run sudo service netfilter-persistent save
run sudo service netfilter-persistent reload

5. Edit the oracle cloud Network rules. to accept http request from anywhere 

6. Running 
run jupyter notebook --allow-root --config=/path/to/jupyter_notebook_config.py