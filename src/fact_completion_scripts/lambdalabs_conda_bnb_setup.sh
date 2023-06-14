# cuda 118

# jupyter setup
wget http://repo.continuum.io/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh
source ~/.bashrc

conda create --name cap
conda activate cap
conda install pip
conda install cudatoolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

git clone https://github.com/daniel-furman/Polyglot-or-Not.git
cd Polyglot-or-Not
pip install -r requirements.txt
cd ..
pip install -U scipy numpy

git clone https://github.com/timdettmers/bitsandbytes.git
cd bitsandbytes
CUDA_VERSION=118 make cuda11x
python setup.py install
python -m bitsandbytes
# should be successfull built

# llama
#cd ..
#gcloud auth login
#sudo apt-get install gcc python3-dev python3-setuptools
#sudo pip3 uninstall crcmod
#sudo pip3 install --no-cache-dir -U crcmod
#mkdir llama
#gsutil -m cp -r gs://calibragpt/llama/hf llama