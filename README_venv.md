# io-mamba-rl
Multiple object tracking with Mamba and reinforcement learning using AFO dataset.



To correctly activate project:

rm -rf .venv (ensure no legacy venvs present)

python3.10 -m venv .venv (force python 3.10)
source ./venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

(after reqs installed)
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack

pip install --no-build-isolation -e .

cd ..
git clone https://github.com/JackWoo0831/Mamba_Trackers.git

note: make sure your interpreter sees the project root (interpreter struggles to find subfolders)
use 
export PYTHONPATH=<<project path>> as needed

developer>reload window on vs code can also be needed 