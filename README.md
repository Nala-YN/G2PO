
# Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning

## Run G2PO on Webshop
### Installation
```bash
conda create -n verl-agent-webshop python==3.10 -y
conda activate verl-agent-webshop
cd ./agent_system/environments/env_package/webshop/webshop
chmod +x setup.sh
chmod +x search_engine/run_indexing.sh
./setup.sh -d all
cd ~/g2po
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install -e .
pip3 install vllm==0.8.5
pip3 install jsonlines seaborn schedule openai
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install ray==2.49.2 numpy==1.26.4 click==8.2.1 opentelemetry-exporter-prometheus==0.48b0
```
### Run G2PO
```bash
bash examples/g2po_trainer/run_webshop.sh
```
## Run G2PO on ALFWorld
### Installation
```bash
conda create -n verl-agent-alfworld python==3.12 -y
conda activate verl-agent-alfworld
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install jsonlines seaborn schedule openai
pip3 install -e .
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip3 install vllm==0.8.5 gymnasium==0.29.1 stable-baselines3==2.6.0
pip3 install opentelemetry-exporter-prometheus==0.48b0 ray==2.49.2
pip3 install alfworld
alfworld-download -f
```
### Run G2PO
```bash
bash examples/g2po_trainer/run_alfworld.sh
```
## Acknowledge

Codes implementations are based on  [verl-agent](https://github.com/langfengQ/verl-agent). Thanks for their great contributions!
