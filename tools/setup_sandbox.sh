#!/bin/bash
# One-time setup of a Daytona H100 sandbox. Before it: ~/.ssh/id_ed25519 is a deploy key with write
# access to okc-muse-kernels, ~/.ssh/attn_key a read-only key for okc-muse-attention-decode, and
# ~/.okc_token an okc API token. After it, a person runs `claude` once to log in.
set -e
cd ~
uv venv -q -p 3.12 okc
. okc/bin/activate
uv pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install -q triton==3.8.0 numpy
curl -fsSL https://openkernelchallenge.dev/install.sh | OKC_NO_MODIFY_PATH=1 sh
curl -fsSL https://claude.ai/install.sh | bash
ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
cat >> ~/.ssh/config <<'CFG'
Host github-attn
  HostName github.com
  IdentityFile ~/.ssh/attn_key
  IdentitiesOnly yes
CFG
for n in layer moe proj; do
  git clone -q git@github.com:tech-cosmos/okc-muse-kernels.git r-$n
  git -C r-$n config user.name "okc agent $n"
  git -C r-$n config user.email "shivam@cosmosinfinity.in"
  git -C r-$n config pull.rebase true
done
git clone -q git@github-attn:tech-cosmos/okc-muse-attention-decode.git attn
chmod -w -R attn
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name())"
OKC_TOKEN="$(cat ~/.okc_token)" ~/.okc/bin/okc whoami
echo "export PATH=\$HOME/.local/bin:\$HOME/.okc/bin:\$PATH" >> ~/.bashrc
echo "export OKC_TOKEN=\$(cat ~/.okc_token)" >> ~/.bashrc
