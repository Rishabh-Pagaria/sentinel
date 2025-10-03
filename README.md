# his
# 1) Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j

# 2) Get a small instruction-tuned model in GGUF (e.g., Llama 3.1 8B Instruct)
# (You can use any 7–9B instruct model you have rights to. Replace paths accordingly.)
# Example filename:
#   llama-3.1-8b-instruct-q4_k_m.gguf
# Put the file here for convenience:
mkdir -p models && mv ~/Downloads/llama-3.1-8b-instruct-q4_k_m.gguf models/
