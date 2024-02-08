huozi3-hf=/path/to/huozi3-hf
huozi3-gguf=/path/to/huozi-gguf/huozi3.gguf
huozi3-gguf-q4_0=/path/to/huozi-gguf/huozi3-q4_0.gguf

python ../llama.cpp/convert.py --outfile "$huozi3_gguf" "$huozi3_hf"
../llama.cpp/quantize "$huozi3_gguf" "$huozi3_gguf_q4_0" q4_0
