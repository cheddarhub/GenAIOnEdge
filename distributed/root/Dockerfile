FROM ubuntu
USER root
RUN apt-get update
# Update package list and install sudo
RUN apt-get update && apt-get install -y sudo

# Your other commands
RUN sudo apt-get install -y curl
RUN apt-get install -y make g++
RUN apt-get install -y git
RUN git clone https://github.com/b4rtaz/distributed-llama.git
WORKDIR /distributed-llama
RUN make dllama
RUN make dllama-api
RUN apt-get install python3 python3-pip -y
RUN pip3 install requests --break-system-packages
ENV RUN_MODEL=false
RUN echo "N" | python3 launch.py llama3-2-1b-instruct-q40
#RUN python3 launch.py llama3_2_3b_instruct_q40
EXPOSE 9990
CMD ["./dllama-api", "--model", "models/llama3_2_1b_instruct_q40/dllama_model_llama3_2_1b_instruct_q40.m", "--tokenizer", "models/llama3_2_1b_instruct_q40/dllama_tokenizer_llama3_2_1b_instruct_q40.t", "--buffer-float-type", "q80", "--prompt", "'Hello?'", "--steps", "32", "--max-seq-len" , "4096", "--nthreads", "3", "--workers", "172.18.0.2:9998"]
#CMD ["./dllama-api", "--model", "models/llama3_1_8b_instruct_q40/dllama_model_llama3_1_8b_instruct_q40.m", "--tokenizer", "models/llama3_1_8b_instruct_q40/dllama_tokenizer_llama3_1_8b_instruct_q40.t", "--buffer-float-type", "q80", "--prompt", "'Hello?'", "--steps", "32", "--nthreads", "3", "--workers", "172.18.0.2:9998"]
#CMD ["./dllama inference --model models/llama3_1_8b_instruct_q40/dllama_model_llama3_1_8b_instruct_q40.m --tokenizer models/llama3_1_8b_instruct_q40/dllama_tokenizer_llama3_1_8b_instruct_q40.t --buffer-float-type q80 --prompt 'what is the meaning of life?' --steps 256 --nthreads 4 --workers 172.18.0.2:9998"]
