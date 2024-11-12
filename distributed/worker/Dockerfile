FROM ubuntu
RUN apt-get update
RUN apt-get install -y make g++
RUN apt-get install -y git
RUN git clone https://github.com/b4rtaz/distributed-llama.git
WORKDIR /distributed-llama
RUN make dllama
RUN make dllama-api
RUN apt-get update && apt-get install -y sudo 
EXPOSE 9998
CMD ["./dllama", "worker", "--port", "9998", "--nthreads", "4"]
