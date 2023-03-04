FROM nvcr.io/nvidia/pytorch:21.04-py3

# install git and clone the repo
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/lucafrost/batchEntailECS.git

# install requirements
WORKDIR batchEntailECS
RUN pip install -r requirements.txt

# print a message to the console
RUN echo "yuhh updated the dockerfile !!"

# run the script
CMD ["python", "BatchEntail.py"]