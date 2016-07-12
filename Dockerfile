FROM quay.io/openai/openai-runtime:latest

RUN pip install PyYAML

COPY . /experiment
WORKDIR /experiment

CMD ["python", "neural_gpu_trainer.py", "--do_batchnorm=0", "--task", "scopy,sdup", "--progressive_curriculum=True", "--do_outchoice=True"]
