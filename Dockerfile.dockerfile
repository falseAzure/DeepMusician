FROM pytorch/pytorch:latest

WORKDIR /app

RUN apt-get update && apt-get install -y python3-pip
RUN pip install --upgrade pip

COPY pyproject.toml .

COPY . .

RUN pip install -e .


ENTRYPOINT ["python3","scripts/train.py"]
CMD ["--path", "--n_epochs", "--batch_size"]
