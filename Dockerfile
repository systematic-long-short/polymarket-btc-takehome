FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /polybench

# Layer heavy deps first for better cache reuse on candidate iteration.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Default entrypoint: show the CLI help. Evaluators override CMD at runtime.
ENTRYPOINT ["polybench"]
CMD ["--help"]
