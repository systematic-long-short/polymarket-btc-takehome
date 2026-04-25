FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /polybench

# Layer heavy deps first for better cache reuse on candidate iteration.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

RUN groupadd --gid 10001 polybench \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin polybench \
    && mkdir -p /submission /output \
    && chown -R polybench:polybench /polybench /submission /output
USER polybench

# Default entrypoint: show the CLI help. Evaluators override CMD at runtime.
ENTRYPOINT ["polybench"]
CMD ["--help"]
