FROM nvcr.io/nvidia/pytorch:25.06-py3

# Install direct dependencies, but not source code.
WORKDIR /stage
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf *

WORKDIR /runtime
