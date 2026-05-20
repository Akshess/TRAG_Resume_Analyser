FROM Python 3.12.0
WORKDIR /common_rag_try
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ..
CMD ["python","main.py"]