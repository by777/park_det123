FROM nvcr.io/nvidia/pytorch:20.10-py3

# Install linux packages
RUN apt update && apt install -y screen

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true

ENV HOST 0.0.0.0
ENV PORT 5001
ENV DEBUG true

COPY . /api
WORKDIR /api

# install requirements

# RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install  -r requirements.txt


# expose the app port
EXPOSE 5001

#RUN pip install gunicorn

# run the app server in production with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "3", "app:app"]
