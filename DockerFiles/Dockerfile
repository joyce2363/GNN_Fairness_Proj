# getting base image ubuntu
FROM ubuntu

MAINTAINER joyce pal <joyceguo2001@gmail.com>

RUN apt-get update

CMB ["echo", "Hello World...! from my first docker image"]

RUN pip install dgl-cu100==0.4.3
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install argparse

