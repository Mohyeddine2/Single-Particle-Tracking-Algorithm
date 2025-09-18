FROM ubuntu:latest
LABEL authors="mohye"

ENTRYPOINT ["top", "-b"]