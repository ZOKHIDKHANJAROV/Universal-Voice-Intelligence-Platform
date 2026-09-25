FROM andrius/asterisk:20

USER root

RUN apt-get update \\
    && apt-get install -y --no-install-recommends curl \\
    && rm -rf /var/lib/apt/lists/*

USER asterisk
