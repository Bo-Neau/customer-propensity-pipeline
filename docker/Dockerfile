FROM apache/airflow:3.2.2

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    # The JDK installs to an arch-specific path (java-17-openjdk-amd64 on
    # x86_64, java-17-openjdk-arm64 on Apple Silicon). Symlink a stable path
    # so JAVA_HOME is correct on both, otherwise PySpark's $JAVA_HOME/bin/java
    # lookup fails and the Spark JVM gateway never starts.
    && ln -s "/usr/lib/jvm/java-17-openjdk-$(dpkg --print-architecture)" /usr/lib/jvm/default-java

ENV JAVA_HOME=/usr/lib/jvm/default-java

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
