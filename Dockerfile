FROM mambaorg/micromamba:1.5.8

WORKDIR /workspace/SurvivalNet-BIO

COPY . .

RUN micromamba create -y -n survivalnet -f environment.yml && \
    micromamba clean -a -y

ENTRYPOINT ["micromamba", "run", "-n", "survivalnet"]
CMD ["snakemake", "--cores", "1"]
