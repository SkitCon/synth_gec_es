FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

LABEL author="Amber Converse"
LABEL description="Default container definition for course project. Based on https://parsertongue.org/tutorials/using-the-ua-hpc/#constructing-a-docker-image"

# This will be our default directory for subsequent commands
WORKDIR /app

# Install requirements.txt dependencies
RUN pip install -r requirements.txt

# pandas, scikit-learn, ignite, etc.
RUN conda install -y pandas ignite -c pytorch \
    && pip install -U scikit-learn tensorboardX crc32c soundfile

# SpaCy
RUN conda install -y spacy cupy -c conda-forge
RUN python -m spacy download es_dep_news_trf

# next, let's install huggingface transformers, tokenizers, and the datasets library
# we'll install a specific version of transformers 
# and the latest versions of tokenizers and datasets that are compatible with that version of transformers 
RUN pip install -U transformers==4.17.0 \
    && pip install -U tokenizers datasets
# let's include ipython as a better default REPL
# and jupyter for running notebooks
RUN conda install -y ipython jupyter ipywidgets widgetsnbextension \
    && jupyter nbextension enable --py widgetsnbextension
# let's define a default command for this image.
# We'll just print the version for our PyTorch installation
CMD ["python", "-c" "\"import torch;print(torch.__version__)\""]

# copy executables to path
COPY . ./
RUN chmod u+x  scripts/* \
    && mv scripts/* /usr/local/bin/ \
    && rmdir scripts

# launch jupyter by default
CMD ["/bin/bash", "launch-notebook"]