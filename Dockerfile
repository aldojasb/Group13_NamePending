# # Docker file for predicting wine quality
# # Author: Son Chau
# # Date: 12/07/2021

# # use Minimal Jupyter Notebook Stack from https://github.com/jupyter/docker-stacks  
# FROM jupyter/minimal-notebook

# USER root

# # RUN apt-get update --yes && \
# #     apt-get install --yes --no-install-recommends \
# #     gnupg2

# # RUN apt-get install chromium-browser -y

# # RUN sed -i -- 's&deb http://deb.debian.org/debian jessie-updates main&#deb http://deb.debian.org/debian jessie-updates main&g' /etc/apt/sources.list \
# #   && apt-get update && apt-get install wget -y
# # ENV CHROME_VERSION "google-chrome-stable"
# # RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
# #   && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list \
# #   && apt-get update && apt-get install ./google-chrome-stable_current_amd64.deb
# # CMD /bin/bash

# # R pre-requisites
# RUN apt-get update --yes && \
#     apt-get install --yes --no-install-recommends \
#     fonts-dejavu \
#     unixodbc \
#     unixodbc-dev \
#     r-cran-rodbc \
#     chromium-chromedriver \
#     chromium-browser \
#     gfortran \
#     gcc && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*  



# # R packages including IRKernel which gets installed globally.
# RUN mamba install --quiet --yes \
#     'r-base' \
#     'r-devtools' \
#     'r-irkernel' && \
#     mamba clean --all -f -y

# # These packages are not easy to install under arm
# RUN set -x && \
#     arch=$(uname -m) && \
#     if [ "${arch}" == "x86_64" ]; then \
#         mamba install --quiet --yes \
#             'r-rmarkdown' \
#             'r-tidymodels' \
#             'r-tidyverse' && \
#             mamba clean --all -f -y; \
#     fi;


# # # install the kableExtra package using install.packages
# RUN Rscript -e "install.packages('kableExtra',repos = 'http://cran.us.r-project.org')"

# # install google chrome
# # RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
# # RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
# # RUN apt-get -y update
# # RUN apt-get install -y google-chrome-stable

# # # install chromedriver
# # RUN apt-get install -yqq unzip
# # RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip
# # RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/

# # set display port to avoid crash
# # ENV DISPLAY=:99

# # install the anaconda distribution of python
# # Install Python 3 packages
# RUN mamba install --quiet --yes \
#     'ipykernel' \
#     'ipython>=7.15' \
#     'pip' \
#     'scikit-learn>=1.0' \
#     'docopt' \
#     'pandas>=1.3.*'&& \
#     mamba clean --all -f -y 


# RUN conda install -c conda-forge altair_saver
# RUN pip install selenium

# RUN conda install -c conda-forge python-chromedriver-binary
# # RUN npm install -g --force vega-lite vega-cli canvas vega --unsafe-perm=true


# # RUN npm install -g --force vega-lite vega-cli canvas vega

# # additional packages required for altair saver. This is a headache >.>
# # RUN apt-get update && apt-get install -y \
# #     software-properties-common \
# #     npm
# # RUN npm install npm@latest -g && \
# #     npm install n -g && \
# #     n latest 

# # RUN conda install -c conda-forge python-chromedriver-binary 
# # RUN npm install vega-lite
# # RUN npm install canvas
# # RUN conda install -c conda-forge vega-lite-cli
# # RUN npm -g config set user root \
# #  && npm install -g canvas \
# #  && npm install -g vega-lite vega-cli


# USER ${NB_UID}

# WORKDIR "${HOME}"

# # docker build --tag v0.1.0 /$(pwd)
# # docker run --rm -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v "${PWD}":/home/jovyan/work v0.1.0


# Docker file for predicting wine quality
# Author: Son Chau
# Date: 12/07/2021

# use Minimal Jupyter Notebook Stack from https://github.com/jupyter/docker-stacks  
FROM jupyter/minimal-notebook

USER root

# R pre-requisites
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    fonts-dejavu \
    unixodbc \
    unixodbc-dev \
    r-cran-rodbc \
    gfortran \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*  

# R packages including IRKernel which gets installed globally.
RUN mamba install --quiet --yes \
    'r-base' \
    'r-devtools' \
    'r-irkernel' && \
    mamba clean --all -f -y

# These packages are not easy to install under arm
RUN set -x && \
    arch=$(uname -m) && \
    if [ "${arch}" == "x86_64" ]; then \
        mamba install --quiet --yes \
            'r-rmarkdown' \
            'r-tidymodels' \
            'r-tidyverse' && \
            mamba clean --all -f -y; \
    fi;


# # install the kableExtra package using install.packages
RUN Rscript -e "install.packages('kableExtra',repos = 'http://cran.us.r-project.org')"

# install the anaconda distribution of python
# Install Python 3 packages
RUN mamba install --quiet --yes \
    'ipykernel' \
    'ipython>=7.15' \
    'pip' \
    'selenium' \
    'scikit-learn>=1.0' \
    'docopt' \
    'pandas>=1.3.*'&& \
    mamba clean --all -f -y 

RUN apt-get update && apt-get install -y chromium-chromedriver
RUN conda install -c conda-forge altair_saver
RUN npm install -g --force vega-lite vega-cli canvas vega --unsafe-perm=true


USER ${NB_UID}

WORKDIR "${HOME}"

# docker build --tag v0.1.0 /$(pwd)
# docker run --rm -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v "${PWD}":/home/jovyan/work v0.1.0