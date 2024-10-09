#!/bin/bash

USER_HOME=~/lexu


sudo apt-get update
sudo apt-get install -y ca-certificates \
                   cmake \
                   doxygen \
                   libboost-all-dev \
                   libcurl4-openssl-dev \
                   libgflags-dev \
                   libgoogle-glog-dev \
                   libgrpc-dev \
                   libgrpc++-dev \
                   libmpich-dev \
                   libprotobuf-dev \
                   libssl-dev \
                   libunwind-dev \
                   libz-dev \
                   protobuf-compiler-grpc \
                   python3-pip \
                   wget \
                   ninja-build

sudo rm -rf /lib/x86_64-linux-gnu/libprotoc.a \
            /lib/x86_64-linux-gnu/libprotobuf.a \
            /lib/x86_64-linux-gnu/libprotobuf-lite.a \
            /lib/x86_64-linux-gnu/libprotobuf.so.23 \
            /lib/x86_64-linux-gnu/libprotobuf.so.23.0.4

sudo ldconfig

# we have to build arrow from source to use the system-wide protobuf
cd $USER_HOME
git clone https://github.com/apache/arrow.git
cd arrow/cpp && git checkout apache-arrow-16.1.0 && mkdir build-release && cd build-release
cmake --preset ninja-release-python -DCMAKE_INSTALL_PREFIX=/usr/ -DProtobuf_PROTOC_LIBRARY=/lib/x86_64-linux-gnu/libprotoc.so.32 ..
cmake --build .
sudo ninja install

sudo pip3 install cython
cd $USER_HOME/arrow/python
sudo python3 setup.py install

cd $USER_HOME
git clone https://github.com/happyandslow/v6d
cd v6d && git checkout 1e02fcea287279330249ec87d4c1624ebb599ae2 && git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DUSE_STATIC_BOOST_LIBS=OFF \
         -DBUILD_VINEYARD_SERVER=ON \
         -DBUILD_VINEYARD_CLIENT=OFF \
         -DBUILD_VINEYARD_PYTHON_BINDINGS=ON \
         -DBUILD_VINEYARD_PYPI_PACKAGES=OFF \
         -DBUILD_VINEYARD_LLM_CACHE=ON \
         -DBUILD_VINEYARD_BASIC=OFF \
         -DBUILD_VINEYARD_GRAPH=OFF \
         -DBUILD_VINEYARD_IO=OFF \
         -DBUILD_VINEYARD_HOSSEINMOEIN_DATAFRAME=OFF \
         -DBUILD_VINEYARD_TESTS=ON \
         -DBUILD_VINEYARD_TESTS_ALL=OFF \
         -DBUILD_VINEYARD_PROFILING=OFF \
         -USE_CUDA=ON
make -j
make vineyard_llm_python -j
sudo make install

# sudo pip3 install cython
# cd $USER_HOME/v6d
# sudo python3 setup.py install
# sudo python3 setup_llm.py install
#########################################################
ETCD_VER=v3.4.33

# choose either URL
GOOGLE_URL=https://storage.googleapis.com/etcd
GITHUB_URL=https://github.com/etcd-io/etcd/releases/download
DOWNLOAD_URL=${GOOGLE_URL}

rm -f /tmp/etcd-${ETCD_VER}-linux-amd64.tar.gz
rm -rf /tmp/etcd-download-test && mkdir -p /tmp/etcd-download-test

curl -L ${DOWNLOAD_URL}/${ETCD_VER}/etcd-${ETCD_VER}-linux-amd64.tar.gz -o /tmp/etcd-${ETCD_VER}-linux-amd64.tar.gz
tar xzvf /tmp/etcd-${ETCD_VER}-linux-amd64.tar.gz -C /tmp/etcd-download-test --strip-components=1
rm -f /tmp/etcd-${ETCD_VER}-linux-amd64.tar.gz

/tmp/etcd-download-test/etcd --version
/tmp/etcd-download-test/etcdctl version

# start a local etcd server
/tmp/etcd-download-test/etcd 1>/dev/null 2>&1 &

# clear
# ETCDCTL_API=3 /tmp/etcd-download-test/etcdctl del "" --from-key=true

##############################################

# start a vineyard server
$USER_HOME/v6d/build/bin/vineyardd --socket /tmp/vineyard_test.sock 1>out.log 2>&1 &
#  build/bin/vineyardd --socket /tmp/vineyard_test.sock --etcdctl_cmd /tmp/etcd-download-test/etcdctl 1>out.log 2>&1 &

#build and install vllm 
cd $USER_HOME && git clone https://github.com/aibrix/vllm.git
cd vllm && git checkout lexu/vineyard-adptation 
pip3 install -e .

#upgrade pyarrow
python -m pip install pyarrow --upgrade
# build vineyard vllm
cd $USER_HOME/v6d && sudo python3 setup.py install && sudo python3 setup_llm.py install
source $USER_HOME/v1_config.sh
#source $USER_HOME/v2_config.sh

##############################################

# #start vllm
# cd $USER_HOME/vllm
# python3 -m vllm.entrypoints.openai.api_server --model=facebook/opt-125m --enable-chunked-prefill

# # export VLLM_LOGGING_LEVEL=DEBUG for debugging purposes

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "facebook/opt-125m",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
}'

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "facebook/opt-125m",
    "prompt": "write a 100% unique, creative and Human-written article in English for the Keyword \"An article on psychology\". The article should include Creative Title (should be in H1 Heading), SEO meta description, Introduction, headings (Shoud be h2 heading), sub headings (h3, h4,h5,h6 headings), bullet points or Numbered list (if needed), faqs and conclusion. Make sure article is plagiarism free. The minimum length of the article should be 800 words. Don'\''t forget to use question mark \(?\) at the end of questions. Try not to change the original An article on psychology while writing the Title. Try to use The \"An article on psychology\" 2-3 times in article. try to include An article on psychology in headings as well. write a content which can easily pass ai detection tools test.",
    "max_tokens": 7,
    "temperature": 0
}'