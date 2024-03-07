git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
python setup.py install
cd ..
pip install xatlas
pip install imageio
# Refer https://kaolin.readthedocs.io/en/latest/notes/installation.html
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install pymeshlab==2023.12
conda install libffi==3.3