

- 添加更新源

  ```
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  conda config --set show_channel_urls yes
  conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  ```

- 创建虚拟环境

  ```
  conda create -n py3-new python=3.6    # 创建一个python3.6的虚拟环境
  conda create -n py2-new python=2.7    # 创建一个python2.7的虚拟环境
  source activate py3-new               # 进入py3-new的虚拟环境
  source deactivate                     # 退出虚拟环境
  ```

- 更新conda

  ```
  conda update anaconda     #升级anaconda
  conda update conda
  ```

- 安装包

  ```
  conda install tensorflow-gpu=1.8.0
  ```

  

