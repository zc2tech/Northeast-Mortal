# Northeast-Mortal

It's a fork project of Mortal porject linked below, the main change is adopting rules for Northeast-Mahjong whose rule is popular for Northeast of China.

[GitHub Mortal project](https://github.com/Equim-chan/Mortal)

This project is for training AI model for [Northeast-Mahjong](https://github.com/zc2tech/Northeast-Mahjong) 

# configure
copy from config.example.toml to a file named config.toml.
Then change the settings you want.

# training
Use below command, for training from start, use '--reset' command parameter.
In windows/linux, maybe the python command is python, not python3

```python
cd mortal
python3 train.py
```
# start inference server:
```python
cd server
python3 inference_server.py
```