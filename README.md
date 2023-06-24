# BabelNet-Codenames-AI
For Cal Poly senior project, advised by Professor Rodrigo Canaan.

Building off of code and ideas from [this paper](https://www.jair.org/index.php/jair/article/view/12665).

### Note:

Any documentation, code, or comments referring to "old BabelNet" are dealing with code from the above-mentioned paper,
parts of which are still kept in this repo for reference.


## Setup

Follow these instructions to set up the codebase locally.

### 1. Clone the Repo
Run your favorite version of the git clone command on this repo. I prefer:

`git clone git@github.com:gwholland3/BabelNet-Codenames-AI.git`

### 2. Download Required Data
Some of the required data was too large to store on GitHub. 

First,
download the [required data](https://drive.google.com/drive/folders/1TAZjF6-P-ZMr19Xl1t0seg2ap7WpCT_l?usp=sharing).
The downloaded folder should be called `cached_babelnet_subgraphs/`.
Once the folder is downloaded, move it into the `babelnet_bots/data/` folder that's in your local copy of the repo.

Optionally, if you will be running the old BabelNet code,
download the [required data](https://drive.google.com/file/d/1F70CtbsoPPPDnV-ZAUq0i0Rrvtv6taoV/view?usp=sharing) for that as well.
The downloaded file should be called `CodenamesData.zip`.
Once the file is downloaded, unzip it.
Inside the `CodenamesData/` folder, there are two items of interest.
Move the `babelnet_v6/` folder and the `word_to_dict2vec_embeddings` file into the `babelnet_bots/data/` folder
that's in your local copy of the repo.
Then rename the `babelnet_v6/` folder to `old_cached_babelnet_data/`.
Feel free to delete the rest of the downloaded data, as it is not used.

### 3. Install Python
This code was developed and run on Python `3.10.10`, but most likely any version of Python `3.10` will do. 
Make sure you have an appropriate version installed locally.

### 4. Install Requirements
I recommend doing this in a fresh Python virtual environment. 
Cd into the repo and run:

`pip3 install -r requirements.txt`

### Optional Steps:

### 5. Obtain a BabelNet API Key
If you will be using any of the online BabelNet-scraping code or the old BabelNet code, then you will need an API key,
as ordained by BabelNet themselves. 
Follow their [instructions](https://babelnet.org/guide) to register a free account and get your own API key
(click the "KEY & LIMITS" tab and read the "How do I obtain a BabelNet API key?" section).
Your key should look like a long hexadecimal number separated by dashes.
Once you have it,
copy the API key and run the following command from the root of the repo
(making sure to replace `{PASTE_API_KEY_HERE}` with your API key):

`echo -n "{PASTE_API_KEY_HERE}" > babelnet_bots/bn_api_key.txt`

### 6. Modify the BabelNet Docker Image

If you will be using any of the local indices BabelNet-scraping code,
then you will need
to fix a small bug in the Docker image provided by BabelNet that is used to read BabelNet data from the local indices.
After installing the `babelnet` Python package, run the following commands from the root of the repo:

```commandline
docker run -d --name babelnet-rpc babelscape/babelnet-rpc:latest
docker cp babelnet_bots/local_index.py babelnet-rpc:/root/miniconda3/envs/pybabelnet/lib/python3.8/site-packages/babelnet/indices/local_index.py
docker commit babelnet-rpc modified-babelnet-rpc
docker stop babelnet-rpc
docker rm babelnet-rpc
```

This will create a new docker image
called `modified-babelnet-rpc` containing a version of the `local_index.py` file that fixes the bug.
To use this modified image instead of the original one,
add a `--print` flag to the end of the normal `babelnet-rpc start ...` command that the BabelNet instructions tell you to run.
Then,
take the resulting docker command you receive
and change the final command-line argument
from `babelscape/babelnet-rpc:latest` to `modified-babelnet-rpc:latest` before running it.

## Run the Codenames Game
To play a sample game, run: 

`python3 codenames_game.py`

It is preconfigured to use the BabelNet spymaster bot,
but both the spymaster and field operative roles can be set to any bot
that conforms to the interface specified in `codenames_bots.py` by changing the lines of code initializing
`spymaster_bot` and `field_operative_bot` in
`codenames_game.py` (where `None` signifies a human player instead of a bot).
