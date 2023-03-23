# BabelNet-Codenames-AI
For senior project.

Building off of code and ideas from [this paper](https://www.jair.org/index.php/jair/article/view/12665).

## Setup

Follow these instructions to set up the codebase locally.

### 1. Clone the Repo
Run your favorite version of the git clone command on this repo. I prefer:

`git clone git@github.com:gwholland3/BabelNet-Codenames-AI.git`

### 2. Download Required Data
Download the [required data](https://drive.google.com/file/d/1F70CtbsoPPPDnV-ZAUq0i0Rrvtv6taoV/view?usp=sharing) that was too large to store on GitHub. The file should be called `CodenamesData.zip`. Once the file is downloaded, unzip it. Inside the `CodenamesData` folder, there are two items of interest. Move the `babelnet_v6` folder and the `word_to_dict2vec_embeddings` file into the `data/` folder that's in your local copy of the repo. Feel free to delete the rest of the downloaded data, as it is not used.

### 3. Install Python
This code was developed and run on Python `3.10.10`, but most likely any version of Python `3.10` will do. Make sure you have an appropriate version installed locally.

### 4. Install Requirements
I recommend doing this in a fresh Python virtual environment. Cd into the repo and run:

`pip3 install -r requirements.txt`

### 5. Obtain a BabelNet API Key
The BabelNet bot needs to query the BabelNet API in order to function, and this requires an API key, as ordained by BabelNet themselves. Follow their [instructions](https://babelnet.org/guide) to register a free account and get your own API key (click the "KEY & LIMITS" tab and read the "How do I obtain a BabelNet API key?" section). Your key should look like a long hexadecimal number separated by dashes. Once you have it, copy the API key and run the following command from the root of the repo (making sure to replace `{PASTE_API_KEY_HERE}` with your API key):

`echo "{PASTE API KEY HERE}" > babelnet_bots/bn_api_key.txt`

## Run the Codenames Game
To play a sample game, run: 

`python3 codenames_game.py`

It is preconfigured to use the BabelNet spymaster bot, but both the spymaster and field operative can be set to any bot that conforms to the interface specified in `codenames_bots.py` by changing the lines of code initializing `spymaster_bot` and `field_operative_bot` in `codenames_game.py` (where `None` signifies a human player instead of a bot).
