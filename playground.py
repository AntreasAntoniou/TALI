# import datasets
# from rich import print

# wit_dataset = datasets.load_dataset(
#     "wikimedia/wit_base",
#     split="train",
#     cache_dir="/data/datasets/tali-wit-2-1-buckets/wit_cache",
# )


# class CustomDataset(wit_dataset.__class__):
#     def __getitem__(self, key):
#         return super().__getitem__(key)


# custom_dataset = CustomDataset(wit_dataset.config)

# # print dataset class full path
# print(custom_dataset.__class__)

# # pretty print all methods available in the dataset
# print(dir(custom_dataset))

# # print all attributes of the dataset
# print(custom_dataset.__dict__)

# # print all attributes of the dataset's first element
# print(custom_dataset[0].__dict__)

from datasets import load_dataset
import torch
import tqdm

# get a list of pokemon from bulbapedia

import requests

def get_pokemon_info(pokemon_url):
    response = requests.get(pokemon_url)

    if response.status_code == 200:
        pokemon = response.json()
        
    else:
        print("Could not retrieve information about Pikachu.")
    return pokemon

from datasets import Dataset
from rich import print
base_url = "https://pokeapi.co/api/v2/pokemon/"
pokemon_ids = [str(i) for i in range(1, 152)]
pokemon_urls = [base_url + pokemon_id for pokemon_id in pokemon_ids]

# print(pokemon_urls)

def gen():
    for url in pokemon_urls:
        yield get_pokemon_info(url)
ds = Dataset.from_generator(gen)
# dataloader = torch.utils.data.DataLoader(ds, batch_size=10)

# for item in ds:
#     print(item)
        