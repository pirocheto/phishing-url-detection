from omegaconf import OmegaConf

your_dict = {"k": "v", "list": {"a": "1", "b": "2", 3: "c"}}
adot_dict = OmegaConf.create(your_dict)

print(adot_dict.k)
print(adot_dict.list.a)
