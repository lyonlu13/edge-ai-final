from hqq.core.quantize import BaseQuantizeConfig

# TODO: Make your own quant config for DeiT-S
def get_quant_config_deit(model):
    quant_config = {}
    
    n_blocks = len(model.blocks)
    q_config_8 = BaseQuantizeConfig(nbits=8, group_size=64)
    q_config_4 = BaseQuantizeConfig(nbits=4, group_size=48)
    
    for i in range(n_blocks):
        quant_config[f'blocks.{i}.attn.qkv'] = q_config_4
        quant_config[f'blocks.{i}.attn.proj'] = q_config_8
        quant_config[f'blocks.{i}.mlp.fc1'] = q_config_4
        quant_config[f'blocks.{i}.mlp.fc2'] = q_config_4
        
    return quant_config

# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model):
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    q4_config = BaseQuantizeConfig(nbits=4, group_size=64)  
    q8_config = BaseQuantizeConfig(nbits=8, group_size=64)  
    
    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q8_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q8_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q8_config
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q8_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_config
        
    return quant_config