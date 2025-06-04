from hqq.core.quantize import BaseQuantizeConfig

# def get_quant_config_slm(model):
#     quant_config = {}
    
#     n_layers = model.config.num_hidden_layers
#     q2_config = BaseQuantizeConfig(nbits=4, group_size=64)  
#     q4_config = BaseQuantizeConfig(nbits=4, group_size=64)  
#     q8_config = BaseQuantizeConfig(nbits=8, group_size=64)  
    
#     for i in range(n_layers):
#         quant_config[f'model.layers.{i}.self_attn.q_proj'] = q4_config
#         quant_config[f'model.layers.{i}.self_attn.k_proj'] = q4_config
#         quant_config[f'model.layers.{i}.self_attn.v_proj'] = q8_config
#         quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_config
        
#         quant_config[f'model.layers.{i}.mlp.gate_proj'] = q8_config
#         quant_config[f'model.layers.{i}.mlp.up_proj'] = q2_config
#         quant_config[f'model.layers.{i}.mlp.down_proj'] = q2_config
        
#     return quant_config

def get_quant_config_slm(model):
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    q2_config = BaseQuantizeConfig(nbits=2, group_size=512)  
    q4_config = BaseQuantizeConfig(nbits=4, group_size=128)  
    q8_config = BaseQuantizeConfig(nbits=8, group_size=64)  
    
    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_config
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_config
        
    return quant_config
