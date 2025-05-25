@staticmethod
def _lecun_normal_reinit(layer: Union[nn.Linear, nn.Conv2d, nn.LayerNorm], mask: torch.Tensor) -> None:
    """Partially re-initializes a layer using Lecun normal."""                                                             
    if isinstance(layer, nn.LayerNorm):                                                                                    
        layer.weight.data[mask] = torch.ones_like(layer.weight.data[mask])                                                 
        layer.bias.data[mask] = torch.zeros_like(layer.bias.data[mask])                                                    
        return                                                                                                             
                                                                                                                            
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)                                                        
    variance = 1.0 / fan_in                                                                                                
    stddev = math.sqrt(variance) / 0.87962566103423978                                                                     
                                                                                                                            
    # Reset weights                                                                                                        
    with torch.no_grad():                                                                                                  
        layer.weight[mask] = nn.init._no_grad_trunc_normal_(                                                               
            layer.weight[mask], mean=0.0, std=1.0, a=-2.0, b=2.0                                                           
        )                                                                                                                  
        layer.weight[mask] *= stddev                                                                                       
                                                                                                                            
        # Reset bias if it exists                                                                                          
        if layer.bias is not None:                                                                                         
            layer.bias.data[mask] = 0.0                                                                                    
                                                                                                                            
def _reset_adam_moments(self, reset_masks) -> None:                                                                        
    """Resets the moments of the Adam optimizer for dormant neurons."""                                                    
    assert isinstance(self.optimizer, optim.Adam), "Moment resetting currently only supported for Adam optimizer"          
                                                                                                                            
    # 获取所有参数列表（假设所有参数都在第一个参数组中）                                                                   
    params = self.optimizer.param_groups[0]['params']                                                                      
                                                                                                                            
    for layer_idx, mask in enumerate(reset_masks):                                                                         
        try:                                                                                                               
            # 获取当前层的权重和偏置参数索引                                                                               
            weight_idx = 2 * layer_idx                                                                                     
            bias_idx = 2 * layer_idx + 1                                                                                   
            next_weight_idx = 2 * (layer_idx + 1)                                                                          
                                                                                                                            
            # 当前层的权重参数                                                                                             
            weight_param = params[weight_idx]                                                                              
            weight_state = self.optimizer.state[weight_param]                                                              
                                                                                                                            
            # 重置权重参数的一阶/二阶矩                                                                                    
            weight_state["exp_avg"][mask, ...] = 0.0                                                                       
            weight_state["exp_avg_sq"][mask, ...] = 0.0                                                                    
            weight_state["step"] = 0                                                                                       
                                                                                                                            
            # 当前层的偏置参数（如果存在）                                                                                 
            if bias_idx < len(params):                                                                                     
                bias_param = params[bias_idx]                                                                              
                if bias_param in self.optimizer.state:                                                                     
                    bias_state = self.optimizer.state[bias_param]                                                          
                    bias_state["exp_avg"][mask] = 0.0                                                                      
                    bias_state["exp_avg_sq"][mask] = 0.0                                                                   
                    bias_state["step"] = 0                                                                                 
                                                                                                                            
            # 下一层的权重参数（输出连接）                                                                                 
            if next_weight_idx < len(params):                                                                              
                next_weight_param = params[next_weight_idx]                                                                
                next_weight_state = self.optimizer.state[next_weight_param]                                                
                                                                                                                            
                # 处理卷积层到线性层的转换                                                                                 
                if len(weight_state["exp_avg"].shape) == 4 and len(next_weight_state["exp_avg"].shape) == 2:               
                    num_repetition = next_weight_state["exp_avg"].shape[1] // mask.shape[0]                                
                    linear_mask = torch.repeat_interleave(mask, num_repetition)                                            
                    next_weight_state["exp_avg"][:, linear_mask] = 0.0                                                     
                    next_weight_state["exp_avg_sq"][:, linear_mask] = 0.0                                                  
                else:                                                                                                      
                    # 标准情况（同类型层间连接）                                                                           
                    next_weight_state["exp_avg"][:, mask, ...] = 0.0                                                       
                    next_weight_state["exp_avg_sq"][:, mask, ...] = 0.0                                                    
                next_weight_state["step"] = 0                                                                              
                                                                                                                            
        except (IndexError, KeyError) as e:                                                                                
            print(f"Warning: Layer {layer_idx} parameter not found in optimizer state")                                    
            continue

def _reset_dormant_neurons(self, model: nn.Module, redo_masks: List[torch.Tensor]) -> nn.Module:                           
        """Re-initializes the weights of dormant neurons."""                                                                   
        # 只获取Conv2d和Linear层                                                                                               
        layers = [(name, layer) for name, layer in model.named_modules()                                                       
                 if isinstance(layer, (nn.Conv2d, nn.Linear))]                                                                 
                                                                                                                               
                                                                                                                               
        assert len(redo_masks) == len(layers) - 1, (                                                                           
            f"Number of masks ({len(redo_masks)}) must match number of layers-1 ({len(layers)-1})"                             
        )                                                                                                                      
                                                                                                                               
        # Reset ingoing weights                                                                                                
        with torch.no_grad():                                                                                                  
            for i in range(len(layers)-1):                                                                                     
                mask = redo_masks[i]                                                                                           
                layer = layers[i][1]                                                                                           
                next_layer = layers[i + 1][1]                                                                                  
                                                                                                                               
                # Skip if no dead neurons                                                                                      
                if torch.all(~mask):                                                                                           
                    continue                                                                                                   
                                                                                                                               
                # Reset weights using specified initialization                                                                 
                if self.use_lecun_init:                                                                                        
                    self._lecun_normal_reinit(layer, mask)                                                                     
                else:                                                                                                          
                    self._kaiming_uniform_reinit(layer, mask)                                                                  
                                                                                                                               
        return model