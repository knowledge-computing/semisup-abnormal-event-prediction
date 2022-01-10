import sys
sys.path.append('../../')

from trainers.tgconv_ngat_trainer import TGConvNGATTrainer
from trainers.tgconv_ngat_ae_trainer import TGConvNGATAETrainer
from trainers.tgconv_ngat_cc_trainer import TGConvNGATCCTrainer
from trainers.tgconv_ngat_cc1_trainer import TGConvNGATCC1Trainer
from trainers.tgconv_ngat_cluster_trainer import TGConvNGATClusterTrainer


def load_trainer(data, model, device, params):
    
    if params['model_name'] == 'TGConvNGATAE':
        trainer = TGConvNGATAETrainer(data=data, 
                                      model=model, 
                                      device=device, 
                                      epochs=params['epochs'],
                                      lr=params['lr'],
                                      lr_milestones=(40,80),
                                      weight_decay=params['weight_decay'],
                                      res_path=params['res_path'],)
        
    elif params['model_name'] == 'TGConvNGAT':
        trainer = TGConvNGATTrainer(data=data, model=model, 
                                    device=device, 
                                    eta=params['eta'],
                                    epochs=params['epochs'],
                                    lr=params['lr'],
                                    weight_decay=params['weight_decay'],
                                    patience=params['patience'],
                                    predict_func=params['predict_func'],
                                    res_path=params['res_path'],)      
        
    elif params['model_name'] == 'TGConvNGATCC':
        trainer = TGConvNGATCCTrainer(data=data, model=model, 
                                      device=device, 
                                      epochs=params['epochs'],
                                      lr=params['lr'],
                                      lr_center=params['lr_center'],
                                      weight_decay=params['weight_decay'],
                                      patience=params['patience'],
                                      predict_func=params['predict_func'],
                                      res_path=params['res_path'],)        
        
    elif params['model_name'] == 'TGConvNGATCC1':
        trainer = TGConvNGATCC1Trainer(data=data, model=model, 
                                      device=device, 
                                      epochs=params['epochs'],
                                      lr=params['lr'],
                                      lr_center=params['lr_center'],
                                      weight_decay=params['weight_decay'],
                                      patience=params['patience'],
                                      predict_func=params['predict_func'],
                                      res_path=params['res_path'],)     
        
    elif params['model_name'] == 'TGConvNGATCluster':
        trainer = TGConvNGATClusterTrainer(data=data, model=model, 
                                           device=device, 
                                           epochs=params['epochs'],
                                           lr=params['lr'],
                                           lr_center=params['lr_center'],
                                           weight_decay=params['weight_decay'],
                                           patience=params['patience'],
                                           predict_func=1,
                                           res_path=params['res_path'],
                                           eta=params['eta'])       
   
    else:
        print('Model type does not find.')
        raise NotImplementedError
        
    return trainer