from anomaly_detection.models.gatconv_encoder import GATConvEncoder
from anomaly_detection.models.gatconv_cvae import GATConvCVAE
from anomaly_detection.models.gatconvlstm_cvae import GATConvLSTMCVAE
from anomaly_detection.models.stconv_cvae import STConvCVAE
from anomaly_detection.models.st_conv_ngat import TGConvNGATEncDec

from anomaly_detection.scripts.train_graph import *


def initialize_model(dataset, args):
    
    model_type = args.model_name.split('_')[1]
    num_features = args.num_features if args.num_features > 0 else dataset.num_features
    num_nodes = dataset[0].num_nodes

    print('expr | num_features: {}/{}'.format(num_features, dataset.num_features)) 
    graph_aux_dim=args.use_aux_attr * dataset[0].graph_attributes.size(-1)
    
    if model_type == 'GATConvEncoder':
        model = GATConvEncoder(in_dim=num_features,
                               h_dim=args.graph_h_dim,
                               out_dim=args.num_classes,
                               num_attn_layers=args.num_attn_layers,
                               graph_aux_dim=graph_aux_dim)
        
    elif model_type == 'GATConvCVAE':
        model = GATConvCVAE(in_dim=num_features,
                            h_dim=args.graph_h_dim,
                            out_dim=args.num_classes,
                            en_h_dims=[int(i) for i in args.en_h_dims.split(',')], 
                            de_h_dims=[int(i) for i in args.de_h_dims.split(',')],
                            num_attn_layers=args.num_attn_layers,
                            graph_aux_dim=graph_aux_dim)
    
    elif model_type == 'GATConvLSTMCVAE':
        model = GATConvLSTMCVAE(in_dim=num_features,
                                lstm_h_dim=args.h_dim,
                                h_dim=args.graph_h_dim,
                                out_dim=args.num_classes,
                                en_h_dims=[int(i) for i in args.en_h_dims.split(',')], 
                                de_h_dims=[int(i) for i in args.de_h_dims.split(',')],
                                num_attn_layers=args.num_attn_layers,
                                graph_aux_dim=graph_aux_dim)
        
    elif model_type == 'GATGConvLSTMCVAE':
        model = GATGConvLSTMCVAE(in_dim=num_features,
                                 h_dim=args.graph_h_dim,
                                 out_dim=args.num_classes,
                                 en_h_dims=[int(i) for i in args.en_h_dims.split(',')], 
                                 de_h_dims=[int(i) for i in args.de_h_dims.split(',')],
                                 num_attn_layers=args.num_attn_layers,
                                 graph_aux_dim=graph_aux_dim)
    
    elif model_type == 'STConvCVAE':
        model = STConvCVAE(num_nodes=num_nodes,
                           in_dim=num_features,
                           conv_h_dim=args.h_dim,
                           h_dim=args.graph_h_dim,
                           out_dim=args.num_classes,
                           en_h_dims=[int(i) for i in args.en_h_dims.split(',')], 
                           de_h_dims=[int(i) for i in args.de_h_dims.split(',')],
                           num_attn_layers=args.num_attn_layers,
                           graph_aux_dim=graph_aux_dim)
        
    elif model_type == 'TGConvNGATEncDec':
        model = TGConvNGATEncDec(num_nodes=num_nodes,
                                 in_dim=num_features,
                                 conv_hidden_channels=args.h_dim*2,
                                 conv_out_channels=args.h_dim,
                                 graph_h_dim=args.graph_h_dim,
                                 out_dim=args.num_classes,
                                 num_attn_layers=args.num_attn_layers,
                                 graph_aux_dim=graph_aux_dim)
        
        
#     elif model_type == 'GATConvLSTM':
#         model = GATConvLSTMEncoder(in_dim=240,
#                                    h_dim=args.h_dim,
#                                    out_dim=args.num_classes,
#                                    num_attn_layers=args.num_attn_layers,
#                                    node_aux_dim=dataset.num_features-240,
#                                    graph_aux_dim=args.use_aux_attr * dataset[0].graph_attributes.size(-1))
    
#     elif model_type == 'GATConvLSTMCVAE' or model_type == 'SemiGATConvLSTMCVAE':
#         model = GATConvLSTMCVAE(in_dim=num_features,
#                                 h_dim=args.h_dim,
#                                 out_dim=args.num_classes,
#                                 en_h_dims=[int(i) for i in args.en_h_dims.split(',')], 
#                                 de_h_dims=[int(i) for i in args.de_h_dims.split(',')], 
#                                 num_attn_layers=args.num_attn_layers,
#                                 graph_aux_dim=args.use_aux_attr * dataset[0].graph_attributes.size(-1))
    
#     elif model_type == 'GATGConvLSTMCVAE' or model_type == 'SemiGATGConvLSTMCVAE':
#         model = GATGConvLSTMCVAE(in_dim=num_features,
#                                  h_dim=args.h_dim,
#                                  out_dim=args.num_classes,
#                                  en_h_dims=[int(i) for i in args.en_h_dims.split(',')], 
#                                  de_h_dims=[int(i) for i in args.de_h_dims.split(',')], 
#                                  num_attn_layers=args.num_attn_layers,
#                                  graph_aux_dim=args.use_aux_attr * dataset[0].graph_attributes.size(-1))
    
#     elif model_type == 'GATConv2LSTMCenterLoss':
#         model = GATConv2LSTMEncoder(in_dim=240,
#                                     h_dim=args.h_dim,
#                                     num_classes=args.num_classes,
#                                     num_attn_layers=args.num_attn_layers,
#                                     node_aux_dim=dataset.num_features-240,
#                                     graph_aux_dim=args.use_aux_attr * dataset[0].graph_attributes.size(-1))

    else:
        print('Model type does not find.')
        raise NotImplementedError
        
    return model


def run_train_func(dataset, train_loader, model, optimizer, loss_func, device, args):

    train_mode = args.model_name.split('_')[0]
    model_type = args.model_name.split('_')[1]
    loss_type = args.model_name.split('_')[2]
    if_cvae = 'CVAE' in model_type

    if train_mode == 'Sup' and not if_cvae and loss_type == 'None':
        loss = train(data_loader=train_loader, 
                     model=model, 
                     optimizer=optimizer, 
                     loss_func=loss_func, 
                     device=device)

    elif train_mode == 'Sup' and not if_cvae and loss_type == 'CenterLoss':
        loss = train_center_loss(data_loader=train_loader, 
                                 model=model, 
                                 optimizer=optimizer, 
                                 loss_func=loss_func, 
                                 device=device)

    elif train_mode == 'Sup' and if_cvae and loss_type == 'None':
        loss = train_cvae(data_loader=train_loader, 
                          model=model, 
                          optimizer=optimizer, 
                          loss_func=loss_func, 
                          device=device)

    elif train_mode == 'Sup' and if_cvae and loss_type == 'CenterLoss':
        loss = train_cvae_center_loss(data_loader=train_loader, 
                                      model=model, 
                                      optimizer=optimizer, 
                                      loss_func=loss_func, 
                                      device=device)
        
    elif train_mode == 'Semi' and if_cvae and loss_type == 'CenterLoss':
        loss = train_semi_cave_center_loss(data_loader=train_loader, 
                                           model=model, 
                                           optimizer=optimizer, 
                                           loss_func=loss_func, 
                                           device=device)
        
    elif train_mode == 'Semi' and not if_cvae and loss_type == 'CenterLoss':
        loss = train_semi_center_loss(data_loader=train_loader, 
                                      model=model, 
                                      optimizer=optimizer, 
                                      loss_func=loss_func, 
                                      device=device)
        
#     elif model_type == 'SemiGATGConvLSTMCVAE' and loss_type == 'CenterLoss':
#         loss = train_semi_gat_cave_center_loss(data_loader=train_loader, 
#                                                model=model, 
#                                                optimizer=optimizer, 
#                                                loss_func=loss_func, 
#                                                device=device)
    else:
        print('Model type does not find.')
        raise NotImplementedError

    return loss