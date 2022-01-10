import sys
sys.path.append('../../')

from models.tgconv_ngat_ae import TGConvNGATAE
from models.tgconv_ngat import TGConvNGAT
from models.tgconv_ngat_cluster import TGConvNGATCluster


def load_model(data, params):
    
    num_features = data.num_features
    num_features += params['use_node_aux_attr'] * data.node_aux_dim
    num_nodes =  data.num_nodes
    
    net_name = params['model_name'].split('_')[0]
    graph_aux_dim = params['use_graph_aux_attr'] * data.graph_aux_dim
    
    if params['model_name'] == 'TGConvNGATAE':
        model = TGConvNGATAE(num_nodes=num_nodes,
                             in_dim=num_features,
                             conv_hidden_channels=params['h_dim'] * 2,
                             conv_out_channels=params['h_dim'])
        
    elif params['model_name'] == 'TGConvNGAT':
        model = TGConvNGAT(num_nodes=num_nodes,
                           in_dim=num_features,
                           conv_hidden_channels=params['h_dim'] * 2,
                           conv_out_channels=params['h_dim'],
                           graph_h_dim=params['graph_h_dim'],
                           out_dim=params['num_classes'])
    
    elif params['model_name'] in ['TGConvNGATCC', 'TGConvNGATCC1']:
        model = TGConvNGAT(num_nodes=num_nodes,
                           in_dim=num_features,
                           conv_hidden_channels=params['h_dim'] * 2,
                           conv_out_channels=params['h_dim'],
                           graph_h_dim=params['graph_h_dim'],
                           out_dim=params['num_classes'])
    
    elif params['model_name'] in ['TGConvNGATCluster', 'TGConvNGATCluster1', 'TGConvNGATCluster2']:
        model = TGConvNGAT(num_nodes=num_nodes,
                           in_dim=num_features,
                           conv_hidden_channels=params['h_dim'] * 2,
                           conv_out_channels=params['h_dim'],
                           graph_h_dim=params['graph_h_dim'],
                           out_dim=params['num_classes'])

#     if net_name == 'GATConvEncoder':
#         model = GATConvEncoder(in_dim=num_features,
#                                h_dim=args.graph_h_dim,
#                                out_dim=args.num_classes,
#                                num_attn_layers=args.num_attn_layers,
#                                graph_aux_dim=graph_aux_dim)
        
#     elif net_name == 'GATConvCVAE':
#         model = GATConvCVAE(in_dim=num_features,
#                             h_dim=args.graph_h_dim,
#                             out_dim=args.num_classes,
#                             en_h_dims=[int(i) for i in args.en_h_dims.split(',')], 
#                             de_h_dims=[int(i) for i in args.de_h_dims.split(',')],
#                             num_attn_layers=args.num_attn_layers,
#                             graph_aux_dim=graph_aux_dim)
    
#     elif net_name == 'GATConvLSTMCVAE':
#         model = GATConvLSTMCVAE(in_dim=num_features,
#                                 lstm_h_dim=args.h_dim,
#                                 h_dim=args.graph_h_dim,
#                                 out_dim=args.num_classes,
#                                 en_h_dims=[int(i) for i in args.en_h_dims.split(',')], 
#                                 de_h_dims=[int(i) for i in args.de_h_dims.split(',')],
#                                 num_attn_layers=args.num_attn_layers,
#                                 graph_aux_dim=graph_aux_dim)
        
#     elif net_name == 'GATGConvLSTMCVAE':
#         model = GATGConvLSTMCVAE(in_dim=num_features,
#                                  h_dim=args.graph_h_dim,
#                                  out_dim=args.num_classes,
#                                  en_h_dims=[int(i) for i in args.en_h_dims.split(',')], 
#                                  de_h_dims=[int(i) for i in args.de_h_dims.split(',')],
#                                  num_attn_layers=args.num_attn_layers,
#                                  graph_aux_dim=graph_aux_dim)
    
#     elif net_name == 'STConvCVAE':
#         model = STConvCVAE(num_nodes=num_nodes,
#                            in_dim=num_features,
#                            conv_h_dim=args.h_dim,
#                            h_dim=args.graph_h_dim,
#                            out_dim=args.num_classes,
#                            en_h_dims=[int(i) for i in args.en_h_dims.split(',')], 
#                            de_h_dims=[int(i) for i in args.de_h_dims.split(',')],
#                            num_attn_layers=args.num_attn_layers,
#                            graph_aux_dim=graph_aux_dim)
        
       
        
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