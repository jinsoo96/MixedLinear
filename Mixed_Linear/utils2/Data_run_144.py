import torch
import argparse
from exp.exp_main import Exp_Main

# arg_set 함수 정의
def arg_set(folder_path, data, model_name):
    args = argparse.Namespace(
        # 기본 설정
        is_training = 1,  # 기본값 1
        train_only = False,  # 기본값 False
        model_id = f'{data}_{model_name}',
        model = 'Mixed_Linear2',
        decomp_kernel_sizes=[25, 49],  # 기본값 설정

        # 데이터 로더 설정
        data = 'custom',  # 데이터 이름
        root_path = folder_path,  # 데이터 폴더 경로
        data_path = data,  # 데이터 파일 경로
        features = 'M',  # 예측 작업의 특징 설정 (M: 멀티밴드, S: 단일밴드, MS: 멀티밴드->단일밴드)
        target = '현재수요(MW)',  # 타겟 피처 (컬럼 이름)
        freq = '5min',
        checkpoints = './checkpoints/',

        # 예측 작업 설정
        seq_len = 96,
        
        
        label_len = 96,
        pred_len = 144,  # 향후 6시간의 5분당 예측 전력수요 예측 길이 설정

        individual = False,

        # 모델 설정
        embed_type = 0,
        enc_in = 8,
        dec_in = 7,
        c_out = 7,
        d_model = 16,
        n_heads = 8,
        e_layers = 2,
        d_layers = 1,
        d_ff = 64,
        moving_avg = 30,
        factor = 1,
        distil = True,
        dropout = 0.1,
        embed = 'timeF',
        activation = 'gelu',
        output_attention = False,
        do_predict = True,

        # 최적화 설정
        num_workers = 4,
        itr = 1,
        train_epochs = 5,
        batch_size = 16,
        patience = 2,
        learning_rate = 0.001,
        des = 'Exp',
        loss = 'mse',
        lradj = 'type1',
        use_amp = False,

        # GPU 설정
        use_gpu = True,
        gpu = 0,
        use_multi_gpu = False,
        devices = '0,1,2,3',
        test_flop = False
    )

    # channels 속성 추가
    args.channels = args.enc_in  # enc_in을 channels로 설정

    return args
    # channels 속성 추가
    
    
    


def model_run(args):
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if not args.train_only:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                    args.model,
                                                                                                    args.data,
                                                                                                    args.features,
                                                                                                    args.seq_len,
                                                                                                    args.label_len,
                                                                                                    args.pred_len,
                                                                                                    args.d_model,
                                                                                                    args.n_heads,
                                                                                                    args.e_layers,
                                                                                                    args.d_layers,
                                                                                                    args.d_ff,
                                                                                                    args.factor,
                                                                                                    args.embed,
                                                                                                    args.distil,
                                                                                                    args.des, ii)

        exp = Exp(args)  # set experiments

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
        torch.cuda.empty_cache()