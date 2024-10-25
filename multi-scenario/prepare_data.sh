#! /bin/bash


MULTI_SCENARIO_DIR='./data/'

wget -P $MULTI_SCENARIO_DIR'models/' 'https://github.com/ajksabva/remote-sensing/releases/download/v1.0.0-remote-sensing/mae_pretrain_vit_base_full.pth'

wget -P $MULTI_SCENARIO_DIR'models/' 'https://github.com/ajksabva/remote-sensing/releases/download/v1.0.0-remote-sensing/multi-scenario.pth'

wget -P $MULTI_SCENARIO_DIR 'https://github.com/ajksabva/remote-sensing/releases/download/v1.0.0-remote-sensing/MASATI-v2.zip' \
    && unzip -d $MULTI_SCENARIO_DIR $MULTI_SCENARIO_DIR'MASATI-v2.zip' && rm $MULTI_SCENARIO_DIR'MASATI-v2.zip'
