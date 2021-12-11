# -*- coding: utf-8 -*-
# @TIME : 2021/5/13 20:51
# @AUTHOR : Xu Bai
# @FILE : logger_test.py
# @DESCRIPTION :
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='logs/log',
                    filemode='w')
logger = logging.getLogger(__name__)

logger.info('start print log')
logger.debug('debugging')
logger.warning('warning!')
