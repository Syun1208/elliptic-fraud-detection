import time
import os
from pathlib import Path

from src.utils.constants import CONFIG_FILE
from src.utils.utils import load_yaml
from src.utils.logger import Logger


FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[2]


data = load_yaml(path=os.path.join(WORK_DIR, CONFIG_FILE))

register = Logger(
    log_dir=data['logger']['log_dir'],
    log_clear_days=data['logger']['log_clear_days']
)

logger = register.get_tracking(__name__)

def time_complexity(name_process):
    
    def decorator_factory(func): 

        def wraper(*args, **kwargs): 
            t1 = time.time() 
            result = func(*args, **kwargs) 
            t2 = time.time() 

            duration = t2-t1
            logger.info(f'DONE PROCESS {name_process} IN {duration:.4f}s')
            return result 
        
        return wraper 
    
    return decorator_factory