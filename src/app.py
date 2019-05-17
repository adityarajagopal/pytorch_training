from checkpointing import Checkpointer
import param_parser as pp

class Application(object):
    
    def __init__(self, configFile):
        self.params = pp.parse_config_file(configFile)


        
        # self.Checkpointer = 
